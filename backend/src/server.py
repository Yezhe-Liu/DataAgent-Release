import asyncio
import contextlib
import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator
from urllib.parse import urlsplit

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from langserve import add_routes
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

try:
    import redis.asyncio as redis_async
except Exception:
    redis_async = None

from src.auth_store import (
    AuthUser,
    authenticate_user,
    auth_store_status,
    create_auth_session,
    create_user,
    delete_auth_session,
    get_user_by_session_token,
    initialize_auth_store,
    is_auth_store_available,
)
from src.agent import get_agent_graph, get_tool_display_name, get_tool_runtime_origin
from src.config import (
    get_auth_settings,
    get_env_bool,
    get_env_float,
    get_env_int,
    get_env_text,
    get_model_settings_summary,
)
from src.data_manager import (
    calculate_correlation,
    get_data_preview,
    get_dataframe,
    load_csv_file,
)
from src.rag_engine import (
    ensure_knowledge_base_loaded,
    format_retrieval_hits,
    get_knowledge_base_stats,
    rebuild_knowledge_base,
    retrieve_knowledge,
)
from src.runtime_context import activate_user_context, reset_user_context

# 1. 强制加载环境变量 (确保 API Key 能被读取)
load_dotenv(override=True)

BASE_DIR = Path(__file__).resolve().parent.parent
TEMP_DIR = BASE_DIR / "temp_data"
USER_TEMP_DIR = TEMP_DIR / "users"
STATIC_DIR = BASE_DIR / "static"
IMAGES_DIR = STATIC_DIR / "images"

MAX_SESSION_MESSAGES = get_env_int("MAX_SESSION_MESSAGES", 16)

_GRAPH_NODE_DISPLAY = {
    "router": "分析问题意图",
    "rewrite": "多角度重写查询",
    "retrieve": "检索知识库",
    "grade": "评估文档相关性",
    "web_search": "外网搜索补充",
    "text_to_sql": "AI 生成 SQL 查询",
    "execute_sql_tool": "执行数据库查询",
    "tool_execute": "执行数据分析",
    "generate": "生成回答",
    "hallucination_check": "事实核查验证",
}
SESSION_MEMORY_BACKEND = get_env_text("SESSION_MEMORY_BACKEND", "redis").lower()
REDIS_URL = get_env_text("REDIS_URL", "redis://127.0.0.1:6379/0")
REDIS_USERNAME = get_env_text("REDIS_USERNAME", "")
REDIS_PASSWORD = get_env_text("REDIS_PASSWORD", "")
REDIS_SESSION_PREFIX = get_env_text("REDIS_SESSION_PREFIX", "dataagent:chat:session:")
REDIS_SESSION_TTL_SECONDS = get_env_int("REDIS_SESSION_TTL_SECONDS", 86400)
AUTH_SETTINGS = get_auth_settings()

SESSION_MEMORY: dict[str, list[dict[str, Any]]] = {}
SESSION_SUMMARIES: dict[str, dict[str, str]] = {}
REDIS_CLIENT: Any | None = None
SESSION_MEMORY_RUNTIME_BACKEND = "memory"
REDIS_SESSION_META_KEY = f"{REDIS_SESSION_PREFIX}__meta__"


def _build_redis_client_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {"decode_responses": True}
    if REDIS_USERNAME:
        kwargs["username"] = REDIS_USERNAME
    if REDIS_PASSWORD:
        kwargs["password"] = REDIS_PASSWORD
    return kwargs


def _get_redis_display_url() -> str:
    parsed = urlsplit(REDIS_URL)
    scheme = parsed.scheme or "redis"
    host = parsed.hostname or parsed.netloc or "localhost"
    port = f":{parsed.port}" if parsed.port else ""
    path = parsed.path or ""
    query = f"?{parsed.query}" if parsed.query else ""
    fragment = f"#{parsed.fragment}" if parsed.fragment else ""
    username = REDIS_USERNAME or (parsed.username or "")
    password = REDIS_PASSWORD or (parsed.password or "")

    if username and password:
        auth = f"{username}:***@"
    elif username:
        auth = f"{username}@"
    elif password:
        auth = ":***@"
    else:
        auth = ""

    return f"{scheme}://{auth}{host}{port}{path}{query}{fragment}"


# =============================================================================
# 2. 初始化 FastAPI 应用
# =============================================================================
@asynccontextmanager
async def lifespan(_app: FastAPI):
    global REDIS_CLIENT, SESSION_MEMORY_RUNTIME_BACKEND
    USER_TEMP_DIR.mkdir(parents=True, exist_ok=True)

    if SESSION_MEMORY_BACKEND == "redis":
        if redis_async is None:
            SESSION_MEMORY_RUNTIME_BACKEND = "memory"
            print("[SessionMemory] redis package unavailable. Using in-memory backend.")
        else:
            try:
                REDIS_CLIENT = redis_async.from_url(REDIS_URL, **_build_redis_client_kwargs())
                await REDIS_CLIENT.ping()
                SESSION_MEMORY_RUNTIME_BACKEND = "redis"
                print(f"[SessionMemory] Redis connected at {_get_redis_display_url()}")
            except Exception as error:
                REDIS_CLIENT = None
                SESSION_MEMORY_RUNTIME_BACKEND = "memory"
                print(f"[SessionMemory] Redis unavailable, using in-memory backend: {error}")
    else:
        SESSION_MEMORY_RUNTIME_BACKEND = "memory"
        print("[SessionMemory] Using in-memory backend.")

    auth_ready, auth_error = initialize_auth_store()
    if auth_ready:
        print("[Auth] MySQL auth store ready.")
    else:
        print(f"[Auth] MySQL auth store unavailable: {auth_error}")

    ensure_knowledge_base_loaded()
    auto_rebuild = get_env_bool("AUTO_REBUILD_KB_ON_STARTUP", False)
    if auto_rebuild:
        result = rebuild_knowledge_base(reset=False)
        print(f"[KB] startup rebuild result: {result}")

    yield

    if REDIS_CLIENT is not None:
        try:
            close_method = getattr(REDIS_CLIENT, "aclose", None)
            if callable(close_method):
                await close_method()
            else:
                await REDIS_CLIENT.close()
        except Exception as error:
            print(f"[SessionMemory] Failed to close Redis client: {error}")
        finally:
            REDIS_CLIENT = None


app = FastAPI(
    title="Data Agent Backend",
    version="2.0",
    description="Agentic RAG + Data Analysis Backend",
    lifespan=lifespan,
)


# =============================================================================
# 3. 配置 CORS
# =============================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_origin_regex=r"https://.*\.figma\.site|http://localhost.*|http://127\.0\.0\.1.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# 4. 挂载静态文件
# =============================================================================
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# =============================================================================
# 5. 请求模型
# =============================================================================
class CorrelationRequest(BaseModel):
    col1: str
    col2: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = None


class RebuildKnowledgeBaseRequest(BaseModel):
    reset: bool = True


# =============================================================================
# 6. 内部辅助函数
# =============================================================================
def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            elif isinstance(item, str) and item.strip():
                parts.append(item.strip())
        return "\n".join(parts)

    if content is None:
        return ""
    return str(content)


def _message_content_to_stream_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)

    if content is None:
        return ""
    return str(content)


def _extract_stream_chunk_text(chunk: Any) -> str:
    content = getattr(chunk, "content", None)
    if isinstance(chunk, dict):
        content = chunk.get("content", content)

    text = _message_content_to_stream_text(content)

    # DeepSeek V4 Pro thinking 阶段: content 为空但 reasoning_content 有 token
    # 不提取会导致前端流式卡顿（几秒无 token 输出）
    if not text:
        additional_kwargs = getattr(chunk, "additional_kwargs", None) or {}
        if isinstance(chunk, dict):
            additional_kwargs = chunk.get("additional_kwargs", additional_kwargs) or {}
        reasoning = additional_kwargs.get("reasoning_content", "")
        if reasoning:
            text = str(reasoning)

    return text


def _extract_ai_message_text(payload: dict[str, Any]) -> str:
    # 1. 优先从新 graph 的 generation 字段提取
    generation = payload.get("generation")
    if isinstance(generation, str) and generation.strip():
        return generation.strip()

    # 2. 从 messages 中提取 AI 消息
    candidate_arrays = [
        payload.get("messages"),
        payload.get("output", {}).get("messages") if isinstance(payload.get("output"), dict) else None,
    ]

    for messages in candidate_arrays:
        if not isinstance(messages, list):
            continue

        for item in reversed(messages):
            msg_type = getattr(item, "type", None)
            content = getattr(item, "content", None)

            if isinstance(item, dict):
                msg_type = item.get("type", msg_type)
                content = item.get("content", content)

            if msg_type in {"ai", "assistant"}:
                text = _message_content_to_text(content).strip()
                if text:
                    return text

    fallback = payload.get("output_text") or payload.get("answer")
    return _message_content_to_text(fallback).strip()


def _trim_history(history: list[dict[str, str]]) -> list[dict[str, str]]:
    if len(history) <= MAX_SESSION_MESSAGES:
        return history
    return history[-MAX_SESSION_MESSAGES:]


def _session_memory_key(user_id: str, session_id: str) -> str:
    return f"{user_id}:{session_id}"


def _redis_session_key(user_id: str, session_id: str) -> str:
    return f"{REDIS_SESSION_PREFIX}{user_id}:{session_id}"


def _redis_session_meta_key(user_id: str) -> str:
    return f"{REDIS_SESSION_PREFIX}{user_id}:__meta__"


def _normalize_history(history: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for item in history:
        msg_type = item.get("type")
        if msg_type == "assistant":
            msg_type = "ai"
        content = item.get("content")
        if msg_type in {"human", "ai"} and isinstance(content, str):
            normalized.append({"type": msg_type, "content": content})
    return normalized


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truncate_text(value: str, max_length: int) -> str:
    text = value.strip()
    if len(text) <= max_length:
        return text
    return f"{text[: max_length - 1]}…"


def _stringify_status_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        for key in ("query", "message", "question", "input", "py_code", "code", "fname", "output", "result"):
            if key not in value:
                continue
            text = _stringify_status_value(value.get(key))
            if text:
                return text
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)
    if isinstance(value, list):
        return " / ".join(filter(None, (_stringify_status_value(item) for item in value[:3])))
    return _message_content_to_text(value)


def _build_status_detail(value: Any, max_length: int = 140) -> str:
    text = " ".join(_stringify_status_value(value).split())
    if not text:
        return ""
    return _truncate_text(text, max_length)


def _build_tool_status_title(tool_name: str, tool_origin: str, status: str) -> str:
    if tool_name == "retrieve_knowledge":
        return {
            "running": "正在检索知识库/数据库",
            "success": "知识库/数据库检索完成",
            "error": "知识库/数据库检索失败",
        }.get(status, "知识库/数据库检索")
    if tool_name == "knowledge_base_status":
        return {
            "running": "正在读取知识库状态",
            "success": "知识库状态读取完成",
            "error": "知识库状态读取失败",
        }.get(status, "知识库状态")
    if tool_name == "python_inter":
        return {
            "running": "正在执行本地 Python 分析",
            "success": "本地 Python 分析完成",
            "error": "本地 Python 分析失败",
        }.get(status, "本地 Python 分析")
    if tool_name == "fig_inter":
        return {
            "running": "正在执行本地绘图",
            "success": "本地绘图完成",
            "error": "本地绘图失败",
        }.get(status, "本地绘图")

    display_name = get_tool_display_name(tool_name)
    prefix = {
        "knowledge_base": "知识库/数据库",
        "local": "本地工具",
        "mcp": "MCP 工具",
    }.get(tool_origin, "工具")
    verb = {
        "running": "正在调用",
        "success": "已完成",
        "error": "调用失败",
    }.get(status, "正在调用")
    return f"{verb}{prefix}：{display_name}"


def _build_model_status_title(model_name: str, status: str) -> str:
    display_name = model_name or "Assistant Model"
    return {
        "running": f"模型 {display_name} 正在思考",
        "success": f"模型 {display_name} 生成完成",
        "error": f"模型 {display_name} 生成失败",
    }.get(status, f"模型 {display_name} 正在运行")


def _build_status_payload(
    status_id: str,
    scope: str,
    status: str,
    title: str,
    detail: str = "",
    tool_name: str = "",
    tool_origin: str = "",
) -> dict[str, Any]:
    return {
        "id": status_id,
        "scope": scope,
        "status": status,
        "title": title,
        "detail": detail,
        "tool_name": tool_name,
        "tool_origin": tool_origin,
        "timestamp": _utc_now_iso(),
    }


def _build_session_summary_payload(
    session_id: str,
    user_message: str,
    ai_message: str,
    previous_summary: dict[str, str] | None = None,
) -> dict[str, str]:
    title = ""
    if previous_summary is not None:
        title = previous_summary.get("title", "").strip()

    clean_user = user_message.strip() or "新对话"
    clean_ai = ai_message.strip()
    return {
        "session_id": session_id,
        "title": title or _truncate_text(clean_user, 24),
        "preview": _truncate_text(clean_ai or clean_user or "暂无内容", 80),
        "updated_at": _utc_now_iso(),
    }


def _normalize_session_summary(session_id: str, payload: dict[str, Any]) -> dict[str, str]:
    title = payload.get("title")
    preview = payload.get("preview")
    updated_at = payload.get("updated_at")
    return {
        "session_id": session_id,
        "title": title.strip() if isinstance(title, str) and title.strip() else "新对话",
        "preview": preview if isinstance(preview, str) else "",
        "updated_at": updated_at if isinstance(updated_at, str) and updated_at else _utc_now_iso(),
    }


def _sort_session_summaries(summaries: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(summaries, key=lambda item: item.get("updated_at", ""), reverse=True)


def _build_session_summary_from_history(session_id: str, history: list[dict[str, Any]]) -> dict[str, str] | None:
    messages = _normalize_session_messages(history)
    if not messages:
        return None

    first_user_message = next(
        (item["content"].strip() for item in messages if item["type"] == "user" and item["content"].strip()),
        "",
    )
    last_message = next((item for item in reversed(messages) if item["content"].strip()), None)
    preview_source = last_message["content"].strip() if last_message is not None else first_user_message or "暂无内容"
    updated_at = last_message["timestamp"] if last_message is not None else _utc_now_iso()

    return {
        "session_id": session_id,
        "title": _truncate_text(first_user_message or "新对话", 24),
        "preview": _truncate_text(preview_source, 80),
        "updated_at": updated_at,
    }


def _save_session_summary_in_memory(user_id: str, session_id: str, user_message: str, ai_message: str) -> dict[str, str]:
    memory_key = _session_memory_key(user_id, session_id)
    summary = _build_session_summary_payload(
        session_id=session_id,
        user_message=user_message,
        ai_message=ai_message,
        previous_summary=SESSION_SUMMARIES.get(memory_key),
    )
    SESSION_SUMMARIES[memory_key] = summary
    return summary


async def _load_session_items(user_id: str, session_id: str) -> list[dict[str, Any]]:
    if SESSION_MEMORY_RUNTIME_BACKEND == "redis" and REDIS_CLIENT is not None:
        key = _redis_session_key(user_id, session_id)
        try:
            raw_items = await REDIS_CLIENT.lrange(key, 0, -1)
            parsed_items: list[dict[str, Any]] = []
            for item in raw_items:
                try:
                    payload = json.loads(item)
                    if isinstance(payload, dict):
                        parsed_items.append(payload)
                except json.JSONDecodeError:
                    continue

            if REDIS_SESSION_TTL_SECONDS > 0:
                await REDIS_CLIENT.expire(key, REDIS_SESSION_TTL_SECONDS)

            return parsed_items[-MAX_SESSION_MESSAGES:]
        except Exception as error:
            print(f"[SessionMemory] Redis read failed, fallback to in-memory: {error}")

    return list(SESSION_MEMORY.get(_session_memory_key(user_id, session_id), []))


async def _read_session_summary(user_id: str, session_id: str) -> dict[str, str] | None:
    if SESSION_MEMORY_RUNTIME_BACKEND == "redis" and REDIS_CLIENT is not None:
        meta_key = _redis_session_meta_key(user_id)
        try:
            raw_payload = await REDIS_CLIENT.hget(meta_key, session_id)
            if raw_payload:
                payload = json.loads(raw_payload)
                if isinstance(payload, dict):
                    return _normalize_session_summary(session_id, payload)
        except Exception as error:
            print(f"[SessionMemory] Redis session summary read failed, fallback to in-memory: {error}")

        history = await _load_session_items(user_id, session_id)
        summary = _build_session_summary_from_history(session_id, history)
        if summary is not None:
            try:
                await REDIS_CLIENT.hset(meta_key, session_id, json.dumps(summary, ensure_ascii=False))
                if REDIS_SESSION_TTL_SECONDS > 0:
                    await REDIS_CLIENT.expire(meta_key, REDIS_SESSION_TTL_SECONDS)
            except Exception as error:
                print(f"[SessionMemory] Redis session summary rebuild failed, fallback to in-memory: {error}")
            return summary

    memory_summary = SESSION_SUMMARIES.get(_session_memory_key(user_id, session_id))
    if memory_summary is None:
        return None
    return _normalize_session_summary(session_id, memory_summary)


async def _save_session_summary(user_id: str, session_id: str, user_message: str, ai_message: str) -> None:
    summary = _save_session_summary_in_memory(user_id, session_id, user_message, ai_message)
    if SESSION_MEMORY_RUNTIME_BACKEND == "redis" and REDIS_CLIENT is not None:
        meta_key = _redis_session_meta_key(user_id)
        try:
            await REDIS_CLIENT.hset(meta_key, session_id, json.dumps(summary, ensure_ascii=False))
            if REDIS_SESSION_TTL_SECONDS > 0:
                await REDIS_CLIENT.expire(meta_key, REDIS_SESSION_TTL_SECONDS)
        except Exception as error:
            print(f"[SessionMemory] Redis session summary write failed, fallback to in-memory: {error}")


async def _list_session_summaries(user_id: str) -> list[dict[str, str]]:
    if SESSION_MEMORY_RUNTIME_BACKEND == "redis" and REDIS_CLIENT is not None:
        meta_key = _redis_session_meta_key(user_id)
        try:
            raw_summary_map = await REDIS_CLIENT.hgetall(meta_key)
            summaries: list[dict[str, str]] = []
            for session_id, raw_payload in raw_summary_map.items():
                try:
                    payload = json.loads(raw_payload)
                except json.JSONDecodeError:
                    continue

                if isinstance(payload, dict):
                    summaries.append(_normalize_session_summary(session_id, payload))

            if summaries:
                return _sort_session_summaries(summaries)
        except Exception as error:
            print(f"[SessionMemory] Redis session summary list failed, fallback to in-memory: {error}")

        try:
            redis_prefix = f"{REDIS_SESSION_PREFIX}{user_id}:"
            keys = await REDIS_CLIENT.keys(f"{redis_prefix}*")
            rebuilt_summaries: list[dict[str, str]] = []
            for key in keys:
                if key == meta_key or not key.startswith(redis_prefix):
                    continue

                session_id = key[len(redis_prefix):]
                if not session_id or session_id == "__meta__":
                    continue

                history = await _load_session_items(user_id, session_id)
                summary = _build_session_summary_from_history(session_id, history)
                if summary is None:
                    continue

                rebuilt_summaries.append(summary)

                try:
                    await REDIS_CLIENT.hset(meta_key, session_id, json.dumps(summary, ensure_ascii=False))
                except Exception as error:
                    print(f"[SessionMemory] Redis session summary backfill failed for {session_id}: {error}")

            if rebuilt_summaries and REDIS_SESSION_TTL_SECONDS > 0:
                await REDIS_CLIENT.expire(meta_key, REDIS_SESSION_TTL_SECONDS)

            if rebuilt_summaries:
                return _sort_session_summaries(rebuilt_summaries)
        except Exception as error:
            print(f"[SessionMemory] Redis session summary rebuild list failed, fallback to in-memory: {error}")

    user_prefix = f"{user_id}:"
    return _sort_session_summaries(
        [
            _normalize_session_summary(memory_key[len(user_prefix):], payload)
            for memory_key, payload in SESSION_SUMMARIES.items()
            if memory_key.startswith(user_prefix)
        ]
    )


def _normalize_session_messages(history: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    fallback_timestamp = _utc_now_iso()
    for item in history:
        msg_type = item.get("type")
        content = item.get("content")
        timestamp = item.get("timestamp")

        if msg_type == "human":
            api_type = "user"
        elif msg_type in {"ai", "assistant"}:
            api_type = "assistant"
        else:
            continue

        if not isinstance(content, str):
            continue

        normalized.append(
            {
                "type": api_type,
                "content": content,
                "timestamp": timestamp if isinstance(timestamp, str) and timestamp else fallback_timestamp,
            }
        )

    return normalized


def _get_memory_manager():
    from src.agent import get_agent_graph
    mgr = getattr(get_agent_graph, "_memory_manager", None)
    if mgr is None:
        from src.memory.manager import MemoryManager
        mgr = MemoryManager()
        get_agent_graph._memory_manager = mgr  # type: ignore[attr-defined]
    return mgr


def _update_long_term_memory(user_id: str, session_id: str, user_msg: str, ai_msg: str) -> None:
    try:
        mgr = _get_memory_manager()
        memory = mgr.long_term.extract_and_store(user_id, user_msg, ai_msg)
        if memory:
            print(f"[Memory] 长期记忆已提取: {memory[:100]}")
    except Exception:
        pass


async def _load_session_history(user_id: str, session_id: str) -> list[dict[str, str]]:
    history = await _load_session_items(user_id, session_id)
    return _normalize_history(history)


async def _build_agent_messages(user_id: str, session_id: str, user_message: str) -> list[dict[str, str]]:
    raw = await _load_session_history(user_id, session_id)

    # 通过 MemoryManager 做上下文增强 (摘要 + 长期记忆)
    try:
        mgr = _get_memory_manager()
        ctx = mgr.get_context(user_id, session_id, user_message, history=raw)
        result: list[dict[str, str]] = []
        if ctx.summary:
            result.append({"type": "system", "content": f"[对话历史摘要]\n{ctx.summary}"})
        if ctx.long_term_hints:
            result.append({"type": "system", "content": ctx.long_term_hints})
        result.extend(ctx.history)
        result.append({"type": "human", "content": user_message})
        return result
    except Exception:
        pass

    return raw + [{"type": "human", "content": user_message}]


def _save_turn_in_memory(user_id: str, session_id: str, user_message: str, ai_message: str) -> None:
    timestamp = _utc_now_iso()
    memory_key = _session_memory_key(user_id, session_id)
    history = SESSION_MEMORY.get(memory_key, [])
    history.extend(
        [
            {"type": "human", "content": user_message, "timestamp": timestamp},
            {"type": "ai", "content": ai_message, "timestamp": timestamp},
        ]
    )
    SESSION_MEMORY[memory_key] = _trim_history(history)
    _save_session_summary_in_memory(user_id, session_id, user_message, ai_message)
    # MemoryManager: 自动提取长期记忆
    _update_long_term_memory(user_id, session_id, user_message, ai_message)


async def _save_turn(user_id: str, session_id: str, user_message: str, ai_message: str) -> None:
    timestamp = _utc_now_iso()
    if SESSION_MEMORY_RUNTIME_BACKEND == "redis" and REDIS_CLIENT is not None:
        key = _redis_session_key(user_id, session_id)
        payload = [
            json.dumps({"type": "human", "content": user_message, "timestamp": timestamp}, ensure_ascii=False),
            json.dumps({"type": "ai", "content": ai_message, "timestamp": timestamp}, ensure_ascii=False),
        ]
        try:
            await REDIS_CLIENT.rpush(key, *payload)
            await REDIS_CLIENT.ltrim(key, -MAX_SESSION_MESSAGES, -1)
            if REDIS_SESSION_TTL_SECONDS > 0:
                await REDIS_CLIENT.expire(key, REDIS_SESSION_TTL_SECONDS)
            await _save_session_summary(user_id, session_id, user_message, ai_message)
            return
        except Exception as error:
            print(f"[SessionMemory] Redis write failed, fallback to in-memory: {error}")

    _save_turn_in_memory(user_id, session_id, user_message, ai_message)


async def _clear_session_memory(user_id: str, session_id: str) -> None:
    if SESSION_MEMORY_RUNTIME_BACKEND == "redis" and REDIS_CLIENT is not None:
        try:
            await REDIS_CLIENT.delete(_redis_session_key(user_id, session_id))
            await REDIS_CLIENT.hdel(_redis_session_meta_key(user_id), session_id)
        except Exception as error:
            print(f"[SessionMemory] Redis clear failed, fallback to in-memory: {error}")

    memory_key = _session_memory_key(user_id, session_id)
    SESSION_MEMORY.pop(memory_key, None)
    SESSION_SUMMARIES.pop(memory_key, None)


def _build_fallback_answer(user_message: str, error: Exception) -> str:
    hits = retrieve_knowledge(query=user_message, top_k=3)
    if hits:
        retrieval_text = format_retrieval_hits(hits)
        return (
            "主模型调用失败，已进入检索回退模式。\n"
            f"错误信息: {error}\n\n"
            "以下结论由知识库召回片段整理得到：\n"
            f"{retrieval_text}\n\n"
            "你可以重试问题，或缩短问题长度以提高稳定性。"
        )

    return (
        "主模型调用失败，且知识库未检索到可用片段。\n"
        f"错误信息: {error}\n"
        "建议检查当前聊天模型配置、服务连通性或 API Key 是否正确。"
    )


async def _invoke_agent(messages: list[dict[str, str]]) -> dict[str, Any]:
    agent_graph = get_agent_graph()
    return await asyncio.to_thread(agent_graph.invoke, {"messages": messages})


async def _astream_agent_events(messages: list[dict[str, str]]) -> AsyncIterator[dict[str, Any]]:
    agent_graph = get_agent_graph()
    async for event in agent_graph.astream_events(
        {"messages": messages},
        version="v2",
    ):
        if isinstance(event, dict):
            yield event


def _session_id_or_new(session_id: str | None) -> str:
    if session_id and session_id.strip():
        return session_id.strip()
    return f"sess-{uuid.uuid4().hex[:12]}"


# =============================================================================
# 8. 核心接口定义
# =============================================================================
@app.get("/")
async def root():
    return {"status": "ok", "message": "Data Agent Backend is running"}


@app.get("/models/current")
async def current_models():
    return {
        "status": "success",
        "models": get_model_settings_summary(),
    }
 
 
@app.get("/data-preview")
async def data_preview(request: Request):
    current_user = await get_current_user(request)
    has_data = get_dataframe(current_user.id) is not None
    return {
        "status": "success",
        "has_data": has_data,
        "preview": get_data_preview(user_id=current_user.id) if has_data else [],
    }
 
 
# --- 接口 A: 上传 CSV ---
@app.post("/upload")
async def upload_csv(request: Request, file: UploadFile = File(...)):
    """
    前端上传 CSV 文件，后端保存并加载到内存 DataFrame
    """
    current_user = await get_current_user(request)
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="只支持 CSV 文件")
 
    safe_filename = Path(file.filename).name
    user_temp_dir = USER_TEMP_DIR / current_user.id
    user_temp_dir.mkdir(parents=True, exist_ok=True)
    file_path = user_temp_dir / safe_filename
 
    try:
        with file_path.open("wb") as f:
            content = await file.read()
            f.write(content)
 
        success, message = load_csv_file(str(file_path), user_id=current_user.id)
        if not success:
            raise HTTPException(status_code=500, detail=message)
 
        preview = get_data_preview(user_id=current_user.id)
        return JSONResponse(content={
            "status": "success",
            "message": message,
            "preview": preview,
            "filename": safe_filename,
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

 
# --- 接口 B: 计算相关性 ---
@app.post("/calculate-correlation")
async def get_correlation(payload: CorrelationRequest, request: Request):
    """
    前端点击两个变量时调用此接口，快速返回相关系数
    """
    current_user = await get_current_user(request)
    result = calculate_correlation(payload.col1, payload.col2, user_id=current_user.id)
 
    desc = "无相关性"
    try:
        corr_val = float(result)
        if abs(corr_val) > 0.8:
            desc = "极强相关"
        elif abs(corr_val) > 0.6:
            desc = "强相关"
        elif abs(corr_val) > 0.4:
            desc = "中等相关"
        elif abs(corr_val) > 0.2:
            desc = "弱相关"
    except Exception:
        pass

    return {
        "status": "success",
        "correlation": result,
        "description": desc,
    }


# --- 接口 C: 知识库管理 ---
@app.get("/kb/stats")
async def kb_stats():
    return {
        "status": "success",
        "stats": get_knowledge_base_stats(),
    }


@app.post("/kb/rebuild")
async def kb_rebuild(payload: RebuildKnowledgeBaseRequest):
    result = await asyncio.to_thread(rebuild_knowledge_base, payload.reset)
    return {
        "status": "success" if result.get("status") in {"ok", "empty"} else "error",
        "result": result,
    }


# --- 接口 D: 自定义 Agent 对话（多轮记忆 + SSE + 异常回退） ---
@app.post("/chat/invoke")
async def chat_invoke(payload: ChatRequest, request: Request):
    current_user = await get_current_user(request)
    session_id = _session_id_or_new(payload.session_id)
    messages = await _build_agent_messages(
        user_id=current_user.id,
        session_id=session_id,
        user_message=payload.message,
    )

    fallback_used = False
    try:
        with _user_runtime_scope(current_user):
            result = await _invoke_agent(messages)
        answer = _extract_ai_message_text(result)
        if not answer:
            answer = "本次响应为空，请重试或缩短问题后再试。"
    except Exception as error:
        fallback_used = True
        answer = _build_fallback_answer(payload.message, error)

    await _save_turn(
        user_id=current_user.id,
        session_id=session_id,
        user_message=payload.message,
        ai_message=answer,
    )

    return {
        "status": "success",
        "session_id": session_id,
        "answer": answer,
        "fallback_used": fallback_used,
    }


@app.post("/chat/stream")
async def chat_stream(payload: ChatRequest, request: Request):
    current_user = await get_current_user(request)
    session_id = _session_id_or_new(payload.session_id)

    async def event_generator():
        messages = await _build_agent_messages(
            user_id=current_user.id,
            session_id=session_id,
            user_message=payload.message,
        )
        agent_status_id = f"agent:{session_id}"
        fallback_used = False
        streaming_started = False
        accumulated_answer = ""
        final_payload: dict[str, Any] | None = None

        yield {
            "event": "meta",
            "data": json.dumps({"session_id": session_id, "status": "processing"}, ensure_ascii=False),
        }
        yield {
            "event": "status",
            "data": json.dumps(
                _build_status_payload(
                    status_id=agent_status_id,
                    scope="system",
                    status="running",
                    title="正在分析问题并规划工具调用",
                ),
                ensure_ascii=False,
            ),
        }

        try:
            with _user_runtime_scope(current_user):
                async for event in _astream_agent_events(messages):
                    event_name = str(event.get("event", ""))
                    run_id = str(event.get("run_id", "")) or f"evt-{uuid.uuid4().hex[:8]}"
                    name = str(event.get("name", ""))
                    data = event.get("data")
                    if not isinstance(data, dict):
                        data = {}

                    if event_name == "on_chain_start":
                        node_name = name
                        if node_name in _GRAPH_NODE_DISPLAY:
                            from src.agent import get_tracer
                            get_tracer().trace_node_start(node_name)
                            yield {
                                "event": "status",
                                "data": json.dumps(
                                    _build_status_payload(
                                        status_id=run_id,
                                        scope="graph",
                                        status="running",
                                        title=_GRAPH_NODE_DISPLAY.get(node_name, node_name),
                                    ),
                                    ensure_ascii=False,
                                ),
                            }
                        continue

                    if event_name == "on_chain_end":
                        node_name = name
                        if node_name in _GRAPH_NODE_DISPLAY:
                            from src.agent import get_tracer
                            get_tracer().trace_node_end(run_id)
                            yield {
                                "event": "status",
                                "data": json.dumps(
                                    _build_status_payload(
                                        status_id=run_id,
                                        scope="graph",
                                        status="success",
                                        title=_GRAPH_NODE_DISPLAY.get(node_name, node_name) + " - 完成",
                                    ),
                                    ensure_ascii=False,
                                ),
                            }
                        continue

                    if event_name == "on_tool_start":
                        tool_origin = get_tool_runtime_origin(name)
                        yield {
                            "event": "status",
                            "data": json.dumps(
                                _build_status_payload(
                                    status_id=run_id,
                                    scope="tool",
                                    status="running",
                                    title=_build_tool_status_title(name, tool_origin, "running"),
                                    detail=_build_status_detail(data.get("input")),
                                    tool_name=name,
                                    tool_origin=tool_origin,
                                ),
                                ensure_ascii=False,
                            ),
                        }
                        continue

                    if event_name == "on_tool_end":
                        tool_origin = get_tool_runtime_origin(name)
                        yield {
                            "event": "status",
                            "data": json.dumps(
                                _build_status_payload(
                                    status_id=run_id,
                                    scope="tool",
                                    status="success",
                                    title=_build_tool_status_title(name, tool_origin, "success"),
                                    detail=_build_status_detail(data.get("output")),
                                    tool_name=name,
                                    tool_origin=tool_origin,
                                ),
                                ensure_ascii=False,
                            ),
                        }
                        continue

                    if event_name == "on_chat_model_start":
                        yield {
                            "event": "status",
                            "data": json.dumps(
                                _build_status_payload(
                                    status_id=run_id,
                                    scope="model",
                                    status="running",
                                    title=_build_model_status_title(name, "running"),
                                ),
                                ensure_ascii=False,
                            ),
                        }
                        continue

                    if event_name == "on_chat_model_end":
                        yield {
                            "event": "status",
                            "data": json.dumps(
                                _build_status_payload(
                                    status_id=run_id,
                                    scope="model",
                                    status="success",
                                    title=_build_model_status_title(name, "success"),
                                ),
                                ensure_ascii=False,
                            ),
                        }
                        continue

                    if event_name.endswith("_end") and isinstance(data.get("output"), dict):
                        final_payload = data.get("output")

                    if event_name != "on_chat_model_stream":
                        continue

                    text_chunk = _extract_stream_chunk_text(data.get("chunk"))
                    if not text_chunk:
                        continue

                    if not streaming_started:
                        streaming_started = True
                        yield {
                            "event": "meta",
                            "data": json.dumps(
                                {
                                    "session_id": session_id,
                                    "fallback_used": fallback_used,
                                    "status": "streaming",
                                },
                                ensure_ascii=False,
                            ),
                        }
                        yield {
                            "event": "status",
                            "data": json.dumps(
                                _build_status_payload(
                                    status_id=agent_status_id,
                                    scope="system",
                                    status="running",
                                    title="正在生成最终回答",
                                ),
                                ensure_ascii=False,
                            ),
                        }

                    accumulated_answer += text_chunk
                    yield {"event": "chunk", "data": text_chunk}

            answer = _extract_ai_message_text(final_payload or {}) or accumulated_answer
            if not answer:
                answer = "本次响应为空，请重试或缩短问题后再试。"
        except Exception as error:
            # HITL: 检查是否为 GraphInterrupt（人工审批暂停）
            error_type = type(error).__name__
            error_module = type(error).__module__ or ""
            if "interrupt" in error_type.lower() or "interrupt" in error_module.lower():
                yield {
                    "event": "approval_required",
                    "data": json.dumps(
                        {
                            "session_id": session_id,
                            "message": f"Agent 暂停等待人工确认: {error}",
                            "node": "unknown",
                        },
                        ensure_ascii=False,
                    ),
                }
                return

            fallback_used = True
            answer = _build_fallback_answer(payload.message, error)
            yield {
                "event": "status",
                "data": json.dumps(
                    _build_status_payload(
                        status_id=agent_status_id,
                        scope="system",
                        status="error",
                        title="主模型失败，已切换回退回答",
                        detail=_build_status_detail(str(error)),
                    ),
                    ensure_ascii=False,
                ),
            }

        if not streaming_started:
            streaming_started = True
            yield {
                "event": "meta",
                "data": json.dumps(
                    {
                        "session_id": session_id,
                        "fallback_used": fallback_used,
                        "status": "streaming",
                    },
                    ensure_ascii=False,
                ),
            }
            if answer:
                yield {"event": "chunk", "data": answer}

        await _save_turn(
            user_id=current_user.id,
            session_id=session_id,
            user_message=payload.message,
            ai_message=answer,
        )

        if not fallback_used:
            yield {
                "event": "status",
                "data": json.dumps(
                    _build_status_payload(
                        status_id=agent_status_id,
                        scope="system",
                        status="success",
                        title="回答生成完成",
                    ),
                    ensure_ascii=False,
                ),
            }

        yield {
            "event": "done",
            "data": json.dumps(
                {
                    "session_id": session_id,
                    "fallback_used": fallback_used,
                    "answer": answer,
                },
                ensure_ascii=False,
            ),
        }

    return EventSourceResponse(event_generator())


@app.delete("/chat/session/{session_id}")
async def clear_chat_session(session_id: str, request: Request):
    current_user = await get_current_user(request)
    await _clear_session_memory(current_user.id, session_id)
    return {
        "status": "success",
        "session_id": session_id,
    }


@app.get("/chat/sessions")
async def list_chat_sessions(request: Request):
    current_user = await get_current_user(request)
    sessions = await _list_session_summaries(current_user.id)
    return {
        "status": "success",
        "runtime_backend": SESSION_MEMORY_RUNTIME_BACKEND,
        "sessions": sessions,
    }


@app.get("/chat/session/{session_id}")
async def get_chat_session(session_id: str, request: Request):
    current_user = await get_current_user(request)
    summary = await _read_session_summary(current_user.id, session_id)
    history = await _load_session_items(current_user.id, session_id)
    if summary is None and not history:
        raise HTTPException(status_code=404, detail="会话不存在")

    session_payload = summary or {
        "session_id": session_id,
        "title": "新对话",
        "preview": "",
        "updated_at": _utc_now_iso(),
    }
    session_payload["messages"] = _normalize_session_messages(history)
    return {
        "status": "success",
        "session": session_payload,
    }


# --- 接口 E: Agent 对话 (LangServe 兼容保留) ---
add_routes(
    app,
    get_agent_graph(),
    path="/agent",
)


# =============================================================================
# 9. 前端兼容路由 (匹配 front1 的 API 调用, 免认证)
# =============================================================================

# ---- HITL 审批持久化 (文件级原子写入, 服务器重启无损恢复) ----
from src.hitl.persistence import get_persistence

_hitl_persistence = get_persistence()

# 向后兼容别名 (deprecated, 请使用 _hitl_persistence)
_PENDING_INTERRUPTS: dict[str, dict[str, Any]] = {}  # 保留引用以兼容旧代码, 实际读写由 _hitl_persistence 管理


@app.post("/chat_stream")
async def chat_stream_compat(payload: dict, request: Request):
    """兼容前端 SSE 格式 + HITL 审批事件。"""
    current_user = await get_current_user(request)
    user_msg = payload.get("query", "")
    session_id = payload.get("session_id", "") or f"conv-{int(datetime.now(timezone.utc).timestamp() * 1000)}"

    async def sse():
        messages = await _build_agent_messages(current_user.id, session_id, user_msg)
        accumulated = ""
        config = {"configurable": {"thread_id": session_id}}
        interrupted = False

        try:
            with _user_runtime_scope(current_user):
                agent_graph = get_agent_graph()
                async for event in agent_graph.astream_events({"messages": messages}, config, version="v2"):
                    event_name = str(event.get("event", ""))
                    name = str(event.get("name", ""))
                    data = event.get("data") or {}

                    if event_name == "on_tool_start":
                        yield json.dumps({"type": "tool_start", "data": {"tool_name": name, "input": data.get("input", {})}}, ensure_ascii=False, default=str)
                        continue

                    if event_name == "on_tool_end":
                        yield json.dumps({"type": "tool_end", "data": {"tool_name": name, "output": str(data.get("output", ""))[:500]}}, ensure_ascii=False, default=str)
                        continue

                    if event_name == "on_chat_model_stream":
                        chunk = _extract_stream_chunk_text(data.get("chunk"))
                        if chunk:
                            accumulated += chunk
                            yield json.dumps({"type": "token", "data": {"content": chunk}}, ensure_ascii=False)
                        continue

            # 检查图是否被 HITL 中断挂起
            try:
                snap = agent_graph.get_state(config)
                if snap.next:
                    interrupted = True
                    interrupt_node = snap.next[0] if isinstance(snap.next, tuple) else snap.next
                    state_values = snap.values or {}
                    # 提取审批信息
                    approval_data = {"session_id": session_id, "node": str(interrupt_node)}
                    if "pending_sql" in state_values:
                        approval_data["sql"] = state_values.get("pending_sql", "")
                        approval_data["reasoning"] = state_values.get("pending_sql_reasoning", "")

                    _hitl_persistence.save(session_id, {
                        "user_id": str(current_user.id),
                        "node": str(interrupt_node),
                        "state": state_values,
                    })
                    yield json.dumps({"type": "approval_required", "data": approval_data}, ensure_ascii=False)
            except Exception:
                pass

            if accumulated and not interrupted:
                yield json.dumps({"type": "finish", "data": {"session_id": session_id, "answer": accumulated}}, ensure_ascii=False)

        except Exception as e:
            yield json.dumps({"type": "error", "data": str(e)[:300]}, ensure_ascii=False)

        if not interrupted:
            await _save_turn(current_user.id, session_id, user_msg, accumulated or "处理完成")

    return EventSourceResponse(sse())


@app.post("/chat/resume")
async def resume_chat(payload: dict, request: Request):
    """HITL 审批恢复端点。

    前端 Approve 后 POST:
      {"session_id": "...", "approved": true}

    后端从 checkpoint 恢复执行，返回 SSE 流。
    """
    session_id = payload.get("session_id", "").strip()
    approved = payload.get("approved", False)
    current_user = await get_current_user(request)

    if not session_id or not _hitl_persistence.exists(session_id):
        raise HTTPException(status_code=404, detail="会话不存在或未处于审批等待状态")

    pending = _hitl_persistence.remove(session_id)
    if not approved:
        return {"status": "rejected", "session_id": session_id}
    user_msg = payload.get("user_msg", "")
    config = {"configurable": {"thread_id": session_id}}

    async def sse():
        accumulated = ""
        try:
            with _user_runtime_scope(current_user):
                agent_graph = get_agent_graph()
                async for event in agent_graph.astream_events(None, config, version="v2"):
                    event_name = str(event.get("event", ""))
                    name = str(event.get("name", ""))
                    data = event.get("data") or {}

                    if event_name == "on_tool_start":
                        yield json.dumps({"type": "tool_start", "data": {"tool_name": name, "input": data.get("input", {})}}, ensure_ascii=False, default=str)
                        continue

                    if event_name == "on_tool_end":
                        yield json.dumps({"type": "tool_end", "data": {"tool_name": name, "output": str(data.get("output", ""))[:500]}}, ensure_ascii=False, default=str)
                        continue

                    if event_name == "on_chat_model_stream":
                        chunk = _extract_stream_chunk_text(data.get("chunk"))
                        if chunk:
                            accumulated += chunk
                            yield json.dumps({"type": "token", "data": {"content": chunk}}, ensure_ascii=False)
                        continue

                if accumulated:
                    yield json.dumps({"type": "finish", "data": {"session_id": session_id, "answer": accumulated}}, ensure_ascii=False)

        except Exception as e:
            yield json.dumps({"type": "error", "data": str(e)[:300]}, ensure_ascii=False)

        await _save_turn(current_user.id, session_id, user_msg, accumulated or "处理完成")

    return EventSourceResponse(sse())


@app.get("/chat/pending_interrupts")
async def list_pending_interrupts():
    """列出所有等待审批的会话 (从磁盘持久化存储读取)。"""
    return {"status": "ok", "pending": _hitl_persistence.list_all()}


@app.post("/chat_invoke")
async def chat_invoke_compat(payload: dict, request: Request):
    """兼容前端 /chat_invoke 路径 + query 字段。"""
    return await chat_invoke(
        ChatRequest(message=payload.get("query", ""), session_id=payload.get("session_id")),
        request,
    )


@app.get("/sessions")
async def list_sessions_compat(request: Request):
    """前端 GET /sessions → 返回会话列表。"""
    try:
        result = await list_chat_sessions(request)
        items = []
        for s in result.get("sessions", []):
            items.append({
                "id": s.get("session_id", ""),
                "title": s.get("title", "新对话"),
                "updated_at": s.get("updated_at", ""),
            })
        return items
    except Exception:
        return []


@app.post("/sessions")
async def create_session_compat(request: Request):
    import uuid as _uuid
    sid = f"conv-{int(datetime.now(timezone.utc).timestamp() * 1000)}-{_uuid.uuid4().hex[:8]}"
    return {"id": sid, "title": "新对话", "updated_at": _utc_now_iso()}


@app.get("/sessions/{session_id}")
async def get_session_compat(session_id: str):
    return {"session": {"id": session_id, "title": "对话", "created_at": _utc_now_iso(), "updated_at": _utc_now_iso()}}


@app.get("/sessions/{session_id}/settings")
async def get_session_settings_compat(session_id: str):
    return {
        "session_id": session_id,
        "active_kb_id": "default",
        "rag_enabled": True,
        "top_k_override": None,
    }


@app.patch("/sessions/{session_id}/settings")
async def update_session_settings_compat(session_id: str, payload: dict):
    return {
        "session_id": session_id,
        "active_kb_id": payload.get("active_kb_id", ""),
        "rag_enabled": payload.get("rag_enabled", True),
        "top_k_override": payload.get("top_k_override"),
    }


@app.patch("/sessions/{session_id}/title")
async def rename_session_compat(session_id: str, payload: dict):
    return {"id": session_id, "title": payload.get("title", "对话"), "updated_at": _utc_now_iso()}


@app.delete("/sessions/{session_id}")
async def delete_session_compat(session_id: str, request: Request):
    try:
        return await clear_chat_session(session_id, request)
    except Exception:
        return {"status": "ok", "session_id": session_id}


@app.get("/history/{session_id}")
async def get_history_compat(session_id: str, request: Request):
    result = await get_chat_session(session_id, request)
    session_data = result.get("session", {})
    return {
        "session": {
            "id": session_id,
            "title": session_data.get("title", "对话"),
            "created_at": _utc_now_iso(),
            "updated_at": session_data.get("updated_at", _utc_now_iso()),
        },
        "messages": session_data.get("messages", []),
        "settings": {"active_kb_id": "default", "rag_enabled": True, "top_k_override": None},
    }


@app.get("/mcp/list")
async def mcp_list_compat():
    from src.agent import _TOOLS
    tools = [{"name": t.name, "description": getattr(t, "description", "")} for t in _TOOLS]
    return {"tools": tools}


class AuthCredentialsRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=80)
    password: str = Field(..., min_length=6, max_length=128)


def _auth_cookie_max_age() -> int:
    return max(3600, AUTH_SETTINGS.session_ttl_hours * 3600)


def _set_auth_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=AUTH_SETTINGS.cookie_name,
        value=token,
        max_age=_auth_cookie_max_age(),
        httponly=True,
        secure=AUTH_SETTINGS.cookie_secure,
        samesite=AUTH_SETTINGS.cookie_samesite,
        path="/",
        domain=AUTH_SETTINGS.cookie_domain or None,
    )


def _clear_auth_cookie(response: Response) -> None:
    response.delete_cookie(
        key=AUTH_SETTINGS.cookie_name,
        path="/",
        domain=AUTH_SETTINGS.cookie_domain or None,
    )


def _ensure_auth_ready() -> None:
    if is_auth_store_available():
        return


async def get_current_user(request: Request) -> AuthUser:
    if not is_auth_store_available():
        return AuthUser(id="guest", username="guest", created_at=_utc_now_iso())
    session_token = request.cookies.get(AUTH_SETTINGS.cookie_name, "")
    user = get_user_by_session_token(session_token)
    if user is None:
        return AuthUser(id="guest", username="guest", created_at=_utc_now_iso())
    return user


@contextlib.contextmanager
def _user_runtime_scope(user: AuthUser):
    tokens = activate_user_context(user.id, user.username)
    try:
        yield
    finally:
        reset_user_context(tokens)


def _auth_user_payload(user: AuthUser) -> dict[str, str]:
    return {
        "id": user.id,
        "username": user.username,
        "created_at": user.created_at,
    }


@app.post("/auth/register")
async def auth_register(payload: AuthCredentialsRequest, response: Response):
    _ensure_auth_ready()
    try:
        user = create_user(payload.username, payload.password)
        session_token = create_auth_session(user.id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))

    _set_auth_cookie(response, session_token)
    return {
        "status": "success",
        "user": _auth_user_payload(user),
    }


@app.post("/auth/login")
async def auth_login(payload: AuthCredentialsRequest, response: Response):
    _ensure_auth_ready()
    try:
        user = authenticate_user(payload.username, payload.password)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))

    if user is None:
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    try:
        session_token = create_auth_session(user.id)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))

    _set_auth_cookie(response, session_token)
    return {
        "status": "success",
        "user": _auth_user_payload(user),
    }


@app.post("/auth/logout")
async def auth_logout(request: Request, response: Response):
    session_token = request.cookies.get(AUTH_SETTINGS.cookie_name, "")
    if session_token:
        try:
            delete_auth_session(session_token)
        except Exception as error:
            print(f"[Auth] Failed to delete auth session during logout: {error}")

    _clear_auth_cookie(response)
    return {"status": "success"}


@app.get("/auth/me")
async def auth_me(request: Request):
    current_user = await get_current_user(request)
    return {
        "status": "success",
        "user": _auth_user_payload(current_user),
    }


# =============================================================================
# 6. 内部辅助函数
# =============================================================================


# =============================================================================
# 9. 启动入口
# =============================================================================
if __name__ == "__main__":
    print(">>> 启动 Data Agent 后端服务...")
    print(">>> API 文档地址: http://localhost:8002/docs")
    print(">>> 图片存储路径:", str(IMAGES_DIR))

    uvicorn.run("src.server:app", host="0.0.0.0", port=8002, reload=True)