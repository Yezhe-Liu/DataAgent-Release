from __future__ import annotations

import os
import re
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv(override=True)


@dataclass(frozen=True)
class ChatSettings:
    provider: str
    model: str          # flash / 快速模型 (默认 deepseek-v4-flash)
    model_pro: str      # pro / 强推理模型 (默认 deepseek-v4-pro)
    temperature: float
    max_tokens: int | None
    ollama_base_url: str
    ollama_num_ctx: int
    ollama_num_gpu: int
    ollama_num_thread: int
    ollama_low_vram: bool
    dashscope_api_key: str
    dashscope_base_url: str
    deepseek_api_key: str
    deepseek_base_url: str
    vllm_base_url: str


@dataclass(frozen=True)
class EmbeddingSettings:
    provider: str
    model: str
    ollama_base_url: str
    dashscope_api_key: str
    dashscope_base_url: str


@dataclass(frozen=True)
class MySQLSettings:
    host: str
    port: int
    database: str
    user: str
    password: str
    charset: str


@dataclass(frozen=True)
class AuthSettings:
    cookie_name: str
    cookie_domain: str
    cookie_secure: bool
    cookie_samesite: str
    session_ttl_hours: int
    password_hash_iterations: int


def _clean_env_value(value: str | None) -> str:
    raw = (value or "").strip()
    if not raw or raw.startswith("#"):
        return ""
    return re.sub(r"\s+#.*$", "", raw).strip()


def get_env_text(name: str, default: str = "") -> str:
    cleaned = _clean_env_value(os.getenv(name))
    return cleaned or default


def get_env_bool(name: str, default: bool = False) -> bool:
    return _to_bool(os.getenv(name), default)


def get_env_int(name: str, default: int) -> int:
    return _to_int(os.getenv(name), default)


def get_env_float(name: str, default: float) -> float:
    return _to_float(os.getenv(name), default)


def _to_bool(value: str | None, default: bool = False) -> bool:
    cleaned = _clean_env_value(value)
    if not cleaned:
        return default
    return cleaned.lower() in {"1", "true", "yes", "on"}


def _to_int(value: str | None, default: int) -> int:
    try:
        return int(_clean_env_value(value))
    except Exception:
        return default


def _to_float(value: str | None, default: float) -> float:
    try:
        return float(_clean_env_value(value))
    except Exception:
        return default


def _to_optional_int(value: str | None) -> int | None:
    raw = _clean_env_value(value)
    if not raw:
        return None
    try:
        return int(raw)
    except Exception:
        return None


def _normalize_provider(value: str | None, default: str = "deepseek") -> str:
    raw = (_clean_env_value(value) or default).strip().lower() or default
    aliases = {
        "qwen": "dashscope",
        "qwen_cloud": "dashscope",
        "aliyun": "dashscope",
        "openai_compatible": "dashscope",
        "vllm": "vllm",
        "openai": "vllm",
    }
    return aliases.get(raw, raw)


def get_chat_settings() -> ChatSettings:
    provider = _normalize_provider(os.getenv("CHAT_MODEL_PROVIDER"), default="deepseek")
    if provider == "deepseek":
        default_model = "deepseek-v4-flash"
        default_model_pro = "deepseek-v4-pro"
    elif provider == "dashscope":
        default_model = "qwen-plus"
        default_model_pro = "qwen-max"
    elif provider == "vllm":
        default_model = os.getenv("VLLM_MODEL_NAME", "")
        default_model_pro = os.getenv("VLLM_MODEL_PRO", default_model)
    else:
        default_model = "qwen3.5:2b"
        default_model_pro = default_model
    model = get_env_text("CHAT_MODEL_NAME", default_model)
    model_pro = get_env_text("CHAT_MODEL_PRO", default_model_pro)
    return ChatSettings(
        provider=provider,
        model=model,
        model_pro=model_pro,
        temperature=_to_float(get_env_text("CHAT_TEMPERATURE", get_env_text("LLM_TEMPERATURE", "0")), 0.0),
        max_tokens=_to_optional_int(get_env_text("CHAT_MAX_TOKENS", "")),
        ollama_base_url=get_env_text("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_num_ctx=get_env_int("OLLAMA_NUM_CTX", 4096),
        ollama_num_gpu=get_env_int("OLLAMA_NUM_GPU", 1),
        ollama_num_thread=get_env_int("OLLAMA_NUM_THREAD", 4),
        ollama_low_vram=get_env_bool("OLLAMA_LOW_VRAM", True),
        dashscope_api_key=get_env_text("DASHSCOPE_API_KEY", ""),
        dashscope_base_url=get_env_text("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        deepseek_api_key=get_env_text("DEEPSEEK_API_KEY", ""),
        deepseek_base_url=get_env_text("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        vllm_base_url=get_env_text("VLLM_BASE_URL", "http://localhost:8000/v1"),
    )


def get_embedding_settings() -> EmbeddingSettings:
    provider = _normalize_provider(os.getenv("EMBEDDING_PROVIDER"), default="ollama")
    default_model = "text-embedding-v3" if provider == "dashscope" else "nomic-embed-text"
    model = get_env_text("EMBEDDING_MODEL_NAME", default_model)
    return EmbeddingSettings(
        provider=provider,
        model=model,
        ollama_base_url=get_env_text("OLLAMA_BASE_URL", "http://localhost:11434"),
        dashscope_api_key=get_env_text("DASHSCOPE_API_KEY", ""),
        dashscope_base_url=get_env_text("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )


def get_mysql_settings() -> MySQLSettings:
    return MySQLSettings(
        host=get_env_text("MYSQL_HOST", "127.0.0.1"),
        port=get_env_int("MYSQL_PORT", 3306),
        database=get_env_text("MYSQL_DATABASE", "dataagent"),
        user=get_env_text("MYSQL_USER", "root"),
        password=get_env_text("MYSQL_PASSWORD", ""),
        charset=get_env_text("MYSQL_CHARSET", "utf8mb4"),
    )


def get_auth_settings() -> AuthSettings:
    cookie_samesite = get_env_text("AUTH_COOKIE_SAMESITE", "lax").strip().lower() or "lax"
    if cookie_samesite not in {"lax", "strict", "none"}:
        cookie_samesite = "lax"

    return AuthSettings(
        cookie_name=get_env_text("AUTH_COOKIE_NAME", "dataagent_auth"),
        cookie_domain=get_env_text("AUTH_COOKIE_DOMAIN", ""),
        cookie_secure=get_env_bool("AUTH_COOKIE_SECURE", False),
        cookie_samesite=cookie_samesite,
        session_ttl_hours=get_env_int("AUTH_SESSION_TTL_HOURS", 168),
        password_hash_iterations=get_env_int("PASSWORD_HASH_ITERATIONS", 120000),
    )


def get_model_settings_summary() -> dict[str, str | int | float | bool | None]:
    chat = get_chat_settings()
    embedding = get_embedding_settings()
    return {
        "chat_provider": chat.provider,
        "chat_model": chat.model,
        "chat_model_pro": chat.model_pro,
        "chat_temperature": chat.temperature,
        "chat_max_tokens": chat.max_tokens,
        "embedding_provider": embedding.provider,
        "embedding_model": embedding.model,
        "ollama_base_url": chat.ollama_base_url,
        "dashscope_base_url": chat.dashscope_base_url,
        "vllm_base_url": chat.vllm_base_url,
    }
