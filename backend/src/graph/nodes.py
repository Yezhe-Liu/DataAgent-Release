"""Agentic RAG 节点工厂

每个节点采用工厂函数模式，通过闭包注入外部依赖。

节点职责:
  router              - 意图分类 (chat/rag/tool/structured_telecom_query)
  rewrite             - Multi-Query 多角度查询重写
  retrieve            - 知识库多路检索 + 合并去重
  grade               - LLM 文档相关性评分
  web_search          - 外网搜索回退
  text_to_sql         - NL→SQL 生成 + MCP 结构化查表
  tool_execute        - 数据分析/绘图工具调用 (mini ReAct)
  generate            - 基于检索结果+SQL结果 生成含引用回答
  hallucination_check - 逐句验证生成内容的文档支撑度
"""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from src.graph.state import (
    AgentState,
    GradeOutput,
    HallucinationOutput,
    RewriteOutput,
    RouterOutput,
    TextToSQLOutput,
)
from src.prompts import PromptRegistry

_MULTI_QUERY_COUNT = 3
_TOP_K_PER_QUERY = 4
_MAX_DOCS_TOTAL = 8

# ---------------------------------------------------------------------------
# 类型别名
# ---------------------------------------------------------------------------
RetrieverFunc = Callable[..., list[Any]]  # (query, top_k) -> list[RetrievalHit]
SearchFunc = Callable[..., str]            # (query, max_results) -> str

# ---------------------------------------------------------------------------
# Prompt 注册表 (单例, 首次访问时加载并对齐所有 static_prefix)
# ---------------------------------------------------------------------------

_registry = PromptRegistry.get_instance()

# ---------------------------------------------------------------------------
# 向后兼容: 旧代码中可能引用的模块级常量
# ---------------------------------------------------------------------------

ROUTER_SYSTEM = _registry.get_static_prefix("router")
REWRITE_SYSTEM = _registry.get_static_prefix("rewrite")
GRADE_SYSTEM = _registry.get_static_prefix("grade")
TOOL_SYSTEM = _registry.get_static_prefix("tool_execute")
TEXT_TO_SQL_SYSTEM = _registry.get_static_prefix("text_to_sql")
GENERATE_SYSTEM = _registry.get_static_prefix("generate")
CHECK_SYSTEM = _registry.get_static_prefix("hallucination_check")


# =============================================================================
# 1. Router — 意图分类
# =============================================================================


def create_router_node(model: BaseChatModel):
    structured_model = model.with_structured_output(RouterOutput)
    router_static = _registry.get_static_prefix("router")
    router_dynamic = _registry.get_dynamic_template("router")

    def router_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        messages = state.get("messages", [])
        if not messages:
            return {"intent": "chat", "intent_reasoning": "无历史消息"}

        last_msg = messages[-1]
        user_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

        response: RouterOutput = structured_model.invoke([
            SystemMessage(content=router_static),
            HumanMessage(content=router_dynamic.format(user_text=user_text)),
        ])

        return {
            "intent": response.intent,
            "intent_reasoning": response.reasoning,
            "loop_count": 0,
        }

    return router_node


# =============================================================================
# 2. Rewrite — Multi-Query 重写
# =============================================================================


def create_rewrite_node(model: BaseChatModel):
    structured_model = model.with_structured_output(RewriteOutput)
    rewrite_static = _registry.get_static_prefix("rewrite").format(n=_MULTI_QUERY_COUNT)
    rewrite_dynamic = _registry.get_dynamic_template("rewrite")

    def rewrite_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        messages = state.get("messages", [])
        last_msg = messages[-1]
        user_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

        response: RewriteOutput = structured_model.invoke([
            SystemMessage(content=rewrite_static),
            HumanMessage(content=rewrite_dynamic.format(user_text=user_text)),
        ])

        return {"rewritten_queries": response.queries}

    return rewrite_node


# =============================================================================
# 3. Retrieve — 多路检索 + 合并去重
# =============================================================================

def create_retrieve_node(retrieve_func: RetrieverFunc):
    def retrieve_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        queries = state.get("rewritten_queries", [])
        if not queries:
            return {"retrieved_docs": [], "graded_docs": []}

        all_docs: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for query in queries:
            hits = retrieve_func(query=query, top_k=_TOP_K_PER_QUERY)
            for hit in hits:
                cid = getattr(hit, "chunk_id", str(hit))
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    all_docs.append({
                        "chunk_id": cid,
                        "source": getattr(hit, "source", ""),
                        "text": getattr(hit, "text", ""),
                        "vector_score": getattr(hit, "vector_score", 0.0),
                        "lexical_score": getattr(hit, "lexical_score", 0.0),
                        "final_score": getattr(hit, "final_score", 0.0),
                    })

        all_docs.sort(key=lambda d: d["final_score"], reverse=True)
        top_docs = all_docs[:_MAX_DOCS_TOTAL]

        print(f"[Retrieve] queries={len(queries)} unique={len(all_docs)} top={len(top_docs)}")
        return {"retrieved_docs": top_docs}

    return retrieve_node


# =============================================================================
# 4. Grade — 文档相关性评分
# =============================================================================


def create_grade_node(model: BaseChatModel):
    structured_model = model.with_structured_output(GradeOutput)
    grade_static = _registry.get_static_prefix("grade")
    grade_dynamic = _registry.get_dynamic_template("grade")

    def grade_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        docs = state.get("retrieved_docs", [])
        if not docs:
            return {"graded_docs": []}

        messages = state.get("messages", [])
        last_msg = messages[-1]
        user_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

        graded: list[dict[str, Any]] = []
        for doc in docs:
            try:
                result: GradeOutput = structured_model.invoke([
                    SystemMessage(content=grade_static),
                    HumanMessage(content=grade_dynamic.format(
                        user_text=user_text,
                        source=doc["source"],
                        text=doc["text"][:800],
                    )),
                ])
                graded.append({**doc, "relevance": result.relevance, "grade_score": result.score})
            except Exception:
                graded.append({**doc, "relevance": "not_relevant", "grade_score": 0.0})

        relevant_count = sum(1 for d in graded if d["relevance"] == "relevant")
        print(f"[Grade] total={len(graded)} relevant={relevant_count}")
        return {"graded_docs": graded}

    return grade_node


# =============================================================================
# 5. Web Search — 外网搜索回退
# =============================================================================

def create_web_search_node(search_func: SearchFunc):
    def web_search_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        messages = state.get("messages", [])
        last_msg = messages[-1]
        user_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

        result = search_func(query=user_text, max_results=3)
        print("[WebSearch] done")
        return {"web_results": str(result)}

    return web_search_node


# =============================================================================
# 6. Tool Execute — 数据分析工具调用 (mini ReAct)
# =============================================================================


def create_tool_execute_node(model: BaseChatModel, tools: list[BaseTool]):
    tool_map = {tool.name: tool for tool in tools}
    model_with_tools = model.bind_tools(tools)
    tool_static = _registry.get_static_prefix("tool_execute")
    tool_dynamic = _registry.get_dynamic_template("tool_execute")

    def tool_execute_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        messages = state.get("messages", [])
        last_msg = messages[-1]
        user_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

        response = model_with_tools.invoke([
            SystemMessage(content=tool_static),
            HumanMessage(content=tool_dynamic.format(user_text=user_text)),
        ])

        if not response.tool_calls:
            return {"messages": [AIMessage(content=response.content or "请提供更具体的数据分析需求。")]}

        tool_messages: list[ToolMessage] = []
        for tc in response.tool_calls:
            tool = tool_map.get(tc["name"])
            if tool is None:
                result = f"未知工具: {tc['name']}"
            else:
                result = str(tool.invoke(tc["args"]))
            tool_messages.append(ToolMessage(content=result, tool_call_id=tc["id"], name=tc["name"]))

        final = model.invoke([
            SystemMessage(content=TOOL_SYSTEM),
            HumanMessage(content=f"用户请求: {user_text}"),
            response,
            *tool_messages,
        ])

        return {"messages": [final]}

    return tool_execute_node


# =============================================================================
# 7. Text-to-SQL — NL→SQL 生成 (不含执行)
# =============================================================================


def create_text_to_sql_node(model: BaseChatModel):
    """生成 SQL 并存入 pending_sql，不执行。执行由 execute_sql_tool_node 负责。"""
    structured_model = model.with_structured_output(TextToSQLOutput)
    sql_static = _registry.get_static_prefix("text_to_sql")
    sql_dynamic = _registry.get_dynamic_template("text_to_sql")

    def text_to_sql_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        messages = state.get("messages", [])
        last_msg = messages[-1]
        user_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

        try:
            result: TextToSQLOutput = structured_model.invoke([
                SystemMessage(content=sql_static),
                HumanMessage(content=sql_dynamic.format(user_text=user_text)),
            ])
        except Exception as e:
            return {
                "pending_sql": "",
                "pending_sql_reasoning": f"生成失败: {e}",
                "sql_query_result": json_dumps({"error": f"SQL 生成失败: {e}"}),
            }

        sql = result.sql.strip()
        print(f"[TextToSQL] reasoning={result.reasoning[:100]}")
        print(f"[TextToSQL] sql={sql[:200]} (pending approval)")

        return {
            "pending_sql": sql,
            "pending_sql_reasoning": result.reasoning,
        }

    return text_to_sql_node


# =============================================================================
# 7b. Execute SQL Tool — 执行 pending_sql (HITL 断点在此之前)
# =============================================================================


def create_execute_sql_tool_node(db_tools: list[Any] | None = None):
    """执行 pending_sql 中的 SQL 语句。依次尝试所有数据库工具直到成功。"""

    tools = db_tools or []

    def execute_sql_tool_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        sql = state.get("pending_sql", "").strip()
        reasoning = state.get("pending_sql_reasoning", "")

        if not sql:
            return {"sql_query_result": json_dumps({"error": "No SQL to execute"})}

        if not tools:
            return {"sql_query_result": json_dumps({"error": "No database tools configured", "sql": sql})}

        print(f"[ExecuteSQL] trying {len(tools)} tool(s), sql={sql[:200]}")

        last_error = ""
        for tool in tools:
            tool_name = getattr(tool, "name", "unknown")
            try:
                tool_result = tool.invoke({"sql": sql})
                result_str = str(tool_result)
                # 如果结果中包含错误, 尝试下一个工具
                if '"error"' in result_str[:200]:
                    last_error = result_str
                    print(f"[ExecuteSQL] {tool_name} returned error, trying next...")
                    continue
                print(f"[ExecuteSQL] {tool_name} success")
                import json as _json
                return {
                    "sql_query_result": _json.dumps({
                        "sql": sql,
                        "reasoning": reasoning,
                        "tool": tool_name,
                        "result": result_str[:3000],
                    }, ensure_ascii=False, indent=2),
                    "pending_sql": "",
                }
            except Exception as e:
                last_error = str(e)
                print(f"[ExecuteSQL] {tool_name} exception: {e}, trying next...")
                continue

        return {"sql_query_result": json_dumps({
            "error": "All database tools failed",
            "sql": sql,
            "last_error": last_error[:500],
        })}

    return execute_sql_tool_node


# =============================================================================
# 8. Generate — 基于检索结果 + SQL 结果 生成回答
# =============================================================================


_CHAT_SYSTEM = "你是一个友好的 AI 助手 DataAgent。请简洁自然地回答用户。"


def create_generate_node(model: BaseChatModel):
    gen_static = _registry.get_static_prefix("generate")
    gen_dynamic = _registry.get_dynamic_template("generate")

    def generate_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        messages = state.get("messages", [])
        last_msg = messages[-1]
        user_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

        graded_docs = state.get("graded_docs", [])
        web_results = state.get("web_results", "")
        sql_result = state.get("sql_query_result", "")
        intent = state.get("intent", "chat")

        # chat 意图：简单对话
        if intent == "chat":
            full = ""
            for chunk in model.stream([
                SystemMessage(content=_CHAT_SYSTEM),
                HumanMessage(content=user_text),
            ]):
                full += chunk.content if hasattr(chunk, "content") and chunk.content else ""
            return {"generation": full, "messages": [AIMessage(content=full)]}

        # 构建检索上下文 (RAG + SQL 双来源)
        context_parts: list[str] = []

        # 知识库文档
        relevant_docs = [d for d in graded_docs if d.get("relevance") == "relevant"]
        if relevant_docs:
            context_parts.append("【知识库检索结果 (ITU-R 等理论文档)】")
            for idx, doc in enumerate(relevant_docs, start=1):
                context_parts.append(f"[{idx}] 来源: {doc['source']} (score: {doc.get('grade_score', 0):.2f})")
                context_parts.append(f"内容: {doc['text'][:600]}\n")

        # 数据库结构化查询结果
        if sql_result:
            context_parts.append("【数据库结构化查询结果 [DB]】")
            context_parts.append(sql_result[:2000])

        if web_results:
            context_parts.append("【外部搜索补充】")
            context_parts.append(web_results)

        if not context_parts:
            context_parts.append("（未检索到相关信息，请根据通用知识回答并说明信息来源不足）")

        # static_prefix (缓存命中) + 上下文 (变动)
        system_prompt = gen_static + "\n\n" + gen_dynamic.format(context="\n".join(context_parts))

        full = ""
        for chunk in model.stream([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_text),
        ]):
            full += chunk.content if hasattr(chunk, "content") and chunk.content else ""

        return {"generation": full, "messages": [AIMessage(content=full)]}

    return generate_node


# json.dumps helper for text_to_sql_node
def json_dumps(obj: Any, **kwargs) -> str:
    import json as _json
    return _json.dumps(obj, ensure_ascii=False, **kwargs)


# =============================================================================
# 8. Hallucination Check — 幻觉检查
# =============================================================================


def create_hallucination_check_node(model: BaseChatModel):
    structured_model = model.with_structured_output(HallucinationOutput)
    check_static = _registry.get_static_prefix("hallucination_check")
    check_dynamic = _registry.get_dynamic_template("hallucination_check")

    def hallucination_check_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        generation = state.get("generation", "")
        graded_docs = state.get("graded_docs", [])
        loop_count = state.get("loop_count", 0) + 1

        if not generation or state.get("intent") == "chat":
            return {"hallucination_score": 1.0, "hallucination_detail": "非 RAG 回答，跳过检查", "loop_count": loop_count}

        relevant_docs = [d for d in graded_docs if d.get("relevance") == "relevant"]
        relevant_texts = "\n---\n".join(
            f"[来源 {i}]: {d['text'][:500]}" for i, d in enumerate(relevant_docs, start=1)
        ) if relevant_docs else "（无相关文档）"

        result: HallucinationOutput = structured_model.invoke([
            SystemMessage(content=check_static),
            HumanMessage(content=check_dynamic.format(
                relevant_texts=relevant_texts,
                generation=generation,
            )),
        ])

        print(f"[HallucinationCheck] score={result.score:.2f} loop={loop_count}")
        return {
            "hallucination_score": result.score,
            "hallucination_detail": result.feedback,
            "loop_count": loop_count,
        }

    return hallucination_check_node
