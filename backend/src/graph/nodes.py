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

_MULTI_QUERY_COUNT = 3
_TOP_K_PER_QUERY = 4
_MAX_DOCS_TOTAL = 8

# ---------------------------------------------------------------------------
# 类型别名
# ---------------------------------------------------------------------------
RetrieverFunc = Callable[..., list[Any]]  # (query, top_k) -> list[RetrievalHit]
SearchFunc = Callable[..., str]            # (query, max_results) -> str


# =============================================================================
# 1. Router — 意图分类
# =============================================================================

ROUTER_SYSTEM = """你是一个意图分类器。分析用户最后一条消息，判断意图:

- chat: 闲聊、问候、自我介绍等无需检索或计算的对话
- rag:  需要从企业知识库检索（政策、流程、规范、FAQ 等）
- tool: 需要 Python 数据分析或生成图表
- structured_telecom_query: 需要查结构化数据库获取精确数值，如:
  * 电信气象: 毫米波衰减、水汽密度、大气衰减、信号传播损耗、气温/气压统计
  * 企业制度: 考勤规定、报销标准、人员编制、合规风险等级
  * 关键词: 统计/平均/最高/最低/对比/排名/数量/百分比/按分类

仅输出 JSON。"""


def create_router_node(model: BaseChatModel):
    structured_model = model.with_structured_output(RouterOutput)

    def router_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        messages = state.get("messages", [])
        if not messages:
            return {"intent": "chat", "intent_reasoning": "无历史消息"}

        last_msg = messages[-1]
        user_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

        response: RouterOutput = structured_model.invoke([
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"用户消息: {user_text}"),
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

REWRITE_SYSTEM = """你是一个查询优化器。将用户问题改写为 {n} 个不同角度的检索查询。

要求:
1. 每个查询聚焦问题的不同侧面
2. 至少一个查询保留原问题的核心措辞
3. 查询应多样化，用不同表达方式覆盖可能的文档措辞

仅输出 JSON，queries 字段包含 {n} 个查询字符串。"""


def create_rewrite_node(model: BaseChatModel):
    structured_model = model.with_structured_output(RewriteOutput)

    def rewrite_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        messages = state.get("messages", [])
        last_msg = messages[-1]
        user_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

        response: RewriteOutput = structured_model.invoke([
            SystemMessage(content=REWRITE_SYSTEM.format(n=_MULTI_QUERY_COUNT)),
            HumanMessage(content=f"用户问题: {user_text}"),
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

GRADE_SYSTEM = """你是一个文档相关性评估器。给定用户问题和一篇文档片段，判断该文档是否与问题相关。

评分标准:
- relevant: 文档内容直接回答或部分回答了用户问题
- not_relevant: 文档内容与问题无关或几乎没有帮助

score 评分: 1.0=完美匹配, 0.7+=高度相关, 0.4-0.7=部分相关, 0.4以下=不相关

仅输出 JSON。"""


def create_grade_node(model: BaseChatModel):
    structured_model = model.with_structured_output(GradeOutput)

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
                    SystemMessage(content=GRADE_SYSTEM),
                    HumanMessage(content=f"用户问题: {user_text}\n\n文档片段 (来源: {doc['source']}):\n{doc['text'][:800]}"),
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

TOOL_SYSTEM = """你是一个数据分析助手。使用 python_inter 和 fig_inter 工具完成数据分析和可视化。

规则:
1. 变量 df 已加载，可以直接使用 pd 和 np
2. 绘图时先创建 fig 对象，再调用 fig_inter 保存
3. 代码执行结果会返回给你，根据结果决定是否继续"""


def create_tool_execute_node(model: BaseChatModel, tools: list[BaseTool]):
    tool_map = {tool.name: tool for tool in tools}
    model_with_tools = model.bind_tools(tools)

    def tool_execute_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        messages = state.get("messages", [])
        last_msg = messages[-1]
        user_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

        response = model_with_tools.invoke([
            SystemMessage(content=TOOL_SYSTEM),
            HumanMessage(content=f"用户请求: {user_text}"),
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

TEXT_TO_SQL_SYSTEM = """你是一个 SQL 专家。将用户的自然语言查询转为数据库 SQL 语句。

目标数据库为 telecom_weather (电信气象)，包含全国 400+ 城市数据。参考工具 description 中的表结构生成 SQL。

规则:
1. 仅生成只读查询 (SELECT/WITH)
2. 表名和列名严格与工具 description 一致
3. 如查询涉及城市名，使用 city_name 列 (如 city_name='北京')
4. 如涉及频率，使用 frequency_ghz 列 (常用值: 28, 39, 60, 73)
5. 默认返回前 20 条，除非用户指定更多
6. 对数值列做聚合时使用 AVG/MAX/MIN/SUM

仅输出 JSON。"""


def create_text_to_sql_node(model: BaseChatModel):
    """生成 SQL 并存入 pending_sql，不执行。执行由 execute_sql_tool_node 负责。"""
    structured_model = model.with_structured_output(TextToSQLOutput)

    def text_to_sql_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        messages = state.get("messages", [])
        last_msg = messages[-1]
        user_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

        try:
            result: TextToSQLOutput = structured_model.invoke([
                SystemMessage(content=TEXT_TO_SQL_SYSTEM),
                HumanMessage(content=f"用户查询: {user_text}"),
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

GENERATE_SYSTEM = """你是一个企业知识库问答助手 DataAgent。基于以下检索到的文档片段和数据库查询结果回答用户问题。

要求:
1. 先给出结论，再展开细节
2. 引用文档时必须标注来源编号，如 [1]、[2]
3. 结构化数据查询结果标注为 [DB]
4. 如果检索结果中没有完整答案，诚实说明并给出已知信息
5. 不要编造检索结果中没有的信息
6. 对于电信气象类问题，结合数据库数值 (精确) 与 ITU-R 文档 (理论依据) 共同回答

{context}"""


def create_generate_node(model: BaseChatModel):
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
                SystemMessage(content="你是一个友好的 AI 助手 DataAgent。请简洁自然地回答用户。"),
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

        system_prompt = GENERATE_SYSTEM.format(context="\n".join(context_parts))

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

CHECK_SYSTEM = """你是一个事实核查器。检查 AI 回答中的每条陈述是否被提供的文档片段所支撑。

评分指南:
- 1.0: 所有陈述都有文档直接支撑
- 0.8+: 大部分有支撑，少量通用知识补充
- 0.5-0.8: 部分有支撑，部分不确定
- 0.5以下: 存在明显编造或无依据的陈述

feedback 字段: 逐句说明哪些陈述有文档支撑，哪些缺乏支撑。

仅输出 JSON。"""


def create_hallucination_check_node(model: BaseChatModel):
    structured_model = model.with_structured_output(HallucinationOutput)

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
            SystemMessage(content=CHECK_SYSTEM),
            HumanMessage(content=f"参考文档:\n{relevant_texts}\n\nAI 回答:\n{generation}"),
        ])

        print(f"[HallucinationCheck] score={result.score:.2f} loop={loop_count}")
        return {
            "hallucination_score": result.score,
            "hallucination_detail": result.feedback,
            "loop_count": loop_count,
        }

    return hallucination_check_node
