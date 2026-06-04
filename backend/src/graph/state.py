"""DataAgent StateGraph 状态定义 — Supervisor/Worker 主从多智能体架构

SupervisorState (主图 8 字段):
  - messages          全局对话历史 (add_messages reducer)
  - intent            路由分类结果
  - pending_sql       text_to_sql 在主图生成, HITL 断点前就绪
  - active_worker     委派目标 Worker 名称
  - worker_output     Worker 回传的最终生成文本

WorkerState (子图基类):
  - worker_messages   子图内部消息 (与主图 messages 隔离)
  - worker_input      Supervisor 传入上下文
  - generation        最终生成文本

Pydantic Structured Output (5 个):
  - RouterOutput / RewriteOutput / GradeOutput / HallucinationOutput / TextToSQLOutput
"""

from __future__ import annotations

from typing import Annotated, Any, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


# =============================================================================
# Supervisor 主图状态 — 8 个核心字段
# =============================================================================


class SupervisorState(TypedDict):
    """Supervisor 主图全局状态。

    只有 8 个字段，所有领域特定字段下沉到各 WorkerState 中。
    text_to_sql 节点留在主图，在 supervisor_handoff 前执行，
    确保 HITL 断点时 pending_sql 已就绪可供前端展示。
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    """全局对话历史，add_messages reducer 自动去重追加"""

    intent: str
    """router 分类结果: chat / rag / tool / structured_telecom_query"""

    intent_reasoning: str
    """router LLM 分类理由 (可观测)"""

    pending_sql: str
    """text_to_sql 节点在主图生成的 SQL，HITL 断点前已写入"""

    pending_sql_reasoning: str
    """LLM 生成 SQL 的推理说明 (HITL 审批展示)"""

    loop_count: int
    """全局回退计数，跨 Worker 生效 (上限 2)"""

    active_worker: str
    """当前委派的 Worker 名称: rag / sql / tool / chat"""

    worker_output: str
    """Worker 子图回传的最终生成文本 (AIMessage.content 或 generation)"""


# =============================================================================
# Worker 子图基类状态
# =============================================================================


class WorkerState(TypedDict):
    """Worker 子图基类状态 — 与 SupervisorState 字段命名空间隔离。

    关键安全约束:
      - worker_messages 与主图 messages 字段名不同，避免 add_messages
        reducer 自动合并导致消息翻倍。
      - 退出时通过 extract_delta() 提取增量，走单向强契约通道回传父图。
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    """子图内部消息历史 (SystemMessage/HumanMessage/AIMessage)，与主图 messages 共享字段名以兼容节点工厂"""

    worker_input: dict[str, Any]
    """Supervisor 通过 handoff 传入的上下文:
      - query: 用户原始问题
      - intent: 路由意图
      - pending_sql: 已审批 SQL (仅 SQLWorker)
      - pending_sql_reasoning: SQL 推理 (仅 SQLWorker)
      - active_worker: 委派目标名称
    """

    generation: str
    """Worker 最终生成文本，回写到 SupervisorState.worker_output"""


# =============================================================================
# 领域特定 Worker 状态 — 继承 WorkerState 语义
# =============================================================================


class RAGWorkerState(TypedDict):
    """RAGWorker 子图状态 (rewrite→retrieve→grade→web_search→generate→hal_check)"""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    worker_input: dict[str, Any]
    generation: str
    rewritten_queries: list[str]
    retrieved_docs: list[dict[str, Any]]
    graded_docs: list[dict[str, Any]]
    web_results: str
    hallucination_score: float
    hallucination_detail: str
    loop_count: int  # RAGWorker 内部回退计数 (上限 2)


class SQLWorkerState(TypedDict):
    """SQLWorker 子图状态 (execute_sql→generate) — 仅执行已审批的 SQL"""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    worker_input: dict[str, Any]
    generation: str
    sql_query_result: str  # execute_sql 查表结果 (JSON)


class ToolWorkerState(TypedDict):
    """ToolWorker 子图状态 (tool_execute mini ReAct)"""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    worker_input: dict[str, Any]
    generation: str


class ChatWorkerState(TypedDict):
    """ChatWorker 子图状态 (简单对话生成)"""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    worker_input: dict[str, Any]
    generation: str


# =============================================================================
# 向后兼容 — 旧代码仍可 import AgentState
# =============================================================================

AgentState = SupervisorState
"""向后兼容别名: 旧代码中的 AgentState → 新的 SupervisorState"""


# =============================================================================
# 图内结构化数据模型 (Retrieval)
# =============================================================================


class RetrievedDoc(BaseModel):
    chunk_id: str
    source: str
    text: str
    vector_score: float = 0.0
    lexical_score: float = 0.0
    final_score: float = 0.0


class GradedDoc(RetrievedDoc):
    relevance: str = "not_relevant"  # "relevant" | "not_relevant"
    grade_score: float = 0.0


# =============================================================================
# Pydantic Structured Output Schema (5 个, 不变)
# =============================================================================


class RouterOutput(BaseModel):
    intent: str = Field(
        description=(
            "用户意图分类:\n"
            "- 'chat': 闲聊/问候/自我介绍\n"
            "- 'rag': 需要检索企业知识库 (政策/流程/规范/FAQ)\n"
            "- 'tool': 需要 Python 数据分析或绘图\n"
            "- 'structured_telecom_query': 需要查询结构化数据库获取精确数值, 包括:\n"
            "  * 电信气象/毫米波衰减/水汽密度/信号传播 (query_telecom_weather_db)\n"
            "  * 企业政策/制度/合规/部门信息 (query_enterprise_policy_db)\n"
            "  * 其他可通过 SQL 查询的结构化数据"
        )
    )
    reasoning: str = Field(description="分类理由")


class RewriteOutput(BaseModel):
    queries: list[str] = Field(
        description="3 个不同角度/措辞的检索查询，用于提高召回率"
    )


class GradeOutput(BaseModel):
    relevance: str = Field(description="'relevant' 或 'not_relevant'")
    score: float = Field(ge=0.0, le=1.0, description="相关性评分 0-1")


class HallucinationOutput(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="事实准确性评分")
    feedback: str = Field(description="逐句评估反馈")


class TextToSQLOutput(BaseModel):
    sql: str = Field(description="生成的只读 SQL 语句 (SELECT/WITH)")
    reasoning: str = Field(description="生成 SQL 的推理过程, 解释如何将自然语言映射到数据库表结构")
