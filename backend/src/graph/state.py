"""Agentic RAG StateGraph 状态定义

支撑 Router -> Rewrite -> Retrieve -> Grade -> Generate -> HallucinationCheck
+ Text-to-SQL 结构化电信气象查询分支。
"""

from __future__ import annotations

from typing import Annotated, Any, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 图内结构化数据
# ---------------------------------------------------------------------------


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


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    intent: str  # "chat" | "rag" | "tool" | "structured_telecom_query"
    intent_reasoning: str
    rewritten_queries: list[str]
    retrieved_docs: list[dict[str, Any]]
    graded_docs: list[dict[str, Any]]
    web_results: str
    pending_sql: str  # text_to_sql 生成的 SQL, execute_sql_tool 执行前供 HITL 审批
    pending_sql_reasoning: str  # LLM 生成 SQL 的推理说明
    sql_query_result: str  # execute_sql_tool 查表结果 (JSON)
    hallucination_score: float
    hallucination_detail: str
    loop_count: int


# ---------------------------------------------------------------------------
# Structured Output Schema
# ---------------------------------------------------------------------------


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
