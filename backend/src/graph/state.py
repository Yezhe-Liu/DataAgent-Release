"""Agentic RAG StateGraph 状态定义

基于 Self-RAG / CRAG 模式扩展的状态字段，支撑整个
Router -> Rewrite -> Retrieve -> Grade -> Generate -> HallucinationCheck 流程。
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
    intent: str  # "chat" | "rag" | "tool"
    intent_reasoning: str
    rewritten_queries: list[str]
    retrieved_docs: list[dict[str, Any]]
    graded_docs: list[dict[str, Any]]
    web_results: str
    hallucination_score: float
    hallucination_detail: str
    loop_count: int


# ---------------------------------------------------------------------------
# Structured Output Schema
# ---------------------------------------------------------------------------


class RouterOutput(BaseModel):
    intent: str = Field(
        description="用户意图: 'chat'(闲聊/问候), 'rag'(需要检索知识库回答问题), 'tool'(需要Python数据分析/绘图)"
    )
    reasoning: str = Field(description="分类理由")


class RewriteOutput(BaseModel):
    queries: list[str] = Field(
        description="3 个不同角度/措辞的检索查询，用于提高召回率"
    )


class GradeOutput(BaseModel):
    relevance: str = Field(description="'relevant'(文档与问题密切相关) 或 'not_relevant'(文档不相关)")
    score: float = Field(ge=0.0, le=1.0, description="相关性评分 0-1")


class HallucinationOutput(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="事实准确性评分: 1.0=完全有据可查, 0.0=存在编造")
    feedback: str = Field(description="逐句评估反馈，指出哪些陈述缺乏文档支撑")
