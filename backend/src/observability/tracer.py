"""可观测性抽象接口

定义 Tracer 协议，支持 Console (开发) / LangFuse (生产) 两种实现。
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceSpan:
    span_id: str
    name: str
    started_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class Tracer(ABC):
    """可观测性追踪器抽象接口。

    每个方法对应一种可追踪事件:
      - trace_node: 图节点执行
      - trace_llm:   LLM 调用
      - trace_retrieval: 检索执行
    """

    @abstractmethod
    def trace_node_start(self, node_name: str, input_summary: str = "") -> str:
        """图节点开始执行, 返回 span_id。"""
        ...

    @abstractmethod
    def trace_node_end(self, span_id: str, output_summary: str = "") -> None:
        """图节点执行结束。"""
        ...

    @abstractmethod
    def trace_llm(self, model: str, latency_ms: float, token_count: int = 0) -> None:
        """LLM 调用完成。"""
        ...

    @abstractmethod
    def trace_retrieval(self, query: str, hit_count: int, latency_ms: float) -> None:
        """检索执行完成。"""
        ...

    @abstractmethod
    def flush(self) -> None:
        """刷新缓冲区。"""
        ...


class NoopTracer(Tracer):
    """空实现, 不追踪。"""

    def trace_node_start(self, node_name: str, input_summary: str = "") -> str:
        return ""

    def trace_node_end(self, span_id: str, output_summary: str = "") -> None:
        pass

    def trace_llm(self, model: str, latency_ms: float, token_count: int = 0) -> None:
        pass

    def trace_retrieval(self, query: str, hit_count: int, latency_ms: float) -> None:
        pass

    def flush(self) -> None:
        pass
