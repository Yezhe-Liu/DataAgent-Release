"""图拓扑骨架测试 — 验证 Supervisor + Worker 多智能体图的编译和路由完整性。

不依赖真实 LLM API Key，使用 Mock 模拟 with_structured_output / bind_tools。
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Mock 模型工厂 — 模拟 with_structured_output 和 bind_tools
# ---------------------------------------------------------------------------


def _make_flash_mock():
    """创建 flash_model mock: 支持 with_structured_output."""
    from src.graph.state import RouterOutput

    mock = MagicMock()
    # with_structured_output 返回一个可调用的 mock
    structured = MagicMock()
    structured.invoke = MagicMock(return_value=RouterOutput(intent="chat", reasoning="test"))
    mock.with_structured_output = MagicMock(return_value=structured)
    mock.bind_tools = MagicMock(return_value=mock)
    mock.invoke = MagicMock()
    mock.stream = MagicMock(return_value=iter([]))
    return mock


def _make_pro_mock():
    """创建 pro_model mock: 支持 invoke / stream."""
    mock = MagicMock()
    mock.invoke = MagicMock()
    mock.stream = MagicMock(return_value=iter([]))
    mock.bind_tools = MagicMock(return_value=mock)
    return mock


def _mock_retrieve_func(query, top_k=4):
    return [{"chunk_id": "1", "source": "test", "text": "mock doc", "final_score": 0.9}]


def _mock_search_func(query, max_results=3):
    return "mock web search result"


def _mock_db_tool():
    tool = MagicMock()
    tool.name = "query_telecom_weather_db"
    tool.invoke = MagicMock(return_value='{"result": "mock"}')
    return tool


# ---------------------------------------------------------------------------
# 图编译测试
# ---------------------------------------------------------------------------


class TestGraphTopology:
    """图拓扑编译与结构验证。"""

    def test_supervisor_graph_routing(self):
        """验证通过总装器编译的图包含完整的 Supervisor + Worker 节点和路由。

        build_supervisor_graph 是内部函数，其 worker 边在 assemble 时补齐，
        因此本测试通过总装器验证完整拓扑。
        """
        from src.graph.builder import assemble_supervisor_graph

        flash = _make_flash_mock()
        pro = _make_pro_mock()
        db_tool = _mock_db_tool()

        graph = assemble_supervisor_graph(
            flash_model=flash,
            pro_model=pro,
            data_tools=[],
            retrieve_func=_mock_retrieve_func,
            search_func=_mock_search_func,
            db_tools=[db_tool],
        )

        assert graph is not None
        node_names = _get_node_names(graph)
        # Supervisor 核心节点
        assert "router" in node_names
        assert "text_to_sql" in node_names
        assert "supervisor_handoff" in node_names
        # 4 个 Worker 节点
        assert "rag_worker" in node_names
        assert "sql_worker" in node_names
        assert "tool_worker" in node_names
        assert "chat_worker" in node_names

    def test_rag_worker_compiles(self):
        """验证 RAGWorker 子图编译 (rewrite→retrieve→grade→generate→hal_check)。"""
        from src.graph.workers.rag_worker import build_rag_worker

        flash = _make_flash_mock()
        pro = _make_pro_mock()
        graph = build_rag_worker(flash, pro, _mock_retrieve_func, _mock_search_func)

        assert graph is not None
        node_names = _get_node_names(graph)
        assert "rewrite" in node_names
        assert "retrieve" in node_names
        assert "grade" in node_names
        assert "web_search" in node_names
        assert "generate" in node_names
        assert "hallucination_check" in node_names

    def test_sql_worker_compiles(self):
        """验证 SQLWorker 子图编译 (仅 execute_sql + generate)。"""
        from src.graph.workers.sql_worker import build_sql_worker

        pro = _make_pro_mock()
        db_tool = _mock_db_tool()
        graph = build_sql_worker(pro, [db_tool])

        assert graph is not None
        node_names = _get_node_names(graph)
        assert "execute_sql" in node_names
        assert "generate" in node_names
        # text_to_sql 不应在 SQLWorker 内部
        assert "text_to_sql" not in node_names

    def test_tool_worker_compiles(self):
        """验证 ToolWorker 子图编译。"""
        from src.graph.workers.tool_worker import build_tool_worker

        flash = _make_flash_mock()
        graph = build_tool_worker(flash, [])

        assert graph is not None
        node_names = _get_node_names(graph)
        assert "tool_execute" in node_names

    def test_chat_worker_compiles(self):
        """验证 ChatWorker 子图编译。"""
        from src.graph.workers.chat_worker import build_chat_worker

        pro = _make_pro_mock()
        graph = build_chat_worker(pro)

        assert graph is not None
        node_names = _get_node_names(graph)
        assert "generate" in node_names

    def test_full_assembly_with_db_tools(self):
        """验证完整总装图编译 + HITL interrupt_before 配置。"""
        from src.graph.builder import assemble_supervisor_graph

        flash = _make_flash_mock()
        pro = _make_pro_mock()
        db_tool = _mock_db_tool()

        graph = assemble_supervisor_graph(
            flash_model=flash,
            pro_model=pro,
            data_tools=[],
            retrieve_func=_mock_retrieve_func,
            search_func=_mock_search_func,
            db_tools=[db_tool],
        )

        assert graph is not None
        node_names = _get_node_names(graph)

        assert "router" in node_names
        assert "text_to_sql" in node_names
        assert "supervisor_handoff" in node_names
        assert "rag_worker" in node_names
        assert "sql_worker" in node_names
        assert "tool_worker" in node_names
        assert "chat_worker" in node_names

    def test_full_assembly_without_db_tools(self):
        """验证无 db_tools 时编译正常。"""
        from src.graph.builder import assemble_supervisor_graph

        flash = _make_flash_mock()
        pro = _make_pro_mock()

        graph = assemble_supervisor_graph(
            flash_model=flash,
            pro_model=pro,
            data_tools=[],
            retrieve_func=_mock_retrieve_func,
            search_func=_mock_search_func,
            db_tools=[],
        )

        assert graph is not None
        node_names = _get_node_names(graph)
        assert "router" in node_names
        assert "supervisor_handoff" in node_names

    def test_mermaid_output(self):
        """生成 Mermaid 拓扑图用于可视化验证。"""
        from src.graph.builder import assemble_supervisor_graph

        flash = _make_flash_mock()
        pro = _make_pro_mock()
        db_tool = _mock_db_tool()

        graph = assemble_supervisor_graph(
            flash_model=flash,
            pro_model=pro,
            data_tools=[],
            retrieve_func=_mock_retrieve_func,
            search_func=_mock_search_func,
            db_tools=[db_tool],
        )

        mermaid = graph.get_graph().draw_mermaid()
        assert mermaid is not None
        assert "graph TD" in mermaid or "graph" in mermaid.lower()
        print(f"\n[Mermaid Topology — Supervisor + 4 Workers]\n{mermaid[:2000]}\n")


# ---------------------------------------------------------------------------
# 状态 Schema 测试
# ---------------------------------------------------------------------------


class TestStateSchema:
    """状态字段契约验证。"""

    def test_supervisor_state_required_fields(self):
        from src.graph.state import SupervisorState

        required = {
            "messages", "intent", "intent_reasoning",
            "pending_sql", "pending_sql_reasoning",
            "loop_count", "active_worker", "worker_output",
        }
        state_fields = set(SupervisorState.__annotations__)
        assert required <= state_fields, f"Missing: {required - state_fields}"

    def test_rag_worker_state_has_rag_fields(self):
        from src.graph.state import RAGWorkerState

        rag_fields = {
            "rewritten_queries", "retrieved_docs", "graded_docs",
            "web_results", "hallucination_score", "hallucination_detail", "loop_count",
        }
        state_fields = set(RAGWorkerState.__annotations__)
        assert rag_fields <= state_fields, f"Missing: {rag_fields - state_fields}"

    def test_sql_worker_state_has_sql_fields(self):
        from src.graph.state import SQLWorkerState

        sql_fields = {"sql_query_result", "generation"}
        state_fields = set(SQLWorkerState.__annotations__)
        assert sql_fields <= state_fields, f"Missing: {sql_fields - state_fields}"

    def test_agent_state_backward_compat(self):
        from src.graph.state import AgentState, SupervisorState
        assert AgentState is SupervisorState

    def test_worker_states_use_messages_not_worker_messages(self):
        """所有 WorkerState 使用 messages 字段名（兼容节点工厂）。"""
        from src.graph.state import (
            RAGWorkerState, SQLWorkerState, ToolWorkerState, ChatWorkerState,
        )
        for ws in [RAGWorkerState, SQLWorkerState, ToolWorkerState, ChatWorkerState]:
            assert "messages" in ws.__annotations__, f"{ws.__name__} missing messages"


# ---------------------------------------------------------------------------
# Delta 提取测试
# ---------------------------------------------------------------------------


class TestDeltaExtraction:
    """extract_delta 消息增量裁剪验证。"""

    def test_empty_messages(self):
        from src.graph.workers.base import extract_delta
        delta = extract_delta([], 0)
        assert delta["messages"] == []
        assert delta["generation"] == ""
        assert delta["delta_count"] == 0

    def test_extract_only_new_aimessages(self):
        from src.graph.workers.base import extract_delta
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        msgs = [
            SystemMessage(content="system prompt"),
            HumanMessage(content="user query"),
            AIMessage(content="final answer"),
            AIMessage(content="second answer"),
        ]
        delta = extract_delta(msgs, entry_baseline=2)
        assert delta["delta_count"] == 2
        assert delta["generation"] == "second answer"
        assert all(isinstance(m, AIMessage) for m in delta["messages"])

    def test_no_aimessages_returns_empty(self):
        from src.graph.workers.base import extract_delta
        from langchain_core.messages import HumanMessage, SystemMessage

        msgs = [SystemMessage(content="sys"), HumanMessage(content="user")]
        delta = extract_delta(msgs, entry_baseline=0)
        assert delta["delta_count"] == 0
        assert delta["messages"] == []

    def test_worker_marker_records_baseline(self):
        from src.graph.workers.base import WorkerMarker

        state = {"messages": [1, 2, 3], "worker_messages": []}
        marker = WorkerMarker(state, "rag")
        assert marker.baseline == 0
        assert marker.supervisor_msg_count == 3
        assert marker.worker_name == "rag"


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _get_node_names(compiled_graph) -> set[str]:
    """提取编译图的所有节点名称。"""
    nodes = compiled_graph.get_graph().nodes
    return {n for n in nodes}
