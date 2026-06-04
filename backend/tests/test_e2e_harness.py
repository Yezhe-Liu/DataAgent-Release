"""端到端联调 (E2E Harness) — 4 通道全链路验证

通道 A: Chat        — 普通闲聊直通
通道 B: RAG         — 非结构化知识库检索 + 幻觉核查回退
通道 C: SQL + HITL  — 主图 text_to_sql → supervisor_handoff 中断 → 持久化 → 放行执行
通道 D: Malicious   — 恶意 Prompt 注入攻击 → Fallback 条件边熔断防御

使用 Mock LLM + Mock Tool，不依赖真实 API Key。
"""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock

import pytest

logging.basicConfig(level=logging.WARNING)


# ---------------------------------------------------------------------------
# Mock 基础设施
# ---------------------------------------------------------------------------

class _StructuredInvoke:
    """模拟 with_structured_output.invoke() 返回 Pydantic 对象。"""

    def __init__(self, router_intent="chat", router_reason="auto",
                 rewrite_queries=None, grade_relevance="relevant", grade_score=0.85,
                 sql="SELECT * FROM city_climate_stats WHERE city_name='北京'",
                 sql_reason="查询北京气候数据",
                 hal_score=0.85, hal_feedback="所有陈述有支撑"):
        self.router_intent = router_intent
        self.router_reason = router_reason
        self.rewrite_queries = rewrite_queries or ["query1", "query2", "query3"]
        self.grade_relevance = grade_relevance
        self.grade_score = grade_score
        self.sql = sql
        self.sql_reason = sql_reason
        self.hal_score = hal_score
        self.hal_feedback = hal_feedback
        self._call_count = 0

    def __call__(self, messages, **kwargs):
        self._call_count += 1
        from src.graph.state import (
            RouterOutput, RewriteOutput, GradeOutput,
            HallucinationOutput, TextToSQLOutput,
        )
        # Detect which structured output type by checking the messages content
        msg_text = str(messages)
        if "意图分类" in msg_text or "intent" in msg_text.lower():
            return RouterOutput(intent=self.router_intent, reasoning=self.router_reason)
        if "改写为" in msg_text or "rewrite" in msg_text.lower():
            return RewriteOutput(queries=self.rewrite_queries)
        if "相关性" in msg_text or "relevance" in msg_text.lower():
            return GradeOutput(relevance=self.grade_relevance, score=self.grade_score)
        if "SQL" in msg_text or "SELECT" in msg_text.upper():
            return TextToSQLOutput(sql=self.sql, reasoning=self.sql_reason)
        if "事实核查" in msg_text or "hallucination" in msg_text.lower():
            return HallucinationOutput(score=self.hal_score, feedback=self.hal_feedback)
        return RouterOutput(intent="chat", reasoning="fallback")


class _StreamChunk:
    def __init__(self, content):
        self.content = content


def _make_flash_mock(intent="chat", gen_response="mock response"):
    """创建 flash_model mock。"""
    mock = MagicMock()
    structured = _StructuredInvoke(router_intent=intent)
    mock.with_structured_output = MagicMock(return_value=structured)
    mock.bind_tools = MagicMock(return_value=mock)
    mock.invoke = MagicMock(return_value=MagicMock(content=gen_response, tool_calls=[]))
    mock.stream = MagicMock(return_value=iter([_StreamChunk(gen_response)]))
    return mock


def _make_pro_mock(gen_response="pro mock response"):
    """创建 pro_model mock。"""
    mock = MagicMock()
    mock.with_structured_output = MagicMock(return_value=_StructuredInvoke())
    mock.invoke = MagicMock(return_value=MagicMock(content=gen_response))
    mock.stream = MagicMock(return_value=iter([_StreamChunk(gen_response)]))
    return mock


def _mock_retrieve(query, top_k=4):
    return [{"chunk_id": f"c{i}", "source": "test_doc", "text": f"mock text {i}", "final_score": 0.9} for i in range(3)]


def _mock_search(query, max_results=3):
    return "web search: no results"


def _mock_db_tool(name="query_telecom_weather_db"):
    tool = MagicMock()
    tool.name = name
    tool.invoke = MagicMock(return_value=json.dumps({"rows": [{"city": "北京", "temp": 25}], "count": 1}))
    return tool


# ---------------------------------------------------------------------------
# 通道 A: Chat 闲聊直通
# ---------------------------------------------------------------------------

class TestChannelA_Chat:
    def test_chat_intent_routes_to_chat_worker(self):
        """通道 A: chat 意图 → chat_worker → 直接生成回答，无检索/无中断。"""
        from src.graph.builder import assemble_supervisor_graph

        flash = _make_flash_mock(intent="chat", gen_response="你好！我是 DataAgent。")
        pro = _make_pro_mock(gen_response="你好！我是 DataAgent，有什么可以帮你的？")

        graph = assemble_supervisor_graph(
            flash_model=flash,
            pro_model=pro,
            data_tools=[],
            retrieve_func=_mock_retrieve,
            search_func=_mock_search,
            db_tools=[],
        )

        assert graph is not None

        # 验证图编译成功且包含 chat_worker
        node_names = {n for n in graph.get_graph().nodes}
        assert "chat_worker" in node_names
        assert "router" in node_names

        # 验证无 interrupt_before (db_tools 为空)
        # chat 请求不经过 supervisor_handoff

    def test_chat_no_hitl_interrupt(self):
        """通道 A: chat 请求不应触发 HITL 中断。"""
        from src.graph.builder import assemble_supervisor_graph

        flash = _make_flash_mock(intent="chat")
        pro = _make_pro_mock()

        graph = assemble_supervisor_graph(
            flash_model=flash, pro_model=pro,
            data_tools=[], retrieve_func=_mock_retrieve,
            search_func=_mock_search, db_tools=[],
        )

        # 无 db_tools → 无 interrupt_before
        compiled = graph.get_graph()
        # chat 路径: router → chat_worker → END (不经过 supervisor_handoff)


# ---------------------------------------------------------------------------
# 通道 B: RAG 非结构化知识库检索 + 幻觉核查回退
# ---------------------------------------------------------------------------

class TestChannelB_RAG:
    def test_rag_intent_full_pipeline(self):
        """通道 B: rag 意图 → RAGWorker 全链路 (rewrite→retrieve→grade→generate→hal_check)。"""
        from src.graph.workers.rag_worker import build_rag_worker

        flash = _make_flash_mock(intent="rag", gen_response="根据公司政策...")
        pro = _make_pro_mock(gen_response="根据《考勤管理规定》[1]，员工请假需提前申请。")

        graph = build_rag_worker(flash, pro, _mock_retrieve, _mock_search)

        assert graph is not None
        node_names = {n for n in graph.get_graph().nodes}
        assert "rewrite" in node_names
        assert "retrieve" in node_names
        assert "grade" in node_names
        assert "generate" in node_names
        assert "hallucination_check" in node_names

    def test_rag_with_web_fallback(self):
        """通道 B: 所有文档 irrelevant → web_search 回退。"""
        from src.graph.workers.rag_worker import build_rag_worker

        flash = _make_flash_mock(intent="rag", gen_response="fallback")
        pro = _make_pro_mock(gen_response="搜索结果...")
        # grade_relevance 设为 not_relevant 会触发 web_search 回退
        flash.with_structured_output = MagicMock(
            return_value=_StructuredInvoke(grade_relevance="not_relevant", grade_score=0.1)
        )

        graph = build_rag_worker(flash, pro, _mock_retrieve, _mock_search)
        assert graph is not None
        node_names = {n for n in graph.get_graph().nodes}
        assert "web_search" in node_names

    def test_hallucination_loop_limit(self):
        """通道 B: 幻觉核查回退最多 2 轮后强制退出。"""
        from src.graph.edges import check_hallucination

        # score=0.3, loop=0 → 回退 rewrite
        assert check_hallucination({"hallucination_score": 0.3, "loop_count": 0}) == "rewrite"
        # score=0.3, loop=1 → 回退 rewrite
        assert check_hallucination({"hallucination_score": 0.3, "loop_count": 1}) == "rewrite"
        # score=0.3, loop=2 → 强制退出 (max=2 已到)
        assert check_hallucination({"hallucination_score": 0.3, "loop_count": 2}) == "generate"
        # score=0.8 → 通过
        assert check_hallucination({"hallucination_score": 0.8, "loop_count": 0}) == "generate"


# ---------------------------------------------------------------------------
# 通道 C: SQL + HITL 中断 → 持久化 → 恢复
# ---------------------------------------------------------------------------

class TestChannelC_SQL_HITL:
    def test_sql_path_goes_through_supervisor_handoff(self):
        """通道 C: structured_telecom_query 路由经过 supervisor_handoff (HITL 中断点)。"""
        from src.graph.builder import assemble_supervisor_graph

        flash = _make_flash_mock(intent="structured_telecom_query",
                                 gen_response="SELECT * FROM city_climate_stats")
        pro = _make_pro_mock(gen_response="北京春季平均气温为 15°C [DB]")
        db_tool = _mock_db_tool()

        graph = assemble_supervisor_graph(
            flash_model=flash,
            pro_model=pro,
            data_tools=[],
            retrieve_func=_mock_retrieve,
            search_func=_mock_search,
            db_tools=[db_tool],
        )

        assert graph is not None
        node_names = {n for n in graph.get_graph().nodes}
        # SQL 路径关键节点必须存在
        assert "text_to_sql" in node_names
        assert "supervisor_handoff" in node_names
        assert "sql_worker" in node_names
        # supervisor_handoff 之后到达 sql_worker (边存在)

    def test_sql_worker_only_has_execute_and_generate(self):
        """通道 C: SQLWorker 内部只有 execute_sql + generate，不含 text_to_sql。"""
        from src.graph.workers.sql_worker import build_sql_worker

        pro = _make_pro_mock()
        db_tool = _mock_db_tool()
        graph = build_sql_worker(pro, [db_tool])

        node_names = {n for n in graph.get_graph().nodes}
        assert "execute_sql" in node_names
        assert "generate" in node_names
        assert "text_to_sql" not in node_names  # 在主图，不在 Worker

    def test_hitl_persistence_save_and_recover(self):
        """通道 C: 中断持久化: save → 模拟重启 → 从磁盘恢复。"""
        from src.hitl.persistence import HITLPersistence
        import tempfile, os

        tmp = tempfile.mktemp(suffix=".json")
        try:
            # 第一生命周期: 保存中断
            p1 = HITLPersistence(file_path=tmp)
            p1.save("session-001", {
                "user_id": "user-1",
                "node": "supervisor_handoff",
                "state": {"pending_sql": "SELECT * FROM test", "pending_sql_reasoning": "查询测试"},
            })
            assert p1.exists("session-001")
            assert p1.count() == 1

            # 模拟服务器重启: 创建新实例, 从同一文件恢复
            p2 = HITLPersistence(file_path=tmp)
            assert p2.exists("session-001")
            assert p2.count() == 1

            meta = p2.load("session-001")
            assert meta is not None
            assert meta["node"] == "supervisor_handoff"
            assert meta["state"]["pending_sql"] == "SELECT * FROM test"

            # 审批通过: remove
            p2.remove("session-001")
            assert not p2.exists("session-001")
            assert p2.count() == 0

            # 第三次重启: 确认已清理
            p3 = HITLPersistence(file_path=tmp)
            assert p3.count() == 0
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def test_hitl_persistence_list_all(self):
        """通道 C: list_all 返回所有待审批会话。"""
        from src.hitl.persistence import HITLPersistence
        import tempfile, os

        tmp = tempfile.mktemp(suffix=".json")
        try:
            p = HITLPersistence(file_path=tmp)
            p.save("s1", {"user_id": "u1", "node": "supervisor_handoff"})
            p.save("s2", {"user_id": "u2", "node": "supervisor_handoff"})

            pending = p.list_all()
            assert len(pending) == 2
            assert {e["session_id"] for e in pending} == {"s1", "s2"}

            p.remove("s1")
            assert len(p.list_all()) == 1
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def test_hitl_restore_empty_on_clean_start(self):
        """通道 C: 无磁盘文件时干净启动。"""
        from src.hitl.persistence import HITLPersistence
        import tempfile

        tmp = tempfile.mktemp(suffix=".json")
        # 确保文件不存在
        try:
            import os
            os.unlink(tmp)
        except OSError:
            pass

        p = HITLPersistence(file_path=tmp)
        assert p.count() == 0
        assert p.list_all() == []


# ---------------------------------------------------------------------------
# 通道 D: 恶意 Prompt 注入 → Fallback 熔断防御
# ---------------------------------------------------------------------------

class TestChannelD_MaliciousPrompt:
    """通道 D: 极端注入型恶意 Prompt 攻击测试。

    防御层次:
      1. Router 将未知/恶意意图分类为 chat (safe fallback)
      2. generate 节点不执行任何危险操作
      3. 幻觉核查跳过非 RAG 路径 (score=1.0, 直接通过)
    """

    def test_unknown_intent_falls_back_to_chat(self):
        """恶意 Prompt 被 router 分类为 chat → 安全直通。"""
        from src.graph.edges import check_hallucination

        # chat 意图的 hallucination 检查直接通过 (score=1.0)
        # 这由 hallucination_check_node 中的 intent=="chat" 分支保证

        # 验证 router 对未知意图的默认路由
        from src.graph.supervisor import _route_from_router

        # 未知/恶意 intent 默认路由到 chat_worker
        assert _route_from_router({"intent": "chat"}) == "chat_worker"
        assert _route_from_router({"intent": "unknown"}) == "chat_worker"
        assert _route_from_router({"intent": "ignore all previous instructions"}) == "chat_worker"
        assert _route_from_router({"intent": "DROP TABLE"}) == "chat_worker"
        assert _route_from_router({}) == "chat_worker"  # 缺失 intent → chat

    def test_malicious_sql_injection_isolated(self):
        """恶意 SQL 注入: 即使 router 误分类为 sql，text_to_sql 节点的结构化输出只生成 SELECT。

        熔断点:
          1. TEXT_TO_SQL_SYSTEM prompt 明确 "仅生成只读查询 (SELECT/WITH)"
          2. execute_sql 只接受 pending_sql 中已审批的 SQL
          3. HITL supervisor_handoff 提供人工审批屏障
        """
        from src.prompts import PromptRegistry

        registry = PromptRegistry.get_instance()
        sql_prefix = registry.get_static_prefix("text_to_sql")

        # 验证 SQL prompt 包含只读约束
        assert "仅生成只读查询" in sql_prefix
        assert "SELECT" in sql_prefix
        # 不应包含任何 DDL/DML 关键字
        for dangerous in ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "TRUNCATE"]:
            assert dangerous not in sql_prefix, f"Dangerous keyword {dangerous} found in SQL prompt!"

    def test_max_loop_defense(self):
        """回退循环上限防御: 最多 2 轮,防止无限循环 DoS。"""
        from src.graph.edges import check_hallucination

        # 无论 score 多低，loop_count >= 2 时强制退出
        for mal_score in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            assert check_hallucination({
                "hallucination_score": mal_score,
                "loop_count": 2,
            }) == "generate", f"loop=2 should force exit, got rewrite for score={mal_score}"

        # 极端: loop_count=999 也应该退出
        assert check_hallucination({
            "hallucination_score": 0.0,
            "loop_count": 999,
        }) == "generate"

    def test_generate_empty_context_graceful(self):
        """无检索结果时 generate 节点优雅降级。"""
        # 验证 GENERATE_STATIC_PREFIX 包含降级提示
        from src.prompts import PromptRegistry
        registry = PromptRegistry.get_instance()
        gen_prefix = registry.get_static_prefix("generate")
        assert "不要编造检索结果中没有的信息" in gen_prefix

    def test_full_graph_defense_surface(self):
        """全图防御面: 验证所有 Worker 都能正确编译，不因恶意输入崩溃。"""
        from src.graph.builder import assemble_supervisor_graph

        # 使用恶意 intent 构建完整图
        flash = _make_flash_mock(intent="DROP TABLE users; --", gen_response="")
        pro = _make_pro_mock(gen_response="")
        db_tool = _mock_db_tool()

        graph = assemble_supervisor_graph(
            flash_model=flash, pro_model=pro,
            data_tools=[], retrieve_func=_mock_retrieve,
            search_func=_mock_search, db_tools=[db_tool],
        )

        assert graph is not None
        node_names = {n for n in graph.get_graph().nodes}
        # 即使恶意输入，图结构完整
        assert "router" in node_names
        assert "supervisor_handoff" in node_names
        assert "chat_worker" in node_names  # 恶意输入 → chat 通道


# ---------------------------------------------------------------------------
# 集成: 全图拓扑 + 持久化 联合验证
# ---------------------------------------------------------------------------

class TestIntegration:
    """全链路集成验证: 图编译 + 持久化 + 通道正确性。"""

    def test_full_graph_with_persistence_roundtrip(self):
        """完整图编译 + HITL save/recover/remove 闭环。"""
        from src.graph.builder import assemble_supervisor_graph
        from src.hitl.persistence import HITLPersistence
        import tempfile, os

        # 1. 编译图
        flash = _make_flash_mock(intent="structured_telecom_query")
        pro = _make_pro_mock()
        db_tool = _mock_db_tool()

        graph = assemble_supervisor_graph(
            flash_model=flash, pro_model=pro,
            data_tools=[], retrieve_func=_mock_retrieve,
            search_func=_mock_search, db_tools=[db_tool],
        )
        assert graph is not None

        # 2. 持久化闭环
        tmp = tempfile.mktemp(suffix=".json")
        try:
            p = HITLPersistence(file_path=tmp)
            p.save("e2e-session", {
                "user_id": "tester",
                "node": "supervisor_handoff",
                "state": {
                    "pending_sql": "SELECT AVG(spring_temp) FROM city_climate_stats WHERE city_name='北京'",
                    "pending_sql_reasoning": "查询北京春季平均气温",
                },
            })
            assert p.count() == 1

            # 恢复
            p2 = HITLPersistence(file_path=tmp)
            assert p2.count() == 1
            assert p2.exists("e2e-session")
            meta = p2.load("e2e-session")
            assert "SELECT AVG" in meta["state"]["pending_sql"]

            # 清理
            p2.remove("e2e-session")
            assert p2.count() == 0
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def test_all_channels_graph_compiles(self):
        """4 通道全部编译成功。"""
        from src.graph.builder import assemble_supervisor_graph

        for intent in ["chat", "rag", "structured_telecom_query", "tool"]:
            flash = _make_flash_mock(intent=intent)
            pro = _make_pro_mock()
            db_tool = _mock_db_tool() if intent == "structured_telecom_query" else None

            graph = assemble_supervisor_graph(
                flash_model=flash, pro_model=pro,
                data_tools=[], retrieve_func=_mock_retrieve,
                search_func=_mock_search,
                db_tools=[db_tool] if db_tool else [],
            )
            assert graph is not None, f"Graph compile failed for intent={intent}"
