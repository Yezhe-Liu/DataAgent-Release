# DataAgent — 基于 LangGraph 的 Agentic RAG 智能问答系统

> 面向企业知识库场景，用自定义 **StateGraph** 实现 Self-RAG / Corrective RAG，从零构建 **Router → Rewrite → Retrieve → Grade → Generate → HallucinationCheck** 全流程。

---

## 一句话定位

不是 LangChain `create_agent()` 套壳。每个节点职责、输入输出、路由逻辑都是显式可控的——面试能画出完整的 StateGraph 图。

---

## 技术架构

```
                 ┌─────────┐
用户消息 ──────→ │ ROUTER  │ ← LLM 意图分类 (chat / rag / tool)
                 └────┬────┘
          ┌───────────┼───────────┐
          ↓           ↓           ↓
       [chat]      [rag]       [tool]
          │           │           │
          │      ┌────▼────┐      │    HITL 中断点:
          │      │ REWRITE │      │    tool_execute — 执行代码前暂停
          │      └────┬────┘      │
          │           ↓           │
          │      ┌──────────┐     │
          │      │ RETRIEVE │  ChromaDB 多粒度索引 + BM25/Vector/Keyword 融合
          │      └────┬─────┘     │
          │           ↓           │
          │      ┌──────────┐     │
          │      │  GRADE   │  LLM with_structured_output
          │      └────┬─────┘     │
          │       ┌───┴───┐       │
          │       ↓       ↓       │
          │    相关     无关       │
          │       │       │       │
          │       │  ┌──────────┐ │
          │       │  │WEB_SEARCH│ │ DuckDuckGo
          │       │  └────┬─────┘ │
          │       └───┬───┘       │
          │           ↓           │
          │      ┌──────────┐     │
          │      │ GENERATE │  带引用来源
          │      └────┬─────┘     │
          │           ↓           │
          │      ┌────────────┐   │
          │      │HALLUCINATION│ 逐句验证
          │      │   CHECK    │   │
          │      └─────┬──────┘   │
          │        ┌───┴───┐     │
          │        ↓       ↓     │
          │      通过    不通过   │
          │        │       │     │
          │        │  (回REWRITE) │ ← 最多2轮
          └────────┼─────────────┘
                   ↓
              ┌───────┐
              │  END  │
              └───────┘
```

### 关键决策

| 决策 | 选型 | 原因 |
|---|---|---|
| Agent 框架 | LangGraph `StateGraph` | 节点/边显式编排，面试可画图 |
| RAG 模式 | Self-RAG / CRAG | Router 分流 + Grade 评分 + HallucinationCheck 闭环 |
| 分块策略 | Semantic / Recursive / Fixed 三种 | 按需切换，覆盖不同文档类型 |
| 检索融合 | BM25(0.3) + Vector(0.5) + Keyword(0.2) | 稀疏/稠密互补 |
| 索引 | 多粒度 (摘要层 + 段落层) | 粗筛→精排，减少无效计算 |
| 重排 | Scoring / Cross-Encoder / LLM Listwise | 三级可选，成本-精度可调 |
| 查询处理 | Multi-Query 重写 + HyDE + 同义词扩展 | 多重覆盖提升召回 |
| 幻觉控制 | 逐句验证 + 最多 2 轮回退 | 工程防御性设计 |
| 流式输出 | `model.stream()` 原生 token 级 | 非字符拼接 |
| HITL | LangGraph `interrupt_before` | 可配置审批策略 |
| 上下文 | SummarizationBuffer + VectorMemory | 短期摘要 + 长期向量记忆 |
| 可观测性 | Tracer 抽象 + Console/LangFuse 双实现 | 接口统一，策略可切换 |
| 模型 | DeepSeek v4-pro | 支持 thinking 模式关闭以兼容 tool_choice |

---

## 模块解耦

```
agent.py (组合根)
  ├─ llm_factory.py          → LLM 实例
  ├─ rag/ (RAGFacade)        → 分块/索引/检索/重排/查询
  ├─ memory/ (MemoryManager) → 短期摘要 + 长期向量记忆
  ├─ hitl/ (approvals)       → 人工审批中断策略
  ├─ observability/ (Tracer) → Console/LangFuse 追踪
  ├─ mcp_tools.py            → 外部工具源接入
  └─ graph/ (StateGraph)     → 8 节点编排
```

**各模块间零直接依赖**——graph/ 不 import rag/，rag/ 不 import memory/，全部通过 agent.py 参数注入。

---

## 项目结构

```
DataAgent/
├── backend/src/
│   ├── graph/               # 自定义 StateGraph (8 节点)
│   │   ├── state.py, nodes.py, edges.py, builder.py
│   ├── rag/                 # RAG 深度优化
│   │   ├── chunking.py      #   语义/递归/固定 三种分块策略
│   │   ├── indexing.py      #   多粒度索引 (摘要层+段落层)
│   │   ├── retrieval.py     #   BM25 + Vector + Keyword 三路融合
│   │   ├── reranker.py      #   Scoring/Cross-Encoder/LLM 三级重排
│   │   └── query.py         #   QueryExpander + HyDE
│   ├── memory/              # 短期+长期上下文管理
│   │   ├── short_term.py    #   SummarizationBuffer (LLM 摘要)
│   │   ├── long_term.py     #   VectorMemory (LLM 自动提取)
│   │   └── manager.py       #   MemoryManager 统一入口
│   ├── hitl/                # Human-in-the-Loop
│   │   └── approvals.py     #   LangGraph interrupt 策略
│   ├── observability/       # 可观测性
│   │   ├── tracer.py        #   抽象接口
│   │   ├── console.py       #   Console 结构化日志
│   │   └── langfuse.py      #   LangFuse 实现
│   ├── agent.py             # 组合根
│   ├── server.py            # FastAPI + SSE + 前端兼容路由
│   ├── rag_engine.py        # 向后兼容 (委托 rag/)
│   ├── tools.py             # Python 执行/绘图/外网搜索
│   ├── llm_factory.py       # LLM 工厂 (DeepSeek/DashScope/Ollama)
│   ├── config.py            # 12-Factor 配置
│   ├── mcp_tools.py         # MCP 协议工具源
│   ├── data_manager.py      # CSV 加载 + 多用户隔离
│   ├── runtime_context.py   # ContextVar 用户上下文
│   └── auth_store.py        # 认证 (可选, guest 降级)
├── front1/                  # React 前端
│   ├── components/          #   工作流追踪/MCP工具/知识库管理
│   └── utils/               #   SSE 客户端/API 封装
└── public/                  # 截图
```

---

## 快速启动

**前置**: Python 3.10+ / [uv](https://docs.astral.sh/uv/) / [Ollama](https://ollama.com) (Embedding) / DeepSeek API Key

```bash
# 后端
cd backend
cp .env.example .env          # 填入 DEEPSEEK_API_KEY
uv sync
uv run python -m src.server   # → http://localhost:8002

# 前端
cd front1
npm install && npm run dev    # → http://localhost:5173

# 构建知识库
curl -X POST http://localhost:8002/kb/rebuild
```

---

## 技术亮点 (面试 Talking Points)

1. **自定义 StateGraph** — 8 节点显式编排，能画完整节点/边/条件路由图
2. **Self-RAG + CRAG** — Router 分流 + Multi-Query + Grade + HallucinationCheck + 2 轮回退
3. **RAG 全链路优化** — 语义分块 / 多粒度索引 / BM25+Vector+Keyword 融合 / 三级重排 / HyDE
4. **依赖注入** — graph/ 不 import tools/rag_engine，节点可独立单测
5. **Structured Output** — Router/Grade/HallucinationCheck 全部 Pydantic schema
6. **HITL 人工审批** — LangGraph interrupt_before，可配置策略
7. **短期+长期记忆** — SummarizationBuffer 摘要压缩 + VectorMemory LLM 自动提取
8. **可观测性** — Tracer 抽象接口 + Console/LangFuse 双实现，换后端不改代码
9. **MCP 协议** — 支持动态加载外部工具源，失败自动回退
10. **Token-level Streaming** — model.stream() 原生流式，前端逐 token 渲染
