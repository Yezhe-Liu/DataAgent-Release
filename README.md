# DataAgent — 基于 LangGraph 的 Agentic RAG 智能问答系统

> 面向企业知识库场景，用自定义 **StateGraph** 实现 Self-RAG / Corrective RAG 模式，替代黑盒 Agent 封装。

![Data Agent](./public/show.png)

---

## 一句话定位

**DataAgent** 不是 LangChain `create_agent()` 的套壳——它用 LangGraph `StateGraph` 从零构建了完整的 **Router → Rewrite → Retrieve → Grade → Generate → HallucinationCheck** 工作流，每个节点的职责、输入输出、路由逻辑都是显式可控的。

---

## 技术架构

### 自定义 Agentic RAG 工作流

```
                 ┌─────────┐
用户消息 ──────→ │ ROUTER  │ ← LLM 意图分类 (chat / rag / tool)
                 └────┬────┘
                      │
          ┌───────────┼───────────┐
          ↓           ↓           ↓
       [chat]      [rag]       [tool]
          │           │           │
          │      ┌────▼────┐      │
          │      │ REWRITE │  Multi-Query 重写 (1→3)
          │      └────┬────┘      │
          │           ↓           │
          │      ┌──────────┐     │
          │      │ RETRIEVE │  ChromaDB + 混合检索 + Reranker
          │      └────┬─────┘     │
          │           ↓           │
          │      ┌──────────┐     │
          │      │  GRADE   │  LLM with_structured_output 文档评分
          │      └────┬─────┘     │
          │       ┌───┴───┐       │
          │       ↓       ↓       │
          │    相关     无关       │
          │       │       │       │
          │       │  ┌──────────┐ │
          │       │  │WEB_SEARCH│ │ DuckDuckGo 外网补充
          │       │  └────┬─────┘ │
          │       └───┬───┘       │
          │           ↓           │
          │      ┌──────────┐     │
          │      │ GENERATE │  带引用来源的 RAG 生成
          │      └────┬─────┘     │
          │           ↓           │
          │      ┌────────────┐   │
          │      │HALLUCINATION│ 逐句验证文档支撑度
          │      │   CHECK    │   │
          │      └─────┬──────┘   │
          │        ┌───┴───┐     │
          │        ↓       ↓     │
          │      通过    不通过   │
          │        │       │     │
          │        │  (回REWRITE) │ ← 最多2轮
          │        │             │
          └────────┼─────────────┘
                   ↓
              ┌───────┐
              │  END  │
              └───────┘
```

### 关键技术决策

| 决策 | 选型 | 原因 |
|---|---|---|
| Agent 框架 | LangGraph `StateGraph` | 节点/边显式可控，比黑盒 `create_agent` 更适合面试展示架构能力 |
| 检索增强 | Self-RAG / CRAG 模式 | Router 分流 + Grade 评分 + HallucinationCheck 闭环，工业界主流范式 |
| 查询重写 | Multi-Query (1→3) | 无需额外推理开销，多角度覆盖提升召回 |
| 文档评分 | `with_structured_output` | Pydantic 结构化输出，比 prompt 字符串解析可靠 |
| 幻觉控制 | 逐句验证 + 2轮回退 | 面试可讲指标：循环上限是工程防御性设计 |
| 流式输出 | `model.stream()` 原生 token级 | 非字符拼接，前端可逐token渲染 |
| 模型 | DeepSeek (deepseek-v4-flash) | 高性价比云端 API，支持 128K 上下文 |
| Embedding | Ollama `nomic-embed-text` | 免费本地，不可用时自动回退 hash 向量 |
| 向量库 | ChromaDB | 持久化 + HNSW 索引 + 余弦相似度 |

### 模块解耦设计

```
agent.py (组合根)
  ├─ create_chat_model()    ──→ llm_factory.py
  ├─ retrieve_knowledge()   ──→ rag_engine.py
  ├─ [python_inter, fig_inter] ──→ tools.py
  └─ build_graph(model, tools, retrieve, search)  ──→ graph/
                                                         ├─ nodes.py  (8个工厂函数)
                                                         ├─ edges.py  (条件路由)
                                                         └─ state.py  (AgentState)
```

**graph 模块不直接 import `tools.py` 或 `rag_engine.py`**——依赖通过 `build_graph()` 的参数注入，节点可独立单测。

---

## 项目结构

```
DataAgent/
├── backend/
│   ├── src/
│   │   ├── graph/               # 自定义 StateGraph (核心)
│   │   │   ├── state.py         #   AgentState 定义 + Structured Output Schema
│   │   │   ├── nodes.py         #   8个节点工厂函数 (依赖注入)
│   │   │   ├── edges.py         #   条件路由: route_intent / grade_documents / check_hallucination
│   │   │   └── builder.py       #   StateGraph 组装 + 编译
│   │   ├── agent.py             # 组合根: 创建依赖并注入到 graph
│   │   ├── server.py            # FastAPI + SSE 流式端点
│   │   ├── rag_engine.py        # 混合检索 (向量+关键词) + ChromaDB
│   │   ├── reranker.py          # 多信号融合重排 / Cross-Encoder
│   │   ├── tools.py             # Python代码执行 / 绘图 / 外网搜索
│   │   ├── llm_factory.py       # 统一 LLM 工厂 (DeepSeek / DashScope / Ollama)
│   │   ├── config.py            # 12-Factor 配置管理
│   │   ├── mcp_tools.py         # MCP 协议工具源接入
│   │   ├── data_manager.py      # CSV 加载 + 预处理 + 多用户隔离
│   │   ├── runtime_context.py   # ContextVar 用户上下文传递
│   │   └── auth_store.py        # MySQL 用户认证 (可选)
│   ├── eval/                    # RAG 检索质量评测
│   │   ├── eval_dataset.json    #   20 题覆盖精确/语义/跨文档/否定场景
│   │   └── run_eval.py          #   Hit@K / MRR / Keyword Recall
│   ├── knowledge_base/
│   │   └── docs/                # 企业知识库文档 (5 篇客服知识)
│   └── pyproject.toml
│
├── front1/                      # React 前端
│   ├── components/              #   工作流追踪 / MCP工具 / 知识库管理
│   └── utils/                   #   SSE 客户端 / RAG API / API 封装
│
└── public/                      # 截图 / 静态资源
```

---

## 快速启动

### 前置要求

- Python 3.10+ + [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com) (本地 Embedding) 或有效的 DashScope API Key
- 可选: Node.js 18+ (前端)

### 1. 后端

```bash
cd backend
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY
uv sync
uv run python -m src.server
# → http://localhost:8002
```

### 2. 知识库 (可选)

```bash
curl -X POST http://localhost:8002/kb/rebuild
curl http://localhost:8002/kb/stats
```

### 3. 前端 (可选)

```bash
cd front1
npm install && npm run dev
# → http://localhost:5173
```

---

## API 一览

| 端点 | 方法 | 说明 |
|---|---|---|
| `/chat/stream` | POST | SSE 流式对话 (meta→status→chunk→done) |
| `/chat/invoke` | POST | 同步对话 |
| `/chat/sessions` | GET | 会话列表 |
| `/chat/session/{id}` | GET/DELETE | 会话详情/删除 |
| `/kb/rebuild` | POST | 重建知识库索引 |
| `/kb/stats` | GET | 知识库状态 |
| `/kb/retrieve` | POST | 直接检索 (不走Agent) |
| `/upload/csv` | POST | 上传 CSV 数据 |
| `/models/current` | GET | 当前模型配置 |

---

## RAG 评估

内置 20 题评测集覆盖 5 种查询类型：

```bash
cd backend
uv run python eval/run_eval.py --rebuild --top_k 4
```

指标: **Hit@K** / **MRR** / **Keyword Recall**，每题输出预期来源与关键词命中详情。

---

## 技术亮点 (面试 Talking Points)

1. **自定义 StateGraph 而非 create_agent()** — 8 节点显式编排，能画出完整的节点/边/条件路由图
2. **Self-RAG + Corrective RAG** — Router 分流 + Multi-Query 重写 + Grade 评分 + HallucinationCheck + 最多2轮回退
3. **依赖注入解耦** — `graph/` 模块通过工厂函数注入依赖，不直接耦合 `tools.py` / `rag_engine.py`
4. **Structured Output** — 路由、评分、幻觉检查全部用 Pydantic schema，比字符串解析可靠
5. **双向量后端 + Reranker** — ChromaDB + JSON 索引，支持 Cross-Encoder 精排
6. **Token-level Streaming** — `model.stream()` 原生流式，非手动 chunk 切分
7. **MCP 协议** — 支持动态加载外部工具源，失败自动回退本地工具
8. **离线评估体系** — 20题多场景评测 + Hit@K/MRR/Keyword Recall
