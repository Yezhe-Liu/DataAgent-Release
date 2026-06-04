# DataAgent — 基于 LangGraph 的 Agentic RAG 智能问答系统

## 项目概述

DataAgent 是一个面向企业知识管理的 Agentic RAG 系统，覆盖非结构化文档检索与结构化数据库查询两类典型企业场景。系统以 **LangGraph StateGraph** 为 Agent 编排引擎，**FastAPI + SSE** 为服务框架，**ChromaDB** 为向量存储，**React + TypeScript** 为交互前端，**DeepSeek** 为推理模型，**Ollama** 提供本地 Embedding。

核心技术栈涵盖：自定义 StateGraph 编排（10 节点 + 条件路由）、Self-RAG / Corrective RAG 检索增强、NL→SQL 结构化查询、Model Context Protocol 工具集成、LangGraph `interrupt_before` 人工审批断点、ConversationSummaryBuffer 短期记忆与 LLM 驱动的 VectorMemory 长期记忆、配置驱动的多数据库 MCP Server、SQL 注入五层安全校验、Tracer 抽象可观测层。

RAG 检索链路包含语义分块（Semantic / Recursive / Fixed）、多粒度索引（摘要层 + 段落层）、BM25 + Dense Vector + Keyword Jaccard 三路融合检索、Scoring / Cross-Encoder / LLM Listwise 三级重排、Multi-Query 重写与 HyDE 查询增强。会话管理支持 Redis 持久化与内存降级，用户认证提供 MySQL / Guest 双模式。

---

## 技术架构

```
                       ┌─────────┐
     用户消息 ──────→  │ ROUTER  │ ← LLM 意图分类 (chat / rag / tool / structured_query)
                       └────┬────┘
                ┌───────────┼───────────┐
                ↓           ↓           ↓
             [chat]    [rag + db]    [tool]
                │           │           │
                │    ┌──────▼──────┐    │  HITL 中断点:
                │    │ TEXT_TO_SQL │    │  tool_execute — 代码执行前
                │    └──────┬──────┘    │  execute_sql_tool — SQL 执行前
                │           │           │
                │    ┌──────▼─────────┐ │
                │    │ ⏸ HITL 断点   │ │  生成 SQL 后挂起
                │    │  等待审批     │ │  前端审核 SQL
                │    └──────┬─────────┘ │
                │           │           │
                │    ┌──────▼─────────┐ │
                │    │ EXECUTE_SQL    │ │  多工具 fallback 执行
                │    │ TOOL           │ │
                │    └──────┬─────────┘ │
                │           ↓           │
                │    ┌────▼────┐        │
                │    │ REWRITE │  Multi-Query 重写
                │    └────┬────┘        │
                │         ↓             │
                │    ┌──────────┐       │
                │    │ RETRIEVE │  ChromaDB 多粒度索引 + BM25/Vector/Keyword
                │    └────┬─────┘       │
                │         ↓             │
                │    ┌──────────┐       │
                │    │  GRADE   │  LLM with_structured_output 相关性评分
                │    └────┬─────┘       │
                │     ┌───┴───┐         │
                │     ↓       ↓         │
                │  相关     无关         │
                │     │       │         │
                │     │  ┌──────────┐   │
                │     │  │WEB_SEARCH│   │ DuckDuckGo 外网补充
                │     │  └────┬─────┘   │
                │     └───┬───┘         │
                │         ↓             │
                │    ┌──────────┐       │
                │    │ GENERATE │  DB 数值 + RAG 文档 + Web 三源融合
                │    └────┬─────┘       │
                │         ↓             │
                │    ┌────────────┐     │
                │    │HALLUCINATION│    │ 逐句验证文档支撑度
                │    │   CHECK    │     │
                │    └─────┬──────┘     │
                │      ┌───┴───┐       │
                │      ↓       ↓       │
                │    通过    不通过     │
                │      │       │       │
                │      │  (回REWRITE)   │ ← 最多 2 轮
                └──────┼───────┘       │
                       ↓               │
                  ┌───────┐            │
                  │  END  │  ←────────┘
                  └───────┘
```

### 数据层：配置驱动 MCP 多数据库

```
mcp_databases.json ──→ mcp_server.py ──→ 动态注册 N 个只读查询工具
                              │
         ┌────────────────────┼────────────────────┐
         ↓                    ↓                    ↓
  telecom_propagation.db  enterprise_policy.db    (可扩展)
  · city_climate_stats    · policies
  · 451 城市气象数据       · compliance_rules
                         · departments
```

新增业务数据库只需在 JSON 配置中加一项，无需修改代码。

---

## 关键决策

| 决策 | 选型 | 原因 |
|---|---|---|
| Agent 框架 | LangGraph `StateGraph` | 节点/边显式编排，面试可逐节点讲解 |
| RAG 模式 | Self-RAG / CRAG | Router 分流 + Grade 评分 + HallucinationCheck 闭环 |
| NL→SQL | text_to_sql → execute_sql_tool 两段式 | SQL 生成后 HITL 断点，人工审批再执行 |
| 分块策略 | Semantic / Recursive / Fixed 三种 | 覆盖不同文档结构 |
| 检索融合 | BM25(0.3) + Vector(0.5) + Keyword(0.2) | 稀疏/稠密/精确三路互补 |
| 索引 | 多粒度 (摘要层 + 段落层) | 粗筛→精排 |
| 重排 | Scoring / Cross-Encoder / LLM Listwise | 三级可选，成本-精度可调 |
| 查询处理 | Multi-Query 重写 + HyDE + 同义词扩展 | 多重覆盖 |
| 幻觉控制 | 逐句验证 + 最多 2 轮回退 | 防御性工程实践 |
| 流式输出 | `model.stream()` 原生 token 级 | 非字符拼接 |
| HITL | LangGraph `interrupt_before` + resume | SQL 执行前、代码执行前双断点 |
| 上下文 | SummarizationBuffer + VectorMemory | LLM 摘要 + 跨会话向量记忆 |
| MCP 数据层 | 配置驱动 + SQLite 只读 + SQL 五层校验 | 多业务热插拔、防注入 |
| 可观测性 | Tracer 抽象 + Console/LangFuse 双实现 | 换后端不动代码 |
| 模型 | DeepSeek v4-pro | thinking 模式关闭兼容 tool_choice |

---

## 模块解耦

```
agent.py (组合根)
  ├─ llm_factory.py          → LLM 实例
  ├─ rag/ (RAGFacade)        → 分块/索引/检索/重排/查询
  ├─ memory/ (MemoryManager) → 短期摘要 + 长期向量记忆
  ├─ hitl/ (approvals)       → 中断策略 (interrupt_before)
  ├─ observability/ (Tracer) → Console/LangFuse 追踪
  ├─ mcp_tools.py            → MCP 客户端 (工具加载)
  ├─ mcp_server.py           → MCP Server (配置驱动多库只读查询)
  └─ graph/ (StateGraph)     → 10 节点编排
```

**模块间零直接依赖**：graph/ 不 import rag/，rag/ 不 import memory/，全部通过 agent.py 参数注入。

---

## 项目结构

```
DataAgent/
├── backend/
│   ├── data/
│   │   ├── mcp_databases.json       # MCP 多数据库配置
│   │   ├── telecom_propagation.db   # 电信气象 SQLite (451 城市)
│   │   └── enterprise_policy.db     # 企业政策 SQLite (8 policies + 8 rules)
│   ├── src/
│   │   ├── graph/                    # 自定义 StateGraph (10 节点)
│   │   │   ├── state.py, nodes.py, edges.py, builder.py
│   │   ├── rag/                      # RAG 全链路
│   │   │   ├── chunking.py           #   语义/递归/固定分块
│   │   │   ├── indexing.py           #   多粒度索引
│   │   │   ├── retrieval.py          #   BM25+Vector+Keyword 融合
│   │   │   ├── reranker.py           #   三级重排
│   │   │   └── query.py             #   QueryExpander + HyDE
│   │   ├── memory/                   # 上下文管理
│   │   │   ├── short_term.py         #   SummarizationBuffer
│   │   │   ├── long_term.py          #   VectorMemory (LLM 提取)
│   │   │   └── manager.py            #   MemoryManager
│   │   ├── hitl/                     # Human-in-the-Loop
│   │   │   └── approvals.py         #   interrupt 策略
│   │   ├── observability/            # 可观测性
│   │   │   ├── tracer.py             #   抽象接口
│   │   │   ├── console.py            #   Console 实现
│   │   │   └── langfuse.py          #   LangFuse 实现
│   │   ├── agent.py                  # 组合根
│   │   ├── server.py                 # FastAPI + SSE + HITL resume
│   │   ├── mcp_server.py             # MCP Server (配置驱动)
│   │   ├── mcp_tools.py              # MCP Client
│   │   ├── init_telecom_db.py        # 电信库初始化
│   │   ├── init_enterprise_policy_db.py  # 企业政策库初始化
│   │   ├── rag_engine.py             # 向后兼容 (委托 rag/)
│   │   ├── tools.py                  # Python 执行/绘图/外网搜索
│   │   ├── llm_factory.py            # LLM 工厂
│   │   ├── config.py                 # 12-Factor 配置
│   │   ├── data_manager.py           # CSV 加载 + 多用户隔离
│   │   └── runtime_context.py        # ContextVar 上下文
│   ├── knowledge_base/docs/          # 企业知识库文档
│   └── eval/                         # RAG 评估
├── front1/                           # React 前端
│   ├── components/                   #   工作流追踪 / MCP / 知识库管理
│   └── utils/                        #   SSE 客户端 / API 封装
└── public/                           # 截图
```

---

## 快速启动

**前置**: Python 3.10+ / [uv](https://docs.astral.sh/uv/) / [Ollama](https://ollama.com) / DeepSeek API Key

```bash
# 后端
cd backend
cp .env.example .env              # 填入 DEEPSEEK_API_KEY
uv sync
uv run python -m src.init_telecom_db           # 初始化电信气象库
uv run python -m src.init_enterprise_policy_db  # 初始化企业政策库
uv run python -m src.server                    # → http://localhost:8002

# 前端
cd front1
npm install && npm run dev        # → http://localhost:5173

# 构建知识库
curl -X POST http://localhost:8002/kb/rebuild
```

---

## API

| 端点 | 方法 | 说明 |
|---|---|---|
| `/chat_stream` | POST | SSE 流式对话 (token + tool + approval_required 事件) |
| `/chat/resume` | POST | HITL 审批恢复 `{"session_id":"...","approved":true}` |
| `/chat/pending_interrupts` | GET | 列出等待审批的会话 |
| `/sessions` | GET/POST/DELETE | 会话管理 |
| `/kb/rebuild` | POST | 重建知识库索引 |
| `/kb/stats` | GET | 知识库状态 |
| `/models/current` | GET | 当前模型配置 |

---

## 技术亮点

1. **自定义 StateGraph，非黑盒封装** — 10 个节点全部显式定义：Router 按意图分流，rewrite/retrieve/grade/generate/hallucination_check 构成 RAG 闭环，text_to_sql/execute_sql_tool 构成结构化查询分支。每个节点的工厂函数通过闭包注入外部依赖，不与具体模块耦合。

2. **Self-RAG + Corrective RAG 融合** — 检索结果不直接喂给生成器，而是先经 LLM `with_structured_output` 逐条评分为 "relevant / not_relevant"。全部无关时触发 Web Search 回退；生成后由 HallucinationCheck 逐句验证文档支撑度，未通过则回退 rewrite 重新检索，最多 2 轮。

3. **NL → SQL 两段式 + HITL 审批** — 自然语言查询先经 TextToSQL 节点生成 SQL 并写入 `pending_sql` 状态，图在 `execute_sql_tool` 前通过 `interrupt_before` 强制挂起。前端审核 SQL 后调用 `/chat/resume` 恢复执行。该模式将不可逆的数据库操作置于人工审阅之下。

4. **配置驱动的 MCP 多数据库** — `mcp_databases.json` 声明式定义每个工具的名称、数据库路径、完整表结构和 SQL 示例。MCP Server 启动时动态注册，`execute_sql_tool` 按顺序 fallback 尝试所有工具直到命中正确数据库。新增业务场景只需追加 JSON 配置。

5. **SQL 安全防护五层** — (1) 语句类型白名单（仅 SELECT/WITH/EXPLAIN），(2) DROP/DELETE/INSERT/UPDATE 等危险关键词全局拦截，(3) 注释符 `--`、`/* */` 和多语句 `;` 注入模式检测，(4) 无 LIMIT 的查询自动补 `LIMIT 500`，(5) 连接层 `file:...?mode=ro` 只读打开。

6. **RAG 检索全链路优化** — 分段策略：Semantic（按段落/Markdown 标题）/ Recursive（段落→句子→字符递归降级）/ Fixed 三种可切换。索引结构：文档摘要层 + 段落层两级，粗筛后精排。检索融合：BM25(0.3) + Dense Vector(0.5) + Keyword Jaccard(0.2) 三路加权。重排器：Scoring（unigram+bigram+原始分+长度惩罚）/ Cross-Encoder（sentence-transformers）/ LLM Listwise 三级可选。查询处理：Multi-Query 重写 + HyDE 假设文档生成 + 同义词扩展。

7. **上下文管理：短期摘要 + 长期向量记忆** — 会话消息超过窗口阈值时 `SummarizationBuffer` 调用 LLM 将前半段压缩为摘要，避免直接丢弃。`VectorMemory` 每轮对话后由 LLM 自动判断是否值得记忆（偏好、决策、背景），提取后存入独立 ChromaDB collection，跨会话检索注入 System Prompt。

8. **模块零耦合的依赖注入架构** — `graph/`、`rag/`、`memory/`、`hitl/`、`observability/` 五个核心模块互相零 import。`agent.py` 作为唯一组合根，通过 `build_graph(model, tools, retrieve_func, search_func, db_tools, interrupt_before)` 将全部依赖参数化注入。

9. **Structured Output 全覆盖** — Router（intent + reasoning）、Grade（relevance + score）、HallucinationCheck（score + feedback）、TextToSQL（sql + reasoning）四个 LLM 输出节点全部使用 Pydantic schema + `with_structured_output`，避免正则解析 prompt 输出的不稳定性。

10. **可观测性抽象层** — 定义 `Tracer` 抽象接口（`trace_node_start/end`、`trace_llm`、`trace_retrieval`），提供 `ConsoleTracer`（结构化 JSON 日志）和 `LangFuseTracer`（云端 Tracing）两种实现，通过 `TRACER_BACKEND` 环境变量切换，业务代码零改动。

11. **Token 级原生流式** — `generate` 和 `tool_execute` 节点使用 `model.stream()` 替代 `model.invoke()`，LangGraph `astream_events` 捕获 token 事件后经 EventSourceResponse 推送到前端，前端逐 token 增量渲染。非字符拼接的假流式，延迟可测。

12. **会话隔离与 Redis 持久化** — 内存/Redis 双后端会话存储，通过 `session_id` + `user_id` 复合键隔离。Redis 不可用时自动降级为内存模式，会话 TTL 自动过期。前端兼容路由采用 guest 降级，无 MySQL 时不影响使用。
