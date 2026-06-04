# DataAgent — 基于 LangGraph 的主从多智能体 RAG 系统

## 项目概述

DataAgent 是一个面向企业知识管理的 Agentic RAG 系统，采用 **Supervisor + Worker 主从多智能体网络** 替代传统单体图编排。系统以 **LangGraph StateGraph** 为 Agent 编排引擎，**FastAPI + SSE** 为服务框架，**ChromaDB** 为向量存储，**React + TypeScript** 为交互前端，**DeepSeek V4 Pro/Flash 双模型** 为推理核心，**Ollama** 提供本地 Embedding。

核心技术栈：Supervisor 主编排器 (router + text-to-sql + HITL 全局中断) + 4 个独立 Worker 子图 (RAGWorker / SQLWorker / ToolWorker / ChatWorker)、Self-RAG / Corrective RAG 检索增强、NL→SQL 两段式主图层审批、vLLM 静态前缀 Prompt Caching 优化、Model Context Protocol 工具集成、HITL 审批文件级原子持久化 (启动恢复)、ConversationSummaryBuffer 短期记忆与 LLM 驱动的 VectorMemory 长期记忆、配置驱动的多数据库 MCP Server、SQL 注入五层安全校验、Tracer 抽象可观测层。

---

## 技术架构

```
                        ┌──────────────────────────────────────────┐
                        │            SupervisorGraph               │
                        │  (interrupt_before=["supervisor_handoff"]) │
  __start__             │                                          │
      │                 │   ┌──────────┐                           │
      ▼                 │   │ router   │──intent=sql──┐            │
 [SupervisorGraph]      │   │ (flash)  │              │            │
      │                 │   └────┬─────┘              ▼            │
      │                 │        │            ┌──────────────┐     │
      │                 │        │            │ text_to_sql  │     │
      │                 │        │            │   (flash)    │     │
      │                 │        │            └──────┬───────┘     │
      │                 │        │                   ▼             │
      │                 │        │       ┌──────────────────────┐  │
      │                 │        └──────▶│ supervisor_handoff   │  │
      │                 │                │  ⏸ HITL 全局中断点   │  │
      │                 │                │  pending_sql 已就绪  │  │
      │                 │                └──────────┬───────────┘  │
      └─────────────────┤                           │              │
                        │        handoff_to_worker  │              │
                        └───────────────────────────┼──────────────┘
                            │                       │
        ┌───────────────────┼───────────────────────┼──────────────┐
        ▼                   ▼                       ▼              ▼
  ┌──────────┐      ┌──────────────┐        ┌──────────┐   ┌──────────┐
  │RAGWorker │      │  SQLWorker   │        │ToolWorker│   │ChatWorker│
  │ (子图)   │      │   (子图)     │        │ (子图)   │   │ (子图)   │
  │          │      │              │        │          │   │          │
  │ rewrite  │      │ execute_sql  │        │  tool_   │   │ generate │
  │ retrieve │      │   (已审批)   │        │  execute │   │  (pro)   │
  │ grade    │      │              │        │ (flash)  │   └────┬─────┘
  │web_search│      │ generate     │        └────┬─────┘        │
  │ generate │      │   (pro)      │             │              │
  │hal_check │      └──────┬───────┘             │              │
  └────┬─────┘             │                     │              │
       └───────────────────┼─────────────────────┼──────────────┘
                           ▼
                    return_to_supervisor
                           │
                           ▼
                          END
```

### 模型分配

| 模型 | thinking | 适用节点 |
|---|---|---|
| `deepseek-v4-flash` | disabled | router / text_to_sql / rewrite / grade / tool_execute |
| `deepseek-v4-pro` | enabled | generate / hallucination_check |

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
| Agent 架构 | Supervisor + Worker 主从多智能体 | 子图隔离各领域逻辑，主图统一 HITL 中断寻址 |
| 模型 | DeepSeek V4 Pro/Flash 双模型 | flash 保证结构化输出稳定性，pro 最大化深度推理 |
| RAG 模式 | Self-RAG / CRAG | Router 分流 + Grade 评分 + HallucinationCheck 闭环 |
| NL→SQL | text_to_sql(主图) → supervisor_handoff(HITL) → SQLWorker | SQL 生成在主图，HITL 断点时 pending_sql 已就绪 |
| Prompt Cache | vLLM 静态前缀 16-token 对齐 | 7 节点 System Prompt 全部对齐 block 边界 |
| 分块策略 | Semantic / Recursive / Fixed 三种 | 覆盖不同文档结构 |
| 检索融合 | BM25(0.3) + Vector(0.5) + Keyword(0.2) | 稀疏/稠密/精确三路互补 |
| 索引 | 多粒度 (摘要层 + 段落层) | 粗筛→精排 |
| 重排 | Scoring / Cross-Encoder / LLM Listwise | 三级可选，成本-精度可调 |
| 查询处理 | Multi-Query 重写 + HyDE + 同义词扩展 | 多重覆盖 |
| 幻觉控制 | 逐句验证 + 最多 2 轮回退 | 防御性工程实践 |
| 流式输出 | `model.stream()` + reasoning_content 提取 | 思维链 token 不丢帧 |
| HITL 持久化 | 文件级原子写入 + 启动自动恢复 | 服务器重启不丢失待审批会话 |
| 消息隔离 | extract_delta 增量裁剪 | 子图内部 SystemMessage 不泄露到全局对话 |
| 上下文 | SummarizationBuffer + VectorMemory | LLM 摘要 + 跨会话向量记忆 |
| MCP 数据层 | 配置驱动 + SQLite 只读 + SQL 五层校验 | 多业务热插拔、防注入 |
| 可观测性 | Tracer 抽象 + Console/LangFuse 双实现 + vLLM Cache Metric 日志 | 换后端不动代码 |
| Checkpointer | MemorySaver (v1) → RedisSaver (v2 预留) | 分布式迁移接口已就绪 |

---

## 模块解耦

```
agent.py (组合根)
  ├─ llm_factory.py             → flash_model + pro_model (双模型工厂)
  ├─ prompts/ (PromptRegistry)  → 7 节点 Prompt 静态前缀 16-token 对齐
  ├─ rag/ (RAGFacade)           → 分块/索引/检索/重排/查询
  ├─ memory/ (MemoryManager)    → 短期摘要 + 长期向量记忆
  ├─ hitl/
  │   ├─ approvals.py           → 中断策略配置
  │   └─ persistence.py         → HITL 审批原子持久化 (文件级)
  ├─ observability/ (Tracer)    → Console/LangFuse 追踪
  ├─ mcp_tools.py               → MCP 客户端 (工具加载)
  ├─ mcp_server.py              → MCP Server (配置驱动多库只读查询)
  └─ graph/
      ├─ state.py                → SupervisorState + 5 WorkerState
      ├─ supervisor.py           → Supervisor 主图 (router + text_to_sql + handoff)
      ├─ workers/                → 4 个 Worker 子图
      │   ├─ rag_worker.py       →   RAGWorker (rewrite→retrieve→grade→generate→hal_check)
      │   ├─ sql_worker.py       →   SQLWorker (execute_sql→generate)
      │   ├─ tool_worker.py      →   ToolWorker (tool_execute mini ReAct)
      │   └─ chat_worker.py      →   ChatWorker (generate)
      ├─ nodes.py                → 节点工厂 (接入 PromptRegistry)
      ├─ edges.py                → 条件路由 (grade / hallucination)
      └─ builder.py              → assemble_supervisor_graph() 总装
```

**模块间零直接依赖**：graph/ 不 import rag/，rag/ 不 import memory/，全部通过 agent.py 参数注入。

---

## 项目结构

```
DataAgent/
├── backend/
│   ├── data/
│   │   ├── mcp_databases.json           # MCP 多数据库配置
│   │   ├── telecom_propagation.db       # 电信气象 SQLite (451 城市)
│   │   ├── enterprise_policy.db         # 企业政策 SQLite (8 policies + 8 rules)
│   │   └── pending_interrupts.json      # HITL 审批持久化 (运行时生成)
│   ├── src/
│   │   ├── graph/                        # Supervisor + Worker 主从多智能体图
│   │   │   ├── state.py                  #   SupervisorState + 5 WorkerState
│   │   │   ├── supervisor.py             #   主图 (router + text_to_sql + handoff)
│   │   │   ├── nodes.py                  #   节点工厂
│   │   │   ├── edges.py                  #   条件路由
│   │   │   ├── builder.py                #   总装 + HITL 配置 + RedisSaver 预留
│   │   │   └── workers/                  #   4 个 Worker 子图
│   │   │       ├── base.py               #     Delta 消息裁剪 + WorkerMarker
│   │   │       ├── rag_worker.py         #     RAGWorker
│   │   │       ├── sql_worker.py         #     SQLWorker
│   │   │       ├── tool_worker.py        #     ToolWorker
│   │   │       └── chat_worker.py        #     ChatWorker
│   │   ├── prompts/                      # 提示词中台
│   │   │   ├── __init__.py               #   PromptRegistry + 16-token 对齐
│   │   │   ├── router.py                 #   Router Prompt
│   │   │   ├── text_to_sql.py            #   Text-to-SQL Prompt
│   │   │   ├── rag_worker.py             #   RAG Worker 4 节点 Prompt
│   │   │   └── tool_worker.py            #   Tool Worker Prompt
│   │   ├── rag/                          # RAG 全链路
│   │   │   ├── chunking.py / indexing.py / retrieval.py / reranker.py / query.py
│   │   ├── memory/                       # 上下文管理
│   │   │   ├── short_term.py / long_term.py / manager.py
│   │   ├── hitl/                         # Human-in-the-Loop
│   │   │   ├── approvals.py              #   中断策略
│   │   │   └── persistence.py            #   文件级原子持久化
│   │   ├── observability/                # 可观测性
│   │   ├── agent.py                      # 组合根 (双模型注入)
│   │   ├── server.py                     # FastAPI + SSE + reasoning_content 流式
│   │   ├── mcp_server.py / mcp_tools.py  # MCP 工具层
│   │   ├── llm_factory.py                # 双模型工厂 (flash + pro + vllm)
│   │   └── config.py                     # 12-Factor 配置
│   ├── tests/
│   │   ├── test_graph_topology.py        # 图拓扑骨架测试 (17)
│   │   └── test_e2e_harness.py           # 端到端 4 通道联调 (17)
│   └── knowledge_base/docs/              # 企业知识库文档
├── front1/                               # React 前端
└── public/                               # 截图
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

# 运行测试
cd backend && uv run python -m pytest tests/ -v
```

---

## API

| 端点 | 方法 | 说明 |
|---|---|---|
| `/chat_stream` | POST | SSE 流式对话 (token + tool + approval_required 事件) |
| `/chat/resume` | POST | HITL 审批恢复 `{"session_id":"...","approved":true}` |
| `/chat/pending_interrupts` | GET | 列出等待审批的会话 (从磁盘持久化读取) |
| `/sessions` | GET/POST/DELETE | 会话管理 |
| `/kb/rebuild` | POST | 重建知识库索引 |
| `/kb/stats` | GET | 知识库状态 |
| `/models/current` | GET | 当前模型配置 |

---

## 技术亮点

1. **Supervisor + Worker 主从多智能体网络** — Supervisor 主图持有 router / text_to_sql / supervisor_handoff 三个核心节点，4 个 Worker 子图独立封装各领域逻辑。子图退出时通过 `extract_delta()` 增量裁剪只回传 AIMessage，杜绝 `add_messages` Reducer 消息翻倍。主图 `interrupt_before` 只配置 `supervisor_handoff` 唯一全局断点，确保分布式 Checkpointer 状态机序列化句柄绝对可寻址。

2. **DeepSeek V4 双模型 + Thinking 动态护栏** — flash 模型 (thinking=disabled) 用于 Router / Grade / Text-to-SQL / Rewrite 等需要 `with_structured_output` 的节点，保证 JSON 输出稳定性；pro 模型 (thinking=enabled) 用于 Generate / HallucinationCheck 节点，最大化深度推理质量。流式层在 `_extract_stream_chunk_text` 中追加 `reasoning_content` 回退提取，防止思维链 token 丢帧导致前端卡顿。

3. **NL → SQL 主图层审批 + HITL 原子持久化** — text_to_sql 保留在主图，在 supervisor_handoff 之前执行，确保 HITL 断点时 `pending_sql` 已就绪可供前端展示。审批数据通过文件级原子写入 (`tmp → os.replace`)，服务器重启后自动从 `pending_interrupts.json` 恢复所有待审批会话，实现生产级进程韧性。

4. **vLLM 静态前缀 Prompt Caching** — 全部 7 节点 System Prompt 从内联字符串迁移至 `PromptRegistry` 统一管理，通过 `align_static_prefix()` 估算 token 长度并对齐 16-token Block 边界。每个节点首次加载时打点 `[vLLM Cache Metric]` 日志 (static_prefix_tokens / remainder_mod16 / block_aligned)，可直接接入 LangFuse / Grafana 量化评估 Prefix Cache 命中率。

5. **配置驱动的 MCP 多数据库** — `mcp_databases.json` 声明式定义每个工具的名称、数据库路径、完整表结构和 SQL 示例。MCP Server 启动时动态注册，`execute_sql_tool` 按顺序 fallback 尝试所有工具直到命中正确数据库。新增业务场景只需追加 JSON 配置。

6. **SQL 安全防护五层** — (1) 语句类型白名单（仅 SELECT/WITH/EXPLAIN），(2) DROP/DELETE/INSERT/UPDATE 等危险关键词全局拦截，(3) 注释符 `--`、`/* */` 和多语句 `;` 注入模式检测，(4) 无 LIMIT 的查询自动补 `LIMIT 500`，(5) 连接层 `file:...?mode=ro` 只读打开。

7. **RAG 检索全链路优化** — 分段策略：Semantic / Recursive / Fixed 三种可切换。索引结构：文档摘要层 + 段落层两级。检索融合：BM25(0.3) + Dense Vector(0.5) + Keyword Jaccard(0.2) 三路加权。重排器：Scoring / Cross-Encoder / LLM Listwise 三级可选。查询处理：Multi-Query 重写 + HyDE 假设文档生成 + 同义词扩展。

8. **上下文管理：短期摘要 + 长期向量记忆** — `SummarizationBuffer` 调用 LLM 将前半段压缩为摘要；`VectorMemory` 每轮对话后由 LLM 自动判断是否值得记忆，提取后存入独立 ChromaDB collection 跨会话检索注入 System Prompt。

9. **模块零耦合的依赖注入架构** — `graph/`、`rag/`、`memory/`、`hitl/`、`observability/`、`prompts/` 六个核心模块互相零 import。`agent.py` 作为唯一组合根，通过 `build_graph(flash_model, pro_model, tools, retrieve_func, search_func, db_tools, interrupt_before)` 将全部依赖参数化注入。

10. **Structured Output 全覆盖** — Router (intent + reasoning)、Grade (relevance + score)、HallucinationCheck (score + feedback)、TextToSQL (sql + reasoning) 四个 LLM 输出节点全部使用 Pydantic schema + `with_structured_output`，避免正则解析 prompt 输出的不稳定性。

11. **可观测性抽象层** — `Tracer` 抽象接口提供 `ConsoleTracer` 和 `LangFuseTracer` 两种实现，通过 `TRACER_BACKEND` 环境变量切换。vLLM Cache Metric 日志独立埋点，业务代码零改动。

12. **Token 级原生流式 + DeepSeek V4 思维链兼容** — `generate` 和 `tool_execute` 使用 `model.stream()` 原生流式，LangGraph `astream_events` 捕获 token 事件经 EventSourceResponse 推送到前端。`_extract_stream_chunk_text` 兼容 `reasoning_content` 字段，pro 模型思考时不丢帧。
