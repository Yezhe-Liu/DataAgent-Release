import os
# from langchain_deepseek import ChatDeepSeek
# [关键] 使用 LangChain 1.1 标准 API 和 中间件工具
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt

# 导入工具和数据管理器
from src.tools import (
    python_inter,
    fig_inter,
    retrieve_knowledge,
    external_search,
    knowledge_base_status,
)
from src.config import get_chat_settings
from src.llm_factory import create_chat_model
from src.mcp_tools import load_mcp_tools
from src.data_manager import get_data_info
from src.rag_engine import get_knowledge_base_stats
# Redis 存 Checkpoint


# =============================================================================
# 1. 初始化模型
# =============================================================================
# model = ChatDeepSeek(
#     model="deepseek-chat",
#     temperature=0,
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     api_base="https://api.deepseek.com"
# )
_GRAPH = None
_GRAPH_SIGNATURE: tuple[str, str] | None = None

local_tools = [
    retrieve_knowledge,
    external_search,
    python_inter,
    fig_inter,
    knowledge_base_status,
]

tools, mcp_status = load_mcp_tools(local_tools)
LOCAL_TOOL_NAMES = {getattr(tool, "name", "") for tool in local_tools if getattr(tool, "name", "")}
ALL_TOOL_NAMES = {getattr(tool, "name", "") for tool in tools if getattr(tool, "name", "")}
MCP_TOOL_NAMES = ALL_TOOL_NAMES - LOCAL_TOOL_NAMES
KNOWLEDGE_BASE_TOOL_NAMES = {"retrieve_knowledge", "knowledge_base_status"}
TOOL_DISPLAY_NAMES = {
    "retrieve_knowledge": "知识库检索",
    "knowledge_base_status": "知识库状态",
    "python_inter": "本地 Python 分析",
    "fig_inter": "本地绘图",
    "external_search": "外部搜索",
}
print(f"[AgentTools] MCP status: {mcp_status}; total_tools={len(tools)}")


def get_tool_runtime_origin(tool_name: str) -> str:
    if tool_name in KNOWLEDGE_BASE_TOOL_NAMES:
        return "knowledge_base"
    if tool_name in MCP_TOOL_NAMES:
        return "mcp"
    if tool_name in LOCAL_TOOL_NAMES:
        return "local"
    return "unknown"


def get_tool_display_name(tool_name: str) -> str:
    return TOOL_DISPLAY_NAMES.get(tool_name, tool_name)

# =============================================================================
# 2. 定义动态 System Prompt 中间件
# =============================================================================

@dynamic_prompt
def dataset_context_middleware(request) -> str:
    """
    [中间件] 动态提示词生成器
    LangChain 会在每次调用 Agent 之前自动运行这个函数。
    
    request: 包含了当前的 ModelRequest (如 input, messages 等)
    返回: 最新的 System Prompt 字符串
    """
    
    # 1. 实时去内存获取最新的数据摘要（包含文件名、行列数、列名等）
    # 只要用户上传了新文件，get_data_info() 的返回结果就会立刻变化
    data_context = get_data_info()
    kb_stats = get_knowledge_base_stats()
    
    # 2. 返回完整的 System Prompt
    return f"""你是一名面向企业知识库问答场景的 Agentic RAG 助手 DataAgent。

【当前数据集实时状态】
{data_context}

【私有知识库状态】
文档数: {kb_stats.get("doc_count", 0)}
切片数: {kb_stats.get("chunk_count", 0)}
Embedding: {kb_stats.get("embedding_provider", "unknown")} / {kb_stats.get("embedding_model", "unknown")}

【你的职责】
1. 与企业文档、FAQ、流程规范相关的问题，优先调用 `retrieve_knowledge`，再基于检索结果回答。
2. 当私有知识库信息不足且问题依赖公开信息时，再调用 `external_search` 补充。
3. 与上传数据集相关的分析问题，调用 `python_inter` 或 `fig_inter`。
4. 变量 `df` 已内置可直接使用；绘图时请将图像对象赋值给变量，并调用绘图工具。
5. 回答格式遵循：先结论，再依据；引用知识库时必须给出来源编号，如 [1][2]。
6. 若工具执行失败，先解释原因，再给出可执行的下一步方案。
"""

# =============================================================================
# 3. 创建 Agent (带中间件)
# =============================================================================

def get_agent_graph():
    global _GRAPH, _GRAPH_SIGNATURE

    chat_settings = get_chat_settings()
    signature = (chat_settings.provider, chat_settings.model)
    if _GRAPH is not None and _GRAPH_SIGNATURE == signature:
        return _GRAPH

    model = create_chat_model()
    _GRAPH = create_agent(
        model=model,
        tools=tools,
        middleware=[dataset_context_middleware],
    )
    _GRAPH_SIGNATURE = signature
    print(f"[AgentModel] provider={chat_settings.provider} model={chat_settings.model}")
    return _GRAPH


graph = get_agent_graph()