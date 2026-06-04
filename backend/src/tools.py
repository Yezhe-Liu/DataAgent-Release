import os
import io
import contextlib
import platform
import urllib.parse
import urllib.request
import json
import matplotlib
# [关键] 永久设置非交互后端，防止服务器报错
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from src.data_manager import get_dataframe
from src.rag_engine import (
    retrieve_knowledge as kb_retrieve,
    format_retrieval_hits,
    get_knowledge_base_stats,
)
import uuid
from src.runtime_context import get_current_user_id

load_dotenv()

# --- 解决中文乱码 ---
def configure_fonts():
    system_name = platform.system()
    if system_name == 'Windows':
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    elif system_name == 'Darwin':
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC']
    else:
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
configure_fonts()

# -----------------------------------------------------------------------------
# 工具 1: Python 计算 (回归你的单作用域逻辑，但增加 stdout 捕获)
# -----------------------------------------------------------------------------
class PythonCodeInput(BaseModel):
    py_code: str = Field(description="Python代码。可以使用变量 df。")

@tool(args_schema=PythonCodeInput)
def python_inter(py_code: str):
    """
    执行 Python 代码。
    """
    df = get_dataframe()
    if df is None:
        return "错误：当前没有加载任何数据。"

    # [回归原始逻辑]：只使用一个 env 字典，同时充当 globals 和 locals
    # 这样列表推导式就不会报错了
    env = {
        "df": df, 
        "pd": pd, 
        "np": np, 
        "result": None
    }
    
    output_buffer = io.StringIO()
    
    try:
        # 使用 contextlib 捕获 print() 的内容
        with contextlib.redirect_stdout(output_buffer):
            # [关键] 只传一个字典！
            exec(py_code, env)
            
        # 1. 优先获取 stdout (print的内容)
        stdout_content = output_buffer.getvalue().strip()
        
        # 2. 其次获取 result 变量 (如果用户写了 result=...)
        result_content = ""
        if "result" in env and env["result"] is not None:
            result_content = str(env["result"])

        # 3. 实在不行，尝试 eval 最后一行 (模仿你原始代码的逻辑)
        # 这是一个兜底策略，为了让 '2+2' 这种直接返回 4
        eval_result = ""
        if not stdout_content and not result_content:
            try:
                # 尝试计算最后一行表达式
                last_line = py_code.strip().split('\n')[-1]
                # 只有当最后一行看起来像表达式时才 eval
                if not last_line.startswith('print') and '=' not in last_line:
                    eval_result = str(eval(last_line, env))
            except:
                pass

        # 组合输出
        final_output = []
        if stdout_content: final_output.append(f"【打印输出】\n{stdout_content}")
        if result_content: final_output.append(f"【计算结果】\n{result_content}")
        if eval_result: final_output.append(f"【表达式值】\n{eval_result}")
        
        if not final_output:
            return "代码执行成功，但没有输出。请使用 print() 打印结果。"
            
        return "\n\n".join(final_output)

    except Exception as e:
        return f"代码执行报错: {e}"

# -----------------------------------------------------------------------------
# 工具 2: 绘图 (回归简单逻辑，保留路径配置)
# -----------------------------------------------------------------------------
class FigCodeInput(BaseModel):
    py_code: str = Field(description="绘图代码。需生成 fig 对象。")
    fname: str = Field(description="图像变量名，例如 'fig'。")

@tool(args_schema=FigCodeInput)
def fig_inter(py_code: str, fname: str) -> str:
    """
    执行绘图代码并保存。
    """
    df = get_dataframe()
    if df is None: return "错误：无数据。"
    
    print(f">>> 开始绘图: {fname}")
    
    # 清理画布
    plt.clf()
    plt.close('all')

    # [回归原始逻辑]：单字典作用域
    env = {
        "df": df, 
        "pd": pd, 
        "plt": plt, 
        "sns": sns
    }
    
    # 路径配置 (保留这部分，因为必须存到 static 目录前端才能看)
    # 确保这个路径和 server.py 挂载的路径一致
    save_dir = os.path.join(os.getcwd(), "static", "images")
    os.makedirs(save_dir, exist_ok=True)

    try:
        # [关键] 只传一个字典！
        exec(py_code, env)
        
        fig = env.get(fname)
        # 容错：如果用户没赋值给 fname，尝试获取当前 fig
        if not fig:
            fig = plt.gcf()
            
        if fig:
            # 文件名处理
            current_user_id = get_current_user_id().strip() or "shared"
            safe_user_id = current_user_id.replace("/", "_").replace("\\", "_").replace(":", "_")
            file_name = f"{safe_user_id}-{uuid.uuid4().hex[:8]}-{fname}.png"
            abs_path = os.path.join(save_dir, file_name)
            
            # 保存
            fig.savefig(abs_path, bbox_inches='tight', dpi=100)
            print(f">>> 图片保存成功: {abs_path}")
            
            # 返回前端标记
            return f"IMAGE_GENERATED: {file_name}"
        else:
            return "绘图代码执行完毕，但未找到图像对象。"
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"绘图报错: {e}"
    finally:
        plt.close('all')


class KnowledgeRetrieveInput(BaseModel):
    query: str = Field(description="用户的问题或检索查询。")
    top_k: int = Field(default=4, ge=1, le=8, description="召回片段数量。")


@tool(args_schema=KnowledgeRetrieveInput)
def retrieve_knowledge(query: str, top_k: int = 4) -> str:
    """
    从私有知识库中检索最相关的文档片段。
    """
    hits = kb_retrieve(query=query, top_k=top_k)
    if not hits:
        stats = get_knowledge_base_stats()
        if stats.get("chunk_count", 0) <= 0:
            return (
                "当前知识库为空。请先将文档放入 backend/knowledge_base/docs，"
                "然后调用 /kb/rebuild 重建索引。"
            )
        return "未在知识库中检索到高置信度内容，请尝试改写问题后再次检索。"

    header = "以下是知识库检索结果。回答时请优先基于这些片段，并尽量给出来源编号：\n\n"
    return header + format_retrieval_hits(hits)


@tool
def knowledge_base_status() -> str:
    """
    查看当前私有知识库的索引状态。
    """
    stats = get_knowledge_base_stats()
    return (
        f"source_dir={stats.get('source_dir')}\n"
        f"doc_count={stats.get('doc_count')}\n"
        f"chunk_count={stats.get('chunk_count')}\n"
        f"embedding_provider={stats.get('embedding_provider')}\n"
        f"embedding_model={stats.get('embedding_model')}"
    )


class ExternalSearchInput(BaseModel):
    query: str = Field(description="需要联网检索的问题。")
    max_results: int = Field(default=3, ge=1, le=5, description="最多返回结果数。")


def _flatten_related_topics(raw_topics: list) -> list[dict]:
    flat: list[dict] = []
    for item in raw_topics:
        if isinstance(item, dict) and "Topics" in item:
            flat.extend(_flatten_related_topics(item.get("Topics", [])))
            continue
        if isinstance(item, dict):
            flat.append(item)
    return flat


@tool(args_schema=ExternalSearchInput)
def external_search(query: str, max_results: int = 3) -> str:
    """
    使用 DuckDuckGo Instant Answer API 获取外部公开信息（无需 API Key）。
    """
    try:
        params = urllib.parse.urlencode(
            {
                "q": query,
                "format": "json",
                "no_redirect": "1",
                "no_html": "1",
            }
        )
        url = f"https://api.duckduckgo.com/?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "DataAgent/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="ignore"))

        candidates: list[dict] = []
        abstract = (payload.get("AbstractText") or "").strip()
        if abstract:
            candidates.append(
                {
                    "title": payload.get("Heading") or "DuckDuckGo 摘要",
                    "snippet": abstract,
                    "url": payload.get("AbstractURL") or "",
                }
            )

        related = _flatten_related_topics(payload.get("RelatedTopics", []))
        for item in related:
            text = (item.get("Text") or "").strip()
            if not text:
                continue
            candidates.append(
                {
                    "title": item.get("FirstURL", "Result").split("/")[-1] or "Result",
                    "snippet": text,
                    "url": item.get("FirstURL") or "",
                }
            )

        if not candidates:
            return "外部搜索未返回有效结果，建议改写关键词后重试。"

        lines = []
        for idx, item in enumerate(candidates[:max_results], start=1):
            lines.append(
                (
                    f"[{idx}] {item['title']}\n"
                    f"摘要: {item['snippet']}\n"
                    f"链接: {item['url']}"
                )
            )

        return "\n\n".join(lines)
    except Exception as e:
        return f"外部搜索失败: {e}"