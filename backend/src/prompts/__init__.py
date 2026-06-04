"""提示词中台 —— 统一管理所有节点的静态前缀与动态模版。

设计目标:
  1. 将所有 System Prompt 拆分为 STATIC_PREFIX (可被 vLLM Prefix Cache 命中)
     与 DYNAMIC_TEMPLATE (用户Query、检索上下文等变动部分)。
  2. align_static_prefix() 确保 static_prefix 的 token 长度对齐 16 的倍数，
     最大化 vLLM APC (Automatic Prefix Caching) 的 Block 命中率。
  3. 所有对齐操作强制埋入性能度量日志，供 LangFuse/Grafana 量化评估。

用法:
  from src.prompts import PromptRegistry
  registry = PromptRegistry.get_instance()

  # 获取对齐后的 static_prefix
  prefix = registry.get_static_prefix("router")

  # 构建完整 SystemMessage
  system_msg = registry.build_system_message("router")
  user_msg = registry.build_user_message("router", user_text="你好")
"""

from __future__ import annotations

import logging
import re
from typing import ClassVar

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 估算 Tokenizer —— 不依赖远程 Tokenizer 加载，基于字符统计估算
# ---------------------------------------------------------------------------

# 粗略估算: 中文字符 ~1.5 token, 英文/数字 ~0.25 token/char (即 4 char/token)
# 保守取 2 字符 ≈ 1 token
_CHARS_PER_TOKEN = 2.0


def _estimate_tokens(text: str) -> int:
    """估算文本的 token 数量 (保守算法，不依赖远程 Tokenizer)。

    DeepSeek V4 tokenizer 使用 BPE，中文字符通常为 1-2 token，英文 ~0.25 token/char。
    保守估算: 总字符数 / 2，向上取整。
    """
    if not text:
        return 0
    return max(1, int(len(text) / _CHARS_PER_TOKEN + 0.5))


# ---------------------------------------------------------------------------
# 16-Token 块边界对齐
# ---------------------------------------------------------------------------

_VLLM_BLOCK_SIZE = 16


def align_static_prefix(text: str, node_name: str = "unknown") -> str:
    """对齐 static_prefix 到 vLLM 16-token Block 边界。

    vLLM APC 以 16 token 为一个 Block 单元缓存 KV Cache。
    如果 static_prefix 的 token 数不是 16 的整数倍，最后一个 partial block 无法缓存。
    通过追加不可见 padding 字符补齐到最近 16 整数倍边界。

    Padding 策略: 交替追加换行符 (\\n) 和空格，避免连续相同字符被 tokenizer 合并。
    """
    if not text:
        logger.info(f"[vLLM Cache Metric] Node={node_name} static_prefix empty, skipped")
        return text

    estimated = _estimate_tokens(text)
    remainder = estimated % _VLLM_BLOCK_SIZE

    if remainder == 0:
        logger.info(
            f"[vLLM Cache Metric] Node={node_name} "
            f"static_prefix_tokens={estimated} block_aligned=True pad=0"
        )
        return text

    # 计算需要的 padding token 数
    pad_tokens_needed = _VLLM_BLOCK_SIZE - remainder
    # 每个 token 估算为 ~2 字符 (保守估计)，故 padding 字符数 = pad_tokens * 2
    pad_chars_needed = pad_tokens_needed * 2

    # 交替使用空格和换行符作为 padding 字符，避免连续相同字符被 tokenizer 合并
    pad_chars: list[str] = []
    for i in range(pad_chars_needed):
        pad_chars.append(" " if i % 2 == 0 else "\n")

    aligned_text = text + "".join(pad_chars)
    aligned_len = _estimate_tokens(aligned_text)
    # 回退检查：如果仍未对齐，暴力追加到对齐
    while aligned_len % _VLLM_BLOCK_SIZE != 0:
        aligned_text += "\n"
        aligned_len = _estimate_tokens(aligned_text)

    logger.info(
        f"[vLLM Cache Metric] Node={node_name} "
        f"static_prefix_tokens={estimated} remainder_mod16={remainder} "
        f"pad_tokens={pad_tokens_needed} aligned_tokens={aligned_len} "
        f"block_aligned={aligned_len % _VLLM_BLOCK_SIZE == 0}"
    )

    return aligned_text


# ---------------------------------------------------------------------------
# Prompt 注册表定义
# ---------------------------------------------------------------------------

# 每个 Prompt 条目: (static_prefix, dynamic_template)
# static_prefix  — 系统人设 + 固定约束 + Few-Shot (对齐到 16-token 边界后缓存)
# dynamic_template — 用户 Query 模板、上下文注入占位符 (每次请求变动)
PromptSpec = tuple[str, str]


class PromptRegistry:
    """提示词注册表 — 单例模式，集中管理所有节点 Prompt。

    注册表在首次访问时自动加载所有 Prompt 模块并执行 16-token 对齐。
    """

    _instance: ClassVar[PromptRegistry | None] = None
    _registry: dict[str, PromptSpec]

    def __init__(self) -> None:
        self._registry = {}
        self._loaded = False

    @classmethod
    def get_instance(cls) -> PromptRegistry:
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load_all()
        return cls._instance

    # ------------------------------------------------------------------
    # 注册
    # ------------------------------------------------------------------

    def register(self, node_name: str, static_prefix: str, dynamic_template: str) -> None:
        """注册一个节点的 Prompt 并执行对齐。"""
        aligned = align_static_prefix(static_prefix, node_name=node_name)
        self._registry[node_name] = (aligned, dynamic_template)

    def _load_all(self) -> None:
        """加载所有内置 Prompt 模块。"""
        if self._loaded:
            return
        self._loaded = True

        from src.prompts import router as _router
        from src.prompts import text_to_sql as _text_to_sql
        from src.prompts import rag_worker as _rag_worker
        from src.prompts import tool_worker as _tool_worker

        # Router
        self.register("router", _router.STATIC_PREFIX, _router.DYNAMIC_TEMPLATE)

        # Text-to-SQL
        self.register("text_to_sql", _text_to_sql.STATIC_PREFIX, _text_to_sql.DYNAMIC_TEMPLATE)

        # RAG Worker 内部节点
        self.register("rewrite", _rag_worker.REWRITE_STATIC_PREFIX, _rag_worker.REWRITE_DYNAMIC_TEMPLATE)
        self.register("grade", _rag_worker.GRADE_STATIC_PREFIX, _rag_worker.GRADE_DYNAMIC_TEMPLATE)
        self.register("generate", _rag_worker.GENERATE_STATIC_PREFIX, _rag_worker.GENERATE_DYNAMIC_TEMPLATE)
        self.register("hallucination_check", _rag_worker.CHECK_STATIC_PREFIX, _rag_worker.CHECK_DYNAMIC_TEMPLATE)

        # Tool Worker
        self.register("tool_execute", _tool_worker.STATIC_PREFIX, _tool_worker.DYNAMIC_TEMPLATE)

        logger.info(
            "[PromptRegistry] Loaded %d prompt entries with 16-token alignment",
            len(self._registry),
        )

    # ------------------------------------------------------------------
    # 查询 API
    # ------------------------------------------------------------------

    def get_static_prefix(self, node_name: str) -> str:
        """获取对齐后的 static_prefix。"""
        spec = self._registry.get(node_name)
        if spec is None:
            logger.warning("[PromptRegistry] Unknown node '%s', returning empty string", node_name)
            return ""
        return spec[0]

    def get_dynamic_template(self, node_name: str) -> str:
        """获取动态模版字符串。"""
        spec = self._registry.get(node_name)
        if spec is None:
            return ""
        return spec[1]

    def get_spec(self, node_name: str) -> PromptSpec:
        """获取完整的 (static_prefix, dynamic_template) 对。"""
        return self._registry.get(node_name, ("", ""))

    def build_system_message(self, node_name: str, **format_kwargs) -> str:
        """构建对齐后的 SystemMessage 内容。

        Args:
            node_name: 注册的节点名称
            **format_kwargs: 动态模版的格式化参数 (如 n=3, user_text=...)

        Returns:
            对齐后的 static_prefix + 格式化后的 dynamic 内容拼接字符串
        """
        static, dynamic = self.get_spec(node_name)
        if not static:
            return dynamic.format(**format_kwargs) if format_kwargs else dynamic
        if format_kwargs and dynamic:
            return f"{static}\n\n{dynamic.format(**format_kwargs)}"
        return static

    def list_nodes(self) -> list[str]:
        """列出所有已注册的节点名。"""
        return list(self._registry.keys())


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------

def get_static_prefix(node_name: str) -> str:
    """获取对齐后的 static_prefix (便捷函数)。"""
    return PromptRegistry.get_instance().get_static_prefix(node_name)


def get_dynamic_template(node_name: str) -> str:
    """获取动态模版 (便捷函数)。"""
    return PromptRegistry.get_instance().get_dynamic_template(node_name)
