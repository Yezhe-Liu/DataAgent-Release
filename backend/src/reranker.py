"""
重排器模块 —— 在混合检索初筛之后对候选片段做二次精排。

支持两种后端（通过环境变量 KB_RERANK_BACKEND 切换）：

1. scoring  （默认）基于多信号融合的轻量重排，无需额外模型，适合低资源环境。
2. cross-encoder  使用 sentence-transformers CrossEncoder 做语义精排，需安装
   sentence-transformers 并首次启动时自动下载模型（默认 cross-encoder/ms-marco-MiniLM-L-2-v2，~20 MB）。
   模型运行在 CPU 上，不占用 GPU 显存。
"""

from __future__ import annotations

import math
import os
from typing import Any


def _tokenize(text: str) -> list[str]:
    """延迟复用 rag_engine 中的统一分词器，避免循环导入。"""
    from src.rag_engine import _tokenize as _rag_tokenize  # noqa: WPS433

    return _rag_tokenize(text)


def _bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    return {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}


# ---------------------------------------------------------------------------
# 1. 多信号融合重排（Scoring Reranker）
# ---------------------------------------------------------------------------

def _scoring_rerank(
    query: str,
    hits: list[Any],
    top_k: int,
) -> list[Any]:
    """
    综合四项信号做重排：
      S1  query-term 覆盖率（unigram coverage）
      S2  bigram 重叠率
      S3  原始混合检索得分（保留初筛信号）
      S4  片段长度惩罚（过短片段信息量低）
    """
    query_tokens = _tokenize(query)
    query_token_set = set(query_tokens)
    query_bi = _bigrams(query_tokens)

    scored: list[tuple[float, Any]] = []
    for hit in hits:
        chunk_tokens = _tokenize(hit.text)
        chunk_token_set = set(chunk_tokens)

        # S1: unigram coverage
        if query_token_set:
            coverage = len(query_token_set & chunk_token_set) / len(query_token_set)
        else:
            coverage = 0.0

        # S2: bigram overlap
        chunk_bi = _bigrams(chunk_tokens)
        if query_bi:
            bigram_overlap = len(query_bi & chunk_bi) / len(query_bi)
        else:
            bigram_overlap = 0.0

        # S3: 原始混合检索得分
        original_score = getattr(hit, "final_score", 0.0)

        # S4: 长度惩罚 —— 太短的片段信息量低
        length_factor = min(1.0, math.log2(max(len(chunk_tokens), 1) + 1) / 6.0)

        # 加权融合
        rerank_score = (
            0.35 * original_score
            + 0.30 * coverage
            + 0.20 * bigram_overlap
            + 0.15 * length_factor
        )
        scored.append((rerank_score, hit))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [hit for _, hit in scored[:top_k]]


# ---------------------------------------------------------------------------
# 2. Cross-Encoder 重排（可选，需 sentence-transformers）
# ---------------------------------------------------------------------------

_CROSS_ENCODER = None


def _get_cross_encoder():
    global _CROSS_ENCODER
    if _CROSS_ENCODER is not None:
        return _CROSS_ENCODER

    model_name = os.getenv(
        "KB_RERANK_MODEL",
        "cross-encoder/ms-marco-MiniLM-L-2-v2",
    )
    try:
        from sentence_transformers import CrossEncoder

        _CROSS_ENCODER = CrossEncoder(model_name, device="cpu")
        print(f"[Reranker] CrossEncoder loaded: {model_name} (CPU)")
        return _CROSS_ENCODER
    except Exception as err:
        print(f"[Reranker] CrossEncoder 加载失败，回退到 scoring 重排: {err}")
        return None


def _cross_encoder_rerank(
    query: str,
    hits: list[Any],
    top_k: int,
) -> list[Any]:
    encoder = _get_cross_encoder()
    if encoder is None:
        return _scoring_rerank(query, hits, top_k)

    pairs = [(query, hit.text) for hit in hits]
    try:
        scores = encoder.predict(pairs, show_progress_bar=False)
    except Exception as err:
        print(f"[Reranker] CrossEncoder predict 失败: {err}")
        return _scoring_rerank(query, hits, top_k)

    scored = sorted(zip(scores, hits), key=lambda x: x[0], reverse=True)
    return [hit for _, hit in scored[:top_k]]


# ---------------------------------------------------------------------------
# 3. 对外统一入口
# ---------------------------------------------------------------------------

def rerank_hits(
    query: str,
    hits: list[Any],
    top_k: int,
) -> list[Any]:
    """
    对混合检索候选结果做二次重排。

    环境变量
    --------
    KB_RERANK_BACKEND : str
        "scoring"（默认）或 "cross-encoder"
    """
    if len(hits) <= top_k:
        return hits

    backend = os.getenv("KB_RERANK_BACKEND", "scoring").strip().lower()

    if backend == "cross-encoder":
        return _cross_encoder_rerank(query, hits, top_k)

    return _scoring_rerank(query, hits, top_k)
