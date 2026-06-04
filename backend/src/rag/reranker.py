"""重排器模块

支持三种重排策略:
  - scoring:       多信号融合 (unigram + bigram + 原始分 + 长度惩罚)
  - cross-encoder: sentence-transformers CrossEncoder 语义精排
  - llm:           LLM Listwise Rerank (一次调用评多个候选)
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

# ---------------------------------------------------------------------------
# Tokenizer (复用)
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    from src.rag_engine import _tokenize as _rag_tokenize
    return _rag_tokenize(text)


def _bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    return {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}


# ---------------------------------------------------------------------------
# 1. Scoring Reranker (多信号融合)
# ---------------------------------------------------------------------------


def scoring_rerank(query: str, hits: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    if len(hits) <= top_k:
        return hits

    qt = _tokenize(query)
    qt_set = set(qt)
    qt_bi = _bigrams(qt)

    scored: list[tuple[float, dict]] = []
    for hit in hits:
        text = hit.get("text", "")
        ct = _tokenize(text)
        ct_set = set(ct)

        # S1: unigram coverage
        coverage = len(qt_set & ct_set) / len(qt_set) if qt_set else 0.0
        # S2: bigram overlap
        cbi = _bigrams(ct)
        bi_overlap = len(qt_bi & cbi) / len(qt_bi) if qt_bi else 0.0
        # S3: original score
        orig = hit.get("final_score", hit.get("score", 0.0))
        # S4: length factor
        length_factor = min(1.0, math.log2(max(len(ct), 1) + 1) / 6.0)

        rerank = 0.35 * orig + 0.30 * coverage + 0.20 * bi_overlap + 0.15 * length_factor
        scored.append((rerank, hit))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {**hit, "rerank_score": s, "rerank_method": "scoring"}
        for s, hit in scored[:top_k]
    ]


# ---------------------------------------------------------------------------
# 2. Cross-Encoder Reranker
# ---------------------------------------------------------------------------

_CROSS_ENCODER = None


def _get_cross_encoder():
    global _CROSS_ENCODER
    if _CROSS_ENCODER is not None:
        return _CROSS_ENCODER
    model_name = os.getenv("KB_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-2-v2")
    try:
        from sentence_transformers import CrossEncoder
        _CROSS_ENCODER = CrossEncoder(model_name, device="cpu")
        print(f"[Reranker] CrossEncoder loaded: {model_name}")
        return _CROSS_ENCODER
    except Exception as err:
        print(f"[Reranker] CrossEncoder load failed: {err}")
        return None


def cross_encoder_rerank(query: str, hits: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    if len(hits) <= top_k:
        return hits

    encoder = _get_cross_encoder()
    if encoder is None:
        return scoring_rerank(query, hits, top_k)

    pairs = [(query, hit.get("text", "")) for hit in hits]
    try:
        scores = encoder.predict(pairs, show_progress_bar=False)
    except Exception as err:
        print(f"[Reranker] CrossEncoder predict failed: {err}")
        return scoring_rerank(query, hits, top_k)

    scored = sorted(zip(scores, hits), key=lambda x: x[0], reverse=True)
    return [
        {**hit, "rerank_score": float(s), "rerank_method": "cross-encoder"}
        for s, hit in scored[:top_k]
    ]


# ---------------------------------------------------------------------------
# 3. LLM-as-Reranker (Listwise)
# ---------------------------------------------------------------------------


LLM_RERANK_PROMPT = """你是一个文档相关性评分器。给定用户查询和一组候选文档，为每个文档打分 (0-100)。

评分标准: 100=完美回答查询, 70+=高度相关, 40-70=部分相关, <40=不相关

用户查询: {query}

候选文档:
{documents}

返回 JSON 格式: {{"scores": [{{"index": 0, "score": 85, "reason": "..."}}, ...]}}
仅输出 JSON，不要有其他文字。"""


class LLMReranker:
    def __init__(self, model):
        self.model = model

    def rerank(self, query: str, hits: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if len(hits) <= top_k:
            return hits

        docs_text = "\n\n".join(
            f"--- 文档 [{i}] ---\n{hit.get('text', '')[:400]}"
            for i, hit in enumerate(hits)
        )

        try:
            response = self.model.invoke(LLM_RERANK_PROMPT.format(query=query, documents=docs_text))
            content = response.content if hasattr(response, "content") else str(response)
            # 提取 JSON
            content = content.strip()
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            data = json.loads(content)
            score_map = {item["index"]: item["score"] / 100.0 for item in data.get("scores", [])}
        except Exception as err:
            print(f"[Reranker] LLM rerank failed: {err}")
            return scoring_rerank(query, hits, top_k)

        scored = [(score_map.get(i, 0.0), hit) for i, hit in enumerate(hits)]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {**hit, "rerank_score": s, "rerank_method": "llm"}
            for s, hit in scored[:top_k]
        ]


# ---------------------------------------------------------------------------
# Unified entry
# ---------------------------------------------------------------------------


def rerank_hits(
    query: str,
    hits: list[dict[str, Any]],
    top_k: int,
    backend: str = "scoring",
    llm_reranker: LLMReranker | None = None,
) -> list[dict[str, Any]]:
    if backend == "cross-encoder":
        return cross_encoder_rerank(query, hits, top_k)
    if backend == "llm" and llm_reranker is not None:
        return llm_reranker.rerank(query, hits, top_k)
    return scoring_rerank(query, hits, top_k)
