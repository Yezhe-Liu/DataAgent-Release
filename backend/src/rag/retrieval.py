"""融合检索: BM25 稀疏 + 稠密向量 + 关键词

三路融合加权: 0.3×BM25 + 0.5×Vector + 0.2×Keyword
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Tokenizer (复用项目中日混合分词逻辑)
# ---------------------------------------------------------------------------

_CJK = re.compile(r"[一-鿿]")
_TOKEN = re.compile(r"[一-鿿]|[a-zA-Z0-9_]+")


def _tokenize(text: str) -> list[str]:
    cleaned = (text or "").lower()
    if not cleaned:
        return []
    matches = _TOKEN.findall(cleaned)
    tokens: list[str] = list(matches)
    prev = None
    for m in matches:
        if _CJK.match(m):
            if prev is not None:
                tokens.append(prev + m)
            prev = m
        else:
            prev = None
    return tokens


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------


class BM25Retriever:
    def __init__(self):
        self._bm25: BM25Okapi | None = None
        self._chunks: list[dict[str, Any]] = []
        self._tokenized: list[list[str]] = []

    def index(self, chunks: list[dict[str, Any]]) -> None:
        self._chunks = chunks
        self._tokenized = [_tokenize(c.get("text", "")) for c in chunks]
        if self._tokenized:
            self._bm25 = BM25Okapi(self._tokenized)
        else:
            self._bm25 = None

    def search(self, query: str, top_k: int = 8) -> list[tuple[int, float]]:
        if self._bm25 is None or not self._tokenized:
            return []
        qt = _tokenize(query)
        scores = self._bm25.get_scores(qt)
        # 归一化到 [0, 1]
        smax = float(scores.max())
        if smax > 0:
            scores = scores / smax
        ranked = np.argsort(scores)[::-1]
        return [(int(ranked[i]), float(scores[ranked[i]])) for i in range(min(top_k, len(ranked))) if scores[ranked[i]] > 0]


# ---------------------------------------------------------------------------
# Keyword
# ---------------------------------------------------------------------------


def _keyword_score(query_tokens: set[str], text: str, source: str = "") -> float:
    doc_tokens = set(_tokenize(text))
    source_tokens = set(_tokenize(source.replace("/", " ").replace("_", " ")))
    doc_tokens.update(source_tokens)
    if not doc_tokens or not query_tokens:
        return 0.0
    common = len(doc_tokens.intersection(query_tokens))
    base = common / max(1.0, np.sqrt(len(doc_tokens) * len(query_tokens)))
    source_boost = 0.12 * (len(source_tokens.intersection(query_tokens)) / len(query_tokens)) if source_tokens else 0.0
    return min(1.0, base + source_boost)


# ---------------------------------------------------------------------------
# Fusion Retriever
# ---------------------------------------------------------------------------


class FusionRetriever:
    """BM25(0.3) + Vector(0.5) + Keyword(0.2) 三路融合。"""

    def __init__(self, vector_search_fn=None):
        self._bm25 = BM25Retriever()
        self._vector_search = vector_search_fn  # (query, top_k) -> list[dict]
        self._chunks: list[dict[str, Any]] = []

    def index(self, chunks: list[dict[str, Any]]) -> None:
        self._chunks = chunks
        self._bm25.index(chunks)

    def search(self, query: str, top_k: int = 4) -> list[dict[str, Any]]:
        qt = set(_tokenize(query))
        n = len(self._chunks)
        candidate_k = min(top_k * 3, n)
        if candidate_k == 0:
            return []

        # BM25 分量
        bm25_results = self._bm25.search(query, candidate_k)
        bm25_map = {idx: score for idx, score in bm25_results}

        # Vector 分量
        vector_map: dict[int, float] = {}
        if self._vector_search:
            try:
                vec_hits = self._vector_search(query, candidate_k)
                for h in vec_hits:
                    idx = h.get("chunk_index", -1)
                    score = h.get("score", h.get("final_score", 0.0))
                    if idx >= 0:
                        vector_map[idx] = max(vector_map.get(idx, 0.0), score)
            except Exception:
                pass

        # 融合
        all_indices = set(bm25_map.keys()) | set(vector_map.keys())
        scored: list[tuple[int, float]] = []
        for idx in all_indices:
            if idx >= n:
                continue
            bm25_s = bm25_map.get(idx, 0.0)
            vec_s = vector_map.get(idx, 0.0)
            kw_s = _keyword_score(qt, self._chunks[idx].get("text", ""), self._chunks[idx].get("source", ""))
            fused = 0.3 * bm25_s + 0.5 * vec_s + 0.2 * kw_s
            scored.append((idx, fused))

        scored.sort(key=lambda x: x[1], reverse=True)

        hits: list[dict[str, Any]] = []
        for idx, fused_score in scored[:top_k]:
            chunk = self._chunks[idx]
            hits.append({
                **chunk,
                "bm25_score": bm25_map.get(idx, 0.0),
                "vector_score": vector_map.get(idx, 0.0),
                "keyword_score": _keyword_score(qt, chunk.get("text", ""), chunk.get("source", "")),
                "final_score": fused_score,
            })

        return hits
