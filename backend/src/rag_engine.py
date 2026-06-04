"""RAG 引擎 — 已重构到 src/rag/ 模块

本文件保留向后兼容接口，内部委托给 src/rag/RAGFacade。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.config import get_embedding_settings, get_env_bool, get_env_text
from src.rag import RAGFacade

# ----- 保留的数据类 (向后兼容) -----


@dataclass
class KnowledgeChunk:
    chunk_id: str
    source: str
    text: str
    start: int
    end: int


@dataclass
class RetrievalHit:
    chunk_id: str
    source: str
    text: str
    vector_score: float
    lexical_score: float
    final_score: float


# ----- 单例 RAGFacade -----

_KB_RERANK_ENABLED = get_env_bool("KB_RERANK_ENABLED", True)
_facade: RAGFacade | None = None


def _get_facade() -> RAGFacade:
    global _facade
    if _facade is None:
        rerank_backend = get_env_text("KB_RERANK_BACKEND", "scoring")
        persist_dir = get_env_text("CHROMA_PERSIST_DIR", "") or str(
            Path(__file__).resolve().parent.parent / "knowledge_base" / "chroma_db"
        )
        chunk_strategy = get_env_text("KB_CHUNK_STRATEGY", "semantic")
        _facade = RAGFacade(
            model=None,  # 延迟注入，RAGFacade 可选
            persist_dir=persist_dir,
            chunk_strategy=chunk_strategy,
            rerank_backend=rerank_backend,
        )
    return _facade


def set_rag_model(model) -> None:
    """注入 LLM 模型（用于 LLM Reranker / HyDE）。"""
    facade = _get_facade()
    facade.model = model
    from src.rag.reranker import LLMReranker
    facade.llm_reranker = LLMReranker(model)


# ----- 兼容接口 -----


def rebuild_knowledge_base(reset: bool = True) -> dict[str, Any]:
    source_dir = Path(get_env_text("KB_SOURCE_DIR", "")) or (
        Path(__file__).resolve().parent.parent / "knowledge_base" / "docs"
    )
    facade = _get_facade()
    result = facade.rebuild(source_dir=str(source_dir), reset=reset)
    return result


def ensure_knowledge_base_loaded(auto_rebuild: bool = True) -> dict[str, Any]:
    facade = _get_facade()
    if facade._doc_count == 0 and auto_rebuild:
        return rebuild_knowledge_base(reset=True)
    return {"doc_count": facade._doc_count, "chunk_count": len(facade._chunks)}


def get_knowledge_base_stats() -> dict[str, Any]:
    facade = _get_facade()
    return facade.get_stats()


def retrieve_knowledge(query: str, top_k: int = 4, min_score: float = 0.18) -> list[RetrievalHit]:
    facade = _get_facade()
    hits = facade.retrieve(query, top_k=top_k * 3 if _KB_RERANK_ENABLED else top_k, expand_query=False)

    result: list[RetrievalHit] = []
    for h in hits:
        score = h.get("final_score", h.get("score", 0.0))
        if score < min_score and len(result) >= 1:
            continue
        result.append(RetrievalHit(
            chunk_id=h.get("chunk_id", ""),
            source=h.get("source", ""),
            text=h.get("text", ""),
            vector_score=h.get("vector_score", 0.0),
            lexical_score=h.get("keyword_score", h.get("bm25_score", 0.0)),
            final_score=score,
        ))
        if len(result) >= top_k:
            break

    return result


def format_retrieval_hits(hits: list[RetrievalHit]) -> str:
    if not hits:
        return "知识库中未检索到高置信度内容。"

    blocks: list[str] = []
    for idx, hit in enumerate(hits, start=1):
        snippet = hit.text.replace("\n", " ").strip()
        if len(snippet) > 260:
            snippet = snippet[:260] + "..."
        blocks.append(
            f"[{idx}] 来源: {hit.source}\n"
            f"chunk_id: {hit.chunk_id}\n"
            f"score: {hit.final_score:.3f} (vector={hit.vector_score:.3f}, lexical={hit.lexical_score:.3f})\n"
            f"内容: {snippet}"
        )

    return "\n\n".join(blocks)
