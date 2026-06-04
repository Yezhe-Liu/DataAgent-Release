"""多粒度索引管理

支持文档摘要层 + 段落层两级索引:
  - 摘要层 (summary): 每篇文档生成一个摘要向量
  - 段落层 (paragraph): 每个分块一个向量

两阶段检索: 粗筛 (摘要层) → 精排 (段落层)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import get_embedding_settings
from src.rag.chunking import Chunk, create_chunker

# ---------------------------------------------------------------------------
# Embedding helper (reuses existing provider logic)
# ---------------------------------------------------------------------------


def _build_embedding_fn():
    """构建 ChromaDB embedding function。复用 rag_engine 的 embedding 逻辑，但以 Chroma EF 形式暴露。"""
    settings = get_embedding_settings()

    if settings.provider == "ollama":
        try:
            from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
            return OllamaEmbeddingFunction(
                model_name=settings.model,
                url=settings.ollama_base_url,
            )
        except Exception:
            pass

    if settings.provider == "dashscope":
        from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
        return OpenAIEmbeddingFunction(
            model_name=settings.model,
            api_key=settings.dashscope_api_key,
            api_base=settings.dashscope_base_url,
        )

    # fallback: use default (all-MiniLM-L6-v2)
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    return DefaultEmbeddingFunction()


# ---------------------------------------------------------------------------
# Multi-Granularity Index
# ---------------------------------------------------------------------------


class MultiGranularIndex:
    """两级索引: summary (粗筛) + paragraph (精排)。"""

    SUMMARY_COLLECTION = "kb_summaries"
    PARAGRAPH_COLLECTION = "kb_paragraphs"

    def __init__(self, persist_dir: str | Path, embedding_fn=None):
        self.persist_dir = str(persist_dir)
        self.embedding_fn = embedding_fn or _build_embedding_fn()
        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    # ----- Collection management -----

    def reset(self) -> None:
        for name in [self.SUMMARY_COLLECTION, self.PARAGRAPH_COLLECTION]:
            try:
                self._client.delete_collection(name)
            except Exception:
                pass

    @property
    def summary_col(self):
        return self._client.get_or_create_collection(
            name=self.SUMMARY_COLLECTION,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def paragraph_col(self):
        return self._client.get_or_create_collection(
            name=self.PARAGRAPH_COLLECTION,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    # ----- Indexing -----

    def index_documents(self, doc_paths: list[Path], chunk_strategy: str = "semantic") -> dict[str, int]:
        chunker = create_chunker(chunk_strategy)
        summary_ids, summary_docs, summary_meta = [], [], []
        para_ids, para_docs, para_meta = [], [], []

        for path in doc_paths:
            source = path.name
            text = path.read_text(encoding="utf-8", errors="ignore")
            chunks = chunker.chunk(text, source=source)

            if not chunks:
                continue

            # 摘要层: 取文档前 500 字作为概要
            summary_text = text[:500].replace("\n", " ")
            summary_ids.append(f"summary::{source}")
            summary_docs.append(summary_text)
            summary_meta.append({"source": source, "chunk_count": len(chunks)})

            # 段落层
            for c in chunks:
                para_ids.append(c.chunk_id)
                para_docs.append(c.text)
                para_meta.append({
                    "source": c.source,
                    "start": c.start_char,
                    "end": c.end_char,
                    "chunk_index": c.chunk_index,
                })

        # Batch upsert
        if summary_ids:
            self._batch_upsert(self.summary_col, summary_ids, summary_docs, summary_meta)
        if para_ids:
            self._batch_upsert(self.paragraph_col, para_ids, para_docs, para_meta)

        return {
            "documents": len(doc_paths),
            "summary_chunks": len(summary_ids),
            "paragraph_chunks": len(para_ids),
        }

    # ----- Retrieval -----

    def retrieve(self, query: str, top_k: int = 4) -> list[dict[str, Any]]:
        """两阶段检索: 粗筛 (summaries) → 精排 (paragraphs)。"""
        # Stage 1: 粗筛 — 找到相关文档
        try:
            sum_results = self.summary_col.query(
                query_texts=[query],
                n_results=min(3, max(1, self.summary_col.count())),
                include=["metadatas"],
            )
            relevant_sources: set[str] = set()
            if sum_results and sum_results.get("metadatas"):
                for meta_list in sum_results["metadatas"]:
                    for meta in meta_list:
                        if meta and meta.get("source"):
                            relevant_sources.add(meta["source"])
        except Exception:
            relevant_sources = set()

        # Stage 2: 精排 — 在相关文档的段落中检索
        para_count = self.paragraph_col.count()
        if para_count == 0:
            return []

        n_results = min(top_k * 2, para_count)
        try:
            para_results = self.paragraph_col.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return []

        if not para_results or not para_results.get("documents"):
            return []

        docs = para_results["documents"][0]
        metas = para_results["metadatas"][0]
        dists = para_results["distances"][0]

        hits: list[dict[str, Any]] = []
        for doc, meta, dist in zip(docs, metas, dists):
            source = meta.get("source", "unknown") if meta else "unknown"
            score = max(0.0, 1.0 - float(dist))

            # 粗筛加权: 来自匹配源文档的段落加分
            if source in relevant_sources:
                score += 0.05

            hits.append({
                "chunk_id": meta.get("chunk_id", source) if meta else source,
                "source": source,
                "text": doc,
                "score": score,
                "chunk_index": meta.get("chunk_index", 0) if meta else 0,
            })

        hits.sort(key=lambda h: h["score"], reverse=True)
        return hits[:top_k]

    # ----- Helpers -----

    def get_stats(self) -> dict[str, Any]:
        return {
            "summary_count": self.summary_col.count(),
            "paragraph_count": self.paragraph_col.count(),
            "persist_dir": self.persist_dir,
        }

    @staticmethod
    def _batch_upsert(collection, ids, docs, metas, batch_size=256):
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            collection.upsert(
                ids=ids[i:end],
                documents=docs[i:end],
                metadatas=metas[i:end],
            )
