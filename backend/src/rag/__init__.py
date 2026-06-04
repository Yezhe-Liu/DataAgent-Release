"""RAG 模块 — 统一门面

对外暴露 RAGFacade，封装分块/索引/检索/重排/查询处理全部流程。
模块间零依赖：通过构造函数注入 model + embedding_fn。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.rag.chunking import Chunk, create_chunker
from src.rag.indexing import MultiGranularIndex
from src.rag.query import QueryProcessor
from src.rag.reranker import LLMReranker, rerank_hits
from src.rag.retrieval import FusionRetriever


class RAGFacade:
    """RAG 统一入口，组合所有子模块。"""

    def __init__(
        self,
        model: Any = None,
        persist_dir: str | Path = "",
        chunk_strategy: str = "semantic",
        rerank_backend: str = "scoring",
    ):
        self.model = model
        self.chunk_strategy = chunk_strategy
        self.rerank_backend = rerank_backend

        # 索引
        persist = str(persist_dir) if persist_dir else "backend/knowledge_base/chroma_db"
        self.index = MultiGranularIndex(persist_dir=persist)

        # 检索器 (vector function 由 index 内部管理)
        self.retriever = FusionRetriever(
            vector_search_fn=self._vector_search_wrapper,
        )

        # 重排器
        self.llm_reranker = LLMReranker(model) if model and rerank_backend == "llm" else None

        # 查询处理
        self.query_processor = QueryProcessor(model=model)

        # 内部状态
        self._chunks: list[dict[str, Any]] = []
        self._doc_count = 0

    # ----- Public API -----

    def rebuild(self, source_dir: str | Path, reset: bool = True) -> dict[str, Any]:
        """完整重建流程: 扫描文档 → 分块 → 嵌入 → 索引。"""
        source = Path(source_dir)
        if not source.exists():
            return {"status": "empty", "message": f"目录不存在: {source_dir}"}

        doc_paths = self._list_docs(source)
        if not doc_paths:
            return {"status": "empty", "message": f"未在 {source_dir} 发现支持的文档"}

        if reset:
            self.index.reset()

        # Phase 1: 分块
        chunker = create_chunker(self.chunk_strategy)
        all_chunks: list[Chunk] = []
        for path in doc_paths:
            text = path.read_text(encoding="utf-8", errors="ignore")
            all_chunks.extend(chunker.chunk(text, source=path.name))

        if not all_chunks:
            return {"status": "empty", "message": "没有生成任何分块"}

        # Phase 2: 建索引
        stats = self.index.index_documents(doc_paths, self.chunk_strategy)

        # Phase 3: 初始化 FusionRetriever (BM25 + Vector)
        self._chunks = [{
            "chunk_id": c.chunk_id,
            "source": c.source,
            "text": c.text,
            "start_char": c.start_char,
            "end_char": c.end_char,
            "chunk_index": c.chunk_index,
            "metadata": c.metadata,
        } for c in all_chunks]
        self.retriever.index(self._chunks)
        self._doc_count = len(doc_paths)

        return {
            "status": "ok",
            "doc_count": stats["documents"],
            "summary_chunks": stats["summary_chunks"],
            "paragraph_chunks": stats["paragraph_chunks"],
            "chunk_strategy": self.chunk_strategy,
        }

    def retrieve(self, query: str, top_k: int = 4, expand_query: bool = False) -> list[dict[str, Any]]:
        """完整检索流程: 查询处理 → 融合检索 → 重排。"""
        if not self._chunks:
            return []

        # Step 1: 查询处理
        if expand_query:
            queries = self.query_processor.process(query, use_expand=True)
            all_hits: list[dict[str, Any]] = []
            seen: set[str] = set()
            for q in queries:
                for hit in self.retriever.search(q, top_k=top_k * 2):
                    cid = hit.get("chunk_id", "")
                    if cid not in seen:
                        seen.add(cid)
                        all_hits.append(hit)
            all_hits.sort(key=lambda h: h.get("final_score", 0), reverse=True)
            hits = all_hits[:top_k * 2]
        else:
            hits = self.retriever.search(query, top_k=top_k * 2)

        # Step 2: 重排
        if len(hits) > top_k:
            hits = rerank_hits(
                query=query,
                hits=hits,
                top_k=top_k,
                backend=self.rerank_backend,
                llm_reranker=self.llm_reranker,
            )

        return hits[:top_k]

    def get_stats(self) -> dict[str, Any]:
        idx_stats = self.index.get_stats()
        return {
            "doc_count": self._doc_count,
            "chunk_count": len(self._chunks),
            "chunk_strategy": self.chunk_strategy,
            "rerank_backend": self.rerank_backend,
            **idx_stats,
        }

    # ----- Internal -----

    def _vector_search_wrapper(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """将 index.retrieve 适配为 FusionRetriever 需要的签名。"""
        return self.index.retrieve(query, top_k=top_k)

    @staticmethod
    def _list_docs(source_dir: Path) -> list[Path]:
        exts = {".txt", ".md", ".csv", ".json"}
        if not source_dir.exists():
            return []
        return sorted(p for p in source_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts)
