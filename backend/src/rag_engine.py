from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from dotenv import load_dotenv

try:
    import chromadb
    from chromadb.utils.embedding_functions import EmbeddingFunction as _ChromaEFBase
except Exception:  # pragma: no cover
    chromadb = None  # type: ignore[assignment]
    _ChromaEFBase = object  # type: ignore[assignment,misc]

from src.config import get_embedding_settings, get_env_bool, get_env_text
from src.reranker import rerank_hits

load_dotenv()


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_KB_SOURCE_DIR = BASE_DIR / "knowledge_base" / "docs"
DEFAULT_INDEX_PATH = BASE_DIR / "knowledge_base" / "vector_index.json"
DEFAULT_CHROMA_DIR = BASE_DIR / "knowledge_base" / "chroma_db"
SUPPORTED_DOC_EXTENSIONS = {".txt", ".md", ".csv", ".json"}
CJK_CHAR_PATTERN = re.compile(r"[\u4e00-\u9fff]")
TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+")
# 分词器版本：修改分词/embedding 逻辑时请递增，启动时会触发自动重建
TOKENIZER_VERSION = "v2-cjk-char-bigram"

KB_VECTOR_BACKEND = get_env_text("KB_VECTOR_BACKEND", "json").lower()
KB_RERANK_ENABLED = get_env_bool("KB_RERANK_ENABLED", True)
CHROMA_COLLECTION_NAME = get_env_text("CHROMA_COLLECTION_NAME", "knowledge_base")


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


_INDEX_LOCK = Lock()
_REBUILD_LOCK = Lock()
_INDEX_CACHE: dict[str, Any] | None = None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _compute_lexical_score(query_tokens: set[str], text: str, source: str = "") -> float:
    token_space = set(_tokenize(text))
    source_tokens = set(_tokenize(source.replace("/", " ").replace("_", " ")))
    token_space.update(source_tokens)
    if not token_space or not query_tokens:
        return 0.0

    common = len(token_space.intersection(query_tokens))
    base_score = common / max(1.0, np.sqrt(len(token_space) * len(query_tokens)))
    source_common = len(source_tokens.intersection(query_tokens))
    source_boost = 0.12 * (source_common / len(query_tokens)) if source_tokens else 0.0
    return min(1.0, base_score + source_boost)


def _tokenize(text: str) -> list[str]:
    """
    面向中英文混合文本的分词：
      - 英文/数字按词切分
      - 中文按“字 + 相邻二元组 (bigram)”扩展
    这样可以让中文查询词与文档片段产生有效的 token 重叠，
    避免像 `[\\u4e00-\\u9fff]+` 这种整串匹配导致 0 重叠。
    """
    cleaned = (text or "").lower()
    if not cleaned:
        return []

    matches = TOKEN_PATTERN.findall(cleaned)
    tokens: list[str] = list(matches)

    # 追加相邻中文字符形成的 bigram，提升中文短语匹配精度
    previous_cjk: str | None = None
    for match in matches:
        if CJK_CHAR_PATTERN.match(match):
            if previous_cjk is not None:
                tokens.append(previous_cjk + match)
            previous_cjk = match
        else:
            previous_cjk = None

    return tokens


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _hash_embed(texts: list[str], dimension: int = 384) -> list[list[float]]:
    vectors: list[list[float]] = []
    for text in texts:
        vector = np.zeros(dimension, dtype=float)
        for token in _tokenize(text):
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            index = int(digest, 16) % dimension
            vector[index] += 1.0
        vectors.append(_normalize_rows(vector)[0].tolist())
    return vectors


def _embed_with_provider(
    texts: list[str],
    provider: str,
    model: str,
    base_url: str,
    api_key: str = "",
) -> list[list[float]]:
    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        embeddings = OllamaEmbeddings(model=model, base_url=base_url)
    elif provider == "dashscope":
        if not api_key:
            raise RuntimeError("EMBEDDING_PROVIDER=dashscope 时必须设置 DASHSCOPE_API_KEY")
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(model=model, api_key=api_key, base_url=base_url)
    else:
        raise ValueError(f"不支持的 Embedding provider: {provider}")

    vectors = embeddings.embed_documents(texts)
    return _normalize_rows(np.asarray(vectors, dtype=float)).tolist()


def _build_embeddings(texts: list[str]) -> tuple[list[list[float]], dict[str, Any]]:
    embedding_settings = get_embedding_settings()
    base_url = (
        embedding_settings.dashscope_base_url
        if embedding_settings.provider == "dashscope"
        else embedding_settings.ollama_base_url
    )

    if texts:
        try:
            vectors = _embed_with_provider(
                texts,
                provider=embedding_settings.provider,
                model=embedding_settings.model,
                base_url=base_url,
                api_key=embedding_settings.dashscope_api_key,
            )
            return vectors, {
                "provider": embedding_settings.provider,
                "model": embedding_settings.model,
                "base_url": base_url,
                "dimension": len(vectors[0]),
            }
        except Exception as error:
            print(f"[RAG] Embedding 不可用，回退到 hash 向量: {error}")

    vectors = _hash_embed(texts)
    return vectors, {
        "provider": "hash",
        "model": "hash-md5-bow",
        "base_url": "",
        "dimension": len(vectors[0]) if vectors else 384,
    }


def _embed_query(text: str, embedding_meta: dict[str, Any]) -> np.ndarray | None:
    provider = embedding_meta.get("provider", "hash")
    dimension = int(embedding_meta.get("dimension", 384))

    if provider in {"ollama", "dashscope"}:
        embedding_settings = get_embedding_settings()
        model = embedding_meta.get("model") or embedding_settings.model
        if provider == "dashscope":
            base_url = embedding_meta.get("base_url") or embedding_settings.dashscope_base_url
            api_key = embedding_settings.dashscope_api_key
        else:
            base_url = embedding_meta.get("base_url") or embedding_settings.ollama_base_url
            api_key = ""
        try:
            vector = _embed_with_provider(
                [text],
                provider=provider,
                model=model,
                base_url=base_url,
                api_key=api_key,
            )[0]
            return np.asarray(vector, dtype=float)
        except Exception as error:
            print(f"[RAG] Query embedding 失败，当前仅使用关键词检索: {error}")
            return None

    vector = _hash_embed([text], dimension=dimension)[0]
    return np.asarray(vector, dtype=float)


def _split_text(text: str, chunk_size: int = 700, overlap: int = 120) -> list[tuple[int, int, str]]:
    clean_text = (text or "").strip()
    if not clean_text:
        return []

    results: list[tuple[int, int, str]] = []
    start = 0
    text_len = len(clean_text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = clean_text[start:end].strip()
        if chunk:
            results.append((start, end, chunk))
        if end >= text_len:
            break
        start = max(0, end - overlap)

    return results


def _read_document(file_path: Path) -> str:
    if file_path.suffix.lower() == ".json":
        raw = file_path.read_text(encoding="utf-8", errors="ignore")
        try:
            parsed = json.loads(raw)
            return json.dumps(parsed, ensure_ascii=False, indent=2)
        except Exception:
            return raw
    return file_path.read_text(encoding="utf-8", errors="ignore")


def _list_documents(source_dir: Path) -> list[Path]:
    if not source_dir.exists():
        return []
    docs: list[Path] = []
    for path in source_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_DOC_EXTENSIONS:
            continue
        docs.append(path)
    return sorted(docs)


def _load_index(index_path: Path) -> dict[str, Any] | None:
    if not index_path.exists():
        return None
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        return data
    except Exception as error:
        print(f"[RAG] 加载索引失败: {error}")
        return None


def _save_index(index_path: Path, payload: dict[str, Any]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _get_index_path() -> Path:
    env_path = get_env_text("KB_INDEX_PATH", "")
    return Path(env_path) if env_path else DEFAULT_INDEX_PATH


def _get_source_dir() -> Path:
    env_dir = get_env_text("KB_SOURCE_DIR", "")
    return Path(env_dir) if env_dir else DEFAULT_KB_SOURCE_DIR


def rebuild_knowledge_base(reset: bool = True) -> dict[str, Any]:
    source_dir = _get_source_dir()
    index_path = _get_index_path()

    documents = _list_documents(source_dir)
    if not documents:
        return {
            "status": "empty",
            "message": f"未在 {source_dir} 发现可索引文档",
            "doc_count": 0,
            "chunk_count": 0,
        }

    chunks: list[KnowledgeChunk] = []
    chunk_texts: list[str] = []

    for file_path in documents:
        raw_text = _read_document(file_path)
        split_chunks = _split_text(raw_text)
        source_name = str(file_path.relative_to(source_dir)).replace("\\", "/")

        for idx, (start, end, text) in enumerate(split_chunks):
            chunk_id = f"{source_name}::chunk_{idx}"
            chunk = KnowledgeChunk(
                chunk_id=chunk_id,
                source=source_name,
                text=text,
                start=start,
                end=end,
            )
            chunks.append(chunk)
            chunk_texts.append(text)

    vectors, embedding_meta = _build_embeddings(chunk_texts)

    payload = {
        "source_dir": str(source_dir),
        "doc_count": len(documents),
        "chunk_count": len(chunks),
        "tokenizer_version": TOKENIZER_VERSION,
        "embedding": embedding_meta,
        "chunks": [asdict(chunk) for chunk in chunks],
        "embeddings": vectors,
    }

    if reset and index_path.exists():
        index_path.unlink()
    _save_index(index_path, payload)

    global _INDEX_CACHE
    with _INDEX_LOCK:
        _INDEX_CACHE = payload

    # ---------- Chroma 同步建库 ----------
    chroma_status = "skipped"
    if KB_VECTOR_BACKEND == "chroma":
        chroma_status = _rebuild_chroma_collection(chunks, chunk_texts, vectors, reset)

    return {
        "status": "ok",
        "message": "知识库索引重建完成",
        "doc_count": payload["doc_count"],
        "chunk_count": payload["chunk_count"],
        "embedding_provider": embedding_meta.get("provider", "unknown"),
        "embedding_model": embedding_meta.get("model", "unknown"),
        "vector_backend": KB_VECTOR_BACKEND,
        "chroma_status": chroma_status,
    }


def _index_is_stale(index: dict[str, Any] | None) -> bool:
    if not index:
        return True

    stored_version = str(index.get("tokenizer_version", ""))
    if stored_version != TOKENIZER_VERSION:
        return True

    on_disk_docs = len(_list_documents(_get_source_dir()))
    indexed_docs = int(index.get("doc_count", 0) or 0)
    if on_disk_docs != indexed_docs:
        return True

    return False


def ensure_knowledge_base_loaded(auto_rebuild: bool = True) -> dict[str, Any]:
    global _INDEX_CACHE
    needs_rebuild = False

    with _INDEX_LOCK:
        if _INDEX_CACHE is not None and not _index_is_stale(_INDEX_CACHE):
            return _INDEX_CACHE

        index = _load_index(_get_index_path())
        cache_index: dict[str, Any] | None = index
        needs_rebuild = auto_rebuild and _index_is_stale(index) and bool(_list_documents(_get_source_dir()))

        if needs_rebuild:
            source_dir = _get_source_dir()
            if _list_documents(source_dir):
                print(
                    "[RAG] Detected stale/missing knowledge base index "
                    f"(tokenizer_version expected={TOKENIZER_VERSION}). Rebuilding..."
                )
                # 释放锁以允许 rebuild_knowledge_base 重新获取
                pass

        if cache_index is None or _index_is_stale(cache_index):
            # 占位空索引，等待外部触发重建；避免在锁内重入
            cache_index = {
                "source_dir": str(_get_source_dir()),
                "doc_count": 0,
                "chunk_count": 0,
                "tokenizer_version": TOKENIZER_VERSION,
                "embedding": {"provider": "hash", "model": "hash-md5-bow", "dimension": 384},
                "chunks": [],
                "embeddings": [],
            }

        _INDEX_CACHE = cache_index

    # 真正的重建在锁外执行，避免嵌套锁
    if needs_rebuild and _REBUILD_LOCK.acquire(blocking=False):
        try:
            rebuild_knowledge_base(reset=True)
        except Exception as error:
            print(f"[RAG] 自动重建知识库失败: {error}")
        finally:
            _REBUILD_LOCK.release()

    return _INDEX_CACHE or {}


def get_knowledge_base_stats() -> dict[str, Any]:
    index = ensure_knowledge_base_loaded()
    embedding = index.get("embedding", {})
    return {
        "source_dir": index.get("source_dir", str(_get_source_dir())),
        "doc_count": int(index.get("doc_count", 0)),
        "chunk_count": int(index.get("chunk_count", 0)),
        "tokenizer_version": index.get("tokenizer_version", "unknown"),
        "embedding_provider": embedding.get("provider", "unknown"),
        "embedding_model": embedding.get("model", "unknown"),
    }


def retrieve_knowledge(query: str, top_k: int = 4, min_score: float = 0.18) -> list[RetrievalHit]:
    # 重排时先多召回再精选
    initial_k = top_k * 3 if KB_RERANK_ENABLED else top_k

    # ---------- Chroma 检索分支 ----------
    if KB_VECTOR_BACKEND == "chroma":
        hits = _retrieve_from_chroma(query, initial_k, min_score)
    else:
        hits = _retrieve_from_json(query, initial_k, min_score)

    # ---------- Rerank ----------
    if KB_RERANK_ENABLED and len(hits) > top_k:
        hits = rerank_hits(query, hits, top_k)
    elif len(hits) > top_k:
        hits = hits[:top_k]

    return hits


def _retrieve_from_json(query: str, top_k: int, min_score: float) -> list[RetrievalHit]:
    """原有 JSON 索引检索逻辑（混合检索）"""
    index = ensure_knowledge_base_loaded()
    chunk_items = index.get("chunks", [])
    embeddings = index.get("embeddings", [])
    if not chunk_items:
        return []

    query_tokens = set(_tokenize(query))

    lexical_scores = np.zeros(len(chunk_items), dtype=float)
    for idx, chunk in enumerate(chunk_items):
        lexical_scores[idx] = _compute_lexical_score(
            query_tokens,
            chunk.get("text", ""),
            chunk.get("source", ""),
        )

    vector_scores = np.zeros(len(chunk_items), dtype=float)
    query_vec = _embed_query(query, index.get("embedding", {}))
    if query_vec is not None and embeddings:
        embedding_matrix = np.asarray(embeddings, dtype=float)
        if embedding_matrix.ndim == 2 and embedding_matrix.shape[1] == query_vec.shape[0]:
            vector_scores = embedding_matrix @ query_vec

    final_scores = 0.75 * vector_scores + 0.25 * lexical_scores
    ranked_indices = np.argsort(final_scores)[::-1]

    hits: list[RetrievalHit] = []
    for idx in ranked_indices[: max(1, top_k)]:
        score = _safe_float(final_scores[idx])
        if score < min_score and len(hits) >= 1:
            continue

        chunk = chunk_items[idx]
        hits.append(
            RetrievalHit(
                chunk_id=chunk.get("chunk_id", f"chunk_{idx}"),
                source=chunk.get("source", "unknown"),
                text=chunk.get("text", ""),
                vector_score=_safe_float(vector_scores[idx]),
                lexical_score=_safe_float(lexical_scores[idx]),
                final_score=score,
            )
        )

    return hits


def format_retrieval_hits(hits: list[RetrievalHit]) -> str:
    if not hits:
        return "知识库中未检索到高置信度内容。"

    blocks: list[str] = []
    for idx, hit in enumerate(hits, start=1):
        snippet = hit.text.replace("\n", " ").strip()
        if len(snippet) > 260:
            snippet = snippet[:260] + "..."

        blocks.append(
            (
                f"[{idx}] 来源: {hit.source}\n"
                f"chunk_id: {hit.chunk_id}\n"
                f"score: {hit.final_score:.3f} (vector={hit.vector_score:.3f}, lexical={hit.lexical_score:.3f})\n"
                f"内容: {snippet}"
            )
        )

    return "\n\n".join(blocks)


# =============================================================================
# Chroma 向量数据库后端
# =============================================================================

def _get_chroma_dir() -> Path:
    env_dir = get_env_text("CHROMA_PERSIST_DIR", "")
    return Path(env_dir) if env_dir else DEFAULT_CHROMA_DIR


def _get_chroma_client():
    if chromadb is None:
        return None
    persist_dir = _get_chroma_dir()
    persist_dir.mkdir(parents=True, exist_ok=True)
    try:
        return chromadb.PersistentClient(path=str(persist_dir))
    except Exception as err:
        print(f"[RAG] Chroma PersistentClient 创建失败: {err}")
        return None


def _rebuild_chroma_collection(
    chunks: list[KnowledgeChunk],
    chunk_texts: list[str],
    vectors: list[list[float]],
    reset: bool,
) -> str:
    client = _get_chroma_client()
    if client is None:
        return "chroma_unavailable"

    try:
        if reset:
            try:
                client.delete_collection(CHROMA_COLLECTION_NAME)
            except Exception:
                pass

        collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        ids = [c.chunk_id for c in chunks]
        metadatas = [{"source": c.source, "start": c.start, "end": c.end} for c in chunks]

        batch_size = 256
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            collection.upsert(
                ids=ids[i:end],
                documents=chunk_texts[i:end],
                embeddings=vectors[i:end],
                metadatas=metadatas[i:end],
            )

        print(f"[RAG] Chroma 索引重建完成: collection={CHROMA_COLLECTION_NAME}, count={collection.count()}")
        return "ok"

    except Exception as err:
        print(f"[RAG] Chroma 索引重建失败: {err}")
        return f"error: {err}"


def _retrieve_from_chroma(query: str, top_k: int, min_score: float) -> list[RetrievalHit]:
    """从 Chroma 向量库检索，并补充关键词分数做混合排序"""
    client = _get_chroma_client()
    if client is None:
        print("[RAG] Chroma 不可用，回退到 JSON 检索")
        return _retrieve_from_json(query, top_k, min_score)

    try:
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
    except Exception:
        print("[RAG] Chroma collection 不存在，回退到 JSON 检索")
        return _retrieve_from_json(query, top_k, min_score)

    index = ensure_knowledge_base_loaded()
    query_vec = _embed_query(query, index.get("embedding", {}))
    if query_vec is None:
        query_vec = np.asarray(_hash_embed([query])[0], dtype=float)

    try:
        results = collection.query(
            query_embeddings=[query_vec.tolist()],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
    except Exception as err:
        print(f"[RAG] Chroma query 失败: {err}")
        return _retrieve_from_json(query, top_k, min_score)

    if not results or not results.get("documents"):
        return []

    query_tokens = set(_tokenize(query))
    hits: list[RetrievalHit] = []

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    for doc, meta, dist in zip(docs, metas, dists):
        # Chroma cosine distance -> similarity
        vector_score = max(0.0, 1.0 - _safe_float(dist))

        # 关键词分数
        lexical_score = _compute_lexical_score(query_tokens, doc, meta.get("source", ""))

        final_score = 0.75 * vector_score + 0.25 * lexical_score

        hits.append(
            RetrievalHit(
                chunk_id=meta.get("chunk_id", meta.get("source", "unknown")),
                source=meta.get("source", "unknown"),
                text=doc,
                vector_score=vector_score,
                lexical_score=lexical_score,
                final_score=final_score,
            )
        )

    hits.sort(key=lambda h: h.final_score, reverse=True)
    filtered_hits: list[RetrievalHit] = []
    for hit in hits:
        if hit.final_score < min_score and len(filtered_hits) >= 1:
            continue
        filtered_hits.append(hit)
        if len(filtered_hits) >= top_k:
            break
    return filtered_hits
