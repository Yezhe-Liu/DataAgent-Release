"""文档分块策略

支持三种策略:
  - fixed:      固定长度 + overlap (原有方案)
  - semantic:   按自然段落 / Markdown 标题边界切分
  - recursive:  递归降级切分 (段落 → 句子 → 字符)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class Chunk:
    chunk_id: str
    source: str
    text: str
    start_char: int
    end_char: int
    chunk_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 1. Fixed Chunker (固定长度 + overlap)
# ---------------------------------------------------------------------------


class FixedChunker:
    def __init__(self, chunk_size: int = 700, overlap: int = 120):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, source: str = "") -> list[Chunk]:
        clean = (text or "").strip()
        if not clean:
            return []

        chunks: list[Chunk] = []
        start = 0
        text_len = len(clean)
        idx = 0

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            fragment = clean[start:end].strip()
            if fragment:
                chunks.append(Chunk(
                    chunk_id=f"{source}::chunk_{idx}",
                    source=source,
                    text=fragment,
                    start_char=start,
                    end_char=end,
                    chunk_index=idx,
                    metadata={"strategy": "fixed"},
                ))
                idx += 1
            if end >= text_len:
                break
            start = max(0, end - self.overlap)

        return chunks


# ---------------------------------------------------------------------------
# 2. Semantic Chunker (段落 + Markdown 标题)
# ---------------------------------------------------------------------------

_MD_HEADING = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_DOUBLE_NEWLINE = re.compile(r"\n\s*\n")


class SemanticChunker:
    def __init__(self, max_chunk_size: int = 1200):
        self.max_chunk_size = max_chunk_size

    def chunk(self, text: str, source: str = "") -> list[Chunk]:
        clean = (text or "").strip()
        if not clean:
            return []

        # 保护 markdown 标题不被切散 — 在标题前插入分界标记
        protected = _MD_HEADING.sub(lambda m: f"\n__SECTION__\n{m.group()}", clean)
        # 按段落分隔
        sections = _DOUBLE_NEWLINE.split(protected)

        chunks: list[Chunk] = []
        buf = ""
        idx = 0
        char_pos = 0

        for section in sections:
            section = section.strip()
            if not section:
                continue
            # 去掉分界标记
            section = section.replace("__SECTION__", "").strip()
            if not section:
                continue

            candidate = buf + ("\n\n" if buf else "") + section
            if len(candidate) <= self.max_chunk_size:
                buf = candidate
            else:
                if buf.strip():
                    chunks.append(self._make_chunk(buf, source, idx, char_pos))
                    idx += 1
                    char_pos += len(buf)
                # 超长段落用 fixed fallback
                if len(section) > self.max_chunk_size:
                    sub = FixedChunker(self.max_chunk_size, 100).chunk(section, source)
                    for s in sub:
                        s.chunk_index = idx
                        s.metadata["strategy"] = "semantic+fixed"
                        chunks.append(s)
                        idx += 1
                    buf = ""
                else:
                    buf = section

        if buf.strip():
            chunks.append(self._make_chunk(buf, source, idx, char_pos))

        return chunks

    @staticmethod
    def _make_chunk(text: str, source: str, idx: int, start: int) -> Chunk:
        return Chunk(
            chunk_id=f"{source}::chunk_{idx}",
            source=source,
            text=text,
            start_char=start,
            end_char=start + len(text),
            chunk_index=idx,
            metadata={"strategy": "semantic"},
        )


# ---------------------------------------------------------------------------
# 3. Recursive Chunker (递归降级)
# ---------------------------------------------------------------------------

_SEPARATORS = ["\n\n", "\n", "。", ". ", " "]


class RecursiveChunker:
    def __init__(self, max_chunk_size: int = 1000, overlap: int = 100):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    def chunk(self, text: str, source: str = "") -> list[Chunk]:
        clean = (text or "").strip()
        if not clean:
            return []
        raw = self._split_recursive(clean, self.max_chunk_size)
        chunks: list[Chunk] = []
        for idx, fragment in enumerate(raw):
            chunks.append(Chunk(
                chunk_id=f"{source}::chunk_{idx}",
                source=source,
                text=fragment.strip(),
                start_char=0,
                end_char=len(fragment),
                chunk_index=idx,
                metadata={"strategy": "recursive"},
            ))
        return chunks

    def _split_recursive(self, text: str, max_len: int) -> list[str]:
        if len(text) <= max_len:
            return [text] if text.strip() else []

        for sep in _SEPARATORS:
            parts = text.split(sep)
            if len(parts) == 1:
                continue
            result: list[str] = []
            for part in parts:
                result.extend(self._split_recursive(part, max_len))
            # 合并短块
            return self._merge_short(result, max_len, sep)

        # 最终降级: 强制按字符切
        return [text[i:i + max_len] for i in range(0, len(text), max_len - self.overlap)]

    @staticmethod
    def _merge_short(parts: list[str], max_len: int, sep: str) -> list[str]:
        merged: list[str] = []
        buf = ""
        for p in parts:
            if not p.strip():
                continue
            candidate = buf + (sep if buf else "") + p
            if len(candidate) <= max_len:
                buf = candidate
            else:
                if buf.strip():
                    merged.append(buf)
                buf = p
        if buf.strip():
            merged.append(buf)
        return merged


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

CHUNKER_REGISTRY: dict[str, Callable[..., Any]] = {
    "fixed": FixedChunker,
    "semantic": SemanticChunker,
    "recursive": RecursiveChunker,
}


def create_chunker(strategy: str = "semantic", **kwargs) -> Any:
    cls = CHUNKER_REGISTRY.get(strategy, SemanticChunker)
    return cls(**kwargs)
