"""HITL 审批中断持久化 — 文件级原子写入 + 启动自动恢复。

生产级设计:
  - 原子写入: 先写临时文件，再 os.replace 原子替换，杜绝写一半断电的数据损坏
  - 线程安全: threading.Lock 保护所有读写操作
  - 启动恢复: 构造时自动从磁盘加载所有未完成的审批会话
  - 超时清理: 可选的 TTL 过期机制 (默认 24h)

未来迁移路径:
  v1 (当前): 文件系统 JSON 持久化
  v2 (规划): Redis Cluster — 替换为 Redis 版本时只需实现相同接口
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 默认持久化文件路径
DEFAULT_PERSIST_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "pending_interrupts.json"

# 默认 TTL: 24 小时
DEFAULT_TTL_SECONDS = 86400


class HITLPersistence:
    """HITL 审批中断持久化管理器。

    接口契约 (可替换为 Redis 实现):
      - save(session_id, metadata): 保存待审批会话
      - load(session_id):          加载单个会话
      - remove(session_id):        删除已完成/拒绝的会话
      - list_all():                列出所有待审批会话
      - exists(session_id):        检查会话是否存在
      - cleanup_expired():         清理过期会话
    """

    __slots__ = ("_file_path", "_lock", "_store", "_ttl_seconds")

    def __init__(
        self,
        file_path: str | Path | None = None,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> None:
        self._file_path = Path(file_path) if file_path else DEFAULT_PERSIST_PATH
        self._ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._store: dict[str, dict[str, Any]] = {}

        # 启动时从磁盘恢复
        self._load_from_disk()
        # 清理过期会话
        self.cleanup_expired()

        count = len(self._store)
        if count > 0:
            logger.warning(
                "[HITL Persistence] Recovered %d pending interrupt(s) from disk: %s",
                count,
                self._file_path,
            )
        else:
            logger.info("[HITL Persistence] Initialized clean. No pending interrupts on disk.")

    # ------------------------------------------------------------------
    # 公开 API (接口契约)
    # ------------------------------------------------------------------

    def save(self, session_id: str, metadata: dict[str, Any]) -> None:
        """保存待审批会话到磁盘（原子写入）。"""
        with self._lock:
            metadata["_created_at"] = datetime.now(timezone.utc).isoformat()
            metadata["_session_id"] = session_id
            self._store[session_id] = metadata
            self._flush()
            logger.info("[HITL Persistence] Saved session=%s node=%s", session_id, metadata.get("node", "?"))

    def load(self, session_id: str) -> dict[str, Any] | None:
        """加载单个待审批会话。"""
        with self._lock:
            entry = self._store.get(session_id)
            if entry is None:
                return None
            # 检查是否过期
            if self._is_expired(entry):
                self._store.pop(session_id, None)
                self._flush()
                return None
            return dict(entry)

    def remove(self, session_id: str) -> dict[str, Any] | None:
        """删除已完成或拒绝的会话。返回被删除的元数据。"""
        with self._lock:
            entry = self._store.pop(session_id, None)
            if entry is not None:
                self._flush()
                logger.info("[HITL Persistence] Removed session=%s", session_id)
            return entry

    def exists(self, session_id: str) -> bool:
        """检查会话是否存在且未过期。"""
        return self.load(session_id) is not None

    def list_all(self) -> list[dict[str, Any]]:
        """列出所有待审批会话 (不含内部字段)。"""
        with self._lock:
            result: list[dict[str, Any]] = []
            expired: list[str] = []
            for sid, entry in self._store.items():
                if self._is_expired(entry):
                    expired.append(sid)
                else:
                    result.append({
                        "session_id": sid,
                        "node": entry.get("node", "?"),
                        "user_id": entry.get("user_id", "?"),
                        "created_at": entry.get("_created_at", ""),
                    })
            # 清理过期
            for sid in expired:
                self._store.pop(sid, None)
            if expired:
                self._flush()
            return result

    def cleanup_expired(self) -> int:
        """清理过期会话，返回清理数量。"""
        with self._lock:
            expired = [sid for sid, entry in self._store.items() if self._is_expired(entry)]
            for sid in expired:
                self._store.pop(sid, None)
            if expired:
                self._flush()
                logger.info("[HITL Persistence] Cleaned up %d expired sessions", len(expired))
            return len(expired)

    def count(self) -> int:
        """当前待审批会话数。"""
        with self._lock:
            return len(self._store)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _is_expired(self, entry: dict[str, Any]) -> bool:
        created_at_str = entry.get("_created_at", "")
        if not created_at_str:
            return True
        try:
            created_at = datetime.fromisoformat(created_at_str)
            elapsed = (datetime.now(timezone.utc) - created_at).total_seconds()
            return elapsed > self._ttl_seconds
        except (ValueError, TypeError):
            return True

    def _flush(self) -> None:
        """原子写入磁盘: 先写 .tmp 临时文件，再 os.replace 原子替换。"""
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._file_path.with_suffix(".tmp")

        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._store, f, ensure_ascii=False, indent=2, default=str)
            # 原子替换 (Windows 上原子性由 OS 保证)
            os.replace(tmp_path, self._file_path)
        except Exception as e:
            logger.error("[HITL Persistence] Flush failed: %s", e)
            # 清理损坏的临时文件
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    def _load_from_disk(self) -> None:
        """从磁盘恢复所有未完成的审批会话。"""
        if not self._file_path.exists():
            self._store = {}
            return

        try:
            with open(self._file_path, "r", encoding="utf-8") as f:
                content = f.read()
            if not content.strip():
                self._store = {}
                return
            self._store = json.loads(content)
            logger.info(
                "[HITL Persistence] Loaded %d entries from disk",
                len(self._store),
            )
        except (json.JSONDecodeError, OSError) as e:
            logger.error("[HITL Persistence] Failed to load from disk: %s. Starting clean.", e)
            self._store = {}
            # 备份损坏文件
            corrupted_path = self._file_path.with_suffix(".corrupted")
            try:
                os.replace(self._file_path, corrupted_path)
                logger.warning("[HITL Persistence] Corrupted file moved to %s", corrupted_path)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# 模块级单例 (供 server.py 使用)
# ---------------------------------------------------------------------------

_persistence_instance: HITLPersistence | None = None


def get_persistence() -> HITLPersistence:
    """获取 HITL 持久化单例。"""
    global _persistence_instance
    if _persistence_instance is None:
        _persistence_instance = HITLPersistence()
    return _persistence_instance
