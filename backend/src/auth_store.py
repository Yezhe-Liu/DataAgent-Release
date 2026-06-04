from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from src.config import get_auth_settings, get_mysql_settings

try:
    import pymysql
    from pymysql.cursors import DictCursor
except Exception:
    pymysql = None
    DictCursor = None


@dataclass(frozen=True)
class AuthUser:
    id: str
    username: str
    created_at: str


_AUTH_AVAILABLE = False
_AUTH_ERROR = ""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _coerce_datetime(value: Any) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc).isoformat()
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, str) and value:
        return value
    return _utc_now_iso()


def _normalize_username(username: str) -> str:
    return username.strip()


def _connect(include_database: bool = True):
    if pymysql is None or DictCursor is None:
        raise RuntimeError("pymysql is not installed")

    mysql_settings = get_mysql_settings()
    kwargs: dict[str, Any] = {
        "host": mysql_settings.host,
        "port": mysql_settings.port,
        "user": mysql_settings.user,
        "password": mysql_settings.password,
        "charset": mysql_settings.charset,
        "cursorclass": DictCursor,
        "autocommit": True,
    }
    if include_database:
        kwargs["database"] = mysql_settings.database
    return pymysql.connect(**kwargs)


def _password_hash(password: str) -> str:
    auth_settings = get_auth_settings()
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        auth_settings.password_hash_iterations,
    )
    return f"pbkdf2_sha256${auth_settings.password_hash_iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        algorithm, iterations_raw, salt_hex, digest_hex = stored_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt_hex),
            int(iterations_raw),
        )
        return hmac.compare_digest(digest.hex(), digest_hex)
    except Exception:
        return False


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _session_expiry() -> datetime:
    auth_settings = get_auth_settings()
    return _utc_now() + timedelta(hours=max(1, auth_settings.session_ttl_hours))


def initialize_auth_store() -> tuple[bool, str]:
    global _AUTH_AVAILABLE, _AUTH_ERROR

    if pymysql is None:
        _AUTH_AVAILABLE = False
        _AUTH_ERROR = "缺少 pymysql 依赖，无法启用 MySQL 登录。"
        return _AUTH_AVAILABLE, _AUTH_ERROR

    mysql_settings = get_mysql_settings()

    try:
        bootstrap_conn = _connect(include_database=False)
        try:
            with bootstrap_conn.cursor() as cursor:
                cursor.execute(
                    f"CREATE DATABASE IF NOT EXISTS `{mysql_settings.database}` CHARACTER SET {mysql_settings.charset}"
                )
        finally:
            bootstrap_conn.close()

        conn = _connect(include_database=True)
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        id BIGINT PRIMARY KEY AUTO_INCREMENT,
                        username VARCHAR(80) NOT NULL UNIQUE,
                        password_hash VARCHAR(255) NOT NULL,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS auth_sessions (
                        id BIGINT PRIMARY KEY AUTO_INCREMENT,
                        user_id BIGINT NOT NULL,
                        session_token_hash CHAR(64) NOT NULL UNIQUE,
                        expires_at TIMESTAMP NOT NULL,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                        INDEX idx_auth_sessions_user_id (user_id),
                        INDEX idx_auth_sessions_expires_at (expires_at)
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS user_documents (
                        id BIGINT PRIMARY KEY AUTO_INCREMENT,
                        user_id BIGINT NOT NULL,
                        original_name VARCHAR(255) NOT NULL,
                        stored_name VARCHAR(255) NOT NULL,
                        mime_type VARCHAR(120) NOT NULL DEFAULT '',
                        storage_path TEXT NOT NULL,
                        file_size BIGINT NOT NULL DEFAULT 0,
                        checksum CHAR(64) NOT NULL DEFAULT '',
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                        INDEX idx_user_documents_user_id (user_id)
                    )
                    """
                )
                cursor.execute("DELETE FROM auth_sessions WHERE expires_at <= UTC_TIMESTAMP()")
        finally:
            conn.close()
    except Exception as error:
        _AUTH_AVAILABLE = False
        _AUTH_ERROR = str(error)
        return _AUTH_AVAILABLE, _AUTH_ERROR

    _AUTH_AVAILABLE = True
    _AUTH_ERROR = ""
    return _AUTH_AVAILABLE, _AUTH_ERROR


def auth_store_status() -> dict[str, str | bool]:
    return {
        "available": _AUTH_AVAILABLE,
        "error": _AUTH_ERROR,
    }


def is_auth_store_available() -> bool:
    return _AUTH_AVAILABLE


def create_user(username: str, password: str) -> AuthUser:
    clean_username = _normalize_username(username)
    if not clean_username:
        raise ValueError("用户名不能为空")
    if len(clean_username) < 3:
        raise ValueError("用户名至少需要 3 个字符")
    if len(password) < 6:
        raise ValueError("密码至少需要 6 个字符")

    conn = _connect(include_database=True)
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT id FROM users WHERE username = %s LIMIT 1",
                (clean_username,),
            )
            if cursor.fetchone():
                raise ValueError("用户名已存在")

            password_hash = _password_hash(password)
            cursor.execute(
                "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
                (clean_username, password_hash),
            )
            user_id = cursor.lastrowid
            cursor.execute(
                "SELECT id, username, created_at FROM users WHERE id = %s LIMIT 1",
                (user_id,),
            )
            row = cursor.fetchone()
            if not row:
                raise RuntimeError("用户创建失败")
            return AuthUser(
                id=str(row["id"]),
                username=row["username"],
                created_at=_coerce_datetime(row.get("created_at")),
            )
    finally:
        conn.close()


def authenticate_user(username: str, password: str) -> AuthUser | None:
    clean_username = _normalize_username(username)
    if not clean_username or not password:
        return None

    conn = _connect(include_database=True)
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT id, username, password_hash, created_at FROM users WHERE username = %s LIMIT 1",
                (clean_username,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            if not verify_password(password, row.get("password_hash", "")):
                return None
            return AuthUser(
                id=str(row["id"]),
                username=row["username"],
                created_at=_coerce_datetime(row.get("created_at")),
            )
    finally:
        conn.close()


def create_auth_session(user_id: str) -> str:
    token = secrets.token_urlsafe(32)
    conn = _connect(include_database=True)
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO auth_sessions (user_id, session_token_hash, expires_at) VALUES (%s, %s, %s)",
                (int(user_id), _token_hash(token), _session_expiry()),
            )
        return token
    finally:
        conn.close()


def delete_auth_session(token: str) -> None:
    if not token:
        return
    conn = _connect(include_database=True)
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "DELETE FROM auth_sessions WHERE session_token_hash = %s",
                (_token_hash(token),),
            )
    finally:
        conn.close()


def get_user_by_session_token(token: str) -> AuthUser | None:
    if not token:
        return None

    conn = _connect(include_database=True)
    try:
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM auth_sessions WHERE expires_at <= UTC_TIMESTAMP()")
            cursor.execute(
                """
                SELECT users.id, users.username, users.created_at
                FROM auth_sessions
                INNER JOIN users ON users.id = auth_sessions.user_id
                WHERE auth_sessions.session_token_hash = %s
                  AND auth_sessions.expires_at > UTC_TIMESTAMP()
                LIMIT 1
                """,
                (_token_hash(token),),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return AuthUser(
                id=str(row["id"]),
                username=row["username"],
                created_at=_coerce_datetime(row.get("created_at")),
            )
    finally:
        conn.close()
