from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeUserContext:
    user_id: str
    username: str


_CURRENT_USER_ID: ContextVar[str] = ContextVar("current_user_id", default="")
_CURRENT_USERNAME: ContextVar[str] = ContextVar("current_username", default="")


def activate_user_context(user_id: str, username: str = "") -> tuple[Token[str], Token[str]]:
    return _CURRENT_USER_ID.set(user_id), _CURRENT_USERNAME.set(username)


def reset_user_context(tokens: tuple[Token[str], Token[str]]) -> None:
    user_token, username_token = tokens
    _CURRENT_USER_ID.reset(user_token)
    _CURRENT_USERNAME.reset(username_token)


def get_current_user_id() -> str:
    return _CURRENT_USER_ID.get("")


def get_current_username() -> str:
    return _CURRENT_USERNAME.get("")


def get_current_user_context() -> RuntimeUserContext | None:
    user_id = get_current_user_id()
    if not user_id:
        return None
    return RuntimeUserContext(user_id=user_id, username=get_current_username())
