from __future__ import annotations

from typing import Any

from langchain_ollama import ChatOllama

from src.config import get_chat_settings


def create_chat_model():
    settings = get_chat_settings()

    if settings.provider == "ollama":
        return ChatOllama(
            model=settings.model,
            temperature=settings.temperature,
            base_url=settings.ollama_base_url,
            num_ctx=settings.ollama_num_ctx,
            num_gpu=settings.ollama_num_gpu,
            num_thread=settings.ollama_num_thread,
            low_vram=settings.ollama_low_vram,
        )

    if settings.provider == "dashscope":
        if not settings.dashscope_api_key:
            raise RuntimeError("CHAT_MODEL_PROVIDER=dashscope 时必须设置 DASHSCOPE_API_KEY")
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {
            "model": settings.model,
            "temperature": settings.temperature,
            "api_key": settings.dashscope_api_key,
            "base_url": settings.dashscope_base_url,
        }
        if settings.max_tokens is not None:
            kwargs["max_tokens"] = settings.max_tokens
        return ChatOpenAI(**kwargs)

    raise ValueError(f"不支持的聊天模型 provider: {settings.provider}")
