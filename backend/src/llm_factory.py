"""LLM 工厂 —— 统一创建 ChatModel 实例

支持的 provider:
  - deepseek:  DeepSeek 平台, 通过 langchain-deepseek 官方集成 (默认)
  - dashscope: 阿里云百炼 (Qwen 系列), 通过 OpenAI 兼容接口
  - ollama:   本地 Ollama 推理

默认: deepseek
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel

from src.config import get_chat_settings


def create_chat_model() -> BaseChatModel:
    settings = get_chat_settings()
    provider = settings.provider

    if provider == "deepseek":
        if not settings.deepseek_api_key:
            raise RuntimeError(
                "provider=deepseek 需要设置 DEEPSEEK_API_KEY，"
                "请前往 https://platform.deepseek.com/ 获取 API Key"
            )
        from langchain_deepseek import ChatDeepSeek

        return ChatDeepSeek(
            model=settings.model or "deepseek-chat",
            temperature=settings.temperature,
            api_key=settings.deepseek_api_key,
            api_base=settings.deepseek_base_url,
            max_tokens=settings.max_tokens,
        )

    if provider == "dashscope":
        if not settings.dashscope_api_key:
            raise RuntimeError(
                "provider=dashscope 需要设置 DASHSCOPE_API_KEY，"
                "请前往 https://dashscope.console.aliyun.com/ 获取 API Key"
            )
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

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.model,
            temperature=settings.temperature,
            base_url=settings.ollama_base_url,
            num_ctx=settings.ollama_num_ctx,
            num_gpu=settings.ollama_num_gpu,
            num_thread=settings.ollama_num_thread,
            low_vram=settings.ollama_low_vram,
        )

    raise ValueError(f"不支持的 provider: {provider}，可选: deepseek / dashscope / ollama")
