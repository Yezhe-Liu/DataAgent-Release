"""LLM 工厂 —— 双模型架构: flash (快速/结构化输出) + pro (强推理/深度思考)

Provider 支持:
  - deepseek: DeepSeek V4 原生模型 (默认)
    - flash: deepseek-v4-flash (thinking=disabled, 适合 Router/Grade/Text-to-SQL)
    - pro:   deepseek-v4-pro  (thinking=enabled,  适合 Generate/HallucinationCheck)
  - vllm:    自部署 vLLM 推理服务 (OpenAI 兼容接口, --enable-prefix-caching)
  - dashscope: 阿里云百炼 (Qwen 系列)
  - ollama:   本地 Ollama 推理

用法:
  from src.llm_factory import create_flash_model, create_pro_model
  flash_model = create_flash_model()
  pro_model = create_pro_model()
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel

from src.config import get_chat_settings

logger = logging.getLogger(__name__)


def _build_deepseek(settings, thinking_enabled: bool) -> BaseChatModel:
    from langchain_deepseek import ChatDeepSeek

    if not settings.deepseek_api_key:
        raise RuntimeError(
            "provider=deepseek 需要设置 DEEPSEEK_API_KEY，"
            "请前往 https://platform.deepseek.com/ 获取 API Key"
        )
    thinking_config = {"type": "enabled"} if thinking_enabled else {"type": "disabled"}
    model_name = settings.model_pro if thinking_enabled else settings.model
    return ChatDeepSeek(
        model=model_name,
        temperature=settings.temperature,
        api_key=settings.deepseek_api_key,
        api_base=settings.deepseek_base_url,
        max_tokens=settings.max_tokens,
        extra_body={"thinking": thinking_config},
    )


def _build_vllm(settings, thinking_enabled: bool) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    model_name = settings.model_pro if thinking_enabled else settings.model
    logger.info(
        "[vLLM] 创建模型 instance=%s model=%s base_url=%s",
        "pro" if thinking_enabled else "flash",
        model_name,
        settings.vllm_base_url,
    )
    kwargs: dict[str, Any] = {
        "model": model_name,
        "temperature": settings.temperature,
        "api_key": "not-needed",
        "base_url": settings.vllm_base_url,
    }
    if settings.max_tokens is not None:
        kwargs["max_tokens"] = settings.max_tokens
    # vLLM 通过 extra_body 传递 thinking 控制（取决于模型是否支持）
    if thinking_enabled:
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
    else:
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    return ChatOpenAI(**kwargs)


def _build_dashscope(settings) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    if not settings.dashscope_api_key:
        raise RuntimeError(
            "provider=dashscope 需要设置 DASHSCOPE_API_KEY，"
            "请前往 https://dashscope.console.aliyun.com/ 获取 API Key"
        )
    kwargs: dict[str, Any] = {
        "model": settings.model,
        "temperature": settings.temperature,
        "api_key": settings.dashscope_api_key,
        "base_url": settings.dashscope_base_url,
    }
    if settings.max_tokens is not None:
        kwargs["max_tokens"] = settings.max_tokens
    return ChatOpenAI(**kwargs)


def _build_ollama(settings) -> BaseChatModel:
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


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------


def create_flash_model() -> BaseChatModel:
    """创建快速模型实例 (thinking=disabled)。

    适用节点: Router / Rewrite / Grade / Text-to-SQL / ToolExecute
    特点: 低延迟，稳定 JSON 结构化输出，不使用思维链。
    """
    settings = get_chat_settings()
    provider = settings.provider

    if provider == "deepseek":
        return _build_deepseek(settings, thinking_enabled=False)
    if provider == "vllm":
        return _build_vllm(settings, thinking_enabled=False)
    if provider == "dashscope":
        return _build_dashscope(settings)
    if provider == "ollama":
        return _build_ollama(settings)

    raise ValueError(f"不支持的 provider: {provider}，可选: deepseek / vllm / dashscope / ollama")


def create_pro_model() -> BaseChatModel:
    """创建强推理模型实例 (thinking=enabled)。

    适用节点: Generate / HallucinationCheck
    特点: 深度思维链推理，适合多源融合生成和事实核查。
    """
    settings = get_chat_settings()
    provider = settings.provider

    if provider == "deepseek":
        return _build_deepseek(settings, thinking_enabled=True)
    if provider == "vllm":
        return _build_vllm(settings, thinking_enabled=True)
    # dashscope / ollama 不支持 thinking 开关，回退到 flash 模型
    if provider == "dashscope":
        return _build_dashscope(settings)
    if provider == "ollama":
        return _build_ollama(settings)

    raise ValueError(f"不支持的 provider: {provider}，可选: deepseek / vllm / dashscope / ollama")


def create_chat_model() -> BaseChatModel:
    """向后兼容: 返回 flash 模型实例。

    旧代码通过此函数获取单一模型。新代码应直接使用 create_flash_model() / create_pro_model()。
    """
    return create_flash_model()
