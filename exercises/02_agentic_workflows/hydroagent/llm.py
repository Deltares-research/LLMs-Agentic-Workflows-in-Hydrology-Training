"""One function to get a LangChain chat model, so every LangGraph architecture is model-agnostic.

The Claude Agent SDK harness (A4) is Claude-only and does not use this.
"""

from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel

from .config import get_settings

PROVIDERS = ("anthropic", "azure")


def get_chat_model(provider: str | None = None, model: str | None = None, temperature: float | None = None,
                   max_tokens: int = 4096) -> BaseChatModel:
    s = get_settings(provider=provider)
    temp = s.temperature if temperature is None else temperature
    if s.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY missing from .env (or use provider='azure').")
        return ChatAnthropic(model=model or s.anthropic_model, temperature=temp, max_tokens=max_tokens,
                             max_retries=3)
    if s.provider == "azure":
        from langchain_openai import AzureChatOpenAI

        missing = [k for k in ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION")
                   if not os.getenv(k)]
        if missing:
            raise RuntimeError(f"Missing Azure settings in .env: {missing}")
        return AzureChatOpenAI(azure_deployment=model or s.azure_deployment, temperature=temp,
                               api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                               azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                               max_tokens=max_tokens, max_retries=3)
    raise ValueError(f"Unknown provider {s.provider!r}; choose from {PROVIDERS}")


def describe_model(provider: str | None = None) -> str:
    s = get_settings(provider=provider)
    return f"anthropic:{s.anthropic_model}" if s.provider == "anthropic" else f"azure:{s.azure_deployment}"
