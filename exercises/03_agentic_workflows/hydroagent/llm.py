"""One function to get a LangChain chat model, so every LangGraph architecture is model-agnostic.

The Claude Agent SDK harness (A4) is Claude-only and does not use this.
"""

from __future__ import annotations

import functools
import os

from langchain_core.language_models import BaseChatModel

from .config import foundry_base_url, foundry_openai_url, get_settings

PROVIDERS = ("anthropic", "azure", "mistral")


def get_chat_model(provider: str | None = None, model: str | None = None, temperature: float | None = None,
                   max_tokens: int = 4096) -> BaseChatModel:
    s = get_settings(provider=provider)
    temp = s.temperature if temperature is None else temperature
    if s.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        foundry_url = foundry_base_url()
        if foundry_url:  # Claude on Azure Foundry: same Messages API, different base URL + key
            auth = {"base_url": foundry_url, "api_key": os.environ["FOUNDRY_KEY"]}
        elif os.getenv("ANTHROPIC_API_KEY"):
            auth = {}
        else:
            raise RuntimeError("Set FOUNDRY_ENDPOINT + FOUNDRY_KEY (Azure Foundry) or ANTHROPIC_API_KEY in .env "
                               "(or use provider='azure').")
        # No temperature: current Claude models (Opus 5.5, Sonnet 5.5, ...) reject sampling parameters with a 400.
        return ChatAnthropic(model=model or s.anthropic_model, max_tokens=max_tokens, max_retries=3, **auth)
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
    if s.provider == "mistral":
        # Mistral on Azure Foundry, through the resource's OpenAI-compatible endpoint (chat completions)
        from langchain_openai import ChatOpenAI

        url = foundry_openai_url()
        key = os.getenv("MISTRAL_KEY") or os.getenv("FOUNDRY_KEY")
        if not url or not key:
            raise RuntimeError("Mistral needs FOUNDRY_ENDPOINT + FOUNDRY_KEY (or MISTRAL_ENDPOINT + MISTRAL_KEY) "
                               "and DEPLOYMENT_MISTRAL_LARGE_4 in .env.")
        # Mistral Large 4 rejects temperature 0 (greedy sampling) unless top_p is exactly 1. It also reasons by
        # default, and the hidden reasoning tokens count against max_tokens: give it room, or a structured reply
        # gets cut off (LengthFinishReasonError, or an empty tool call -> None).
        return _mistral_chat_class()(model=model or s.mistral_deployment, base_url=url, api_key=key,
                                     temperature=temp, top_p=1, max_tokens=max(max_tokens, 16384), max_retries=3,
                                     use_responses_api=False)
    raise ValueError(f"Unknown provider {s.provider!r}; choose from {PROVIDERS}")


@functools.cache  # one class, built lazily so importing this module does not import langchain_openai
def _mistral_chat_class():
    from langchain_openai import ChatOpenAI

    class MistralFoundryChat(ChatOpenAI):
        """ChatOpenAI for Mistral on Foundry. Mistral's API rejects the OpenAI ``name`` field on messages (422
        'extra_forbidden'), which LangChain agents set on their own messages (e.g. ``name="react_agent"``)."""

        def _get_request_payload(self, input_, *, stop=None, **kwargs):
            payload = super()._get_request_payload(input_, stop=stop, **kwargs)
            for m in payload.get("messages", []):
                m.pop("name", None)
            return payload

    return MistralFoundryChat


def structured(llm: BaseChatModel, schema):
    """``llm.with_structured_output(schema)`` that works on current Claude and Mistral models.

    Both get the API's native structured outputs (``json_schema``: the reply is constrained to the schema).
    LangChain's default for Claude forces a tool call, which Claude Opus 5.5 / Sonnet 5.5 reject (and the call
    silently yields nothing). Mistral Large 4 sometimes ignores a forced tool call too (the A2 critic got None).
    """
    if type(llm).__name__ in ("ChatAnthropic", "MistralFoundryChat"):
        return llm.with_structured_output(schema, method="json_schema")
    return llm.with_structured_output(schema)


def text_of(msg_or_content) -> str:
    """Answer text of a chat message (or its ``.content``), without thinking/tool blocks.

    Claude returns a list of content blocks (e.g. a thinking block with a signature, then text);
    GPT returns a plain string. Only the ``text`` blocks are the answer.
    """
    c = getattr(msg_or_content, "content", msg_or_content)
    if isinstance(c, str):
        return c
    parts = [b if isinstance(b, str) else b.get("text", "") for b in c or []
             if isinstance(b, str) or (isinstance(b, dict) and b.get("type") == "text")]
    return "\n".join(p for p in parts if p)


def describe_model(provider: str | None = None) -> str:
    s = get_settings(provider=provider)
    return {"anthropic": f"anthropic:{s.anthropic_model}", "azure": f"azure:{s.azure_deployment}",
            "mistral": f"mistral:{s.mistral_deployment}"}.get(s.provider, s.provider)
