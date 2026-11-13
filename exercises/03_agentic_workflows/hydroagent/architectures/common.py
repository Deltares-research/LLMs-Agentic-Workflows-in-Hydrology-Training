"""Shared pieces for the LangGraph architectures: state, domain context, session factory."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated, Any, Callable

from langchain_core.messages import AnyMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from ..llm import get_chat_model
from ..runtime import GraphSession
from ..tracing import Event, Recorder

DOMAIN_CONTEXT = """You support hydrologists working with a Wflow (wflow_sbm) model of the upper Moselle
upstream of Epinal (France, ~1220 km2, 1 km grid), built with HydroMT and run with Wflow.jl.
Simulation period 2019-2020 (daily, E-OBS forcing); statistics exclude a 3-month spin-up.
Scenarios are made by multiplying model parameters (e.g. ksat_vertical 0.7 = 30% lower soil permeability).
The main gauge is GRDC 9116044 (Moselle at Epinal). Units: discharge in m3/s.
Be quantitative and concise; always state which run_ids your conclusions are based on."""


class HydroState(TypedDict, total=False):
    query: str
    messages: Annotated[list[AnyMessage], add_messages]
    answer: str
    run_ids: Annotated[list[str], operator.add]
    plots: Annotated[list[str], operator.add]
    note: str
    # free-form scratch space for specific architectures
    data: dict


@dataclass
class ArchInfo:
    key: str
    name: str
    pattern: str            # designation in Agentic_Overview.drawio
    summary: str
    use_when: str
    trade_off: str
    example_query: str
    hitl: bool = False
    claude_only: bool = False


def harness_for(provider: str | None) -> str:
    """The provider's own code-agent harness ("anthropic" = Claude Agent SDK, "mistral" = Mistral Vibe).

    Each vendor's model runs in that vendor's harness. GPT (Azure) has no harness here, so it falls back to Claude.
    """
    from .. import config as C

    return "mistral" if (provider or C.get_settings().provider) == "mistral" else "anthropic"


def make_session(name: str, build: Callable[..., Any], provider: str | None = None,
                 on_event: Callable[[Event], None] | None = None, recursion_limit: int = 60, **opts) -> GraphSession:
    from ..llm import describe_model

    rec = Recorder(on_event, model=describe_model(provider))
    llm = None if opts.pop("no_llm", False) else get_chat_model(provider)
    graph = build(llm=llm, checkpointer=MemorySaver(), rec=rec, provider=provider, **opts)
    return GraphSession(name, graph, rec, recursion_limit=recursion_limit)
