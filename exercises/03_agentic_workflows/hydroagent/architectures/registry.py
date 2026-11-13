"""Registry of the pre-built architectures: one place for the CLI, notebook and Streamlit app."""

from __future__ import annotations

from . import a0_plain_llm, a1_react, a2_critic_loop, a3_director, a4_claude_harness, a4m_vibe_harness, a5_composed
from .. import config as C
from .common import ArchInfo, harness_for, make_session

MODULES = {m.INFO.key: m for m in (a0_plain_llm, a1_react, a2_critic_loop, a3_director, a4_claude_harness,
                                    a4m_vibe_harness, a5_composed)}
HARNESSES = ("a4", "a4m")  # not LangGraph graphs: a harness session with start() only
INFOS: dict[str, ArchInfo] = {k: m.INFO for k, m in MODULES.items()}

# Each vendor's model runs in that vendor's own harness: the harness patterns ignore the provider setting.
HARNESS_VENDOR = {"a4": "anthropic", "a4m": "mistral"}
VENDOR_HARNESS = {"anthropic": "Claude Agent SDK", "mistral": "Mistral Vibe"}  # provider -> its native harness


def uses_provider(key: str) -> bool:
    """True if the architecture runs on the chosen LLM provider (LangGraph patterns), False for fixed harnesses."""
    return key not in HARNESSES


def model_label(key: str, provider: str | None = None) -> str:
    """Human-readable 'model · harness' that the architecture will actually use."""
    s = C.get_settings(provider=provider)
    models = {"anthropic": s.anthropic_model, "mistral": s.mistral_deployment, "azure": s.azure_deployment}
    if key in HARNESS_VENDOR:
        v = HARNESS_VENDOR[key]
        return f"{models[v]} + {VENDOR_HARNESS[v]}"
    label = f"{models[s.provider]} + LangGraph"
    if key == "a5":
        v = harness_for(s.provider)
        label += f" (open-ended step: {models[v]} + {VENDOR_HARNESS[v]})"
    return label


def create_session(key: str, provider: str | None = None, on_event=None, **opts):
    """Return a session with ``start(query)`` / ``resume(answer)`` for any architecture key (a0..a5)."""
    mod = MODULES[key]
    if key in HARNESSES:
        return mod.make_session(provider=provider, on_event=on_event, **opts)
    return make_session(key, mod.build, provider=provider, on_event=on_event, **opts)


def graph_structure(key: str, provider: str | None = "azure") -> tuple[list[str], list[tuple[str, str, str]]]:
    """Nodes and edges of an architecture's graph (for drawing). Builds the graph without calling an LLM."""
    if key in HARNESSES:
        node = "claude_harness" if key == "a4" else "vibe_harness"
        return (["__start__", node, "__end__"], [("__start__", node, ""), (node, "__end__", "")])
    from unittest.mock import MagicMock

    llm = MagicMock()  # graph structure does not depend on the model
    g = MODULES[key].build(llm=llm).get_graph(xray=False)
    nodes = list(g.nodes)
    edges = [(e.source, e.target, str(e.data) if e.data else ("cond" if e.conditional else "")) for e in g.edges]
    return nodes, edges
