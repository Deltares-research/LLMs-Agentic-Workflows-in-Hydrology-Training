"""A5 - Composed (drawio section F): a framework for the structure, a harness for the hard open-ended step.

    route (LLM classifier + confidence)
       |--simple question-------> plain LLM call
       |--parameter scenario----> ReAct agent (A1) as a node
       '--open-ended / code-----> the provider's own harness as a node (Claude Agent SDK = A4, Mistral Vibe = A4m)
    --> human check (only if router confidence < threshold) --> result

Rule of thumb: "framework for the structure, harness for the hard open-ended steps, plain LLM call for
anything simple."
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from ..harness.claude_runner import run_harness
from ..harness.vibe_runner import run_vibe_harness
from ..llm import structured, text_of
from ..runtime import _last_ai_text
from . import a1_react
from .common import DOMAIN_CONTEXT, ArchInfo, HydroState, harness_for

INFO = ArchInfo(
    key="a5",
    name="A5 - Composed: router -> {LLM | ReAct | harness} -> human check",
    pattern="Composing: harness agents inside your own framework (section F)",
    summary="A router sends each request to the cheapest component that can do it: plain LLM, the ReAct agent, "
            "or the provider's own code-agent harness (Claude Agent SDK or Mistral Vibe). A human checks the "
            "result when the router is unsure.",
    use_when="Mixed workloads where most requests are simple but some need a code agent.",
    trade_off="Saves cost on simple requests; adds a routing step that can be wrong (hence the human check).",
    example_query="What is the difference between KsatVer and KsatHorFrac in wflow_sbm?",
    hitl=True,
)


class Route(BaseModel):
    route: Literal["simple_answer", "parameter_scenario", "open_ended_code"] = Field(
        description="simple_answer: knowledge question, no model run. parameter_scenario: uniform parameter "
                    "multipliers + standard statistics. open_ended_code: needs custom code, spatial edits or custom "
                    "analysis of NetCDF files.")
    confidence: float = Field(description="0-1")
    reason: str


def build(llm, checkpointer=None, rec=None, confidence_threshold: float = 0.7, harness_max_turns: int = 40,
          provider: str | None = None, **_):
    router = structured(llm, Route)
    # same vendor for the whole graph: Mistral routes to Vibe, Claude (and GPT, which has no harness here) to Claude
    run_code_agent = run_vibe_harness if harness_for(provider) == "mistral" else run_harness
    react = a1_react.build(llm)

    def route(state: HydroState):
        r: Route = router.invoke(f"{DOMAIN_CONTEXT}\n\nRoute this request:\n{state['query']}")
        return {"data": {"route": r.model_dump()}, "note": r.model_dump_json()}

    def simple_answer(state: HydroState):
        msg = llm.invoke(f"{DOMAIN_CONTEXT}\n\nAnswer briefly (no model runs available here):\n{state['query']}")
        return {"answer": text_of(msg)}

    def parameter_scenario(state: HydroState):
        res = react.invoke({"messages": [HumanMessage(state["query"])]})
        return {"answer": _last_ai_text(res["messages"])}

    def open_ended_code(state: HydroState):
        r = run_code_agent(state["query"], recorder=rec, max_turns=harness_max_turns)
        return {"answer": r.answer or r.error, "run_ids": r.run_ids, "plots": r.plots}

    def human_check(state: HydroState):
        r = state["data"]["route"]
        if r["confidence"] >= confidence_threshold:
            return {"note": "confidence above threshold - no human check"}
        ans = interrupt({"question": f"Router was unsure (confidence {r['confidence']:.2f}, route "
                                     f"'{r['route']}'). Accept this answer? (yes / no + comment)",
                         "answer": state.get("answer", "")[:1500]})
        if str(ans).strip().lower().startswith(("y", "ok")):
            return {"note": "human accepted"}
        return {"answer": f"[REJECTED by reviewer: {ans}]\n\n{state.get('answer', '')}"}

    g = StateGraph(HydroState)
    for name, fn in [("route", route), ("simple_answer", simple_answer), ("parameter_scenario", parameter_scenario),
                     ("open_ended_code", open_ended_code), ("human_check", human_check)]:
        g.add_node(name, fn)
    g.add_edge(START, "route")
    g.add_conditional_edges("route", lambda s: s["data"]["route"]["route"],
                            ["simple_answer", "parameter_scenario", "open_ended_code"])
    for n in ("simple_answer", "parameter_scenario", "open_ended_code"):
        g.add_edge(n, "human_check")
    g.add_edge("human_check", END)
    return g.compile(checkpointer=checkpointer)
