"""A3 - Director: supervisor + specialist subagents + human-in-the-loop + scenario allow-list.

Pattern adapted from the Agent4Hydro project (Director -> Hydrology / Summarizer specialists), applied to the
Moselle Wflow model. Only the *architecture* is reused, not its tools.

    classify (LLM) --missing info--> clarify (HUMAN) --> classify
             |--out of scope--> blocked
             '--ok--> plan (code: scenario template) --> approve (HUMAN gate)
                  --> modeller (subagent, retries on error) --> analyst (subagent) --> finalize

Guarantees you get from the framework that a single ReAct agent cannot promise:
* only scenarios in the allow-list run (stage gating), with validated parameter ranges;
* a human approves the plan before compute is spent (HITL);
* fixed step order + audit trail; errors are retried, then escalated.
"""

from __future__ import annotations

from typing import Literal, Optional

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from .. import wflow_tools as W
from ..runtime import _last_ai_text
from ..tools import ARTIFACTS, get_tools
from ..llm import structured
from .common import DOMAIN_CONTEXT, ArchInfo, HydroState

INFO = ArchInfo(
    key="a3",
    name="A3 - Director (supervisor + subagents + HITL)",
    pattern="Dynamic orchestration > Subagents (supervisor) + Human-in-the-loop + Custom logic",
    summary="A Director classifies the request against an allow-list of scenarios, asks the user for missing "
            "inputs, gets plan approval, then dispatches a Modeller and an Analyst subagent.",
    use_when="You need guarantees: only approved scenario types, human sign-off, audit trail, error escalation.",
    trade_off="More code, more LLM calls and slower; less flexible for questions outside the allow-list.",
    example_query="What if the soils in the Moselle become less permeable?",
    hitl=True,
)

# --- Scenario allow-list: what the Director is allowed to run ---------------------------------------------
SCENARIOS = {
    "soil_permeability": ("ksat_vertical", "Change in soil permeability (e.g. compaction, sealing)."),
    "soil_depth": ("soil_thickness", "Change in effective soil depth (e.g. erosion)."),
    "snowmelt_rate": ("degree_day", "Faster/slower snow melt (warmer/colder spring)."),
    "river_roughness": ("river_manning_n", "River-bed roughness (vegetation, restoration)."),
    "evaporation_demand": ("crop_factor", "Higher/lower evaporative demand (warmer climate, vegetation change)."),
}


class Classification(BaseModel):
    scenario: Literal["soil_permeability", "soil_depth", "snowmelt_rate", "river_roughness", "evaporation_demand",
                      "out_of_scope"]
    change_pct: Optional[float] = Field(None, description="signed % change requested, e.g. -30 for '30% less'. "
                                                          "null if the user did not give a magnitude")
    gauge: str = Field("9116044", description="GRDC gauge id, default Epinal 9116044")
    reason: str = Field(description="one sentence")


MODELLER_PROMPT = DOMAIN_CONTEXT + """
You are the MODELLER specialist. Execute exactly the run(s) requested by the Director using run_scenario.
Do not interpret results. Reply with the run_id(s) and their status only."""

ANALYST_PROMPT = DOMAIN_CONTEXT + """
You are the ANALYST specialist. Given run_ids, use compare_runs/get_run_summary and plot_hydrographs
(include "baseline"), then write a short technical note: numbers first (% change mean/peak/high/low flow),
then physical interpretation, then one caveat."""


def build(llm, checkpointer=None, require_approval: bool = True, max_retries: int = 2, **_):
    classifier = structured(llm, Classification)
    modeller = create_agent(llm, get_tools("parameters", "simulation"), system_prompt=MODELLER_PROMPT,
                            name="modeller")
    analyst = create_agent(llm, get_tools("analysis"), system_prompt=ANALYST_PROMPT, name="analyst")
    allow = "\n".join(f"- {k}: {v[1]}" for k, v in SCENARIOS.items())

    def classify(state: HydroState):
        c: Classification = classifier.invoke(
            f"{DOMAIN_CONTEXT}\n\nClassify the user's request into one allowed scenario type:\n{allow}\n"
            f"- out_of_scope: anything else.\n\nConversation:\n{state['query']}")
        return {"data": {**(state.get("data") or {}), "cls": c.model_dump()}, "note": c.model_dump_json()}

    def route_after_classify(state: HydroState):
        c = state["data"]["cls"]
        if c["scenario"] == "out_of_scope":
            return "blocked"
        return "clarify" if c["change_pct"] is None else "plan"

    def clarify(state: HydroState):
        c = state["data"]["cls"]
        answer = interrupt(f"Scenario '{c['scenario']}' recognised. By how many percent should "
                           f"'{SCENARIOS[c['scenario']][0]}' change? (e.g. -30 for 30% lower)")
        return {"query": f"{state['query']}\nUser clarification: {answer}", "messages": [HumanMessage(str(answer))]}

    def blocked(state: HydroState):
        return {"answer": "This request is outside the scenarios I am allowed to run:\n" + allow +
                          "\n\nPlease rephrase as one of these, or use the single agent (A1) / harness (A4)."}

    def plan(state: HydroState):
        c = state["data"]["cls"]
        key = SCENARIOS[c["scenario"]][0]
        lo, hi = W.PARAMETERS[key].bounds
        mult = round(min(max(1 + c["change_pct"] / 100, lo), hi), 3)
        todo = [
            {"agent": "modeller", "task": f"Run scenario multipliers={{'{key}': {mult}}} label='{c['scenario']} "
                                          f"{c['change_pct']:+g}%'."},
            {"agent": "analyst", "task": f"Analyse the run(s) at gauge {c['gauge']} vs baseline and plot them."},
        ]
        return {"data": {**state["data"], "todo": todo, "multipliers": {key: mult}}, "note": str(todo)}

    def approve(state: HydroState):
        if not require_approval:
            return {"note": "auto-approved"}
        todo = state["data"]["todo"]
        ans = interrupt({"question": "Approve this plan? (yes / no)", "plan": todo})
        ok = str(ans).strip().lower() in ("y", "yes", "ok", "approve", "approved", "true")
        return {"data": {**state["data"], "approved": ok}, "note": f"human: {ans}"}

    def modeller_node(state: HydroState):
        d = state["data"]
        task = d["todo"][0]["task"]
        arts = ARTIFACTS.get() or {"run_ids": []}
        before = set(arts.get("run_ids", []))
        err = ""
        for attempt in range(max_retries + 1):
            res = modeller.invoke({"messages": [HumanMessage(task + (f"\nPrevious attempt failed: {err}" if err else ""))]})
            text = _last_ai_text(res["messages"])
            new = [r for r in arts.get("run_ids", []) if r not in before]
            done = [r for r in new if W.load_run(r).status == "done"]
            if done:
                return {"run_ids": done, "data": {**d, "modeller_report": text}, "note": f"runs {done}"}
            err = text[:500]
        return {"data": {**d, "escalate": f"Modeller failed after {max_retries + 1} attempts: {err}"},
                "note": "escalated"}

    def analyst_node(state: HydroState):
        d = state["data"]
        res = analyst.invoke({"messages": [HumanMessage(f"{d['todo'][1]['task']} Run ids: {state['run_ids']}. "
                                                        f"User question: {state['query']}")]})
        return {"data": {**d, "analysis": _last_ai_text(res["messages"])}}

    def finalize(state: HydroState):
        d = state["data"]
        if d.get("escalate"):
            return {"answer": f"ESCALATED to a human expert: {d['escalate']}"}
        if d.get("approved") is False:
            return {"answer": "Plan rejected by the user; nothing was run."}
        return {"answer": d.get("analysis", "")}

    g = StateGraph(HydroState)
    for name, fn in [("classify", classify), ("clarify", clarify), ("blocked", blocked), ("plan", plan),
                     ("approve", approve), ("modeller", modeller_node), ("analyst", analyst_node),
                     ("finalize", finalize)]:
        g.add_node(name, fn)
    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", route_after_classify, ["blocked", "clarify", "plan"])
    g.add_edge("clarify", "classify")
    g.add_edge("blocked", END)
    g.add_edge("plan", "approve")
    g.add_conditional_edges("approve", lambda s: "modeller" if s["data"].get("approved", True) else "finalize",
                            ["modeller", "finalize"])
    g.add_conditional_edges("modeller", lambda s: "finalize" if s["data"].get("escalate") else "analyst",
                            ["analyst", "finalize"])
    g.add_edge("analyst", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)
