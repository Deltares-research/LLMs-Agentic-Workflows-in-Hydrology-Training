"""A0 - No agent: plain LLM calls inside a fixed (sequential) workflow.

    interpret (LLM, structured output) -> simulate (code) -> report (LLM)

The LLM never chooses what happens next; *your code* fixes the order. Two LLM calls, always.
Drawio section E: "Can one LLM call do it? -> plain LLM call (no agent needed)".
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .. import wflow_tools as W
from .common import DOMAIN_CONTEXT, ArchInfo, HydroState

INFO = ArchInfo(
    key="a0",
    name="A0 - Plain LLM + fixed workflow",
    pattern="Deterministic workflow > Sequential (no agent)",
    summary="LLM translates the question into multipliers (structured output); code runs Wflow; LLM writes the report.",
    use_when="The steps are always the same and only the inputs vary.",
    trade_off="Cheapest and most predictable; cannot handle questions that need extra steps or several runs.",
    example_query="What happens to peak and low flows at Epinal if soils become 30% less permeable?",
)


class Multiplier(BaseModel):
    parameter: str = Field(description="parameter key from the catalogue")
    multiplier: float


class ScenarioSpec(BaseModel):
    """Translation of a what-if question into Wflow parameter multipliers."""
    label: str = Field(description="short label, e.g. 'ksat -30%'")
    multipliers: list[Multiplier]
    rationale: str = Field(description="one sentence: why these parameters")


def build(llm, checkpointer=None, **_):
    catalogue = "\n".join(f"- {p['key']} (allowed x{p['allowed_multiplier'][0]}-x{p['allowed_multiplier'][1]}): "
                          f"{p['description']}" for p in W.list_parameters())
    planner = llm.with_structured_output(ScenarioSpec)

    def interpret(state: HydroState):
        spec: ScenarioSpec = planner.invoke(
            f"{DOMAIN_CONTEXT}\n\nParameter catalogue:\n{catalogue}\n\n"
            f"Translate this question into ONE scenario (parameter multipliers):\n{state['query']}")
        return {"data": {"spec": spec.model_dump()}, "note": spec.model_dump_json()}

    def simulate(state: HydroState):
        spec = state["data"]["spec"]
        W.ensure_baseline()
        info = W.run_scenario({m["parameter"]: m["multiplier"] for m in spec["multipliers"]}, spec["label"])
        summary = W.summarise_run(info.run_id)
        plot = str(W.plot_runs(["baseline", info.run_id]))
        return {"data": {**state["data"], "summary": summary}, "run_ids": [info.run_id], "plots": [plot],
                "note": f"run {info.run_id}: {info.status}"}

    def report(state: HydroState):
        msg = llm.invoke(f"{DOMAIN_CONTEXT}\n\nQuestion: {state['query']}\n\nScenario: {state['data']['spec']}\n"
                         f"Model result (vs baseline): {state['data']['summary']}\n\n"
                         "Write a short answer for a hydrologist (numbers first, then one-line explanation).")
        return {"answer": msg.content if isinstance(msg.content, str) else str(msg.content)}

    g = StateGraph(HydroState)
    g.add_node("interpret", interpret)
    g.add_node("simulate", simulate)
    g.add_node("report", report)
    g.add_edge(START, "interpret")
    g.add_edge("interpret", "simulate")
    g.add_edge("simulate", "report")
    g.add_edge("report", END)
    return g.compile(checkpointer=checkpointer)
