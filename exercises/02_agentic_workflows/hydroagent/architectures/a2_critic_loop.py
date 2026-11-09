"""A2 - Loop / generator-critic: calibrate the model against observed discharge.

    generator (LLM proposes multipliers) -> simulate + score (code: Wflow, KGE)
        -> critic (LLM or rule: approve / feedback) --feedback--> generator
                                                   --approved--> report

The loop structure, exit conditions and the scoring are fixed in code; the LLM only proposes
parameter values and critiques. "Observed" discharge comes from the twin experiment (twin.py).
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .. import twin
from .. import wflow_tools as W
from .common import DOMAIN_CONTEXT, ArchInfo, HydroState

INFO = ArchInfo(
    key="a2",
    name="A2 - Generator-critic loop (calibration)",
    pattern="Deterministic workflow > Loop / generator-critic",
    summary="Generator LLM proposes parameter multipliers, code runs Wflow and computes KGE vs observed, "
            "a critic approves or sends feedback. Max N iterations.",
    use_when="Iterative improvement towards a measurable target (calibration, report review).",
    trade_off="Quality improves per iteration but each loop costs LLM calls and a model run; needs a clear stop rule.",
    example_query="Calibrate ksat_vertical, soil_thickness and ksat_horizontal_ratio so the Epinal discharge "
                  "matches observations (target KGE 0.95).",
)


class Multiplier(BaseModel):
    parameter: str
    multiplier: float


class Proposal(BaseModel):
    """Next parameter set to try."""
    reasoning: str = Field(description="1-2 sentences: what you learned from previous tries and why this set")
    multipliers: list[Multiplier]


class Verdict(BaseModel):
    approved: bool = Field(description="true if the calibration is good enough to stop")
    feedback: str = Field(description="concrete advice for the next proposal (direction/size of changes)")


def build(llm, checkpointer=None, max_iter: int = 5, target_kge: float = 0.95, critic: str = "llm",
          params: tuple[str, ...] = tuple(twin.CALIBRATION_PARAMS), **_):
    catalogue = "\n".join(f"- {p['key']} (x{p['allowed_multiplier'][0]}-x{p['allowed_multiplier'][1]}): "
                          f"{p['description']}" for p in W.list_parameters() if p["key"] in params)
    generator = llm.with_structured_output(Proposal)
    critic_llm = llm.with_structured_output(Verdict)

    def _hist(state) -> str:
        rows = state.get("data", {}).get("history", [])
        return "\n".join(f"iter {h['iter']}: {h['multipliers']} -> {h['score']}" for h in rows) or "(none yet)"

    def generate(state: HydroState):
        d = state.get("data") or {"history": [], "feedback": ""}
        p: Proposal = generator.invoke(
            f"{DOMAIN_CONTEXT}\n\nTask: {state['query']}\nCalibrate ONLY these parameters:\n{catalogue}\n"
            f"Scores: KGE (1 best), r=timing, alpha=variability ratio sim/obs, beta=volume ratio sim/obs.\n"
            f"History:\n{_hist(state)}\nCritic feedback: {d.get('feedback') or '-'}\nPropose the next set.")
        mult = {m.parameter: m.multiplier for m in p.multipliers if m.parameter in params}
        for k, v in list(mult.items()):  # clip into bounds instead of failing
            lo, hi = W.PARAMETERS[k].bounds
            mult[k] = min(max(v, lo), hi)
        return {"data": {**d, "proposal": mult, "reasoning": p.reasoning}, "note": f"{mult} | {p.reasoning}"}

    def simulate(state: HydroState):
        d = state["data"]
        it = len(d["history"]) + 1
        info = W.run_scenario(d["proposal"], f"calib iter {it}")
        if info.status != "done":
            score = {"error": info.error}
        else:
            score = twin.score_run(info.run_id)
        hist = d["history"] + [{"iter": it, "run_id": info.run_id, "multipliers": d["proposal"], "score": score}]
        return {"data": {**d, "history": hist}, "run_ids": [info.run_id], "note": f"iter {it}: {score}"}

    def critique(state: HydroState):
        d = state["data"]
        last = d["history"][-1]["score"]
        kge = last.get("kge", -99)
        if kge >= target_kge:
            return {"data": {**d, "approved": True, "feedback": f"KGE {kge} >= target {target_kge}"},
                    "note": "target reached (rule)"}
        if len(d["history"]) >= max_iter:
            return {"data": {**d, "approved": True, "feedback": "max iterations reached"}, "note": "max iterations"}
        if critic == "rule":
            fb = (f"beta={last.get('beta')}: {'reduce' if last.get('beta', 1) > 1 else 'increase'} volume; "
                  f"alpha={last.get('alpha')}: {'dampen' if last.get('alpha', 1) > 1 else 'sharpen'} the response.")
            return {"data": {**d, "approved": False, "feedback": fb}, "note": fb}
        v: Verdict = critic_llm.invoke(
            f"{DOMAIN_CONTEXT}\n\nYou review a calibration. Target KGE {target_kge}. "
            f"Iteration {len(d['history'])}/{max_iter}.\nHistory:\n{_hist(state)}\n"
            "Should we stop? If not, give concrete feedback on which parameter to move in which direction.")
        return {"data": {**d, "approved": v.approved, "feedback": v.feedback}, "note": v.feedback}

    def report(state: HydroState):
        d = state["data"]
        ok = [h for h in d["history"] if "kge" in h["score"]]
        best = max(ok, key=lambda h: h["score"]["kge"]) if ok else None
        plots = []
        if best:
            plots.append(str(W.plot_runs(["baseline", best["run_id"]], obs=twin.load_observed(),
                                         title="Calibration: best run vs observed")))
        msg = llm.invoke(f"{DOMAIN_CONTEXT}\n\nSummarise this calibration for a hydrologist in <=8 lines: "
                         f"best parameter set, KGE progression, and caveats (equifinality).\n"
                         f"History:\n{_hist(state)}")
        text = msg.content if isinstance(msg.content, str) else str(msg.content)
        return {"answer": text, "plots": plots, "data": {**d, "best": best}}

    g = StateGraph(HydroState)
    g.add_node("generator", generate)
    g.add_node("simulate", simulate)
    g.add_node("critic", critique)
    g.add_node("report", report)
    g.add_edge(START, "generator")
    g.add_edge("generator", "simulate")
    g.add_edge("simulate", "critic")
    g.add_conditional_edges("critic", lambda s: "report" if s["data"].get("approved") else "generator",
                            {"report": "report", "generator": "generator"})
    g.add_edge("report", END)
    return g.compile(checkpointer=checkpointer)
