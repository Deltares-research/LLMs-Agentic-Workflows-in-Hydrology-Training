"""Starter graphs for the canvas (and examples of the JSON spec)."""

from __future__ import annotations


def _n(id, type, x, y, **config):
    return {"id": id, "type": type, "pos": [x, y], **({"config": config} if config else {})}


def _e(s, t, label=""):
    return {"source": s, "target": t, "label": label}


TEMPLATES = {
    "Single agent": {
        "name": "single_agent",
        "nodes": [_n("start", "start", 0, 0), _n("agent", "llm_agent", 0, 120), _n("end", "end", 0, 240)],
        "edges": [_e("start", "agent"), _e("agent", "end")],
    },
    "Agent + critic loop": {
        "name": "agent_critic",
        "nodes": [_n("start", "start", 0, 0), _n("agent", "llm_agent", 0, 120), _n("critic", "critic", 0, 240),
                  _n("summary", "plain_llm", 0, 360), _n("end", "end", 0, 480)],
        "edges": [_e("start", "agent"), _e("agent", "critic"), _e("critic", "agent", "revise"),
                  _e("critic", "summary", "approved"), _e("summary", "end")],
    },
    "Router: answer / model / code": {
        "name": "router",
        "nodes": [_n("start", "start", 200, 0), _n("router", "router", 200, 120),
                  _n("explain", "plain_llm", 0, 260, instruction="Answer the user's question briefly; no model runs."),
                  _n("agent", "llm_agent", 200, 260), _n("harness", "claude_harness", 420, 260),
                  _n("human", "human", 200, 400), _n("end", "end", 200, 520)],
        "edges": [_e("start", "router"), _e("router", "explain", "knowledge question, no simulation"),
                  _e("router", "agent", "uniform parameter scenario"),
                  _e("router", "harness", "needs custom code or spatial NetCDF edits"),
                  _e("explain", "human"), _e("agent", "human"), _e("harness", "human"), _e("human", "end")],
    },
    "Fixed pipeline (no agent)": {
        "name": "pipeline",
        "nodes": [_n("start", "start", 0, 0), _n("baseline", "tool_step", 0, 120, action="baseline"),
                  _n("agent", "llm_agent", 0, 240, toolsets=["simulation"],
                     system_prompt="Run exactly one scenario for the user's request with run_scenario, then stop."),
                  _n("plot", "tool_step", 0, 360, action="plot_all_runs"),
                  _n("report", "plain_llm", 0, 480), _n("end", "end", 0, 600)],
        "edges": [_e("start", "baseline"), _e("baseline", "agent"), _e("agent", "plot"), _e("plot", "report"),
                  _e("report", "end")],
    },
}
