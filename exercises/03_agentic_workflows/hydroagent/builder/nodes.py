"""Node catalogue for the build-your-own-agent canvas.

Each node type maps to a drawio legend colour:
deterministic code (grey), plain LLM call (blue), agent / harness (green / purple), human (orange).
A graph spec is plain JSON, so it can be drawn in Streamlit, edited by hand, and run from the CLI:

    {"name": "...",
     "nodes": [{"id": "start", "type": "start"}, {"id": "agent", "type": "llm_agent", "config": {...}}, ...],
     "edges": [{"source": "start", "target": "agent", "label": ""}, ...]}
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NodeType:
    type: str
    label: str
    colour: str
    description: str
    defaults: dict = field(default_factory=dict)
    branch_labels: tuple[str, ...] = ()   # required labels on outgoing edges (empty = exactly one plain edge)


NODE_TYPES: dict[str, NodeType] = {n.type: n for n in [
    NodeType("start", "Start", "#ffffff", "Entry point: the user's request."),
    NodeType("end", "End", "#ffffff", "Exit: the last message is the answer."),
    NodeType("llm_agent", "LLM agent (ReAct)", "#d5e8d4",
             "An LLM with tools, looping until done. Choose its tool sets and instructions.",
             {"system_prompt": "You are a hydrological modelling assistant. Use the tools to answer with model "
                               "evidence and give key numbers.",
              "toolsets": ["parameters", "simulation", "analysis"]}),
    NodeType("plain_llm", "Plain LLM call", "#dae8fc",
             "One LLM call, no tools. Good for summarising, rewriting, explaining.",
             {"instruction": "Summarise the findings above for a water manager in 5 bullet points."}),
    NodeType("tool_step", "Code step (deterministic)", "#f5f5f5",
             "Runs fixed Python code, no LLM: e.g. a baseline run, a plot of all runs, a score vs observations.",
             {"action": "plot_all_runs"}),
    NodeType("critic", "Critic (LLM judge)", "#e1d5e7",
             "Judges the last answer against criteria. Outgoing edges must be labelled 'approved' and 'revise'.",
             {"criteria": "The answer quotes numbers from a model run (run_id) and explains them physically.",
              "max_loops": 2}, ("approved", "revise")),
    NodeType("router", "Router (LLM classifier)", "#fff2cc",
             "Chooses one outgoing edge. Label every outgoing edge with a short description of when to take it.",
             {"instruction": "Pick the branch that best fits the user's request."}, ("*",)),
    NodeType("human", "Human check", "#ffe6cc",
             "Pauses for a person. With two outgoing edges, label them 'approved' and 'rejected'.",
             {"question": "Do you approve the result above?"}),
    NodeType("claude_harness", "Claude harness (code agent)", "#d0cee2",
             "Claude Agent SDK agent with skills that can write and run code on the NetCDF files. Claude-only.",
             {"task": "{query}", "max_turns": 30}),
]}

TOOL_ACTIONS = {
    "baseline": "Make sure the baseline run exists and report its statistics.",
    "plot_all_runs": "Plot all runs made in this session (plus baseline).",
    "summarise_runs": "Flow statistics of all runs in this session vs baseline.",
    "score_runs": "KGE/NSE of all runs in this session against the observed (twin) discharge.",
}
