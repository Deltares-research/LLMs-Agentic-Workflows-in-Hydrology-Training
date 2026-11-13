"""Validate a graph spec (JSON) and compile it into a runnable LangGraph StateGraph."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field, create_model

from .. import twin
from .. import wflow_tools as W
from ..llm import structured, text_of
from ..architectures.common import DOMAIN_CONTEXT, HydroState, make_session
from ..harness.claude_runner import run_harness
from ..runtime import GraphSession, _last_ai_text
from ..tools import ARTIFACTS, get_tools
from .edit import auto_label
from .nodes import NODE_TYPES, TOOL_ACTIONS


class GraphSpecError(ValueError):
    pass


def validate(spec: dict) -> list[str]:
    """Return a list of problems (empty = valid). Missing branch labels are filled in first (see auto_label)."""
    spec, _notes = auto_label(spec)
    errs: list[str] = []
    nodes = {n["id"]: n for n in spec.get("nodes", [])}
    edges = spec.get("edges", [])
    starts = [i for i, n in nodes.items() if n["type"] == "start"]
    ends = [i for i, n in nodes.items() if n["type"] == "end"]
    if len(starts) != 1:
        errs.append(f"Need exactly one Start node (found {len(starts)}).")
    if not ends:
        errs.append("Need at least one End node.")
    for n in nodes.values():
        if n["type"] not in NODE_TYPES:
            errs.append(f"Unknown node type {n['type']!r} ({n['id']}).")
    out: dict[str, list[dict]] = {i: [] for i in nodes}
    for e in edges:
        if e["source"] not in nodes or e["target"] not in nodes:
            errs.append(f"Edge {e['source']} -> {e['target']} points to a missing node.")
            continue
        out[e["source"]].append(e)
    for i, n in nodes.items():
        t, outs = n["type"], out.get(i, [])
        labels = [(e.get("label") or "").strip() for e in outs]
        if t == "end":
            if outs:
                errs.append(f"End node {i} must not have outgoing edges.")
        elif t == "critic":
            if sorted(labels) != ["approved", "revise"]:
                errs.append(f"Critic {i} needs two connections: 'approved' (go on) and 'revise' (back to the agent). "
                        f"It has {len(outs)}. Tip: add the critic with 'Insert on connection' and both are made for you.")
        elif t == "router":
            if len(outs) < 2 or any(not lb for lb in labels) or len(set(labels)) != len(labels):
                errs.append(f"Router {i} needs at least 2 branches (has {len(outs)}). Connect it to each option; "
                            "the branch descriptions are filled in for you and can be edited by clicking an edge.")
        elif t == "human" and len(outs) == 2:
            if sorted(labels) != ["approved", "rejected"]:
                errs.append(f"Human node {i} with two edges needs labels 'approved' and 'rejected'.")
        elif len(outs) > 1:
            errs.append(f"{i} ({NODE_TYPES[t].label if t in NODE_TYPES else t}) goes to {len(outs)} next steps "
                        f"({', '.join(e['target'] for e in outs)}). A normal step has one next step: keep one "
                        "connection, or put a Router or Critic after it to choose between paths.")
        elif not outs:
            errs.append(f"{i} ({NODE_TYPES[t].label if t in NODE_TYPES else t}) has no next step. "
                        "Connect it to another node or to End.")
    if len(starts) == 1:  # reachability
        seen, stack = set(), [starts[0]]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack += [e["target"] for e in out.get(cur, [])]
        for i in nodes:
            if i not in seen:
                errs.append(f"Node {i} is not reachable from Start.")
        if ends and not any(e in seen for e in ends):
            errs.append("No End node is reachable from Start.")
    return errs


def _transcript(state, n: int = 12) -> str:
    lines = []
    for m in state.get("messages", [])[-n:]:
        who = getattr(m, "name", None) or m.type
        c = text_of(m)
        if c.strip():
            lines.append(f"[{who}] {c.strip()[:3000]}")
    return "\n".join(lines)


def _session_runs() -> list[str]:
    a = ARTIFACTS.get() or {}
    return list(a.get("run_ids", []))


def compile_spec(spec: dict, llm, checkpointer=None, rec=None, **_):
    spec, _notes = auto_label(spec)
    errs = validate(spec)
    if errs:
        raise GraphSpecError("\n".join(errs))
    nodes = {n["id"]: n for n in spec["nodes"]}
    out: dict[str, list[dict]] = {i: [] for i in nodes}
    for e in spec["edges"]:
        out[e["source"]].append(e)

    def target(i: str) -> str:
        t = nodes[i]["type"]
        return START if t == "start" else END if t == "end" else i

    def make(nid: str, n: dict):
        t = n["type"]
        cfg = {**NODE_TYPES[t].defaults, **(n.get("config") or {})}

        if t == "llm_agent":
            agent = create_agent(llm, get_tools(*cfg["toolsets"]) if cfg["toolsets"] else [],
                                 system_prompt=DOMAIN_CONTEXT + "\n\n" + cfg["system_prompt"], name=nid)

            def fn(state):
                msgs = list(state["messages"])
                # After a tool step / plain LLM / another agent the conversation ends on an assistant message.
                # Mistral rejects that ("Expected last role User or Tool"), so hand the turn back to the user
                # (only for this call; the shared state is unchanged).
                if msgs and isinstance(msgs[-1], AIMessage):
                    msgs.append(HumanMessage("Continue with your step, using the results above."))
                res = agent.invoke({"messages": msgs})
                return {"messages": [AIMessage(_last_ai_text(res["messages"]), name=nid)]}

        elif t == "plain_llm":
            def fn(state):
                msg = llm.invoke(f"{DOMAIN_CONTEXT}\n\nConversation so far:\n{_transcript(state)}\n\n"
                                 f"Instruction: {cfg['instruction']}")
                return {"messages": [AIMessage(text_of(msg),
                                               name=nid)]}

        elif t == "tool_step":
            def fn(state):
                act = cfg["action"]
                runs = [r for r in _session_runs() if r != "baseline"]
                if act == "baseline":
                    W.ensure_baseline()
                    res = W.summarise_run("baseline", reference="")
                elif act == "plot_all_runs":
                    p = W.plot_runs(["baseline", *runs])
                    a = ARTIFACTS.get()
                    if a is not None:
                        a.setdefault("plots", []).append(str(p))
                    res = f"Plot written: {p}"
                elif act == "summarise_runs":
                    res = [W.summarise_run(r) for r in runs] or "No runs yet."
                elif act == "score_runs":
                    res = {r: twin.score_run(r) for r in runs} or "No runs yet."
                else:
                    res = f"Unknown action {act!r}; choose from {list(TOOL_ACTIONS)}"
                return {"messages": [AIMessage(f"[{act}] {json.dumps(res, default=str)[:4000]}", name=nid)]}

        elif t == "critic":
            class Verdict(BaseModel):
                approved: bool
                feedback: str = Field(description="what to improve, concretely")

            judge = structured(llm, Verdict)

            def fn(state):
                d = dict(state.get("data") or {})
                loops = d.get(f"{nid}_loops", 0)
                if loops >= int(cfg["max_loops"]):
                    d[f"{nid}_verdict"] = "approved"
                    return {"data": d, "note": "max loops reached -> approved"}
                v: Verdict = judge.invoke(f"Criteria: {cfg['criteria']}\n\nConversation:\n{_transcript(state)}\n\n"
                                          "Does the latest answer meet the criteria?")
                d[f"{nid}_loops"] = loops + 1
                d[f"{nid}_verdict"] = "approved" if v.approved else "revise"
                msgs = [] if v.approved else [HumanMessage(f"Reviewer feedback: {v.feedback}", name=nid)]
                return {"data": d, "messages": msgs, "note": f"{d[f'{nid}_verdict']}: {v.feedback}"}

        elif t == "router":
            labels = [e["label"].strip() for e in out[nid]]
            Choice = create_model("Choice", branch=(Literal[tuple(labels)], ...), reason=(str, ...))
            chooser = structured(llm, Choice)

            def fn(state):
                c = chooser.invoke(f"{DOMAIN_CONTEXT}\n\n{cfg['instruction']}\nBranches: {labels}\n\n"
                                   f"Conversation:\n{_transcript(state)}")
                d = dict(state.get("data") or {})
                d[f"{nid}_choice"] = c.branch
                return {"data": d, "note": f"{c.branch}: {c.reason}"}

        elif t == "human":
            def fn(state):
                ans = interrupt({"question": cfg["question"], "last_message": _transcript(state, 1)[-1500:]})
                d = dict(state.get("data") or {})
                d[f"{nid}_ok"] = str(ans).strip().lower().startswith(("y", "ok", "approve"))
                return {"data": d, "messages": [HumanMessage(f"Human reviewer: {ans}", name=nid)]}

        elif t == "claude_harness":
            def fn(state):
                task = cfg["task"].replace("{query}", state.get("query", ""))
                task += "\n\nContext from previous steps:\n" + _transcript(state, 6)
                r = run_harness(task, recorder=rec, max_turns=int(cfg["max_turns"]))
                a = ARTIFACTS.get()
                if a is not None:
                    a["run_ids"] += [x for x in r.run_ids if x not in a["run_ids"]]
                return {"messages": [AIMessage(r.answer or r.error, name=nid)], "plots": r.plots}
        else:
            raise GraphSpecError(f"Cannot compile node type {t}")
        return fn

    g = StateGraph(HydroState)
    for nid, n in nodes.items():
        if n["type"] not in ("start", "end"):
            g.add_node(nid, make(nid, n))
    for nid, n in nodes.items():
        t, outs = n["type"], out[nid]
        if t == "end":
            continue
        if t in ("critic", "router") or (t == "human" and len(outs) == 2):
            key = {"critic": f"{nid}_verdict", "router": f"{nid}_choice"}.get(t)
            mapping = {e["label"].strip(): target(e["target"]) for e in outs}
            if t == "human":
                g.add_conditional_edges(nid, lambda s, nid=nid: "approved" if s["data"].get(f"{nid}_ok") else "rejected",
                                        mapping)
            else:
                g.add_conditional_edges(nid, lambda s, key=key: s["data"][key], mapping)
        else:
            g.add_edge(target(nid), target(outs[0]["target"]))
    return g.compile(checkpointer=checkpointer)


def load_spec(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def make_custom_session(spec: dict, provider: str | None = None, on_event=None) -> GraphSession:
    return make_session(f"custom:{spec.get('name', 'graph')}", lambda **kw: compile_spec(spec, **kw), provider=provider,
                        on_event=on_event)
