"""Helpers that make graph specs easier to build: insert a step on a connection, auto-label branches,
and one-click fixes for the common validation errors. All functions take and return plain JSON specs.
"""

from __future__ import annotations

import copy

from .nodes import NODE_TYPES, TOOL_ACTIONS

BRANCHING = ("critic", "router", "human")


def _outs(spec: dict, nid: str) -> list[dict]:
    return [e for e in spec["edges"] if e["source"] == nid]


def _reaches(spec: dict, src: str, dst: str) -> bool:
    """True if dst can be reached from src by following edges."""
    seen, stack = set(), [src]
    while stack:
        cur = stack.pop()
        if cur == dst:
            return True
        if cur in seen:
            continue
        seen.add(cur)
        stack += [e["target"] for e in spec["edges"] if e["source"] == cur]
    return False


def _router_label(spec: dict, target: str) -> str:
    """A default branch description for a router edge, from what the target node does."""
    n = next((n for n in spec["nodes"] if n["id"] == target), None)
    if n is None:
        return target
    t, cfg = n["type"], n.get("config") or {}
    return {
        "plain_llm": f"{target}: answer from knowledge, no model run",
        "llm_agent": f"{target}: run model scenarios with tools",
        "claude_harness": f"{target}: needs custom code or NetCDF edits",
        "tool_step": f"{target}: {TOOL_ACTIONS.get(cfg.get('action', ''), 'fixed code step')}",
        "end": "done, nothing more to do",
    }.get(t, f"{target}: {NODE_TYPES[t].label if t in NODE_TYPES else t}")


def auto_label(spec: dict) -> tuple[dict, list[str]]:
    """Fill in missing branch labels. Returns (new spec, notes on what was changed).

    Critic / human with two edges: the edge that loops back (its target leads back to the node) gets
    'revise' / 'rejected', the other 'approved'. Router: every unlabelled edge gets a description of its target.
    """
    spec = copy.deepcopy(spec)
    notes: list[str] = []
    types = {n["id"]: n["type"] for n in spec["nodes"]}
    for nid, t in types.items():
        outs = _outs(spec, nid)
        if t in ("critic", "human") and len(outs) == 2:
            yes, no = ("approved", "revise") if t == "critic" else ("approved", "rejected")
            labels = [(e.get("label") or "").strip() for e in outs]
            if sorted(labels) == sorted([yes, no]):
                continue
            back = [_reaches(spec, e["target"], nid) for e in outs]
            if back.count(True) == 1:  # exactly one edge loops back -> unambiguous
                for e, b in zip(outs, back):
                    e["label"] = no if b else yes
                notes.append(f"Labelled {nid}'s edges '{yes}' and '{no}' (the loop back is '{no}').")
            elif labels.count("") == 1 and any(lb in (yes, no) for lb in labels):
                given = next(lb for lb in labels if lb)
                for e in outs:
                    if not (e.get("label") or "").strip():
                        e["label"] = no if given == yes else yes
                notes.append(f"Labelled {nid}'s second edge.")
        elif t == "router":
            for e in outs:
                if not (e.get("label") or "").strip():
                    e["label"] = _router_label(spec, e["target"])
                    notes.append(f"Router {nid}: labelled the edge to {e['target']} - edit it to describe "
                                 "when to take that branch.")
    return spec, notes


def insert_on_edge(spec: dict, node: dict, source: str, target: str) -> dict:
    """Put a new node on the connection source -> target (source -> node -> target).

    A critic also gets its loop: critic -'revise'-> source. A router gets one branch to target;
    connect a second branch yourself. Nodes below the insertion point move down to make room.
    """
    spec = copy.deepcopy(spec)
    edge = next((e for e in spec["edges"] if e["source"] == source and e["target"] == target), None)
    if edge is None:
        raise ValueError(f"No connection {source} -> {target}")
    pos = {n["id"]: n.get("pos", [0, 0]) for n in spec["nodes"]}
    tx, ty = pos[target]
    for n in spec["nodes"]:  # make room: shift everything at or below the target down
        if n.get("pos", [0, 0])[1] >= ty and n["id"] != source:
            n["pos"] = [n["pos"][0], n["pos"][1] + 130]
    node = {**node, "pos": [tx, ty]}
    spec["nodes"].append(node)
    nid, t = node["id"], node["type"]
    label = edge.get("label", "")
    spec["edges"].remove(edge)
    spec["edges"].append({"source": source, "target": nid, "label": label})
    if t == "critic":
        spec["edges"] += [{"source": nid, "target": target, "label": "approved"},
                          {"source": nid, "target": source, "label": "revise"}]
    elif t == "router":
        spec["edges"].append({"source": nid, "target": target, "label": _router_label(spec, target)})
    else:
        spec["edges"].append({"source": nid, "target": target, "label": ""})
    return spec


def keep_only(spec: dict, source: str, target: str) -> dict:
    """Remove all outgoing edges of source except one to target (duplicates are dropped too)."""
    spec = copy.deepcopy(spec)
    kept, edges = False, []
    for e in spec["edges"]:
        if e["source"] == source:
            if e["target"] != target or kept:
                continue
            kept = True
        edges.append(e)
    spec["edges"] = edges
    return spec


def connect(spec: dict, source: str, target: str, label: str = "") -> dict:
    spec = copy.deepcopy(spec)
    spec["edges"].append({"source": source, "target": target, "label": label})
    return spec


def problems(spec: dict) -> list[dict]:
    """Fixable problems per node: {'node', 'kind': 'too_many'|'none', 'targets': [...]}. Used for fix buttons."""
    out = []
    for n in spec["nodes"]:
        nid, t = n["id"], n["type"]
        if t in ("end", *BRANCHING):
            if t in BRANCHING and not _outs(spec, nid):
                out.append({"node": nid, "kind": "none", "targets": []})
            continue
        targets = [e["target"] for e in _outs(spec, nid)]
        if len(targets) > 1:
            out.append({"node": nid, "kind": "too_many", "targets": targets})
        elif not targets:
            out.append({"node": nid, "kind": "none", "targets": []})
    return out
