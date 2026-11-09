"""Run any compiled LangGraph graph with tracing and human-in-the-loop (interrupt / resume).

``GraphSession.start(query)`` returns an ``Outcome``. If the graph paused for a human
(``langgraph.types.interrupt``), ``outcome.status == "interrupted"`` and ``outcome.question``
holds the question; answer it with ``session.resume(answer)``. This works the same in the
CLI (``input()``), the notebook (``input()``) and Streamlit (a form + session_state).
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from langgraph.types import Command

from .tools import ARTIFACTS
from .tracing import Recorder


@dataclass
class Outcome:
    status: str                      # done | interrupted | error
    answer: str = ""
    question: Any = None
    run_ids: list[str] = field(default_factory=list)
    plots: list[str] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    state: dict = field(default_factory=dict)
    error: str = ""


class GraphSession:
    def __init__(self, name: str, graph, recorder: Recorder, recursion_limit: int = 60):
        self.name = name
        self.graph = graph
        self.rec = recorder
        self.thread_id = f"{name}-{uuid.uuid4().hex[:8]}"
        self.recursion_limit = recursion_limit
        self.artifacts: dict = {"run_ids": [], "plots": []}

    @property
    def config(self) -> dict:
        return {"configurable": {"thread_id": self.thread_id}, "callbacks": [self.rec.callback()],
                "recursion_limit": self.recursion_limit}

    def _invoke(self, payload) -> Outcome:
        token = ARTIFACTS.set(self.artifacts)
        try:
            for chunk in self.graph.stream(payload, self.config, stream_mode="updates"):
                for node, upd in (chunk or {}).items():
                    if node == "__interrupt__":
                        continue
                    note = upd.get("note") if isinstance(upd, dict) else None
                    self.rec.emit("node", f"node: {node}", note or "")
        except Exception as e:  # surface errors to the user instead of a stack trace in the UI
            self.rec.emit("error", type(e).__name__, str(e))
            return Outcome("error", error=f"{type(e).__name__}: {e}", usage=self.rec.summary(),
                           run_ids=list(self.artifacts.get("run_ids", [])), plots=list(self.artifacts.get("plots", [])))
        finally:
            ARTIFACTS.reset(token)
        snap = self.graph.get_state(self.config)
        values = dict(snap.values or {})
        interrupts = [i for t in (snap.tasks or []) for i in (getattr(t, "interrupts", None) or [])]
        run_ids = _merge(values.get("run_ids"), self.artifacts.get("run_ids"))
        plots = _merge(values.get("plots"), self.artifacts.get("plots"))
        base = dict(run_ids=run_ids, plots=plots, usage=self.rec.summary(), state=values)
        if interrupts:
            q = interrupts[0].value
            self.rec.emit("human", "waiting for human", q)
            return Outcome("interrupted", question=q, **base)
        answer = values.get("answer") or _last_ai_text(values.get("messages", []))
        self.rec.emit("final", "final answer", answer)
        return Outcome("done", answer=answer, **base)

    def start(self, query: str, extra_state: dict | None = None) -> Outcome:
        self.rec.emit("info", f"{self.name} started", query)
        from langchain_core.messages import HumanMessage

        payload = {"messages": [HumanMessage(query)], **(extra_state or {})}
        if "query" in getattr(self.graph, "channels", {}):
            payload["query"] = query
        return self._invoke(payload)

    def resume(self, answer: Any) -> Outcome:
        self.rec.emit("human", "human answered", answer)
        return self._invoke(Command(resume=answer))


def run_interactive(session: GraphSession, query: str, ask: Callable[[Any], Any] = None, **kw) -> Outcome:
    """Drive a session to completion, asking the human via ``ask`` (default: ``input()``)."""
    ask = ask or (lambda q: input(f"\n[agent asks] {q}\n> "))
    out = session.start(query, **kw)
    while out.status == "interrupted":
        out = session.resume(ask(out.question))
    return out


def _merge(*lists) -> list:
    out: list = []
    for lst in lists:
        for x in lst or []:
            if x not in out:
                out.append(x)
    return out


def _last_ai_text(messages) -> str:
    for m in reversed(messages or []):
        if getattr(m, "type", "") == "ai":
            c = m.content
            text = c if isinstance(c, str) else " ".join(x.get("text", "") for x in c if isinstance(x, dict))
            if text.strip():
                return text
    return ""


def run_sync(coro):
    """Run a coroutine from sync code, also inside Jupyter/Streamlit where a loop may be running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    box: dict = {}

    def target():
        try:
            box["v"] = asyncio.run(coro)
        except BaseException as e:  # noqa: BLE001
            box["e"] = e

    t = threading.Thread(target=target)
    t.start()
    t.join()
    if "e" in box:
        raise box["e"]
    return box["v"]
