"""Uniform trace of what an agent did: LLM calls, tool calls, tokens, wall time.

Every architecture (and the custom builder) reports through the same ``Recorder``, so the
Streamlit app, the notebook and the CLI can show and compare them side by side.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.callbacks import BaseCallbackHandler


@dataclass
class Event:
    kind: str          # node | llm | tool_call | tool_result | human | info | error | harness | final
    title: str
    detail: str = ""
    t: float = field(default_factory=time.time)
    data: dict = field(default_factory=dict)


def _short(x: Any, n: int = 600) -> str:
    s = x if isinstance(x, str) else json.dumps(x, default=str)
    return s if len(s) <= n else s[:n] + " ..."


class Recorder:
    def __init__(self, on_event: Callable[[Event], None] | None = None):
        self.events: list[Event] = []
        self.on_event = on_event
        self.t0 = time.time()
        self.llm_calls = 0
        self.tool_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0  # only reported by the Claude Agent SDK harness

    def emit(self, kind: str, title: str, detail: Any = "", **data) -> Event:
        ev = Event(kind, title, _short(detail, 4000) if detail else "", data=data)
        self.events.append(ev)
        if self.on_event:
            try:
                self.on_event(ev)
            except Exception:
                pass
        return ev

    def add_usage(self, input_tokens: int = 0, output_tokens: int = 0, cost_usd: float = 0.0) -> None:
        self.input_tokens += int(input_tokens or 0)
        self.output_tokens += int(output_tokens or 0)
        self.cost_usd += float(cost_usd or 0.0)

    def summary(self) -> dict:
        return {"llm_calls": self.llm_calls, "tool_calls": self.tool_calls, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "wall_time_s": round(time.time() - self.t0, 1),
                **({"cost_usd": round(self.cost_usd, 4)} if self.cost_usd else {})}

    def callback(self, tag: str = "") -> "TraceCallback":
        return TraceCallback(self, tag)


class TraceCallback(BaseCallbackHandler):
    """LangChain callback: picks up every model and tool call, including those inside subgraphs."""

    def __init__(self, rec: Recorder, tag: str = ""):
        self.rec = rec
        self.tag = tag
        self._nodes: dict = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs) -> None:
        self._nodes[run_id] = (metadata or {}).get("langgraph_node", self.tag)

    def on_llm_end(self, response, *, run_id=None, **kwargs) -> None:
        self.rec.llm_calls += 1
        try:
            msg = response.generations[0][0].message
        except Exception:
            return
        um = getattr(msg, "usage_metadata", None) or {}
        self.rec.add_usage(um.get("input_tokens", 0), um.get("output_tokens", 0))
        node = self._nodes.pop(run_id, self.tag)
        text = msg.content if isinstance(msg.content, str) else " ".join(
            c.get("text", "") for c in msg.content if isinstance(c, dict))
        calls = [f"{tc['name']}({_short(tc.get('args', {}), 200)})" for tc in getattr(msg, "tool_calls", []) or []]
        detail = (text or "").strip()
        if calls:
            detail = (detail + "\n" if detail else "") + "-> wants to call: " + ", ".join(calls)
        self.rec.emit("llm", f"LLM{(' [' + node + ']') if node else ''}", detail or "(structured output)",
                      tokens=um)

    def on_tool_start(self, serialized, input_str, **kwargs) -> None:
        self.rec.tool_calls += 1
        self.rec.emit("tool_call", f"tool: {serialized.get('name', '?')}", input_str)

    def on_tool_end(self, output, **kwargs) -> None:
        content = getattr(output, "content", output)
        self.rec.emit("tool_result", f"result: {kwargs.get('name') or ''}".strip(), content)

    def on_tool_error(self, error, **kwargs) -> None:
        self.rec.emit("error", "tool error", str(error))
