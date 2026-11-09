"""Uniform trace and metrics: LLM calls, tool calls, tokens, time breakdown, cost estimate.

Every architecture (and the custom builder) reports through the same ``Recorder``, so the
Streamlit app, the notebook and the CLI can show and compare them side by side, and the
scoreboard can rank architectures on cost, time and quality.

Time is split into:
* ``llm_s``    - waiting for the language model (summed over calls; parallel calls can overlap)
* ``model_s``  - running Wflow / HydroMT (the "real work")
* ``other_s``  - everything else (Python tools, plotting, orchestration overhead)
"""

from __future__ import annotations

import json
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml
from langchain_core.callbacks import BaseCallbackHandler

# The recorder of the session that is currently running (set by GraphSession / the harness) so that
# deep code such as the Wflow runner can add its time without being passed the recorder explicitly.
CURRENT_RECORDER: ContextVar["Recorder | None"] = ContextVar("CURRENT_RECORDER", default=None)

PRICING_FILE = Path(__file__).resolve().parents[1] / "pricing.yml"


def load_prices() -> dict:
    try:
        return yaml.safe_load(PRICING_FILE.read_text()) or {}
    except Exception:
        return {}


@dataclass
class Event:
    kind: str          # node | llm | tool_call | tool_result | model | human | info | error | harness | final
    title: str
    detail: str = ""
    t: float = field(default_factory=time.time)
    data: dict = field(default_factory=dict)


def _short(x: Any, n: int = 600) -> str:
    s = x if isinstance(x, str) else json.dumps(x, default=str)
    return s if len(s) <= n else s[:n] + " ..."


class Recorder:
    def __init__(self, on_event: Callable[[Event], None] | None = None, model: str = ""):
        self.events: list[Event] = []
        self.on_event = on_event
        self.model = model  # e.g. "anthropic:claude-sonnet-5-5"; used for pricing
        self.t0 = time.time()
        self.t_end: float | None = None
        self.llm_calls = 0
        self.tool_calls = 0
        self.input_tokens = 0       # uncached input tokens
        self.output_tokens = 0
        self.cache_read_tokens = 0
        self.cache_write_tokens = 0
        self.llm_s = 0.0
        self.tool_s = 0.0
        self.model_s = 0.0          # Wflow + HydroMT wall time
        self.model_runs = 0
        self.startup_s = 0.0        # one-off Julia/Wflow start-up, excluded from wall time
        self.reported_cost_usd = 0.0  # cost reported by the Claude Agent SDK (harness), if any
        self.by_node: dict[str, dict] = {}
        self.by_model: dict[str, dict] = {}  # tokens per model (A5 mixes e.g. GPT router + Claude harness)

    # -- events ---------------------------------------------------------------------------------
    def emit(self, kind: str, title: str, detail: Any = "", **data) -> Event:
        ev = Event(kind, title, _short(detail, 4000) if detail else "", data=data)
        self.events.append(ev)
        if self.on_event:
            try:
                self.on_event(ev)
            except Exception:
                pass
        return ev

    def model_sink(self):
        """Callable for ``wflow_tools.LOG_SINK``: model console output becomes 'model' events."""
        return lambda source, line: self.emit("model", source, line)

    # -- metrics --------------------------------------------------------------------------------
    def add_llm(self, node: str, input_tokens=0, output_tokens=0, cache_read=0, cache_write=0, seconds=0.0,
                calls: int = 1, model: str | None = None) -> None:
        m = self.by_model.setdefault(model or self.model or "?", {"input": 0, "output": 0, "cache_read": 0,
                                                                  "cache_write": 0})
        m["input"] += int(input_tokens or 0)
        m["output"] += int(output_tokens or 0)
        m["cache_read"] += int(cache_read or 0)
        m["cache_write"] += int(cache_write or 0)
        self.llm_calls += calls
        self.input_tokens += int(input_tokens or 0)
        self.output_tokens += int(output_tokens or 0)
        self.cache_read_tokens += int(cache_read or 0)
        self.cache_write_tokens += int(cache_write or 0)
        self.llm_s += float(seconds or 0)
        n = self.by_node.setdefault(node or "-", {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0,
                                                  "cache_read_tokens": 0, "llm_s": 0.0})
        n["llm_calls"] += calls
        n["input_tokens"] += int(input_tokens or 0)
        n["output_tokens"] += int(output_tokens or 0)
        n["cache_read_tokens"] += int(cache_read or 0)
        n["llm_s"] = round(n["llm_s"] + float(seconds or 0), 2)

    def add_model_time(self, seconds: float, runs: int = 1) -> None:
        self.model_s += seconds
        self.model_runs += runs

    def add_startup(self, seconds: float) -> None:
        if seconds:
            self.startup_s += seconds
            self.t0 += seconds  # do not charge the first architecture for starting Julia

    def finish(self) -> None:
        self.t_end = time.time()

    def cost_estimate(self) -> tuple[float | None, str]:
        """(USD, source). Uses pricing.yml; falls back to the SDK-reported cost for the harness."""
        prices = load_prices()
        usd, missing = 0.0, []
        for model, t in self.by_model.items():
            if not any(t.values()):
                continue
            p = prices.get(model.split(":", 1)[-1])
            if not p:
                missing.append(model)
                continue
            usd += sum(t[k] * p.get(k, 0) for k in ("input", "output", "cache_read", "cache_write")) / 1e6
        if missing and self.reported_cost_usd and len(missing) == len([m for m in self.by_model if any(
                self.by_model[m].values())]):
            return round(self.reported_cost_usd, 4), "claude-agent-sdk"
        if missing:
            return (round(usd, 4) if usd else None), f"no price in pricing.yml for {missing}"
        return round(usd, 4), "pricing.yml"

    def summary(self) -> dict:
        wall = (self.t_end or time.time()) - self.t0
        cost, src = self.cost_estimate()
        return {
            "model": self.model,
            "llm_calls": self.llm_calls, "tool_calls": self.tool_calls, "wflow_runs": self.model_runs,
            "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens, "cache_write_tokens": self.cache_write_tokens,
            "total_tokens": self.input_tokens + self.output_tokens + self.cache_read_tokens + self.cache_write_tokens,
            "wall_time_s": round(wall, 1), "llm_s": round(self.llm_s, 1), "model_s": round(self.model_s, 1),
            "other_s": round(max(wall - self.llm_s - self.model_s, 0.0), 1),
            "julia_startup_s": round(self.startup_s, 1),
            "cost_usd": cost, "cost_source": src,
            "by_node": self.by_node,
        }

    def callback(self, tag: str = "") -> "TraceCallback":
        return TraceCallback(self, tag)


class TraceCallback(BaseCallbackHandler):
    """LangChain callback: picks up every model and tool call, including those inside subgraphs."""

    def __init__(self, rec: Recorder, tag: str = ""):
        self.rec = rec
        self.tag = tag
        self._llm: dict = {}   # run_id -> (node, t_start)
        self._tool: dict = {}  # run_id -> t_start

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs) -> None:
        self._llm[run_id] = ((metadata or {}).get("langgraph_node", self.tag), time.time())

    def on_llm_end(self, response, *, run_id=None, **kwargs) -> None:
        node, t0 = self._llm.pop(run_id, (self.tag, time.time()))
        try:
            msg = response.generations[0][0].message
        except Exception:
            self.rec.add_llm(node, seconds=time.time() - t0)
            return
        um = getattr(msg, "usage_metadata", None) or {}
        det = um.get("input_token_details") or {}
        cache_read, cache_write = det.get("cache_read", 0) or 0, det.get("cache_creation", 0) or 0
        # LangChain's input_tokens includes cached tokens; store them separately for pricing
        uncached = max((um.get("input_tokens", 0) or 0) - cache_read - cache_write, 0)
        self.rec.add_llm(node, uncached, um.get("output_tokens", 0), cache_read, cache_write, time.time() - t0)
        text = msg.content if isinstance(msg.content, str) else " ".join(
            c.get("text", "") for c in msg.content if isinstance(c, dict))
        calls = [f"{tc['name']}({_short(tc.get('args', {}), 200)})" for tc in getattr(msg, "tool_calls", []) or []]
        detail = (text or "").strip()
        if calls:
            detail = (detail + "\n" if detail else "") + "-> wants to call: " + ", ".join(calls)
        self.rec.emit("llm", f"LLM{(' [' + node + ']') if node else ''}", detail or "(structured output)",
                      tokens=um, seconds=round(time.time() - t0, 2))

    def on_tool_start(self, serialized, input_str, *, run_id=None, **kwargs) -> None:
        self.rec.tool_calls += 1
        self._tool[run_id] = time.time()
        self.rec.emit("tool_call", f"tool: {serialized.get('name', '?')}", input_str)

    def on_tool_end(self, output, *, run_id=None, **kwargs) -> None:
        self.rec.tool_s += time.time() - self._tool.pop(run_id, time.time())
        content = getattr(output, "content", output)
        self.rec.emit("tool_result", f"result: {kwargs.get('name') or ''}".strip(), content)

    def on_tool_error(self, error, *, run_id=None, **kwargs) -> None:
        self._tool.pop(run_id, None)
        self.rec.emit("error", "tool error", str(error))
