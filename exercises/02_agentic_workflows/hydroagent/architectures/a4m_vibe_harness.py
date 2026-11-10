"""A4m - Agent harness + skills with Mistral Vibe: the Mistral counterpart of A4, for a like-for-like comparison.

    User -> Mistral Vibe loop [skills (SKILL.md), read/write/edit/shell, MCP: hydro_run_wflow_scenario] -> answer

Same workspace, skills, Wflow tools and guard-rails as A4; different harness (Mistral Vibe) and model (your
Mistral deployment on Azure Foundry). Needs ``uv tool install mistral-vibe`` once.
"""

from __future__ import annotations

from typing import Callable

from ..harness.vibe_runner import run_vibe_harness
from ..runtime import Outcome
from ..tracing import Event, Recorder
from .a4_claude_harness import INFO as A4_INFO
from .common import ArchInfo

INFO = ArchInfo(
    key="a4m",
    name="A4m - Mistral Vibe harness + skills (code agent)",
    pattern="Agent harness (Mistral Vibe) + Skills: the Mistral counterpart of A4",
    summary="Mistral's coding agent in the same sandboxed workspace as A4, with the same skills and Wflow MCP "
            "tools. Compare it with A4 on the same question: quality, tokens, time, cost.",
    use_when="Comparing harnesses/models on open-ended tasks; when you want a non-Claude code agent.",
    trade_off="Same pattern as A4, different model and harness; needs the Vibe CLI installed.",
    example_query=A4_INFO.example_query,
    claude_only=False,
)


class VibeHarnessSession:
    """Same interface as GraphSession (start/resume) so the UI can treat A4m like the others."""

    def __init__(self, name: str = "a4m", on_event: Callable[[Event], None] | None = None,
                 skills: list[str] | None = None, max_turns: int = 40, model: str | None = None,
                 max_price: float | None = None):
        self.name = name
        self.rec = Recorder(on_event)
        self.skills = skills
        self.max_turns = max_turns
        self.model = model
        self.max_price = max_price
        self.result = None
        self.log = True  # write finished runs to the scoreboard

    def start(self, query: str, **_) -> Outcome:
        self.rec.emit("info", f"{self.name} started", query)
        r = run_vibe_harness(query, skills=self.skills, max_turns=self.max_turns, model=self.model,
                             max_price=self.max_price, recorder=self.rec)
        self.result = r
        out = Outcome("done" if r.status == "done" else "error", answer=r.answer, run_ids=r.run_ids,
                      plots=r.plots, usage=r.usage, error=r.error,
                      state={"workspace": r.workspace, "skills_loaded": r.skills_loaded})
        if self.log:
            from .. import scoreboard

            out.entry_id = scoreboard.record(self.name, query, out)
        return out

    def resume(self, answer) -> Outcome:  # the harness does not pause for humans in this exercise
        raise RuntimeError("A4m does not use interrupts")


def make_session(provider: str | None = None, on_event=None, **opts) -> VibeHarnessSession:
    return VibeHarnessSession(on_event=on_event, **opts)
