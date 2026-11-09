"""A4 - Agent harness + skills (Claude Agent SDK): the agent writes and executes its own code.

    User -> Claude harness loop [skills (SKILL.md), Read/Write/Bash, MCP: run_wflow_scenario] -> answer

Use for open-ended work that fixed tools cannot express, e.g. a parameter change in one sub-catchment
only, or a custom analysis of output.nc. Claude-only (harness), unlike A0-A3 which are model-agnostic.
"""

from __future__ import annotations

from typing import Callable

from ..harness.claude_runner import run_harness
from ..runtime import Outcome
from ..tracing import Event, Recorder
from .common import ArchInfo

INFO = ArchInfo(
    key="a4",
    name="A4 - Claude Agent SDK harness + skills (code agent)",
    pattern="Agent harness (Claude Agent SDK) + Skills (progressive disclosure)",
    summary="A Claude Code agent in a sandboxed workspace. Loads skills (wflow-toml, netcdf-analysis), writes "
            "Python to edit/analyse NetCDF files, and runs Wflow through an MCP tool.",
    use_when="Open-ended tasks that need code: spatially varying edits, custom analyses, new plots.",
    trade_off="Most capable but least predictable; more tokens; needs sandboxing and review. Claude-only.",
    example_query="Halve the soil permeability (soil_ksat_vertical) only in the Fresse-sur-Moselle sub-catchment "
                  "(GRDC 9116039), run the model, and compare the annual-maximum discharge at Epinal and at "
                  "Fresse-sur-Moselle with the baseline. Make one figure.",
    claude_only=True,
)


class HarnessSession:
    """Same interface as GraphSession (start/resume) so the UI can treat A4 like the others."""

    def __init__(self, name: str = "a4", on_event: Callable[[Event], None] | None = None,
                 skills: list[str] | None = None, max_turns: int = 40, model: str | None = None):
        self.name = name
        self.rec = Recorder(on_event)
        self.skills = skills
        self.max_turns = max_turns
        self.model = model
        self.result = None

    def start(self, query: str, **_) -> Outcome:
        self.rec.emit("info", f"{self.name} started", query)
        r = run_harness(query, skills=self.skills, max_turns=self.max_turns, model=self.model, recorder=self.rec)
        self.result = r
        return Outcome("done" if r.status == "done" else "error", answer=r.answer, run_ids=r.run_ids,
                       plots=r.plots, usage=r.usage, error=r.error,
                       state={"workspace": r.workspace, "skills_loaded": r.skills_loaded})

    def resume(self, answer) -> Outcome:  # the harness does not pause for humans in this exercise
        raise RuntimeError("A4 does not use interrupts")


def make_session(provider: str | None = None, on_event=None, **opts) -> HarnessSession:
    return HarnessSession(on_event=on_event, **opts)
