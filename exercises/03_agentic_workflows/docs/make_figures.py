"""Draw one diagram per agent architecture (A0-A5) for the notebook and README.

    uv run python exercises/03_agentic_workflows/docs/make_figures.py

Writes docs/figures/a0.png ... a5.png. Colours follow the builder legend (hydroagent/builder/nodes.py):
grey = deterministic code, blue = plain LLM call, green = agent, purple = Claude harness, orange = human,
yellow = router, lilac = critic. Re-run after changing an architecture.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"

COLOURS = {
    "io": ("#ffffff", "#555555"),
    "code": ("#f5f5f5", "#666666"),
    "llm": ("#dae8fc", "#6c8ebf"),
    "agent": ("#d5e8d4", "#82b366"),
    "harness": ("#d0cee2", "#56517e"),
    "human": ("#ffe6cc", "#d79b00"),
    "router": ("#fff2cc", "#d6b656"),
    "critic": ("#e1d5e7", "#9673a6"),
    "wflow": ("#cfe2f3", "#3d85c6"),
}
LEGEND = {"code": "code (no LLM)", "llm": "plain LLM call", "agent": "agent (LLM + tools, loops)",
          "harness": "agent harness", "human": "human", "router": "router (LLM)", "critic": "critic (LLM judge)",
          "wflow": "Wflow model run"}

W, H = 1.9, 0.78  # default box size (data units)


class Diagram:
    def __init__(self, title: str, size: tuple[float, float], xlim: tuple[float, float], ylim: tuple[float, float]):
        self.fig, self.ax = plt.subplots(figsize=size, dpi=150)
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.ax.annotate(title, (0, 1), xycoords="axes fraction", xytext=(0, 26), textcoords="offset points",
                         fontsize=12, fontweight="bold", ha="left", va="bottom")
        self.boxes: dict[str, FancyBboxPatch] = {}
        self.kinds: set[str] = set()

    def box(self, key: str, x: float, y: float, text: str, kind: str, w: float = W, h: float = H,
            fontsize: float = 8.5, bold_first: bool = True) -> None:
        face, edge = COLOURS[kind]
        style = "round,pad=0.02,rounding_size=0.35" if kind == "io" else "round,pad=0.02,rounding_size=0.12"
        p = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=style, fc=face, ec=edge, lw=1.3,
                           ls="--" if kind == "human" else "-", zorder=2)
        self.ax.add_patch(p)
        self.boxes[key] = p
        if kind != "io":
            self.kinds.add(kind)
        lines = text.split("\n")
        if bold_first and len(lines) > 1:
            self.ax.text(x, y + h * 0.17, lines[0], ha="center", va="center", fontsize=fontsize,
                         fontweight="bold", zorder=3)
            self.ax.text(x, y - h * 0.16, "\n".join(lines[1:]), ha="center", va="center",
                         fontsize=fontsize - 1.5, color="#333333", zorder=3)
        else:
            self.ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                         fontweight="bold" if bold_first else "normal", zorder=3)

    def frame(self, x: float, y: float, w: float, h: float, label: str, kind: str) -> None:
        """A large container (e.g. the harness) drawn behind its inner boxes."""
        face, edge = COLOURS[kind]
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.2", fc=face,
                                         ec=edge, lw=1.6, alpha=0.55, zorder=1))
        self.ax.text(x + 0.15, y + h - 0.22, label, ha="left", va="center", fontsize=9, fontweight="bold",
                     color="#222222", zorder=3)
        self.kinds.add(kind)

    def arrow(self, a: str, b: str, label: str = "", rad: float = 0.0, colour: str = "#444444",
              ls: str = "-", label_pos: float = 0.5, label_offset: tuple[float, float] = (0, 0.14)) -> None:
        pa, pb = self.boxes[a], self.boxes[b]
        ca = (pa.get_x() + pa.get_width() / 2, pa.get_y() + pa.get_height() / 2)
        cb = (pb.get_x() + pb.get_width() / 2, pb.get_y() + pb.get_height() / 2)
        arr = FancyArrowPatch(ca, cb, patchA=pa, patchB=pb, arrowstyle="-|>", mutation_scale=13, lw=1.3,
                              color=colour, ls=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2,
                              zorder=1.5)
        self.ax.add_patch(arr)
        if label:
            # label near the (curved) path: midpoint pushed sideways by the curvature
            mx, my = ca[0] + (cb[0] - ca[0]) * label_pos, ca[1] + (cb[1] - ca[1]) * label_pos
            dx, dy = cb[0] - ca[0], cb[1] - ca[1]
            mx, my = mx + rad * dy * 0.5 + label_offset[0], my - rad * dx * 0.5 + label_offset[1]
            self.ax.text(mx, my, label, ha="center", va="center", fontsize=7.5, style="italic", color=colour,
                         bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85), zorder=4)

    def note(self, _x: float, _y: float, text: str, fontsize: float = 8.5) -> None:
        """Subtitle under the title, above the drawing (position arguments kept for readability only)."""
        self.ax.annotate(text, (0, 1), xycoords="axes fraction", xytext=(0, 8), textcoords="offset points",
                         fontsize=fontsize, color="#333333", ha="left", va="bottom")

    def save(self, name: str) -> Path:
        handles = [Patch(fc=COLOURS[k][0], ec=COLOURS[k][1], label=LEGEND[k], ls="--" if k == "human" else "-")
                   for k in LEGEND if k in self.kinds]
        self.ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(handles),
                       fontsize=7.5, frameon=False, handlelength=1.4, columnspacing=1.2)
        OUT.mkdir(parents=True, exist_ok=True)
        p = OUT / f"{name}.png"
        self.fig.savefig(p, bbox_inches="tight", facecolor="white")
        plt.close(self.fig)
        return p


def a0() -> Path:
    d = Diagram("A0 · Plain LLM calls in a fixed workflow (no agent)", (11, 2.4), (-0.3, 12.3), (-1.05, 0.8))
    d.box("q", 0.8, 0.3, "Question", "io", w=1.4)
    d.box("interpret", 3.2, 0.3, "interpret\nquestion → multipliers", "llm", w=2.3)
    d.box("simulate", 5.9, 0.3, "simulate\nrun scenario + compare", "code", w=2.3)
    d.box("wflow", 5.9, -0.75, "Wflow", "wflow", w=1.2, h=0.42, fontsize=7.5)
    d.box("report", 8.6, 0.3, "report\nexplain the numbers", "llm", w=2.3)
    d.box("a", 11.0, 0.3, "Answer", "io", w=1.4)
    for a, b in [("q", "interpret"), ("interpret", "simulate"), ("simulate", "report"), ("report", "a")]:
        d.arrow(a, b)
    d.arrow("simulate", "wflow")
    d.note(-0.2, 1.05, "Your code fixes the order. Always exactly 2 LLM calls; the LLM never decides what happens next.")
    return d.save("a0")


def a1() -> Path:
    d = Diagram("A1 · Single agent (ReAct): one LLM in a loop with tools", (11, 3.0), (-0.3, 12.3), (-1.35, 1.35))
    d.box("q", 0.8, 0.5, "Question", "io", w=1.4)
    d.box("agent", 4.0, 0.5, "ReAct agent\nLLM decides the next step", "agent", w=2.6, h=0.9)
    d.box("tools", 8.0, 0.5, "Tools\nparameters · run scenario ·\ncompare · plot", "code", w=2.8, h=0.9)
    d.box("wflow", 8.0, -0.75, "Wflow", "wflow", w=1.2, h=0.42, fontsize=7.5)
    d.box("a", 4.0, -0.85, "Answer", "io", w=1.4)
    d.arrow("q", "agent")
    d.arrow("agent", "tools", "tool call", rad=-0.35)
    d.arrow("tools", "agent", "result", rad=-0.35)
    d.arrow("tools", "wflow")
    d.arrow("agent", "a", "done")
    d.note(-0.2, 1.45, "Repeats call → observe → decide until it answers. Flexible, but the order, the number of calls "
                       "and the cost are not fixed.")
    return d.save("a1")


def a2() -> Path:
    d = Diagram("A2 · Generator-critic loop (calibration against observed discharge)", (11, 3.2), (-0.3, 12.3),
                (-1.4, 1.65))
    d.box("q", 0.75, 0.3, "Question", "io", w=1.3)
    d.box("gen", 3.0, 0.3, "generator\npropose multipliers", "llm", w=2.2)
    d.box("sim", 5.6, 0.3, "simulate + score\nKGE vs observed", "code", w=2.2)
    d.box("wflow", 5.6, -0.9, "Wflow", "wflow", w=1.2, h=0.42, fontsize=7.5)
    d.box("critic", 8.1, 0.3, "critic\nLLM or rule", "critic", w=1.9)
    d.box("report", 11.1, 0.3, "report\nsummarise", "llm", w=1.8)
    d.box("a", 11.1, -0.9, "Answer", "io", w=1.3)
    d.arrow("q", "gen")
    d.arrow("gen", "sim")
    d.arrow("sim", "wflow")
    d.arrow("sim", "critic")
    d.arrow("critic", "report", "approved")
    d.arrow("critic", "gen", "revise + feedback", rad=0.45, colour="#9673a6")
    d.arrow("report", "a")
    d.note(-0.2, 1.65, "Code fixes the loop, the exit rule (KGE target or max iterations) and the scoring; "
                       "the LLM only proposes values and critiques.")
    return d.save("a2")


def a3() -> Path:
    d = Diagram("A3 · Director: subagents, allow-list and human-in-the-loop", (11, 5.0), (-0.3, 12.3), (-3.25, 2.3))
    # row 1: intake and approval
    d.box("q", 0.8, 0.8, "Question", "io", w=1.4)
    d.box("classify", 3.3, 0.8, "classify\nscenario on allow-list?", "router", w=2.3)
    d.box("clarify", 3.3, 1.85, "clarify\nask for missing info", "human", w=2.3, h=0.7)
    d.box("plan", 6.2, 0.8, "plan\nscenario template", "code", w=2.0)
    d.box("approve", 9.1, 0.8, "approve\nplan before compute", "human", w=2.3)
    # row 2: execution, left to right
    d.box("blocked", 0.8, -0.9, "blocked\nout of scope", "code", w=1.6, h=0.7)
    d.box("modeller", 4.4, -0.9, "modeller\nsubagent, retries", "agent", w=2.2)
    d.box("analyst", 7.2, -0.9, "analyst\nsubagent: stats + plot", "agent", w=2.3)
    d.box("final", 10.4, -0.9, "finalize\nreport + audit trail", "llm", w=2.2)
    d.box("wflow", 4.4, -2.1, "Wflow", "wflow", w=1.2, h=0.42, fontsize=7.5)
    d.box("a", 10.4, -2.15, "Answer", "io", w=1.4)
    d.arrow("q", "classify")
    d.arrow("classify", "clarify", "missing info", label_offset=(0.7, 0))
    d.arrow("clarify", "classify", rad=0.5)
    d.arrow("classify", "blocked", "out of scope", label_offset=(-0.55, 0.05))
    d.arrow("classify", "plan", "ok")
    d.arrow("plan", "approve")
    d.arrow("approve", "modeller", "yes", label_pos=0.35, label_offset=(0, 0.12))
    d.arrow("approve", "final", "no", colour="#d79b00", label_offset=(0.22, 0))
    d.arrow("modeller", "wflow")
    d.arrow("modeller", "analyst")
    d.arrow("analyst", "final")
    d.arrow("modeller", "final", "escalate", rad=0.32, colour="#b85450", label_offset=(0, -0.05))
    d.arrow("final", "a")
    d.arrow("blocked", "a", rad=0.3, ls=":", colour="#888888")
    d.note(-0.2, 2.05, "Guarantees: only allowed scenarios run · a human approves before compute · fixed step order · "
                       "errors retried, then escalated.")
    return d.save("a3")


def a4(mistral: bool = False) -> Path:
    if mistral:
        title = "A4m · Mistral Vibe harness + skills (code agent)"
        frame = "Mistral Vibe harness (agent loop, context management, permissions)"
        agent = "Mistral\nplans, writes + runs code"
        tools = "Built-in tools\nread_file · write_file · edit · grep · shell"
        mcp = "MCP server (stdio): run_wflow_scenario\nmultipliers · edited staticmaps · edited forcing"
        guard = "permissions\nwrites only in workspace"
        note = ("Same workspace, skills and Wflow tools as A4; different harness and model. "
                "Ask both the same question and compare.")
    else:
        title = "A4 · Claude Agent SDK harness + skills (code agent)"
        frame = "Claude harness (agent loop, context management, permissions)"
        agent = "Claude\nplans, writes + runs code"
        tools = "Built-in tools\nRead · Write · Edit · Bash (Python)"
        mcp = "MCP tool: run_wflow_scenario\nmultipliers · edited staticmaps · edited forcing"
        guard = "guard hook\nwrites only in workspace"
        note = "Open-ended: the agent writes its own analysis code. Most flexible, least predictable in time and cost."
    d = Diagram(title, (11, 3.8), (-0.3, 12.3), (-2.1, 1.4))
    d.box("q", 0.8, 0.0, "Question", "io", w=1.4)
    d.frame(2.2, -2.0, 7.6, 3.3, frame, "harness")
    d.box("loop", 4.0, 0.0, agent, "agent", w=2.6, h=0.9)
    d.box("skills", 7.8, 0.55, "Skills (loaded on demand)\nwflow-toml · netcdf-analysis · hydrology-env",
          "code", w=3.4, h=0.75)
    d.box("builtin", 7.8, -0.4, tools, "code", w=3.4, h=0.75)
    d.box("mcp", 7.8, -1.35, mcp, "code", w=3.4, h=0.75)
    d.box("guard", 4.0, -1.35, guard, "code", w=2.6, h=0.75)
    d.box("ws", 11.15, -0.4, "Workspace\nscripts, NetCDF,\nfigures", "code", w=1.9, h=1.0, fontsize=8)
    d.box("wflow", 11.15, -1.35, "Wflow", "wflow", w=1.2, h=0.42, fontsize=7.5)
    d.box("a", 0.8, -1.35, "Answer", "io", w=1.4)
    d.arrow("q", "loop")
    d.arrow("loop", "skills", rad=-0.15)
    d.arrow("loop", "builtin")
    d.arrow("loop", "mcp", rad=0.15)
    d.arrow("builtin", "ws")
    d.arrow("mcp", "wflow")
    d.arrow("guard", "loop", ls=":", colour="#888888")
    d.arrow("loop", "a", rad=0.0)
    d.note(-0.2, 1.88, note)
    return d.save("a4m" if mistral else "a4")


def a5() -> Path:
    d = Diagram("A5 · Composed: a router sends each request to the cheapest component", (11, 3.4), (-0.3, 12.6),
                (-1.7, 1.7))
    d.box("q", 0.8, 0.0, "Question", "io", w=1.4)
    d.box("route", 2.85, 0.0, "route\npath + confidence", "router", w=1.9)
    d.box("simple", 7.0, 1.15, "simple_answer\none LLM call, no tools", "llm", w=2.5)
    d.box("param", 7.0, 0.0, "parameter_scenario\nA1 ReAct agent", "agent", w=2.5)
    d.box("code", 7.0, -1.15, "open_ended_code\nA4 Claude harness", "harness", w=2.5)
    d.box("human", 10.0, 0.0, "human check\nonly if confidence < 0.7", "human", w=2.3)
    d.box("a", 11.95, 0.0, "Answer", "io", w=1.1)
    d.arrow("q", "route")
    d.arrow("route", "simple", "knowledge question", label_offset=(-0.35, 0.2))
    d.arrow("route", "param", "parameter change", label_offset=(0, 0.17))
    d.arrow("route", "code", "custom code / NetCDF edits", label_offset=(-0.45, -0.22))
    for b in ("simple", "param", "code"):
        d.arrow(b, "human")
    d.arrow("human", "a")
    d.note(-0.2, 1.85, "Framework for the structure, harness for the hard open-ended steps, plain LLM call for "
                       "anything simple.")
    return d.save("a5")


if __name__ == "__main__":
    for fn in (a0, a1, a2, a3, a4, lambda: a4(mistral=True), a5):
        print(fn())
