"""Schematic: what happens when you ask a question through the MCP server."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from mcp_server.hydro_server import OUT_DIR

fig, ax = plt.subplots(figsize=(14, 6.5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 6.5)
ax.axis("off")


def box(x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=color, ec="k"))
    ax.text(x + w / 2, y + h - 0.3, title, ha="center", va="top", weight="bold", fontsize=11)
    ax.text(x + w / 2, y + h - 0.85, body, ha="center", va="top", fontsize=9)


def arrow(x1, y1, x2, y2, label, color="k", dy=0.18):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18, color=color, lw=1.8))
    ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, label, ha="center", fontsize=8.5, color=color)


box(0.2, 2.2, 2.6, 3.4, "1. You", 'Ask in plain language:\n"Compare ERA5, CHIRPS\nand E-OBS precipitation\nfor 1990-2020."', "#fff2cc")
box(4.0, 2.2, 3.0, 3.4, "2. Agent (LLM)", "Reads the tool list\nPicks a tool\nFills in arguments:\ncompare_precip(\n  start, end)", "#dae8fc")
box(8.2, 2.2, 2.8, 3.4, "3. MCP server", "hydro_server.py\n(stdio, started with uv)\n\ncompare_precip\ncheck_discharge\nskill_scores\nplot_hydrograph\nwater_balance", "#d5e8d4")
box(11.6, 2.2, 2.2, 3.4, "4. Data", "hydromt_data/\n  era5, chirps, eobs\nexercise/data/\n  wflow output,\n  synthetic obs", "#f8cecc")

arrow(2.85, 4.4, 3.95, 4.4, "question")
arrow(7.05, 4.4, 8.15, 4.4, "tool call (JSON)")
arrow(11.05, 4.4, 11.55, 4.4, "read", dy=0.2)
arrow(11.55, 3.2, 11.05, 3.2, "arrays", dy=-0.35)
arrow(8.15, 3.2, 7.05, 3.2, "numbers (dict)", dy=-0.35)
arrow(3.95, 2.7, 2.85, 2.7, "answer", dy=-0.35)

ax.text(7, 1.3, "The server does the computing. The agent only chooses the tool and explains the result.",
        ha="center", fontsize=11, style="italic")
ax.text(7, 0.6, "Always read the tool output yourself: the agent can misread or over-interpret it.",
        ha="center", fontsize=10, color="firebrick")
ax.set_title("What happens when you ask a question through MCP", fontsize=14, weight="bold")
fig.tight_layout()
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "mcp_schematic.png", dpi=130)
print(OUT_DIR / "mcp_schematic.png")
