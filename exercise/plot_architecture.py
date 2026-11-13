"""Architecture diagram of the hydrology MCP exercise."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from mcp_server.hydro_server import OUT_DIR

fig, ax = plt.subplots(figsize=(15, 9))
ax.set_xlim(0, 15)
ax.set_ylim(0, 9)
ax.axis("off")


def layer(x, y, w, h, title, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=color, ec="grey", lw=1.5, alpha=0.35))
    ax.text(x + 0.15, y + h - 0.12, title, va="top", weight="bold", fontsize=11)


def box(x, y, w, h, text, color="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", fc=color, ec="k"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)


def arrow(x1, y1, x2, y2, label="", both=False, dx=0.0, dy=0.15):
    style = "<|-|>" if both else "-|>"
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=16, lw=1.6))
    if label:
        ax.text((x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy, label, ha="center", fontsize=8.5)


# Layer 1: clients
layer(0.2, 6.9, 14.6, 1.9, "Client (pick one)", "#fff2cc")
box(0.6, 7.1, 3.6, 1.1, "VS Code Copilot\n.vscode/mcp.json")
box(5.7, 7.1, 3.6, 1.1, "Claude Code\n.mcp.json")
box(10.8, 7.1, 3.6, 1.1, "Copilot CLI\n/mcp add hydro-analysis")

# Layer 2: LLM
layer(0.2, 5.2, 14.6, 1.4, "Model", "#dae8fc")
box(4.5, 5.4, 6, 0.8, "LLM: chooses tools, writes arguments, explains results")

# Layer 3: MCP server
layer(0.2, 1.7, 9.6, 3.2, "MCP server  (uv run python -m mcp_server.hydro_server, stdio)", "#d5e8d4")
box(0.6, 3.5, 2.8, 0.9, "compare_precip", "#eaf5e8")
box(3.6, 3.5, 2.8, 0.9, "check_discharge", "#eaf5e8")
box(6.6, 3.5, 2.8, 0.9, "skill_scores", "#eaf5e8")
box(0.6, 2.3, 2.8, 0.9, "plot_hydrograph", "#eaf5e8")
box(3.6, 2.3, 2.8, 0.9, "water_balance", "#eaf5e8")
box(6.6, 2.3, 2.8, 0.9, "pandas / xarray /\nmatplotlib", "#f4f4f4")

# Layer 4: data
layer(10.2, 1.7, 4.6, 3.2, "Data (read only)", "#f8cecc")
box(10.5, 3.6, 4.0, 0.8, "hydromt_data/\nERA5, CHIRPS, E-OBS (.nc)")
box(10.5, 2.5, 4.0, 0.8, "exercise/data/\nwflow output + synthetic obs (.csv)")
ax.text(12.5, 2.0, "plots -> exercise/output/", ha="center", fontsize=9)

# Layer 5: offline pipeline
layer(0.2, 0.1, 14.6, 1.3, "Prepared beforehand (not run by the exercise)", "#e1d5e7")
box(0.6, 0.3, 3.8, 0.7, "hydromt build / update")
box(5.6, 0.3, 3.8, 0.7, "Wflow.jl run (Julia)")
box(10.6, 0.3, 3.8, 0.7, "make_synthetic_obs.py")
arrow(4.4, 0.65, 5.6, 0.65)
arrow(9.4, 0.65, 10.6, 0.65)
arrow(12.5, 1.0, 12.5, 2.5, "")
ax.text(12.65, 1.55, "creates exercise data", fontsize=8.5)

# Vertical flow
arrow(7.5, 7.1, 7.5, 6.2, "", both=True)
arrow(7.5, 5.2, 5.0, 4.95, "", both=True)
ax.text(8.2, 5.05, "MCP (JSON over stdio)", fontsize=8.5)
arrow(9.85, 3.5, 10.45, 3.5, "", both=True)

ax.set_title("Architecture of the hydrology MCP exercise", fontsize=14, weight="bold")
fig.tight_layout()
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "architecture.png", dpi=120)
print(OUT_DIR / "architecture.png")
