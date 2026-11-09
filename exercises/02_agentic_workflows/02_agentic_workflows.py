# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Exercise 2: Agentic architectures for HydroMT / Wflow
#
# You will run the same kind of hydrological question through agents of increasing complexity. For each one,
# ask: *what did the extra structure buy me, and what did it cost?*
#
# | # | Architecture | Pattern (Agentic Overview) |
# |---|---|---|
# | A0 | Plain LLM + fixed workflow | Deterministic workflow, no agent |
# | A1 | Single agent (ReAct) | Dynamic orchestration: single agent |
# | A2 | Generator-critic loop | Deterministic workflow: loop / generator-critic |
# | A3 | Director + subagents + human-in-the-loop | Subagents (supervisor) + HITL + custom logic |
# | A4 | Claude Agent SDK harness + skills | Agent harness + skills (code agent) |
# | A5 | Router -> {LLM, ReAct, harness} -> human check | Composing harness agents inside a framework |
#
# **Model:** wflow_sbm of the upper Moselle to Epinal (HydroMT build from `hydromt/`), forced with E-OBS for
# 2019-2020. Agents change parameters through **multipliers** (A0-A3, A5). The code agent (A4) can also write
# Python that edits the NetCDF files directly.
#
# Prerequisites:
# - the course environment, with `verify_env.py` all OK;
# - Julia `Wflow` 1.x;
# - API keys in a `.env` file at the repo root or in a parent folder. Never paste keys into this notebook.

# %% [markdown]
# ## Step 0: Setup
# Pick the LLM provider for the LangGraph architectures: `"anthropic"` (Claude) or `"azure"` (GPT). The harness
# (A4) always uses Claude. If your Anthropic API key has no credit, set `HARNESS_AUTH = "login"` to use your
# Claude Code login.

# %%
import os
import sys
from pathlib import Path

EXERCISE_DIR = Path.cwd() if (Path.cwd() / "hydroagent").exists() else Path.cwd() / "exercises" / "02_agentic_workflows"
sys.path.insert(0, str(EXERCISE_DIR))

PROVIDER = "anthropic"          # or "azure"
os.environ.setdefault("HARNESS_AUTH", "auto")   # "login" to use the Claude Code login for A4

from IPython.display import Image, Markdown, display

from hydroagent import config as C
from hydroagent import twin
from hydroagent import wflow_tools as W
from hydroagent.architectures import registry as R
from hydroagent.runtime import run_interactive

print("env file:", C.load_env())
print("settings:", C.get_settings(provider=PROVIDER))


def show(ev):
    """Print the agent's trace as it happens. Remove kinds you find noisy ("model" = HydroMT/Wflow console)."""
    if ev.kind in ("llm", "tool_call", "node", "human", "error", "harness", "model"):
        print(f"[{ev.kind}] {ev.title}: {ev.detail[:250]}")


# Stream HydroMT / Wflow console output also when you call the model functions directly (Step 1).
W.LOG_SINK.set(lambda source, line: print(f"  [{source}] {line}"))


def report(out):
    display(Markdown(out.answer or f"**{out.status}** {out.error}"))
    print("usage:", out.usage, "| runs:", out.run_ids)
    for p in out.plots:
        display(Image(p))

# %% [markdown]
# Build the model (once, about 2 min), start Wflow, and run the baseline. The first Julia start takes about
# 1-1.5 min. After that, a run takes about 15 s because the Julia process stays alive.

# %%
W.prepare_base_model()
baseline = W.ensure_baseline()
print(baseline.status, baseline.runtime_s, "s")
W.summarise_run("baseline", reference="")

# %% [markdown]
# ## Step 1: The tools, without any LLM
# Agents are only as good as their tools. These are the functions the agents call. Try them yourself first.

# %%
W.list_parameters()[:4]

# %%
run = W.run_scenario({"ksat_vertical": 0.7}, "ksat -30% (by hand)")
W.summarise_run(run.run_id)

# %%
display(Image(str(W.plot_runs(["baseline", run.run_id]))))

# %% [markdown]
# ## Step 2: A0, plain LLM + fixed workflow
# The LLM only translates the question into multipliers and writes the report. Code fixes the order of steps.

# %%
out = run_interactive(R.create_session("a0", provider=PROVIDER, on_event=show),
                      "What happens to peak and low flows at Epinal if soils become 30% less permeable?")
report(out)

# %% [markdown]
# ## Step 3: A1, single agent (ReAct)
# One LLM with tools in a loop. **Customise it:** change the system prompt or remove a tool set, and see how the
# behaviour changes. Try a question that needs *two* scenarios, for example comparing +20% and -20% soil thickness.

# %%
from hydroagent.architectures import a1_react
from hydroagent.architectures.common import make_session

MY_PROMPT = a1_react.SYSTEM_PROMPT          # edit me
MY_TOOLSETS = ("parameters", "simulation", "analysis")   # try removing "analysis"

session = make_session("a1", a1_react.build, provider=PROVIDER, on_event=show,
                       system_prompt=MY_PROMPT, toolsets=MY_TOOLSETS)
out = run_interactive(session, "Compare +20% and -20% soil thickness: which affects low flows at Epinal more?")
report(out)

# %% [markdown]
# ## Step 4: A2, generator-critic loop (calibration)
# There are no observed discharge series in this repo, so we use a **twin experiment**. A hidden "truth" run
# made with secret multipliers plays the role of observations, and the loop tries to recover it.
# **Knobs:** `max_iter`, `target_kge`, `critic="llm"` vs `"rule"`. Is the LLM critic worth its tokens?

# %%
twin.ensure_truth()
out = run_interactive(R.create_session("a2", provider=PROVIDER, on_event=show, max_iter=4, critic="llm"),
                      R.INFOS["a2"].example_query)
report(out)

# %%
print("the truth was:", twin.reveal_truth())   # debrief: equifinality?

# %% [markdown]
# ## Step 5: A3, Director with subagents and human-in-the-loop
# This follows the Director pattern from the Agent4Hydro project:
# - an allow-list of scenario types;
# - clarification questions when inputs are missing;
# - plan approval before any compute is spent;
# - Modeller and Analyst subagents, with retries and escalation.
#
# The agent will ask you questions in an input box. Try an out-of-scope request too.

# %%
out = run_interactive(R.create_session("a3", provider=PROVIDER, on_event=show),
                      "What if the soils in the Moselle become less permeable?")
report(out)

# %% [markdown]
# ## Step 6: A4, Claude Agent SDK harness + skills (code agent)
# The harness writes and runs its own Python on `staticmaps.nc` / `output.nc`. It works in a sandboxed
# workspace and runs Wflow through an MCP tool.
#
# It loads **skills** on demand:
# - `wflow-toml`
# - `netcdf-analysis`
# - `hydrology-env` (the repo's environment skill)
#
# Watch for `tool: Skill` in the trace.

# %%
from hydroagent.harness.claude_runner import available_skills, run_harness

[(s["name"], s["description"][:80]) for s in available_skills()]

# %%
r = run_harness(R.INFOS["a4"].example_query, on_event=show)
display(Markdown(r.answer or r.error))
print("skills loaded:", r.skills_loaded, "| usage:", r.usage, "| workspace:", r.workspace)
for p in r.plots:
    display(Image(p))

# %% [markdown]
# ### Write your own skill
# Edit the cell below, then run the harness on a task that needs it. Did the agent load it (`skills loaded`)?
# Does it help compared with a run *without* the skill (`skills=[...]` excluding it)?
# Template: `skills/_template/SKILL.md`.

# %%
SKILL_NAME = "flow-duration-curve"
SKILL_MD = """---
name: flow-duration-curve
description: Compute and plot flow-duration curves and exceedance flows (Q10, Q50, Q90, Q95) from Wflow output.csv at a gauge. Use for low-flow / drought or flow-regime questions.
---

# Flow-duration curve

1. Read `<run>/output/output.csv`, column `river_q_<gauge>`; drop dates before 2019-04-01 (spin-up).
2. Sort discharge descending; exceedance probability p = rank / (n + 1).
3. Qx = discharge exceeded x% of the time (Q90 = low flow, Q10 = high flow).
4. Plot with log y-axis, x = exceedance [%]; one line per run; save PNG in the workspace.
5. Report a table of Q10, Q50, Q90, Q95 per run and the % change vs baseline.
"""
d = EXERCISE_DIR / "skills" / SKILL_NAME
d.mkdir(parents=True, exist_ok=True)
(d / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")

r = run_harness("Make flow-duration curves at Epinal for the baseline and the 'ksat -30% (by hand)' run, and report "
                "Q10, Q50, Q90 for both.", on_event=show)
display(Markdown(r.answer or r.error))
print("skills loaded:", r.skills_loaded, "| usage:", r.usage)

# %% [markdown]
# ## Step 7: A5, composed router
# Try a knowledge question, a uniform parameter scenario, and an open-ended code task. Which route does each take?
# What does each cost?

# %%
for q in ["What does the KsatHorFrac parameter do in wflow_sbm?",
          "Increase river roughness by 50%: how does the Epinal peak change?"]:
    out = run_interactive(R.create_session("a5", provider=PROVIDER, on_event=show), q)
    report(out)

# %% [markdown]
# ## Step 8: Build your own graph (JSON)
# This is the same spec the Streamlit canvas produces.
#
# Node types:
# - `llm_agent`
# - `plain_llm`
# - `tool_step`
# - `critic` (edges labelled `approved` / `revise`)
# - `router` (edges labelled with a description of when to take them)
# - `human`
# - `claude_harness`

# %%
from hydroagent.builder.compile import make_custom_session, validate

my_graph = {
    "name": "my_agent",
    "nodes": [
        {"id": "start", "type": "start"},
        {"id": "modeller", "type": "llm_agent", "config": {"toolsets": ["parameters", "simulation"],
                                                         "system_prompt": "Run the scenario(s) needed. Be brief."}},
        {"id": "plot", "type": "tool_step", "config": {"action": "plot_all_runs"}},
        {"id": "critic", "type": "critic", "config": {"criteria": "Gives % change for peak AND low flow with run_ids.",
                                                    "max_loops": 1}},
        {"id": "writer", "type": "plain_llm", "config": {"instruction": "Write a 3-line note for a water manager."}},
        {"id": "end", "type": "end"},
    ],
    "edges": [
        {"source": "start", "target": "modeller"}, {"source": "modeller", "target": "plot"},
        {"source": "plot", "target": "critic"}, {"source": "critic", "target": "modeller", "label": "revise"},
        {"source": "critic", "target": "writer", "label": "approved"}, {"source": "writer", "target": "end"},
    ],
}
print(validate(my_graph) or "valid")
out = run_interactive(make_custom_session(my_graph, provider=PROVIDER, on_event=show),
                      "How do low flows at Epinal respond to a 50% higher crop factor?")
report(out)

# %% [markdown]
# ## Debrief
# Fill in the table from your runs, using `out.usage`:
#
# | Architecture | LLM calls | tokens | time | right answer? | guarantees |
# |---|---|---|---|---|---|
# | A0 | | | | | |
# | A1 | | | | | |
# | A2 | | | | | |
# | A3 | | | | | |
# | A4 | | | | | |
#
# Questions to discuss:
# - Which questions did **A0** handle as well as the agents?
# - Where did **A1** go wrong, if anywhere?
# - What did **A3** guarantee that A1 cannot?
# - When is the **harness (A4)** worth its cost and its unpredictability?
# - Where would a **skill** be better than a new tool, or a new subagent?
