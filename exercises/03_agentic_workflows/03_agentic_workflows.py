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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Exercise 3: Agentic architectures for HydroMT / Wflow
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
# | A4m | Mistral Vibe harness + skills | The same pattern with Mistral's own harness |
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
# `PROVIDER` picks the model for the LangGraph architectures (A0-A3, A5, your own graph): `"anthropic"` (Claude),
# `"mistral"` (Mistral on Azure Foundry, deployment `DEPLOYMENT_MISTRAL_LARGE_4`) or `"azure"` (GPT). Run the same
# step with two providers and compare them on the scoreboard.
#
# The harness architectures do **not** use `PROVIDER`: each vendor's model runs in that vendor's own harness, so
# A4 is always Claude + Claude Agent SDK and A4m is always Mistral + Mistral Vibe. A5's open-ended step follows
# `PROVIDER` the same way (Mistral -> Vibe; Claude, or GPT which has no harness here -> Claude Agent SDK).
#
# Claude auth: with `FOUNDRY_ENDPOINT` + `FOUNDRY_KEY` in `.env`, Claude is called through Azure Foundry
# (deployment `DEPLOYMENT_OPUS_55`). Otherwise `ANTHROPIC_API_KEY` is used, or set `HARNESS_AUTH = "login"` to
# use your Claude Code login for the Claude harness.

# %%
import os
import sys
from pathlib import Path

EXERCISE_DIR = Path.cwd() if (Path.cwd() / "hydroagent").exists() else Path.cwd() / "exercises" / "03_agentic_workflows"
sys.path.insert(0, str(EXERCISE_DIR))

PROVIDER = "anthropic"          # LangGraph architectures only: or "mistral", "azure"
os.environ.setdefault("HARNESS_AUTH", "auto")   # "login" to use the Claude Code login for the Claude harness

from IPython.display import Image, Markdown, display

from hydroagent import config as C
from hydroagent import scoreboard
from hydroagent import twin
from hydroagent import wflow_tools as W
from hydroagent.architectures import registry as R
from hydroagent.notebook_ui import run_with_buttons   # answer agent questions with buttons
from hydroagent.runtime import run_interactive

print("env file:", C.load_env())
print("settings:", C.get_settings(provider=PROVIDER))
for k in R.INFOS:   # what each architecture will actually run on
    print(f"  {k:4s} {R.model_label(k, PROVIDER)}")


def show(ev):
    """Print the agent's trace as it happens. Remove kinds you find noisy ("model" = HydroMT/Wflow console)."""
    if ev.kind in ("llm", "tool_call", "node", "human", "error", "harness", "model"):
        print(f"[{ev.kind}] {ev.title}: {ev.detail[:250]}")


# Stream HydroMT / Wflow console output also when you call the model functions directly (Step 1).
W.LOG_SINK.set(lambda source, line: print(f"  [{source}] {line}"))


def report(out):
    """Show the answer, plots and the cost/time metrics. Every run is also logged to the scoreboard."""
    if out.status == "interrupted":
        q = out.question
        if isinstance(q, dict):  # e.g. plan approval: {"question": ..., "plan": [{"agent", "task"}, ...]}
            steps = [f"{i}. **{s.get('agent', '')}**: {s.get('task', s)}" if isinstance(s, dict) else f"{i}. {s}"
                     for i, s in enumerate(q.get("plan", []), 1)]
            q = "\n".join([q.get("question", ""), ""] + steps)
        display(Markdown(f"### 🙋 The agent asks you\n\n{q}\n\n"
                         "Answer with the buttons below."))
        return
    display(Markdown(out.answer or f"**{out.status}** {out.error}"))
    for p in out.plots:
        display(Image(p))
    u = out.usage
    cost = u.get("cost_usd")
    print(f"LLM calls {u['llm_calls']} | tool calls {u['tool_calls']} | Wflow runs {u['wflow_runs']} | "
          f"tokens {u['total_tokens']:,} (in {u['input_tokens']:,}, out {u['output_tokens']:,}, "
          f"cached {u['cache_read_tokens']:,})")
    print(f"time {u['wall_time_s']} s = LLM {u['llm_s']} + Wflow {u['model_s']} + other {u['other_s']} | "
          f"cost {'$%.4f' % cost if isinstance(cost, (int, float)) else 'n/a'} | runs: {out.run_ids}")
    print(f"scoreboard entry: {out.entry_id}  ->  rate it: scoreboard.rate('{out.entry_id}', 1..5)")

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
# ## How your question reaches each agent
# Every step below has a `QUESTION = "..."` line: edit it to ask your own. All architectures start the same way:
# `session.start(QUESTION)` (or `run_interactive(session, QUESTION)`, which does the same) puts your question
# into the graph's starting state twice:
#
# - `query`: the plain question text, for steps that write their own prompt around it;
# - `messages`: a chat history that starts with your question, for agents that converse.
#
# What differs is **who sees it**:
#
# | | Who reads your question |
# |---|---|
# | **A0** | `interpret` (as part of its prompt) and `report` (with the Wflow results) |
# | **A1** | the ReAct agent, as the first chat message |
# | **A2** | the generator, every round. The critic sees only the scores, not your question |
# | **A3** | `classify`. Your clarifications are appended to it. The modeller subagent gets a task written by the plan, **not** your question |
# | **A4** | Claude, as the task (on top of a system prompt with the model paths) |
# | **A5** | the router, then the chosen branch (prompt, chat message or harness task) |
# | **your graph** | agent nodes get the chat history. Plain-LLM and critic nodes get a transcript. A harness node gets `{query}` |
#
# When an agent pauses to ask *you* something, buttons appear in the cell (`run_with_buttons`). Your click calls
# `session.resume(answer)`, which sends the answer back into the paused step. The run continues from there; it
# does not start over.

# %% [markdown]
# ## Step 2: A0, plain LLM + fixed workflow
#
# ![A0 architecture](docs/figures/a0.png)
#
# The LLM only translates the question into multipliers and writes the report. Code fixes the order of steps.

# %%
QUESTION = "What happens to peak and low flows at Epinal if soils become 30% less permeable?"   # edit me
out = run_interactive(R.create_session("a0", provider=PROVIDER, on_event=show), QUESTION)
report(out)

# %% [markdown]
# ## Step 3: A1, single agent (ReAct)
#
# ![A1 architecture](docs/figures/a1.png)
#
# One LLM with tools in a loop. **Customise it:** change the system prompt or remove a tool set, and see how the
# behaviour changes. Try a question that needs *two* scenarios, for example comparing +20% and -20% soil thickness.

# %%
from hydroagent.architectures import a1_react
from hydroagent.architectures.common import make_session

MY_PROMPT = a1_react.SYSTEM_PROMPT          # edit me
MY_TOOLSETS = ("parameters", "simulation", "analysis")   # try removing "analysis" or "simulation"

session = make_session("a1", a1_react.build, provider=PROVIDER, on_event=show,
                       system_prompt=MY_PROMPT, toolsets=MY_TOOLSETS)
QUESTION = "Compare +20% and -20% soil thickness: which affects low flows at Epinal more?"   # edit me
out = run_interactive(session, QUESTION)
report(out)

# %% [markdown]
# ## Step 4: A2, generator-critic loop (calibration)
#
# ![A2 architecture](docs/figures/a2.png)
#
# There are no observed discharge series in this repo, so we use a **twin experiment**. A hidden "truth" run
# made with secret multipliers plays the role of observations, and the loop tries to recover it.
# **Knobs:** `max_iter`, `target_kge`, `critic="llm"` vs `"rule"`. Is the LLM critic worth its tokens?

# %%
twin.ensure_truth()
QUESTION = ("Calibrate ksat_vertical, soil_thickness and ksat_horizontal_ratio so the Epinal discharge matches "
            "observations (target KGE 0.95).")   # edit me
out = run_interactive(R.create_session("a2", provider=PROVIDER, on_event=show, max_iter=4, critic="llm"), QUESTION)
report(out)

# %%
print("the truth was:", twin.reveal_truth())   # debrief: equifinality?

# %% [markdown]
# ## Step 5: A3, Director with subagents and human-in-the-loop
#
# ![A3 architecture](docs/figures/a3.png)
#
# This follows the Director pattern:
# - an allow-list of scenario types;
# - clarification questions when inputs are missing;
# - plan approval before any compute is spent;
# - Modeller and Analyst subagents, with retries and escalation.
#
# The agent pauses to ask you questions. Answer with the **buttons** that appear in the cell output:
# a text box + **Send** for open questions (try `-30`), **Approve / Reject** for the plan. After each click the
# trace and result appear under the buttons. Try an out-of-scope request too.

# %%
QUESTION = "What if the soils in the Moselle become less permeable?"   # edit me (no % on purpose: it will ask)
a3 = R.create_session("a3", provider=PROVIDER, on_event=show)
a3_run = run_with_buttons(a3, QUESTION, report)   # a3_run["out"] holds the latest outcome

# %% [markdown]
# ## Step 6: A4, Claude Agent SDK harness + skills (code agent)
#
# ![A4 architecture](docs/figures/a4.png)
#
# > **Security: run Step 6 in the course web app.** A4, A4m and the skills lab let the LLM write and run its own
# > code. On your laptop that code would run with your rights (your files, network and keys), so here they run
# > only in the sandboxed Streamlit app (URL and password given in class). On your machine the cells below stop
# > with *"Code-writing agents are disabled on this machine"*. Read them to see how the harness is called, then
# > run the same task on the app's A4 / A4m pages and its skills lab.
#
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
HARNESS_QUESTION = ("Halve the soil permeability (soil_ksat_vertical) only in the Fresse-sur-Moselle sub-catchment "
                    "(GRDC 9116039), run the model, and compare the annual-maximum discharge at Epinal and at "
                    "Fresse-sur-Moselle with the baseline. Make one figure.")   # edit me (also used by A4m below)
out_a4 = R.create_session("a4", on_event=show).start(HARNESS_QUESTION)
report(out_a4)
print("skills loaded:", out_a4.state.get("skills_loaded"), "| workspace:", out_a4.state.get("workspace"))

# %% [markdown]
# ### A4m: the same task with Mistral Vibe
#
# ![A4m architecture](docs/figures/a4m.png)
#
# Mistral's own coding agent, **Mistral Vibe**, with your Mistral deployment on Azure Foundry. It gets the same
# workspace layout, the same skills and the same Wflow tools as A4, so this is a like-for-like comparison of two
# harnesses and two models. Install Vibe once: `uv tool install mistral-vibe`.
#
# Compare the two outputs (and the scoreboard below): did both load a skill? Same numbers? Tokens, time, cost?
# Rate both answers with `scoreboard.rate(...)`.

# %%
out_a4m = R.create_session("a4m", on_event=show).start(HARNESS_QUESTION)
report(out_a4m)
print("skills loaded:", out_a4m.state.get("skills_loaded"), "| workspace:", out_a4m.state.get("workspace"))

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
#
# ![A5 architecture](docs/figures/a5.png)
#
# Try a knowledge question, a uniform parameter scenario, and an open-ended code task. Which route does each take?
# What does each cost? The open-ended route runs `PROVIDER`'s own harness (Mistral -> Vibe, otherwise Claude).
#
# On your laptop the open-ended route stops with the "disabled" message (see Step 6); the other routes run
# locally. Try an open-ended question on the web app's A5 page.

# %%
QUESTIONS = [   # edit me: each question gets its own session; watch which route the router picks
    "What does the KsatHorFrac parameter do in wflow_sbm?",                  # expect: simple_answer
    "Increase river roughness by 50%: how does the Epinal peak change?",     # expect: parameter_scenario
    # "Halve soil_ksat_vertical only in sub-catchment 9116039 and compare annual maxima",  # open_ended_code (slow)
]
for q in QUESTIONS:
    run_with_buttons(R.create_session("a5", provider=PROVIDER, on_event=show), q, report)  # buttons if unsure

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
# - `claude_harness` (web app only: on your laptop it stops with the "disabled" message, see Step 6)

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
QUESTION = "How do low flows at Epinal respond to a 50% higher crop factor?"   # edit me
run_with_buttons(make_custom_session(my_graph, provider=PROVIDER, on_event=show), QUESTION, report)

# %% [markdown]
# ## Debrief: which architecture strikes the right balance?
# Every run above was logged to the **scoreboard** (`output/scoreboard.jsonl`) with its tokens, time split
# (LLM vs Wflow vs other) and estimated cost (`pricing.yml`). Only *you* can judge the quality: rate each
# answer from 1 to 5 with `scoreboard.rate(entry_id, score)`, using the entry ids printed by `report()`.
# Then compare. A fair comparison uses the same question for every architecture, ideally repeated 2-3 times,
# because agents are not deterministic.

# %%
# scoreboard.rate("<entry id>", 4)          # rate your answers first
sb = scoreboard.load()
display(sb[["id", "arch", "status", "quality", "total_tokens", "wall_time_s", "llm_s", "model_s", "cost_usd",
            "question"]].tail(20))
summary = scoreboard.summary(sb)
summary["tokens_per_quality_pt"] = (summary["total_tokens"] / summary["quality"]).round(0)
summary

# %%
# Where does the time go? LLM vs hydrological model, per architecture
summary.set_index("arch")[["llm_s", "model_s"]].plot.barh(stacked=True, figsize=(7, 3), xlabel="seconds");

# %% [markdown]
#
# Questions to discuss:
# - Which questions did **A0** handle as well as the agents?
# - Where did **A1** go wrong, if anywhere?
# - What did **A3** guarantee that A1 cannot?
# - When is the **harness (A4)** worth its cost and its unpredictability?
# - Where would a **skill** be better than a new tool, or a new subagent?
