# Exercise 2: Agentic architectures for HydroMT / Wflow

Run Wflow simulations of the upper Moselle (to Epinal) through agents of increasing complexity. The aim is to
learn **when** an architecture is worth its cost, from a single LLM call up to multi-agent graphs and code-writing
harnesses with skills.

> Rule #1: start with a single agent and good tools. Add structure only when you hit a clear limit.
> (*Agentic Overview*, Google Cloud and LangChain design-pattern guides)

Pick the level that matches your Python experience. All three use the same code (`hydroagent/`).

| Level | Entry point | You will |
|---|---|---|
| 🟢 No code | Streamlit app | run the pre-built architectures, compare them, build your own on a drag-and-drop canvas, write a skill |
| 🟡 Some Python | Jupyter notebook | run each architecture in cells, change prompts, tools and loop settings, write a skill and a graph spec |
| 🔴 Advanced | CLI + source | edit the architectures in `hydroagent/architectures/`, add tools, run custom graphs from JSON |

## Architectures

| Key | Architecture | Pattern (Agentic Overview) | Hydro task |
|---|---|---|---|
| A0 | Plain LLM + fixed workflow | Deterministic workflow, no agent | question → multipliers → Wflow → report |
| A1 | Single agent (ReAct) | Single agent | "What if soils were 30% less permeable?" |
| A2 | Generator-critic loop | Loop / generator-critic | calibration against "observed" discharge (twin experiment) |
| A3 | Director | Subagents (supervisor) + human-in-the-loop + custom logic | allow-listed scenarios, clarification, plan approval, Modeller and Analyst subagents |
| A4 | Claude Agent SDK harness + skills | Agent harness + skills | writes Python to edit and analyse NetCDF (e.g. change one sub-catchment only) |
| A5 | Composed | Harness agents inside your own framework | router → {plain LLM \| ReAct \| harness} → human check if unsure |
| custom | Your graph | anything | built on the canvas or in JSON |

**Notes on A3.** It reuses the *architecture* of the Agent4Hydro project's Director (classify → clarify →
plan → specialists → finalize, with LangGraph `interrupt`). It does not reuse that project's tools.

**How agents change the model:**
- **A0–A3 and A5** change the model only through bounded **parameter multipliers**: Wflow TOML `scale` on
  staticmaps variables, or a multiplied constant `value`.
- **A4** may write code that edits `staticmaps.nc` copies in its own workspace. It can also turn a constant into
  a spatial map.
- The base model is never modified.

## Setup (once)

From the **repo root**:

1. **Install and verify the environment.** Use the `hydrology-env` skill, or run `uv sync` and install Julia +
   `Wflow`. Every row of the verify script must be `OK`, including `agents: langgraph + claude-agent-sdk + streamlit`:

   ```bash
   uv run python .agents/skills/hydrology-env/scripts/verify_env.py
   ```

2. **Add keys to a `.env` file.** It can live at the repo root or in any parent folder. Never paste keys into code.

   ```
   ANTHROPIC_API_KEY=...
   ANTHROPIC_DEPLOYMENT_SONNET55=claude-sonnet-5-5   # model used for Claude
   # optional, for the Azure switch (LangGraph architectures only):
   AZURE_OPENAI_API_KEY=...
   AZURE_OPENAI_API_BASE=...          # or AZURE_OPENAI_ENDPOINT
   AZURE_OPENAI_API_VERSION=...
   AZURE_OPENAI_DEPLOYMENT_GPT41=...  # or AZURE_OPENAI_DEPLOYMENT_NAME
   ```

   Optional environment variables:

   | Variable | Values | Effect |
   |---|---|---|
   | `AGENT_PROVIDER` | `anthropic` \| `azure` | default provider for the LangGraph architectures |
   | `HARNESS_AUTH` | `auto` \| `api_key` \| `login` | `login` makes A4 use the Claude Code login on the machine instead of the API key |
   | `WFLOW_MODE` | `server` \| `subprocess` | `server` (the default) keeps one Julia process alive between runs |

3. **Build the model, run the baseline and create the twin "observations".** This takes about 3 min the first time:

   ```bash
   uv run python exercises/02_agentic_workflows/cli.py prepare
   ```

The first Wflow run in a session starts Julia, which takes about 1–1.5 min. After that a 2-year run takes about
15 s, because the Julia process is reused.

## 🟢 Level 1: Streamlit

```bash
uv run streamlit run exercises/02_agentic_workflows/app/streamlit_app.py
```

1. **Architectures.**
   - Run A0 and A1 on the default question, then open *Full trace*. How many LLM calls did each make? Did the answers differ?
   - Run A3 and answer its questions. Then try an out-of-scope request such as *"estimate flood damage in euros"*.
   - Run A2 with `critic = llm`, then with `critic = rule`. Compare the KGE progression and the tokens used.
   - Run A4 on the default task. Which skill did it load? How many turns and how much cost did it take?
2. **Compare.** Run A0, A1 and A3 on one question. Fill in the debrief table below.
3. **Build your own.**
   - Start from *Agent + critic loop*.
   - Add a `Human check` node before `End`.
   - Label edges by clicking them.
   - Run it, then download the JSON.
4. **Skills lab.** Write a `flow-duration-curve` skill and run the harness with and without it.

## 🟡 Level 2: Notebook

Open [`02_agentic_workflows.ipynb`](02_agentic_workflows.ipynb) and select the project `.venv` kernel, or run
`uv run jupyter lab`. It is paired with `02_agentic_workflows.py` (jupytext). Work through Steps 0–8. Cells
marked *edit me* are the knobs.

## 🔴 Level 3: CLI and source

```bash
CLI="uv run python exercises/02_agentic_workflows/cli.py"
$CLI list                                    # architectures, parameters, skills
$CLI run a1 "Compare +20% and -20% soil thickness for low flows" -p anthropic
$CLI run a2 -o max_iter=3 -o critic='"rule"'  # options are JSON values
$CLI run a3 "What if snow melts faster?"      # answers questions at the prompt
$CLI compare a0 a1 a3 "What if soils become 30% less permeable?" --auto-answer "-30,yes"
$CLI harness "Halve soil_ksat_vertical in sub-catchment 9116039 and compare annual maxima" -s netcdf-analysis
$CLI templates && $CLI build --graph exercises/02_agentic_workflows/output/graphs/agent_critic.json "..."
$CLI runs                                   # all scenario runs
```

Suggested tasks:

- **Add a tool.** Add `seasonal_stats(run_id)` (DJF/MAM/JJA/SON mean flow) to `hydroagent/tools.py` and add it
  to a tool set. Which architectures use it without further changes?
- **Add a scenario.** Add a type to the A3 allow-list (`SCENARIOS` in `a3_director.py`), for example
  `land_roughness`.
- **Use a deterministic critic.** Replace A2's LLM critic with a rule that uses the KGE components
  (`critic="rule"` already exists). Is it worse?
- **Edit the YAML.** Expose `build_model_variant` (it edits the HydroMT build YAML: `ptf_ksatver`, `river_upa`,
  `rivdph_method`) to an agent, and compare `brakensiek` with `cosby`.
- **Restrict the harness.** Tighten A4's guard hook in `harness/claude_runner.py`, for example by blocking Bash
  entirely. What can the agent still do?

## Skills

Skills are folders containing a `SKILL.md` (YAML frontmatter `name` + `description`, then instructions) and
optional `scripts/`. The agent only sees the *description* until it decides to load the skill. This is called
progressive disclosure.

The harness loads skills from two places:
- `exercises/02_agentic_workflows/skills/`, with `wflow-toml`, `netcdf-analysis` and a `_template`;
- the repo's `.agents/skills/`, with `hydrology-env`, the environment-setup skill shared by all exercises.

They are copied into each harness workspace as `.claude/skills/`.

**Tool vs skill vs subagent:**

| Use a… | When |
|---|---|
| tool | a fixed action with typed inputs (run Wflow) |
| skill | know-how and procedures loaded on demand (how Wflow TOML scaling works, how to make an FDC) |
| subagent | a separate context window for a sub-task (Modeller vs Analyst in A3) |

## Debrief table

| Architecture | LLM calls | Tokens | Time | Correct? | Guarantees (validation / approval / audit) |
|---|---|---|---|---|---|
| A0 | | | | | |
| A1 | | | | | |
| A2 | | | | | |
| A3 | | | | | |
| A4 | | | | | |
| A5 | | | | | |

## Files

| Path | Role |
|---|---|
| `hydroagent/wflow_tools.py` | HydroMT build, scenario TOMLs (multipliers), warm Julia Wflow server, metrics, plots. No LLM code. |
| `hydroagent/tools.py` | LangChain tools (thin wrappers) and tool sets |
| `hydroagent/architectures/a0..a5_*.py` | The pre-built architectures (LangGraph and Claude Agent SDK) |
| `hydroagent/harness/claude_runner.py` | Claude Agent SDK harness: workspace, skills, MCP tools, guard hook |
| `hydroagent/builder/` | Node catalogue, graph-spec validation and compiler, templates |
| `hydroagent/twin.py` | Twin experiment ("observed" discharge from hidden multipliers) |
| `hydroagent/tracing.py`, `runtime.py` | Uniform trace (calls, tokens, time), interrupt/resume driver |
| `skills/` | Exercise skills and `_template` |
| `app/streamlit_app.py` | Level 1 |
| `02_agentic_workflows.ipynb` / `.py` | Level 2 |
| `cli.py` | Level 3 |
| `tests/` | Fast tests, no LLM: `uv run pytest exercises/02_agentic_workflows/tests -q` |
| `output/` | Runs, plots, harness workspaces, graphs (gitignored) |

## Troubleshooting

| Problem | Fix |
|---|---|
| `Credit balance is too low` from Anthropic | Top up the key; use `-p azure` for A0–A3, A5; or set `HARNESS_AUTH=login` for A4 |
| A run fails with `Couldn't parse the section under [input]` | The TOML scale syntax is wrong. See `skills/wflow-toml/SKILL.md` |
| The first run is slow | Julia start-up (~1 min). Later runs reuse the warm process |
| `Model not built` | `cli.py prepare` |
| Julia not found | Install via the hydrology-env skill, or set `JULIA_EXE` |
| Windows console shows `?` instead of symbols | The CLI forces UTF-8; for your own scripts set `PYTHONIOENCODING=utf-8` |
