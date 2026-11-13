# LLMs and agentic workflows in hydrology

Training repo for [Hydrology Software Days (Day 3) - LLMs and Agentic Workflows in Hydrology Training](https://softwaredays.deltares.nl/-/hydrology-software-days-day-3-llms-and-agentic-workflows-in-hydrology-training-afternoon-2026).

Python packages live in a project `.venv` managed by uv. Git, uv, and Julia are system tools. Wflow is a **Julia** package (`Wflow.jl` 1.x), not a Python import.

Do not use conda, miniforge, mamba, or pixi.

## Install

Two ways: ask a coding agent to run the skill, or follow the manual steps. Either way, finish with the verify script and continue only when every row is `OK`.

### Use the skill (recommended)

The skill lives at [`.agents/skills/hydrology-env/`](.agents/skills/hydrology-env/). Cursor, Codex, and GitHub Copilot load it from there. Claude Code: the same folder is pointed to from [`CLAUDE.md`](CLAUDE.md).

Open this repo in your agent, then prompt:

```
Install and verify the hydrology training environment using the hydrology-env skill.
```

To check an existing install:

```
Verify the hydrology-env skill: Git, uv >= 0.13.6, Julia, Python 3.13.18, hydromt, hydromt_wflow, Wflow.jl, DLLMForge, python-dotenv, and jupyter.
```

The agent should follow the skill and run `python .agents/skills/hydrology-env/scripts/verify_env.py`. If a tool installer changes PATH, open a new terminal and ask it to verify again.

### Manual install

#### 1. System tools

**Windows (PowerShell)**

```powershell
winget install --id Git.Git -e --source winget --accept-package-agreements --accept-source-agreements
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
winget install --name Julia --id 9NJNWW8PVKMN -e -s msstore --accept-package-agreements --accept-source-agreements
```

If Microsoft Store is blocked for Julia:

```powershell
Add-AppxPackage -AppInstallerFile https://install.julialang.org/Julia.appinstaller
```

**macOS / Linux**

```bash
# Git: use your package manager if `git --version` fails
curl -LsSf https://astral.sh/uv/install.sh | sh
curl -fsSL https://install.julialang.org | sh
```

Open a **new** terminal so Git, uv, and Julia are on PATH. Then:

```bash
uv self update
git --version
uv --version   # need 0.13.6 or newer
julia --version
```

#### 2. Project Python env

From the repo root (do not run `uv init`):

```bash
uv python install 3.13.18
uv python pin 3.13.18
uv sync
```

Optional activate: `.venv/Scripts/Activate.ps1` (Windows) or `source .venv/bin/activate` (Unix). `uv run` uses the venv without activating.

#### 3. Wflow.jl

```bash
julia -e "using Pkg; Pkg.add(\"Wflow\")"
```

PowerShell if quoting fails:

```powershell
@'
using Pkg
Pkg.add("Wflow")
'@ | julia
```

Need Wflow **1.x** (1.0.4 or newer), not 0.8.

#### 4. Verify

```bash
python .agents/skills/hydrology-env/scripts/verify_env.py
```

All rows must be `OK` before course exercises. The script checks Git, uv (>= 0.13.6), Julia, Python 3.13.18+, certifi (>= 2026.10.19), hydromt, hydromt_wflow plugins (`wflow_sbm`, `wflow_sediment`), DLLMForge / python-dotenv / jupyter, and Wflow.jl 1.x.

## Troubleshooting

- **Command not found after install**: PATH was not refreshed. Open a new terminal.
- **conda still on PATH**: ignore it. Use `uv run ...`. Do not `conda activate`.
- **Python 3.14**: hydromt_wflow does not support it. Stay on 3.13.18+.
- **uv < 0.13.6 or Python < 3.13.18**: emSign CA incident. Run `uv self update`, then `uv python install 3.13.18` and recreate `.venv`.
- **`import wflow` in Python**: wrong package. Use Julia `using Wflow`.
- **`import dllmforge` or `import docling` fails**: `uv sync` needs GitHub access to `Deltares-research/DLLMForge` (`dllmforge[api,docling]`, pinned in `pyproject.toml`). Azure keys stay in `.env`; they are not part of the skill.
- **Wflow 0.8.x**: pin the Julia package:

  ```bash
  julia -e "using Pkg; Pkg.add(name=\"Wflow\", version=\"1.0.4\")"
  ```

## Exercise 1: information extraction

After verify is green and `.env` has Azure OpenAI variables, see [exercises/01_information_extraction/](exercises/01_information_extraction/).

## Exercise 2: inspect a Wflow model through MCP

Read-only QA of Wflow inputs and outputs (Moselle at Épinal) via an MCP server in your coding agent.
See [exercises/02_mcp_rag/](exercises/02_mcp_rag/).

## Exercise 3: agentic architectures for HydroMT / Wflow

This exercise runs Wflow scenarios for the upper Moselle through six agent architectures:
- a plain LLM call;
- a ReAct agent;
- a generator-critic calibration loop;
- a Director with subagents and human-in-the-loop;
- a Claude Agent SDK code agent with skills;
- a composed router.

You can also build your own architecture. There are three levels: a Streamlit app, a notebook, and a CLI.
It needs the Anthropic key (and optionally Azure) in `.env`. See [exercises/03_agentic_workflows/](exercises/03_agentic_workflows/).

## Run a model

After HydroMT-Wflow has written a model:

```julia
using Wflow
Wflow.run("path/to/wflow_sbm.toml")
```
