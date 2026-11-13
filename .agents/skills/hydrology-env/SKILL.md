---
name: hydrology-env
description: Installs and verifies the hydrology training stack (Git, uv >= 0.13.6, Julia, Python 3.13.18, hydromt, hydromt_wflow, Wflow.jl, DLLMForge, python-dotenv, jupyter, mcp, pandas, xarray, geopandas, matplotlib). Use when setting up the course environment, installing hydromt or wflow, creating the project venv, or checking package versions.
---

# Hydrology training environment

Bootstrap Git, uv, Julia, Python 3.13, hydromt, hydromt_wflow, Wflow.jl, and DLLMForge (plus python-dotenv and jupyter) for this repo. Use existing tools if they already meet the version gates. Do not re-run `uv init`. Do not write Azure API keys; those stay in `.env`.

## Hard bans

- No conda, miniforge, mamba, or pixi.
- Never `uv add wflow`, `pip install wflow`, or `python -c "import wflow"`. That is an obsolete Python package. The model runtime is Julia `Wflow`.
- Do not install QGIS, Notepad++, or other GUI editors as part of this skill.
- Do not reimplement version checks in prose. Execute `scripts/verify_env.py`.

## Utility scripts

**verify_env.py**: execute this script (do not read it as the check). Run from the repo root:

```bash
python .agents/skills/hydrology-env/scripts/verify_env.py
```

Print the script output. Stop on non-zero exit. Only continue when it reports all OK.

## Workflow

Copy and track:

```
Task Progress:
- [ ] Step 1: Detect OS and refresh PATH if tools were just installed
- [ ] Step 2: Git
- [ ] Step 3: uv
- [ ] Step 4: Julia (juliaup)
- [ ] Step 5: Python 3.13 project env (uv sync)
- [ ] Step 6: Wflow.jl
- [ ] Step 7: Verify (must pass)
```

After every install step that changes PATH, open a new shell or prepend the install location. Then continue.

### Step 1: Detect OS

- Windows -> winget commands below.
- macOS / Linux -> Unix commands below.

Skip Git or Julia if already on PATH. Always ensure uv is **>= 0.13.6** (emSign CA incident).

### Step 2: Git

Windows:

```powershell
winget install --id Git.Git -e --source winget --accept-package-agreements --accept-source-agreements
```

Unix: install Git from the OS package manager if missing (`git --version`).

### Step 3: uv

Windows:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Unix:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

If uv is missing, install with the commands above. If it exists, run `uv self update`. Confirm `uv --version` is **0.13.6 or newer**.

### Step 4: Julia

Windows (juliaup via Microsoft Store):

```powershell
winget install --name Julia --id 9NJNWW8PVKMN -e -s msstore --accept-package-agreements --accept-source-agreements
```

If Store is blocked:

```powershell
Add-AppxPackage -AppInstallerFile https://install.julialang.org/Julia.appinstaller
```

Unix:

```bash
curl -fsSL https://install.julialang.org | sh
```

Confirm `julia --version` (1.10+ is fine).

### Step 5: Python 3.13 project env

Work in the repo root. This repo already has `pyproject.toml`, `uv.lock`, and `.python-version`. Do not `uv init`. `uv sync` also installs `dllmforge[api,docling]` from GitHub plus `python-dotenv` and jupyter.

```bash
uv python install 3.13.18
uv python pin 3.13.18
uv sync
```

If `.venv` was created with an older 3.13, remove it and run `uv sync` again.

`uv run python --version` must be **3.13.18 or newer 3.13.x**, not 3.14.

### Step 6: Wflow.jl

```bash
julia -e "using Pkg; Pkg.add(\"Wflow\")"
```

On PowerShell, pipe the script instead if quoting fails:

```powershell
@'
using Pkg
Pkg.add("Wflow")
'@ | julia
```

Require Wflow **1.x** (1.0.4 or newer). If the registry resolves 0.8.x:

```bash
julia -e "using Pkg; Pkg.add(name=\"Wflow\", version=\"1.0.4\")"
```

First `using Wflow` may take several minutes (precompile).

### Step 7: Verify

```bash
python .agents/skills/hydrology-env/scripts/verify_env.py
```

If it fails: read the FAIL row, fix only that component, re-run the script. Do not proceed with course work until exit code 0.

Expected plugins in `uv run hydromt --plugins`: `wflow_sbm`, `wflow_sediment`.
Expected Python imports via `uv run`: `dllmforge`, `docling`, `dotenv`, `ipykernel`, `jupyter`, plus for Exercise 2 `mcp`, `pandas`, `xarray`, `geopandas`, `matplotlib`, plus for Exercise 3 `langgraph`, `langchain_anthropic`, `langchain_openai`, `claude_agent_sdk` (bundles the Claude Code CLI; no Node needed), `streamlit`, `streamlit_flow`.

## Version gates

| Component | Gate |
|-----------|------|
| uv | >= 0.13.6 |
| Python (via `uv run`) | 3.13.18+ (not 3.14) |
| certifi | >= 2026.10.19 |
| hydromt | >= 1.4.1 |
| hydromt_wflow | >= 1.1.0 |
| Wflow.jl | 1.x (>= 1.0.4) |
| dllmforge | importable (`[api,docling]` extra from git, 0.1.1+) |
| python-dotenv | importable (`dotenv`) |
| jupyter / ipykernel | importable |
| mcp / pandas / xarray / geopandas / matplotlib (Exercise 2) | importable |
| langgraph / claude-agent-sdk / streamlit (Exercise 3) | importable; SDK bundled CLI present |

## Troubleshooting

- **Command not found after install**: PATH not refreshed. New terminal, then retry.
- **conda on PATH**: ignore it. Use `uv run` only. Do not `conda activate`.
- **uv older than 0.13.6 or Python older than 3.13.18**: emSign CA incident. `uv self update`, then `uv python install 3.13.18` and recreate `.venv`.
- **`import wflow` in Python**: wrong package. Use Julia `using Wflow`.
- **Wflow 0.8.x**: pin 1.0.4 as in Step 6. HydroMT-Wflow 1.x models will not run on 0.8.
- **Julia Store blocked (Windows)**: App Installer fallback in Step 4.
- **`import dllmforge` or `import docling` fails**: `uv sync` could not install `dllmforge[api,docling]` from `https://github.com/Deltares-research/DLLMForge.git`. Confirm GitHub access to that repo, then re-run `uv sync` from the repo root. Do not put Azure keys in this skill; they belong in `.env`.
