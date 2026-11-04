# LLMs and agentic workflows in hydrology

Training repo for HydroMT-Wflow model setup/update and Wflow.jl runs, using coding agents.

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
Verify the hydrology-env skill: Git, uv, Julia, Python 3.13, hydromt, hydromt_wflow, and Wflow.jl.
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

Open a **new** terminal so Git, uv, and Julia are on PATH. Check:

```bash
git --version
uv --version
julia --version
```

#### 2. Project Python env

From the repo root (do not run `uv init`):

```bash
uv python install 3.13
uv python pin 3.13
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

All rows must be `OK` before course exercises. The script checks Git, uv, Julia, Python 3.13, hydromt, hydromt_wflow plugins (`wflow_sbm`, `wflow_sediment`), and Wflow.jl 1.x.

## Troubleshooting

- **Command not found after install**: PATH was not refreshed. Open a new terminal.
- **conda still on PATH**: ignore it. Use `uv run ...`. Do not `conda activate`.
- **Python 3.14**: hydromt_wflow does not support it. Stay on 3.13.
- **`import wflow` in Python**: wrong package. Use Julia `using Wflow`.
- **Wflow 0.8.x**: pin the Julia package:

  ```bash
  julia -e "using Pkg; Pkg.add(name=\"Wflow\", version=\"1.0.4\")"
  ```

## Run a model

After HydroMT-Wflow has written a model:

```julia
using Wflow
Wflow.run("path/to/wflow_sbm.toml")
```
