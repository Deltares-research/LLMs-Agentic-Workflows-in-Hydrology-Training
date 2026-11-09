# Agent notes

This is a uv + Julia hydrology training repo. Python 3.13.18+ only (`>=3.13.18,<3.14`). uv >= 0.13.6. Never conda, miniforge, mamba, or pixi.

For environment setup, hydromt/wflow/DLLMForge install, or version checks, follow `.agents/skills/hydrology-env/SKILL.md` and execute:

```bash
python .agents/skills/hydrology-env/scripts/verify_env.py
```

Wflow is the Julia package `Wflow` (1.x). Do not `pip install wflow` or `uv add wflow`.

Exercise 2 (agentic workflows) has its own agent skills in `exercises/02_agentic_workflows/skills/` (`wflow-toml`, `netcdf-analysis`). The Claude Agent SDK harness copies them into each workspace at runtime. Agent outputs go to `exercises/02_agentic_workflows/output/` (gitignored). Never modify `wflow_moselle_epinal/` from an agent.
