# Agent notes

This is a uv + Julia hydrology training repo. Python 3.13.18+ only (`>=3.13.18,<3.14`). uv >= 0.13.6. Never conda, miniforge, mamba, or pixi.

For environment setup, hydromt/wflow install, or version checks, follow `.agents/skills/hydrology-env/SKILL.md` and execute:

```bash
python .agents/skills/hydrology-env/scripts/verify_env.py
```

Wflow is the Julia package `Wflow` (1.x). Do not `pip install wflow` or `uv add wflow`.
