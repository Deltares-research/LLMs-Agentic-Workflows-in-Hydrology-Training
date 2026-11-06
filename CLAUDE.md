# Claude Code

This is a uv + Julia hydrology training repo. Python 3.13.18+ only. uv >= 0.13.6. Never conda, miniforge, mamba, or pixi.

For environment setup, hydromt/wflow/DLLMForge install, or version checks, read and follow `.agents/skills/hydrology-env/SKILL.md` (project skills live there, not under `.claude/skills/`). Then execute:

```bash
python .agents/skills/hydrology-env/scripts/verify_env.py
```

Stop if the script exits non-zero. Wflow is the Julia package `Wflow` (1.x). Do not `pip install wflow` or `uv add wflow`.
