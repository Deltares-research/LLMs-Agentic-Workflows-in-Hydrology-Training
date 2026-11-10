"""Mistral Vibe harness (A4m): the Mistral counterpart of the Claude harness (A4), for a like-for-like comparison.

Mistral Vibe is Mistral's open-source coding agent (``uv tool install mistral-vibe``). Like A4 it has an agent
loop, file and shell tools, skills (SKILL.md) and MCP. This runner gives it the same setup as A4:

* a per-task workspace (``output/harness/vibe-<id>/``) as the working directory, skills copied in;
* the same Wflow tools, served by ``wflow_mcp_server.py`` over stdio (MCP);
* the same guard-rails, as Vibe permissions: writes only inside the workspace (``outside_workdir = "deny"``),
  the base model folder read-only, blocked shell commands, no web tools;
* the model: your Mistral deployment on Azure Foundry (OpenAI-compatible endpoint, see config.py).

Vibe runs headless (``vibe --prompt-file ... --output streaming``); its event stream becomes the same trace,
metrics and scoreboard entry as A4. Vibe's config lives outside the workspace (``VIBE_HOME``), so the agent
cannot change its own permissions, and your own ``~/.vibe`` is not touched.

This is a teaching sandbox, not a security boundary.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable

import yaml

from .. import config as C
from ..tracing import Event, Recorder
from .claude_runner import HarnessResult

ALLOWED_TOOLS = ["read_file", "write_file", "edit", "grep", "skill", "todo", "bash", "git_bash", "hydro_*"]
DISABLED_TOOLS = ["web_fetch", "web_search", "task", "ask_user_question", "powershell"]
# Same intent as BLOCKED_SHELL in claude_runner.py (Vibe matches these globs against the command).
SHELL_DENYLIST = ["*rm -rf*", "*rmdir /s*", "*del /s*", "*del /q*", "*format ?:*", "*git push*", "*pip install*",
                  "*uv add*", "*uv remove*", "*curl *", "*wget *", "*Invoke-WebRequest*", "*shutdown*"]
SERVER_SCRIPT = Path(__file__).with_name("wflow_mcp_server.py")


def vibe_executable() -> str | None:
    return shutil.which("vibe")


def prepare_workspace(skills: list[str] | None = None, name: str | None = None) -> Path:
    """Create a workspace and copy the selected skills (default: all) into ``<ws>/skills/``."""
    C.ensure_dirs()
    ws = C.HARNESS_DIR / (name or f"vibe-{datetime.now():%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}")
    dest = ws / "skills"
    dest.mkdir(parents=True, exist_ok=True)
    for root in C.SKILLS_DIRS:
        if not root.exists():
            continue
        for sk in root.iterdir():
            if not (sk / "SKILL.md").exists() or sk.name.startswith("_"):
                continue
            if skills is not None and sk.name not in skills:
                continue
            shutil.copytree(sk, dest / sk.name, dirs_exist_ok=True)
    _stage_inputs(ws)
    return ws


def _stage_inputs(ws: Path) -> None:
    """Put everything the agent needs INSIDE the workspace.

    Vibe's sandbox (``outside_workdir = "deny"``) refuses shell commands that mention paths outside the workspace,
    and the repo path contains spaces, which its shell check splits. So the agent gets: read-only copies of the model
    inputs in ``model/``, the baseline output in ``runs/baseline/`` (new runs are copied to ``runs/<run_id>/`` by
    the MCP server), and a ``./py`` launcher for the project's Python. Then it only needs relative paths.
    """
    model = ws / "model"
    model.mkdir(exist_ok=True)
    for f in ("staticmaps.nc", "inmaps-eobs-2019-2020.nc", "wflow_sbm.toml"):
        if (C.BASE_MODEL_DIR / f).exists():
            shutil.copy2(C.BASE_MODEL_DIR / f, model / f)
    copy_run_output("baseline", ws)
    py = Path(sys.executable).as_posix()
    (ws / "py").write_text(f'#!/bin/sh\nexec "{py}" "$@"\n', encoding="utf-8", newline="\n")
    (ws / "py.cmd").write_text(f'@"{Path(sys.executable)}" %*\r\n', encoding="utf-8")
    try:
        (ws / "py").chmod(0o755)
    except OSError:
        pass


def copy_run_output(run_id: str, ws: Path) -> Path | None:
    """Copy a run's output.csv / output.nc into ``<ws>/runs/<run_id>/`` (used for the baseline and after each run)."""
    src = C.RUNS_DIR / run_id / "output"
    if not src.exists():
        return None
    dst = ws / "runs" / run_id
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("output.csv", "output.nc"):
        if (src / f).exists():
            shutil.copy2(src / f, dst / f)
    return dst


def _toml_str(s: str) -> str:
    return json.dumps(str(s))  # a JSON string is a valid TOML basic string


def _toml_list(items: list[str]) -> str:
    return "[" + ", ".join(_toml_str(i) for i in items) + "]"


def _price(model: str) -> tuple[float, float]:
    try:
        p = yaml.safe_load((C.EXERCISE_DIR / "pricing.yml").read_text(encoding="utf-8")) or {}
        e = p.get(model) or {}
        return float(e.get("input", 0) or 0), float(e.get("output", 0) or 0)
    except Exception:  # noqa: BLE001
        return 0.0, 0.0


def write_vibe_config(ws: Path, model: str, api_base: str, key_env: str) -> Path:
    """Write a private VIBE_HOME (outside the workspace) with provider, model, skills, MCP server and permissions."""
    home = C.HARNESS_DIR / "_vibe_home" / ws.name
    home.mkdir(parents=True, exist_ok=True)
    base = C.BASE_MODEL_DIR.resolve().as_posix()
    read_ok = [f"{base}/*", f"{C.RUNS_DIR.resolve().as_posix()}/*"]
    protect = [f"{base}/*"]
    pin, pout = _price(model)
    lines = [
        'active_model = "harness-model"',
        f"skill_paths = {_toml_list([(ws / 'skills').resolve().as_posix()])}",
        'outside_workdir = "deny"',
        f"enabled_tools = {_toml_list(ALLOWED_TOOLS)}",
        f"disabled_tools = {_toml_list(DISABLED_TOOLS)}",
        "",
        "[[providers]]",
        'name = "foundry"',
        f"api_base = {_toml_str(api_base)}",
        f"api_key_env_var = {_toml_str(key_env)}",
        'api_style = "openai"',
        'backend = "generic"',
        "",
        "[[models]]",
        f"name = {_toml_str(model)}",
        'provider = "foundry"',
        'alias = "harness-model"',
        f"input_price = {pin}",
        f"output_price = {pout}",
        "",
        "[[mcp_servers]]",
        'name = "hydro"',
        'transport = "stdio"',
        f"command = {_toml_str(Path(sys.executable).as_posix())}",
        f"args = {_toml_list([SERVER_SCRIPT.as_posix(), '--workspace', ws.resolve().as_posix()])}",
        "",
    ]
    tools = {
        "read_file": {"permission": "always", "allowlist": read_ok},
        "grep": {"permission": "always", "allowlist": read_ok},
        "skill": {"permission": "always"},
        "todo": {"permission": "always"},
        "write_file": {"permission": "always", "denylist": protect},
        "edit": {"permission": "always", "denylist": protect},
        "bash": {"permission": "always", "denylist": SHELL_DENYLIST},
        "git_bash": {"permission": "always", "denylist": SHELL_DENYLIST},
        "hydro_list_parameters": {"permission": "always"},
        "hydro_run_wflow_scenario": {"permission": "always"},
        "hydro_get_run_summary": {"permission": "always"},
    }
    for name, cfg in tools.items():
        lines.append(f"[tools.{name}]")
        for k, v in cfg.items():
            lines.append(f"{k} = {_toml_list(v) if isinstance(v, list) else _toml_str(v)}")
        lines.append("")
    (home / "config.toml").write_text("\n".join(lines), encoding="utf-8")
    return home


def _task_prompt(task: str, ws: Path) -> str:
    return f"""You are a hydrological modelling agent with a code sandbox.
Your working directory is your workspace. Everything you need is inside it; access outside it is denied.
ALWAYS use RELATIVE paths (the absolute path contains spaces): e.g. model/staticmaps.nc, not /c/Users/...

Files in the workspace:
- model/staticmaps.nc, model/inmaps-eobs-2019-2020.nc, model/wflow_sbm.toml: copies of the base Wflow model
  (this is the BASE_MODEL_DIR the skills mention). Do not modify them; write edited copies to new files.
- runs/baseline/output.csv and runs/baseline/output.nc: the baseline run.
- runs/<run_id>/output.csv and output.nc: copied here automatically after every run you make.
- ./py: the Python interpreter (has xarray, pandas, numpy, matplotlib, netCDF4). Run scripts as: ./py script.py

Model: wflow_sbm, upper Moselle to Epinal (GRDC 9116044), 2019-2020 daily, first 3 months = spin-up.
Run Wflow ONLY through the MCP tool hydro_run_wflow_scenario. Scenarios can combine uniform parameter multipliers,
an edited copy of staticmaps.nc (pass staticmaps_path, relative to the workspace) and an edited copy of the forcing
inmaps-eobs-2019-2020.nc (pass forcing_path; precip mm/day, pet mm/day, temp degC).
Use your skills when relevant. Write analysis scripts to files (write_file) and run them with ./py using the
shell tool. Do not use long python -c one-liners.
Finish with a concise answer: numbers with units, run_ids used, and paths of figures you made.

TASK:
{task}
"""


def _entry_text(e: dict) -> str:
    return "\n\n".join(b.get("text", "") for b in e.get("content", []) or [] if b.get("type") == "text").strip()


def _get(d: dict, *keys, default=None):
    """camelCase / snake_case tolerant lookup (Vibe's protocol models serialise with camelCase aliases)."""
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return default


class _StreamParser:
    """Turns Vibe's streaming entries into trace events; keeps the final answer and the skills loaded."""

    def __init__(self, rec: Recorder):
        self.rec = rec
        self.seen_calls: set[str] = set()
        self.done_calls: set[str] = set()
        self.answer = ""
        self.skills: list[str] = []

    def feed(self, e: dict) -> None:
        etype = e.get("type")
        if etype == "message" and e.get("role") == "assistant":
            text = _entry_text(e)
            if text and _get(e, "generationStatus", "generation_status") == "completed":
                self.answer = text
                self.rec.emit("llm", "Mistral", text)
        elif etype == "effect":
            eid = e.get("id", "")
            detail = e.get("detail") or {}
            state = e.get("state") or {}
            name = _get(detail, "toolName", "tool_name", default=e.get("title", "tool"))
            if eid not in self.seen_calls:
                self.seen_calls.add(eid)
                self.rec.tool_calls += 1
                inp = detail.get("input")
                if detail.get("kind") == "skill" and isinstance(inp, dict) and inp.get("name"):
                    self.skills.append(inp["name"])
                self.rec.emit("tool_call", f"tool: {name}", inp if inp is not None else e.get("title", ""))
            status = state.get("status")
            if status in ("completed", "failed") and eid not in self.done_calls:
                self.done_calls.add(eid)
                out = _get(state, "outputText", "output_text") or json.dumps(state.get("output"), default=str)
                if status == "failed":
                    err = state.get("error") or {}
                    out = f"{err.get('message', '')} {out}".strip()
                self.rec.emit("error" if status == "failed" else "tool_result", "result", out)
        elif etype == "notice":
            self.rec.emit("harness", "vibe", json.dumps(e.get("content", e.get("title", "")), default=str)[:300])


def _tail(path: Path, rec: Recorder, stop: threading.Event) -> None:
    """Stream the MCP server's Wflow console log into the trace as 'model' events."""
    pos = 0
    while not stop.is_set():
        if path.exists():
            with path.open(encoding="utf-8", errors="replace") as f:
                f.seek(pos)
                for line in f:
                    src, _, msg = line.rstrip("\n").partition("] ")
                    rec.emit("model", src.lstrip("[") or "wflow", msg)
                pos = f.tell()
        stop.wait(0.5)


def run_vibe_harness(task: str, *, skills: list[str] | None = None, model: str | None = None, max_turns: int = 40,
                     max_price: float | None = None, on_event: Callable[[Event], None] | None = None,
                     recorder: Recorder | None = None, workspace: Path | None = None) -> HarnessResult:
    """Run the Mistral Vibe harness on a task (blocking). Works in scripts, Jupyter and Streamlit."""
    rec = recorder or Recorder(on_event)
    s = C.get_settings()
    model = model or s.mistral_deployment
    label = f"mistral:{model}"
    if not rec.model:
        rec.model = label
    exe = vibe_executable()
    if not exe:
        msg = "Mistral Vibe is not installed. Install it once with:  uv tool install mistral-vibe"
        rec.emit("error", "vibe missing", msg)
        return HarnessResult("error", "", "", usage=rec.summary(), error=msg)
    api_base = C.foundry_openai_url()
    key_env = "MISTRAL_KEY" if os.getenv("MISTRAL_KEY") else "FOUNDRY_KEY"
    if not api_base or not os.getenv(key_env):
        msg = "Set FOUNDRY_ENDPOINT + FOUNDRY_KEY (or MISTRAL_ENDPOINT + MISTRAL_KEY) in .env."
        rec.emit("error", "config", msg)
        return HarnessResult("error", "", "", usage=rec.summary(), error=msg)

    ws = workspace or prepare_workspace(skills)
    home = write_vibe_config(ws, model, api_base, key_env)
    (ws / "_task.md").write_text(_task_prompt(task, ws), encoding="utf-8")
    out_dir = C.HARNESS_DIR / "_vibe_runs" / ws.name
    rec.emit("info", "vibe harness workspace", str(ws))
    rec.emit("harness", "session start", f"model={model} skills={sorted(p.name for p in (ws / 'skills').iterdir())}")

    cmd = [exe, "--prompt-file", str(ws / "_task.md"), "--output", "streaming", "--max-turns", str(max_turns),
           "--workdir", str(ws), "--output-dir", str(out_dir)]
    if max_price:
        cmd += ["--max-price", str(max_price)]
    env = {**os.environ, "VIBE_HOME": str(home), "PYTHONIOENCODING": "utf-8"}
    parser = _StreamParser(rec)
    stop = threading.Event()
    tail = threading.Thread(target=_tail, args=(ws / "wflow_console.log", rec, stop), daemon=True)
    tail.start()
    t0 = time.time()
    stderr_lines: list[str] = []
    try:
        proc = subprocess.Popen(cmd, cwd=ws, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                encoding="utf-8", errors="replace")
        err_thread = threading.Thread(target=lambda: stderr_lines.extend(proc.stderr), daemon=True)
        err_thread.start()
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                parser.feed(json.loads(line))
            except json.JSONDecodeError:
                rec.emit("harness", "vibe", line[:300])
        code = proc.wait()
        err_thread.join(timeout=5)
    finally:
        stop.set()
        tail.join(timeout=2)
    wall = time.time() - t0

    # Wflow runs made through the MCP server (run ids + time), and Vibe's own usage/cost report
    run_ids, wflow_s = [], 0.0
    calls = ws / "wflow_calls.jsonl"
    if calls.exists():
        for ln in calls.read_text(encoding="utf-8").splitlines():
            d = json.loads(ln)
            run_ids.append(d["run_id"])
            wflow_s += float(d.get("seconds", 0))
    if run_ids:
        rec.add_model_time(wflow_s, runs=len(run_ids))
    export = {}
    if (out_dir / "export.json").exists():
        export = json.loads((out_dir / "export.json").read_text(encoding="utf-8"))
    u = export.get("usage") or {}
    rec.add_llm("vibe_harness", u.get("input_tokens", 0), u.get("output_tokens", 0), u.get("cached_input_tokens", 0),
                0, max(wall - wflow_s, 0.0), calls=export.get("steps") or 0, model=label)
    if export.get("cost_usd"):
        rec.reported_cost_usd += float(export["cost_usd"])
    rec.emit("harness", "usage", f"outcome={export.get('outcome', code)} steps={export.get('steps')} "
                                 f"tokens={u.get('total_tokens', 0)} Wflow runs={len(run_ids)}")

    status, err = "done", ""
    if code != 0:
        status = "error"
        detail = (export.get("error") or {}).get("message") if isinstance(export.get("error"), dict) else ""
        err = f"vibe exited with code {code} ({export.get('outcome', 'see stderr')}). " + (
            detail or "".join(stderr_lines[-15:]).strip())
    plots = sorted(str(p) for p in ws.rglob("*.png"))
    rec.finish()
    rec.emit("final" if status == "done" else "error", "final answer", parser.answer or err)
    return HarnessResult(status, parser.answer, str(ws), run_ids, plots, parser.skills, rec.summary(), err)
