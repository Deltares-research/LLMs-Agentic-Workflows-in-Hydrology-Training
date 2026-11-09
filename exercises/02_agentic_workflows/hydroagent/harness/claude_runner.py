"""Claude Agent SDK harness: an agent that writes and runs its own code, with skills and guard-rails.

What the harness gives you "for free" (drawio section B): the agent loop, file system + Bash tools,
skills (SKILL.md, progressive disclosure), subagents, context management, permission hooks.
What we add:
* a per-task workspace (``output/harness/<id>/``) used as the agent's cwd;
* skills copied into ``<workspace>/.claude/skills/`` (from ``exercises/02_agentic_workflows/skills``
  and the repo's ``.agents/skills``);
* in-process MCP tools to run Wflow scenarios on the warm Julia server;
* a PreToolUse hook that blocks writes outside the workspace and destructive shell commands.

This is a teaching sandbox, not a security boundary.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from .. import config as C
from .. import wflow_tools as W
from ..runtime import run_sync
from ..tracing import Event, Recorder

DEFAULT_ALLOWED = ["Read", "Write", "Edit", "Glob", "Grep", "Bash", "TodoWrite",
                   "mcp__hydro__run_wflow_scenario", "mcp__hydro__list_parameters", "mcp__hydro__get_run_summary"]

BLOCKED_SHELL = re.compile(r"(rm\s+-rf|rmdir\s+/s|del\s+/[sq]|format\s+[a-z]:|git\s+push|pip\s+install|uv\s+(add|remove)|"
                           r"curl\s|wget\s|Invoke-WebRequest|shutdown)", re.I)


@dataclass
class HarnessResult:
    status: str
    answer: str
    workspace: str
    run_ids: list[str] = field(default_factory=list)
    plots: list[str] = field(default_factory=list)
    skills_loaded: list[str] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    error: str = ""


def prepare_workspace(skills: list[str] | None = None, name: str | None = None) -> Path:
    """Create a workspace and copy the selected skills (default: all) into .claude/skills/."""
    C.ensure_dirs()
    ws = C.HARNESS_DIR / (name or f"{datetime.now():%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}")
    dest = ws / ".claude" / "skills"
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
    return ws


def available_skills() -> list[dict]:
    out = []
    for root in C.SKILLS_DIRS:
        for sk in sorted(root.glob("*/SKILL.md")) if root.exists() else []:
            if sk.parent.name.startswith("_"):
                continue
            meta = _frontmatter(sk.read_text(encoding="utf-8"))
            out.append({"name": meta.get("name", sk.parent.name), "folder": sk.parent.name,
                        "description": meta.get("description", ""), "path": str(sk)})
    return out


def _frontmatter(text: str) -> dict:
    m = re.match(r"^---\s*\n(.*?)\n---", text, re.S)
    if not m:
        return {}
    import yaml

    try:
        return yaml.safe_load(m.group(1)) or {}
    except Exception:
        return {}


def _system_prompt(ws: Path) -> str:
    return f"""You are a hydrological modelling agent with a code sandbox.
Workspace (your cwd, write ONLY here): {ws.as_posix()}
Python interpreter (has xarray, pandas, numpy, matplotlib, netCDF4): "{Path(sys.executable).as_posix()}"
Base Wflow model (READ-ONLY): {C.BASE_MODEL_DIR.as_posix()}  (staticmaps.nc, inmaps-eobs-2019-2020.nc, wflow_sbm.toml)
Baseline run output: {(C.RUNS_DIR / 'baseline' / 'output').as_posix()}  (output.csv, output.nc)
All scenario runs live in {C.RUNS_DIR.as_posix()}/<run_id>/output/.

Model: wflow_sbm, upper Moselle to Epinal (GRDC 9116044), 2019-2020 daily, first 3 months = spin-up.
Run Wflow ONLY through the MCP tool run_wflow_scenario (uniform multipliers and/or an edited staticmaps file).
Use your skills when relevant. Write analysis scripts to files (Write tool) and run them with the interpreter
above using the Bash tool (not PowerShell).
Finish with a concise answer: numbers with units, run_ids used, and paths of figures you made."""


def _make_mcp_server(ws: Path, rec: Recorder, run_ids: list[str]):
    from claude_agent_sdk import create_sdk_mcp_server, tool

    def _txt(x) -> dict:
        return {"content": [{"type": "text", "text": x if isinstance(x, str) else json.dumps(x, indent=1, default=str)}]}

    @tool("list_parameters", "List Wflow parameters that can be multiplied, with allowed ranges.", {})
    async def list_parameters(args):
        return _txt(W.list_parameters())

    @tool("run_wflow_scenario",
          "Run a Wflow scenario (~15-80 s). Args: label (str), multipliers (object, may be {}), staticmaps_path "
          "(str, optional absolute path to an edited staticmaps.nc INSIDE the workspace), static_overrides (object, "
          "optional {wflow_standard_name: netcdf_variable_name} so a parameter is read from a (new) map in that "
          "file, e.g. to turn a constant into a spatial map). Returns run_id + stats and % change vs baseline.",
          {"type": "object", "properties": {
              "label": {"type": "string"},
              "multipliers": {"type": "object", "additionalProperties": {"type": "number"}},
              "staticmaps_path": {"type": "string"},
              "static_overrides": {"type": "object", "additionalProperties": {"type": "string"}}},
           "required": ["label"]})
    async def run_wflow_scenario(args):
        try:
            sp = args.get("staticmaps_path") or None
            if sp:
                p = Path(sp) if Path(sp).is_absolute() else ws / sp
                if ws.resolve() not in p.resolve().parents:
                    return _txt("ERROR: staticmaps_path must be inside the workspace " + ws.as_posix())
                sp = p
            W.ensure_baseline()
            info = W.create_run(args.get("multipliers") or {}, args["label"], static_override=sp,
                                static_map_overrides=args.get("static_overrides") or None)
            info = W.execute_run(info.run_id)
            run_ids.append(info.run_id)
            return _txt(W.summarise_run(info.run_id))
        except Exception as e:
            return _txt(f"ERROR: {e}")

    @tool("get_run_summary", "Flow statistics of a run at a gauge (default Epinal 9116044) vs baseline.",
          {"run_id": str, "gauge": str})
    async def get_run_summary(args):
        try:
            return _txt(W.summarise_run(args["run_id"], args.get("gauge") or C.DEFAULT_GAUGE))
        except Exception as e:
            return _txt(f"ERROR: {e}")

    return create_sdk_mcp_server("hydro", tools=[list_parameters, run_wflow_scenario, get_run_summary])


def _guard_hook(ws: Path, rec: Recorder):
    wsr = ws.resolve()

    async def hook(input_data, tool_use_id, context):
        name = input_data.get("tool_name", "")
        tin = input_data.get("tool_input", {}) or {}
        reason = ""
        if name in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
            fp = Path(tin.get("file_path") or tin.get("notebook_path") or "")
            fp = fp if fp.is_absolute() else ws / fp
            if wsr != fp.resolve() and wsr not in fp.resolve().parents:
                reason = f"Writes are only allowed inside the workspace {ws.as_posix()}"
        elif name == "Bash":
            cmd = tin.get("command", "")
            if BLOCKED_SHELL.search(cmd):
                reason = "This shell command is blocked in the course sandbox."
            elif re.search(r"(>|\bmv\b|\bmove\b|\bdel\b|\brm\b)\s*[\"']?[^\s\"']*" + re.escape(C.BASE_MODEL_DIR.name),
                           cmd):
                reason = "The base model folder is read-only."
        if reason:
            rec.emit("error", f"blocked {name}", reason)
            return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                           "permissionDecisionReason": reason}}
        return {}

    return hook


async def _run(task: str, ws: Path, rec: Recorder, model: str | None, max_turns: int,
               allowed_tools: list[str], auth: str, extra_system: str = "") -> HarnessResult:
    from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions, ClaudeSDKClient, HookMatcher, ResultMessage,
                                  SystemMessage, TextBlock, ToolResultBlock, ToolUseBlock, UserMessage)

    s = C.get_settings()
    run_ids: list[str] = []
    skills_loaded: list[str] = []
    env = {}
    if auth == "login" or (auth == "auto" and not os.getenv("ANTHROPIC_API_KEY")):
        env["ANTHROPIC_API_KEY"] = ""  # fall back to the Claude Code login on this machine
    options = ClaudeAgentOptions(
        model=model or s.anthropic_model,
        cwd=str(ws),
        system_prompt={"type": "preset", "preset": "claude_code", "append": _system_prompt(ws) + extra_system},
        setting_sources=["project"],          # loads <ws>/.claude/skills/*
        skills=sorted(p.name for p in (ws / ".claude" / "skills").iterdir() if p.is_dir()),  # only course skills
        allowed_tools=allowed_tools,
        disallowed_tools=["PowerShell", "WebFetch", "WebSearch"],
        permission_mode="acceptEdits",
        mcp_servers={"hydro": _make_mcp_server(ws, rec, run_ids)},
        hooks={"PreToolUse": [HookMatcher(matcher=None, hooks=[_guard_hook(ws, rec)])]},
        max_turns=max_turns,
        add_dirs=[str(C.RUNS_DIR), str(C.BASE_MODEL_DIR)],
        env=env,
    )
    answer, status, err = "", "done", ""
    async with ClaudeSDKClient(options=options) as client:
        await client.query(task)
        async for msg in client.receive_response():
            if isinstance(msg, SystemMessage) and msg.subtype == "init":
                rec.emit("harness", "session start", f"model={msg.data.get('model')} "
                                                     f"skills={msg.data.get('skills')} tools={len(msg.data.get('tools', []))}")
            elif isinstance(msg, AssistantMessage):
                rec.llm_calls += 1
                u = msg.usage or {}
                rec.add_usage(u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0)
                              + u.get("cache_creation_input_tokens", 0), u.get("output_tokens", 0))
                for b in msg.content:
                    if isinstance(b, TextBlock) and b.text.strip():
                        rec.emit("llm", "Claude", b.text)
                    elif isinstance(b, ToolUseBlock):
                        rec.tool_calls += 1
                        if b.name == "Skill":
                            skills_loaded.append(str(b.input.get("skill") or b.input.get("name") or b.input))
                        rec.emit("tool_call", f"tool: {b.name}", b.input)
            elif isinstance(msg, UserMessage) and isinstance(msg.content, list):
                for b in msg.content:
                    if isinstance(b, ToolResultBlock):
                        c = b.content if isinstance(b.content, str) else json.dumps(b.content, default=str)
                        rec.emit("error" if b.is_error else "tool_result", "result", c)
            elif isinstance(msg, ResultMessage):
                rec.cost_usd += msg.total_cost_usd or 0.0
                answer = msg.result or ""
                if msg.is_error:
                    status, err = "error", answer or msg.subtype
    plots = sorted(str(p) for p in ws.rglob("*.png"))
    rec.emit("final" if status == "done" else "error", "final answer", answer or err)
    return HarnessResult(status, answer, str(ws), run_ids, plots, skills_loaded, rec.summary(), err)


def run_harness(task: str, *, skills: list[str] | None = None, model: str | None = None, max_turns: int = 40,
                allowed_tools: list[str] | None = None, on_event: Callable[[Event], None] | None = None,
                recorder: Recorder | None = None, workspace: Path | None = None,
                auth: str | None = None, extra_system: str = "") -> HarnessResult:
    """Run the Claude Agent SDK harness on a task (blocking). Works in scripts, Jupyter and Streamlit."""
    rec = recorder or Recorder(on_event)
    ws = workspace or prepare_workspace(skills)
    rec.emit("info", "harness workspace", str(ws))
    try:
        return run_sync(_run(task, ws, rec, model, max_turns, allowed_tools or DEFAULT_ALLOWED,
                             auth or C.get_settings().harness_auth, extra_system))
    except Exception as e:
        rec.emit("error", type(e).__name__, str(e))
        return HarnessResult("error", "", str(ws), usage=rec.summary(), error=f"{type(e).__name__}: {e}")
