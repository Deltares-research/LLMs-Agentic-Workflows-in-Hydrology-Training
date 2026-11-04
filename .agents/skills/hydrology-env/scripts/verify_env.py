#!/usr/bin/env python3
"""Verify the hydrology training stack from the repository root.

Stdlib only. Calls `uv run` and `julia` so checks hit the project env, not
whatever `python` happens to be on PATH.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
MIN_HYDROMT = (1, 4, 1)
MIN_HYDROMT_WFLOW = (1, 1, 0)
MIN_WFLOW_JL = (1, 0, 4)
REQUIRED_PLUGINS = ("wflow_sbm", "wflow_sediment")


def parse_version(text: str) -> tuple[int, ...] | None:
    match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", text)
    if not match:
        return None
    major, minor, patch = match.groups()
    return (int(major), int(minor), int(patch or 0))


def version_gte(got: tuple[int, ...], minimum: tuple[int, ...]) -> bool:
    got_p = got + (0,) * (len(minimum) - len(got))
    min_p = minimum + (0,) * (len(got) - len(minimum))
    return got_p >= min_p


def run(args: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def combined(proc: subprocess.CompletedProcess[str]) -> str:
    return f"{proc.stdout}\n{proc.stderr}".strip()


def check_which(name: str) -> tuple[bool, str]:
    path = shutil.which(name)
    if not path:
        return False, f"{name} not found on PATH"
    proc = run([name, "--version"])
    text = combined(proc)
    if proc.returncode != 0:
        return False, text or f"{name} --version failed"
    first = text.splitlines()[0] if text else name
    return True, first


def check_project_python() -> tuple[bool, str]:
    proc = run(["uv", "run", "python", "--version"])
    text = combined(proc)
    if proc.returncode != 0:
        return False, text or "uv run python --version failed"
    version = parse_version(text)
    if version is None:
        return False, text
    if version[0] != 3 or version[1] != 13:
        return False, f"{text.strip()} (need Python 3.13.x, not 3.14+)"
    return True, text.splitlines()[0]


def check_python_packages() -> tuple[bool, str]:
    code = (
        "import hydromt, hydromt_wflow; "
        "print(hydromt.__version__); "
        "print(hydromt_wflow.__version__)"
    )
    proc = run(["uv", "run", "python", "-c", code])
    text = combined(proc)
    if proc.returncode != 0:
        return False, text or "failed to import hydromt / hydromt_wflow"
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        return False, text
    hydromt_v = parse_version(lines[0])
    plugin_v = parse_version(lines[1])
    if hydromt_v is None or plugin_v is None:
        return False, text
    problems = []
    if not version_gte(hydromt_v, MIN_HYDROMT):
        problems.append(f"hydromt {lines[0]} < {'.'.join(map(str, MIN_HYDROMT))}")
    if not version_gte(plugin_v, MIN_HYDROMT_WFLOW):
        problems.append(
            f"hydromt_wflow {lines[1]} < {'.'.join(map(str, MIN_HYDROMT_WFLOW))}"
        )
    detail = f"hydromt {lines[0]}, hydromt_wflow {lines[1]}"
    if problems:
        return False, f"{detail} ({'; '.join(problems)})"
    return True, detail


def check_plugins() -> tuple[bool, str]:
    proc = run(["uv", "run", "hydromt", "--plugins"], timeout=180)
    text = combined(proc)
    if proc.returncode != 0:
        return False, text or "uv run hydromt --plugins failed"
    missing = [name for name in REQUIRED_PLUGINS if name not in text]
    if missing:
        return False, f"missing plugins: {', '.join(missing)}"
    return True, ", ".join(REQUIRED_PLUGINS)


def check_wflow_jl() -> tuple[bool, str]:
    proc = run(
        ["julia", "-e", "using Wflow; println(pkgversion(Wflow))"],
        timeout=180,
    )
    text = combined(proc)
    if proc.returncode != 0:
        return False, text or "julia using Wflow failed"
    version = parse_version(proc.stdout)
    if version is None:
        return False, text
    if version[0] != 1 or not version_gte(version, MIN_WFLOW_JL):
        return False, (
            f"Wflow.jl {proc.stdout.strip()} "
            f"(need 1.x >= {'.'.join(map(str, MIN_WFLOW_JL))}; 0.8 is not compatible)"
        )
    return True, f"Wflow.jl {proc.stdout.strip()}"


def main() -> int:
    if not (REPO_ROOT / "pyproject.toml").is_file():
        print(f"Could not find repo root (looked at {REPO_ROOT})", file=sys.stderr)
        return 2

    checks = [
        ("Git", lambda: check_which("git")),
        ("uv", lambda: check_which("uv")),
        ("Julia", lambda: check_which("julia")),
        ("Python 3.13 (uv run)", check_project_python),
        ("hydromt + hydromt_wflow", check_python_packages),
        ("hydromt wflow plugins", check_plugins),
        ("Wflow.jl 1.x", check_wflow_jl),
    ]

    rows: list[tuple[str, str, str]] = []
    all_ok = True
    for name, fn in checks:
        try:
            ok, detail = fn()
        except FileNotFoundError as exc:
            ok, detail = False, str(exc)
        except subprocess.TimeoutExpired:
            ok, detail = False, "timed out"
        status = "OK" if ok else "FAIL"
        all_ok = all_ok and ok
        rows.append((status, name, detail.replace("\n", " | ")))

    name_width = max(len(name) for _, name, _ in rows)
    print(f"{'STATUS':<6}  {'CHECK':<{name_width}}  DETAIL")
    for status, name, detail in rows:
        print(f"{status:<6}  {name:<{name_width}}  {detail}")

    if all_ok:
        print("\nAll checks passed.")
        return 0
    print("\nOne or more checks failed. See FAIL rows above.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
