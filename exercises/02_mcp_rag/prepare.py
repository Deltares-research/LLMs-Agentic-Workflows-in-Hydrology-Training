"""Run baseline Wflow and a twin 'observed' discharge series for Exercise 2.

The twin uses the same hidden multipliers as Exercise 3 (`hydroagent/twin.py`):
ksat_vertical=0.4, soil_thickness=1.6, ksat_horizontal_ratio=3.0.
Does not import hydroagent.

From the repo root:
    uv run python exercises/02_mcp_rag/prepare.py
    uv run python exercises/02_mcp_rag/prepare.py --force
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import tomlkit

EXERCISE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXERCISE_DIR.parents[1]
BASE_MODEL_DIR = REPO_ROOT / "wflow_moselle_epinal"
BASE_TOML = BASE_MODEL_DIR / "wflow_sbm.toml"
DATA_DIR = EXERCISE_DIR / "data"
WORK_DIR = EXERCISE_DIR / "output" / "_prepare"

DEFAULT_TRUTH = {"ksat_vertical": 0.4, "soil_thickness": 1.6, "ksat_horizontal_ratio": 3.0}
GAUGE = "9116044"
START = "2019-01-01T00:00:00"
END = "2020-12-31T00:00:00"

MAP_PARAMS = {
    "ksat_vertical": "soil_surface_water__vertical_saturated_hydraulic_conductivity",
    "soil_thickness": "soil__thickness",
}
SCALAR_PARAMS = {
    "ksat_horizontal_ratio": "subsurface_water__horizontal_to_vertical_saturated_hydraulic_conductivity_ratio",
}


def find_julia() -> str:
    exe = os.getenv("JULIA_EXE") or shutil.which("julia")
    if exe:
        return exe
    cand = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WindowsApps" / "julia.exe"
    if cand.is_file():
        return str(cand)
    raise FileNotFoundError("julia not on PATH. Install Julia (see hydrology-env skill).")


def _posix(p: Path | str) -> str:
    return Path(p).resolve().as_posix()


def write_run_toml(out_toml: Path, multipliers: dict[str, float]) -> None:
    doc = tomlkit.parse(BASE_TOML.read_text(encoding="utf-8"))
    doc["dir_input"] = _posix(BASE_MODEL_DIR)
    doc["dir_output"] = _posix(out_toml.parent / "output")
    doc["time"]["starttime"] = START
    doc["time"]["endtime"] = END
    doc["model"]["cold_start__flag"] = True
    static = doc["input"]["static"]
    for key, mult in multipliers.items():
        if key in MAP_PARAMS:
            std = MAP_PARAMS[key]
            entry = static.get(std)
            var = entry if isinstance(entry, str) else entry["netcdf_variable_name"]
            t = tomlkit.table()
            t.add("netcdf_variable_name", str(var))
            t.add("scale", float(mult))
            t.add("offset", 0.0)
            static[std] = t
        else:
            std = SCALAR_PARAMS[key]
            static[std]["value"] = float(static[std]["value"]) * float(mult)
    out_toml.parent.mkdir(parents=True, exist_ok=True)
    out_toml.write_text(tomlkit.dumps(doc), encoding="utf-8")


def run_wflow(toml_path: Path) -> None:
    log = toml_path.with_name("julia_stdout.log")
    cmd = [find_julia(), "-t", "4", "-e", "using Wflow; Wflow.run()", str(toml_path)]
    print(f"Wflow: {toml_path}")
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
    if proc.returncode != 0:
        tail = log.read_text(encoding="utf-8", errors="replace")[-1500:]
        raise RuntimeError(f"Wflow failed (exit {proc.returncode}):\n{tail}")
    csv = toml_path.parent / "output" / "output.csv"
    if not csv.is_file():
        raise FileNotFoundError(f"Wflow finished but {csv} is missing")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-run even if CSVs already exist")
    args = parser.parse_args()

    sim_csv = DATA_DIR / "wflow_output_sample.csv"
    obs_csv = DATA_DIR / "observed.csv"
    if sim_csv.is_file() and obs_csv.is_file() and not args.force:
        print(f"Already prepared:\n  {sim_csv}\n  {obs_csv}\nRe-run with --force to regenerate.")
        return 0

    if not BASE_TOML.is_file():
        print(
            f"Base model not found at {BASE_MODEL_DIR}.\n"
            "Build it first (HydroMT), or from Exercise 3:\n"
            "  uv run python exercises/03_agentic_workflows/cli.py prepare",
            file=sys.stderr,
        )
        return 1

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    baseline_dir = WORK_DIR / "baseline"
    twin_dir = WORK_DIR / "twin"
    write_run_toml(baseline_dir / "wflow_sbm.toml", {})
    write_run_toml(twin_dir / "wflow_sbm.toml", DEFAULT_TRUTH)
    run_wflow(baseline_dir / "wflow_sbm.toml")
    run_wflow(twin_dir / "wflow_sbm.toml")

    shutil.copy2(baseline_dir / "output" / "output.csv", sim_csv)
    twin = pd.read_csv(twin_dir / "output" / "output.csv", index_col=0, parse_dates=True)
    col = f"river_q_{GAUGE}"
    if col not in twin.columns:
        raise KeyError(f"{col} not in twin output. Available: {list(twin.columns)}")
    twin[col].rename("q_obs").to_frame().to_csv(obs_csv, index_label="time")
    print(f"Wrote {sim_csv}")
    print(f"Wrote {obs_csv}  (gauge {GAUGE})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
