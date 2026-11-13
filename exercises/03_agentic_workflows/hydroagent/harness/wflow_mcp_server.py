"""Standalone MCP server (stdio) with the Wflow tools, for harnesses that run outside this Python process.

The Claude harness (A4) gets these tools in-process; Mistral Vibe (A4m) starts this server as a subprocess:

    python -m hydroagent.harness.wflow_mcp_server --workspace <dir>

Same three tools as A4: list_parameters, run_wflow_scenario, get_run_summary. Edited NetCDF files must lie inside
the workspace. stdout is the MCP channel, so nothing else may print there: Wflow console output goes to
``<workspace>/wflow_console.log`` and each run's timing to ``<workspace>/wflow_calls.jsonl`` (read by the runner
for the live trace and the time split).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

EXERCISE_DIR = Path(__file__).resolve().parents[2]
if str(EXERCISE_DIR) not in sys.path:
    sys.path.insert(0, str(EXERCISE_DIR))

from mcp.server.mcpserver import MCPServer  # noqa: E402

from hydroagent import config as C  # noqa: E402
from hydroagent import wflow_tools as W  # noqa: E402


def _inside(ws: Path, p: Path) -> bool:
    return ws.resolve() in p.resolve().parents


def build_server(ws: Path) -> MCPServer:
    console = ws / "wflow_console.log"
    calls = ws / "wflow_calls.jsonl"

    def sink(source: str, line: str) -> None:
        with console.open("a", encoding="utf-8") as f:
            f.write(f"[{source}] {line}\n")

    W.LOG_SINK.set(sink)
    server = MCPServer("hydro", instructions="Wflow model of the upper Moselle (Epinal). Run scenarios here.")

    @server.tool()
    def list_parameters() -> str:
        """List Wflow parameters that can be multiplied, with allowed ranges."""
        return json.dumps(W.list_parameters(), indent=1, default=str)

    @server.tool()
    def run_wflow_scenario(label: str, multipliers: dict[str, float] | None = None,
                           staticmaps_path: str | None = None, static_overrides: dict[str, str] | None = None,
                           forcing_path: str | None = None) -> str:
        """Run a Wflow scenario (~15-80 s) and return run_id + flow statistics and % change vs baseline.

        label: short name. multipliers: {parameter: factor}, may be empty. staticmaps_path: path to an edited copy of
        staticmaps.nc INSIDE the workspace. static_overrides: {wflow_standard_name: netcdf_variable_name} so a
        parameter is read from a (new) map in that file. forcing_path: path to an edited copy of
        inmaps-eobs-2019-2020.nc INSIDE the workspace (same grid/time/variables precip, pet, temp).
        """
        paths = {}
        for key, val in (("staticmaps_path", staticmaps_path), ("forcing_path", forcing_path)):
            if val:
                p = Path(val) if Path(val).is_absolute() else ws / val
                if not _inside(ws, p):
                    return f"ERROR: {key} must be inside the workspace {ws.as_posix()}"
                if not p.is_file():
                    return f"ERROR: {key} not found: {p.as_posix()}"
                paths[key] = p
        t0 = time.time()
        try:
            with contextlib.redirect_stdout(sys.stderr):  # keep the MCP channel clean
                W.ensure_baseline()
                info = W.create_run(multipliers or {}, label, static_override=paths.get("staticmaps_path"),
                                    static_map_overrides=static_overrides or None,
                                    forcing_override=paths.get("forcing_path"))
                info = W.execute_run(info.run_id)
                summary = W.summarise_run(info.run_id)
        except Exception as e:  # noqa: BLE001 - report to the agent, do not crash the server
            return f"ERROR: {type(e).__name__}: {e}"
        with calls.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"run_id": info.run_id, "seconds": round(time.time() - t0, 1)}) + "\n")
        from hydroagent.harness.vibe_runner import copy_run_output

        if copy_run_output(info.run_id, ws):  # the agent can only read inside its workspace
            summary = {**summary, "output_files": f"runs/{info.run_id}/output.csv, runs/{info.run_id}/output.nc"}
        return json.dumps(summary, indent=1, default=str)

    @server.tool()
    def get_run_summary(run_id: str, gauge: str = C.DEFAULT_GAUGE) -> str:
        """Flow statistics of a run at a gauge (default Epinal 9116044) compared with the baseline."""
        try:
            return json.dumps(W.summarise_run(run_id, gauge or C.DEFAULT_GAUGE), indent=1, default=str)
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {type(e).__name__}: {e}"

    return server


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workspace", required=True, type=Path)
    args = ap.parse_args()
    build_server(args.workspace.resolve()).run("stdio")


if __name__ == "__main__":
    main()
