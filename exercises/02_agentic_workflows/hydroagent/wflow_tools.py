"""Plain-Python HydroMT/Wflow operations used by every agent architecture.

Nothing in this module knows about LLMs. Agents reach it through the thin tool wrappers in
``tools.py`` (LangGraph) or the MCP tools in ``harness/claude_runner.py`` (Claude Agent SDK).

Design rules (also good agent-design rules):
* the base model in ``wflow_moselle_epinal/`` is never modified;
* every scenario is a new run folder containing only a TOML (+ outputs); it points to the
  shared static maps / forcing with absolute paths;
* parameter changes are *multipliers* with bounds, so simple agents cannot produce nonsense.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tomlkit
import yaml

from . import config as C
from .tracing import CURRENT_RECORDER

# --------------------------------------------------------------------------------------
# Parameter catalogue: what simple agents are allowed to change
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Param:
    key: str            # short name used by agents
    std_name: str       # Wflow 1.x standard name in [input.static]
    kind: str           # "map" (scale a staticmaps variable) or "scalar" (multiply a constant value)
    description: str
    unit: str
    bounds: tuple[float, float]  # allowed multiplier range


PARAMETERS: dict[str, Param] = {p.key: p for p in [
    Param("ksat_vertical", "soil_surface_water__vertical_saturated_hydraulic_conductivity", "map",
          "Vertical saturated hydraulic conductivity at the soil surface (KsatVer). Lower = less infiltration, "
          "more fast runoff.", "mm/day", (0.1, 10.0)),
    Param("ksat_decay_f", "soil_water__vertical_saturated_hydraulic_conductivity_scale_parameter", "map",
          "Exponential decay of KsatVer with depth (f). Higher = Ksat drops faster with depth.", "1/mm", (0.3, 3.0)),
    Param("ksat_horizontal_ratio", "subsurface_water__horizontal_to_vertical_saturated_hydraulic_conductivity_ratio",
          "scalar", "Ratio horizontal/vertical Ksat (KsatHorFrac). Higher = more lateral subsurface flow, "
          "smoother hydrograph.", "-", (0.1, 10.0)),
    Param("soil_thickness", "soil__thickness", "map",
          "Total soil thickness. Thicker = more storage, lower peaks, more baseflow.", "mm", (0.3, 3.0)),
    Param("theta_s", "soil_water__saturated_volume_fraction", "map",
          "Porosity (saturated water content).", "-", (0.7, 1.3)),
    Param("root_depth", "vegetation_root__depth", "map",
          "Rooting depth. Deeper = more transpiration from deeper layers.", "mm", (0.3, 3.0)),
    Param("crop_factor", "vegetation__crop_factor", "map",
          "Crop factor scaling potential evapotranspiration.", "-", (0.5, 1.5)),
    Param("canopy_evap_ratio", "vegetation_canopy_water__mean_evaporation_to_mean_precipitation_ratio", "scalar",
          "Gash interception: mean evaporation / mean precipitation ratio.", "-", (0.2, 3.0)),
    Param("compacted_infiltration", "compacted_soil_surface_water__infiltration_capacity", "scalar",
          "Infiltration capacity of compacted (urban/paved) soil.", "mm/day", (0.1, 5.0)),
    Param("degree_day", "snowpack__degree_day_coefficient", "scalar",
          "Snow melt degree-day factor. Higher = faster snow melt.", "mm/degC/day", (0.3, 3.0)),
    Param("river_manning_n", "river_water_flow__manning_n_parameter", "map",
          "River-bed roughness. Higher = slower, attenuated flood waves.", "s/m^(1/3)", (0.3, 3.0)),
    Param("land_manning_n", "land_surface_water_flow__manning_n_parameter", "map",
          "Overland-flow roughness.", "s/m^(1/3)", (0.3, 3.0)),
]}


def list_parameters() -> list[dict]:
    return [{"key": p.key, "kind": p.kind, "unit": p.unit, "allowed_multiplier": list(p.bounds),
             "description": p.description} for p in PARAMETERS.values()]


def validate_multipliers(multipliers: dict[str, float]) -> dict[str, float]:
    clean: dict[str, float] = {}
    for k, v in (multipliers or {}).items():
        if k not in PARAMETERS:
            raise ValueError(f"Unknown parameter {k!r}. Valid keys: {sorted(PARAMETERS)}")
        v = float(v)
        lo, hi = PARAMETERS[k].bounds
        if not lo <= v <= hi:
            raise ValueError(f"Multiplier for {k} must be within [{lo}, {hi}], got {v}.")
        clean[k] = v
    return clean


# --------------------------------------------------------------------------------------
# Julia / Wflow execution
# --------------------------------------------------------------------------------------


def find_julia() -> str:
    exe = os.getenv("JULIA_EXE") or shutil.which("julia")
    if not exe and os.name == "nt":
        cand = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WindowsApps" / "julia.exe"
        if cand.exists():
            exe = str(cand)
    if not exe:
        raise RuntimeError("Julia not found. Install it via the hydrology-env skill or set JULIA_EXE.")
    return exe


class ModelLog:
    """Forwards console lines of HydroMT / Wflow to the current trace (``LOG_SINK``), throttling progress bars.

    ``LOG_SINK`` is a context variable holding ``callable(source, line)``. Sessions set it, so model output
    shows up in the same live trace as LLM and tool calls (Streamlit, notebook, CLI)."""

    _ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
    _PCT = re.compile(r"Progress:\s+(\d+)%")

    def __init__(self, source: str, sink=None, step: int = 10):
        self.source = source
        self.sink = sink if sink is not None else LOG_SINK.get()
        self.step = step
        self.last_pct = -step
        self.buffer: list[str] = []

    def __call__(self, raw: str) -> None:
        line = self._ANSI.sub("", raw).rstrip()
        if not line.strip():
            return
        self.buffer.append(line)
        if not self.sink:
            return
        # Julia logging prints multi-line blocks: "┌ Info: ...", "│ ...", "└ ...". Join them into one line.
        head = line[:1]
        if head == "┌":
            self._block = [line[1:].strip()]
            return
        if head in ("│", "└") and getattr(self, "_block", None) is not None:
            part = line[1:].strip()
            if not part.startswith("@ "):  # drop source-location lines
                self._block.append(part)
            if head == "└":
                line, self._block = " ".join(self._block), None
            else:
                return
        elif head == "└":  # stray source-location line
            return
        if line.startswith("[ "):
            line = line[2:]
        m = self._PCT.search(line)
        if m:  # progress bar: only every `step` %
            pct = int(m.group(1))
            if pct < self.last_pct:  # a new run started on the warm server
                self.last_pct = -self.step
            if pct == self.last_pct or (pct < 100 and pct - self.last_pct < self.step):
                return
            self.last_pct = pct
            line = f"progress {pct}%"
        elif self.source == "hydromt":
            if " - DEBUG - " in line or line.startswith(("Error in sys.excepthook", "Original exception was")):
                return  # debug chatter / harmless interpreter-shutdown noise
            line = re.sub(r"^\S+ \S+ - [\w.]+ - \w+ - (\w+) - ", r"\1 ", line)  # drop timestamp/logger
        try:
            self.sink(self.source, line[:400])
        except Exception:
            pass


LOG_SINK: ContextVar = ContextVar("LOG_SINK", default=None)


class WflowServer:
    """Persistent Julia process running ``wflow_server.jl``. Thread-safe (one run at a time)."""

    def __init__(self, threads: int = 4):
        self.threads = threads
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()
        self.version: str | None = None
        self.log_path = C.OUTPUT_DIR / "wflow_server.log"
        self._current: ModelLog | None = None  # receives Julia stderr (Wflow logging + progress) while a run is active

    def _pump_stderr(self) -> None:
        for ln in self.proc.stderr:  # type: ignore[union-attr]
            self._log.write(ln)
            self._log.flush()
            cur = self._current
            if cur is not None:
                cur(ln)

    def _start(self) -> None:
        C.ensure_dirs()
        script = Path(__file__).with_name("wflow_server.jl")
        self._log = open(self.log_path, "a", encoding="utf-8")
        self.proc = subprocess.Popen(
            [find_julia(), "-t", str(self.threads), str(script)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        threading.Thread(target=self._pump_stderr, daemon=True).start()
        sink = LOG_SINK.get()
        if sink:
            sink("wflow", "starting Julia + loading Wflow.jl (first time ~1 min) ...")
        line = self._read_marker(timeout=900)
        if not line.startswith("@@WFLOW_READY@@"):
            raise RuntimeError(f"Wflow server failed to start: {line}. See {self.log_path}")
        self.version = line.split()[-1]
        self._warm_up()

    def _warm_up(self) -> None:
        """Julia compiles Wflow.run on its first call (~1 min). Do that now with a 2-day run, so the
        compile time is counted as start-up and not charged to whichever agent happens to run first."""
        if not C.BASE_TOML.exists():
            return
        sink = LOG_SINK.get()
        if sink:
            sink("wflow", "warm-up: compiling Wflow with a 2-day run ...")
        work = C.OUTPUT_DIR / "_warmup"
        work.mkdir(parents=True, exist_ok=True)
        write_scenario_toml(C.BASE_TOML, work / "wflow_sbm.toml", {}, C.BASE_MODEL_DIR,
                            C.DEFAULT_START, "2019-01-03T00:00:00")
        self.proc.stdin.write(str((work / "wflow_sbm.toml").resolve()) + "\n")  # type: ignore[union-attr]
        self.proc.stdin.flush()  # type: ignore[union-attr]
        self._read_marker(timeout=900)

    def _read_marker(self, timeout: float) -> str:
        assert self.proc and self.proc.stdout
        result: list[str] = []

        def reader():
            for ln in self.proc.stdout:  # type: ignore[union-attr]
                if ln.startswith("@@WFLOW_"):
                    result.append(ln.strip())
                    return
            result.append("@@WFLOW_ERROR@@ Julia process exited")

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        t.join(timeout)
        if not result:
            self.stop()
            raise TimeoutError(f"Wflow did not respond within {timeout} s")
        return result[0]

    def run(self, toml_path: Path, timeout: float = 900) -> float:
        """Run one TOML; returns the seconds spent starting Julia (0 when the server was already warm)."""
        startup = 0.0
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                t0 = time.time()
                self._start()
                startup = time.time() - t0
            assert self.proc and self.proc.stdin
            self._current = ModelLog("wflow")
            try:
                self.proc.stdin.write(str(Path(toml_path).resolve()) + "\n")
                self.proc.stdin.flush()
                line = self._read_marker(timeout)
            finally:
                self._current = None
            if not line.startswith("@@WFLOW_DONE@@"):
                raise RuntimeError(line.replace("@@WFLOW_ERROR@@", "Wflow error:").strip())
        return startup

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.write("EXIT\n")  # type: ignore[union-attr]
                self.proc.stdin.flush()  # type: ignore[union-attr]
                self.proc.wait(10)
            except Exception:
                self.proc.kill()
        self.proc = None


_SERVER: WflowServer | None = None


def get_server() -> WflowServer:
    global _SERVER
    if _SERVER is None:
        _SERVER = WflowServer(threads=int(os.getenv("WFLOW_THREADS", "4")))
        atexit.register(_SERVER.stop)
    return _SERVER


def run_wflow_toml(toml_path: Path, mode: str | None = None, timeout: float = 900) -> float:
    """Run Wflow for a TOML. ``mode``: "server" (default, persistent Julia) or "subprocess"."""
    mode = mode or os.getenv("WFLOW_MODE", "server")
    t0 = time.time()
    startup = 0.0
    try:
        if mode == "server":
            startup = get_server().run(Path(toml_path), timeout=timeout)
        else:
            log = Path(toml_path).with_name("julia_stdout.log")
            rc = _stream([find_julia(), "-t", "4", "-e", "using Wflow; Wflow.run()", str(toml_path)], log,
                         ModelLog("wflow"), timeout=timeout)
            if rc != 0:
                tail = log.read_text(encoding="utf-8", errors="replace")[-1500:]
                raise RuntimeError(f"Wflow failed (exit {rc}):\n{tail}")
    finally:
        rec = CURRENT_RECORDER.get()
        if rec is not None:
            rec.add_model_time(time.time() - t0 - startup)
            rec.add_startup(startup)  # one-off Julia start: reported, but not charged to the architecture
    return time.time() - t0


def _stream(cmd: list[str], log: Path, forward: ModelLog, cwd: Path | None = None, timeout: float = 3600) -> int:
    """Run a command, writing its combined output to ``log`` and forwarding each line to the trace."""
    with open(log, "w", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", bufsize=1)
        t0 = time.time()
        for ln in proc.stdout:  # type: ignore[union-attr]
            fh.write(ln)
            forward(ln)
            if time.time() - t0 > timeout:
                proc.kill()
                break
        return proc.wait()


# --------------------------------------------------------------------------------------
# Base model (HydroMT build) and model variants (edited HydroMT YAML)
# --------------------------------------------------------------------------------------


def _hydromt(args: list[str], log: Path) -> None:
    exe = shutil.which("hydromt") or str(Path(os.sys.executable).with_name("hydromt"))
    fwd = ModelLog("hydromt")
    if fwd.sink:
        fwd.sink("hydromt", f"hydromt {' '.join(args[:2])} ...")
    t0 = time.time()
    rc = _stream([exe, *args], log, fwd, cwd=C.REPO_ROOT)
    rec = CURRENT_RECORDER.get()
    if rec is not None:
        rec.add_model_time(time.time() - t0, runs=0)
    if rc != 0:
        raise RuntimeError(f"hydromt {' '.join(args[:2])} failed; see {log}")


def base_model_ready(model_dir: Path = C.BASE_MODEL_DIR) -> bool:
    return (model_dir / "staticmaps.nc").exists() and any(model_dir.glob("inmaps-*.nc"))


def prepare_base_model(force: bool = False, model_dir: Path = C.BASE_MODEL_DIR,
                       build_yml: Path = C.HYDROMT_BUILD_YML) -> Path:
    """Build the Moselle model with HydroMT (once) and add E-OBS forcing. Returns the model dir."""
    if base_model_ready(model_dir) and not force:
        return model_dir
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    _hydromt(["build", "wflow_sbm", str(model_dir), "-i", str(build_yml), "-d", str(C.DATA_CATALOG), "-v"],
             model_dir.parent / f"{model_dir.name}_build.log")
    _hydromt(["update", "wflow_sbm", str(model_dir), "-i", str(C.HYDROMT_FORCING_YML), "-d",
              str(C.DATA_CATALOG), "-v"], model_dir.parent / f"{model_dir.name}_forcing.log")
    return model_dir


HYDROMT_OPTIONS = {
    # option -> (step, key, allowed values). Small, safe surface for agents to edit the build YAML.
    "ptf_ksatver": ("setup_soilmaps", "ptf_ksatver", ["brakensiek", "cosby"]),
    "river_upa": ("setup_rivers", "river_upa", [10, 30, 100]),
    "rivdph_method": ("setup_rivers", "rivdph_method", ["powlaw", "manning"]),
}


def build_model_variant(name: str, options: dict) -> Path:
    """Copy the HydroMT build YAML, change whitelisted options, and build a new model variant."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", name):
        raise ValueError("Variant name must be 1-40 chars of letters, digits, _ or -.")
    cfg = yaml.safe_load(C.HYDROMT_BUILD_YML.read_text())
    for opt, value in options.items():
        if opt not in HYDROMT_OPTIONS:
            raise ValueError(f"Option {opt!r} not editable. Allowed: {list(HYDROMT_OPTIONS)}")
        step, key, allowed = HYDROMT_OPTIONS[opt]
        if value not in allowed:
            raise ValueError(f"{opt} must be one of {allowed}")
        for s in cfg["steps"]:
            if step in s:
                s[step][key] = value
    C.ensure_dirs()
    out = C.VARIANTS_DIR / name
    yml = C.VARIANTS_DIR / f"{name}.yml"
    yml.write_text(yaml.safe_dump(cfg, sort_keys=False))
    if out.exists():
        shutil.rmtree(out)
    return prepare_base_model(force=True, model_dir=out, build_yml=yml)


# --------------------------------------------------------------------------------------
# Scenario runs
# --------------------------------------------------------------------------------------


@dataclass
class RunInfo:
    run_id: str
    label: str
    multipliers: dict[str, float]
    model_dir: str
    start: str
    end: str
    created: str
    status: str = "created"
    runtime_s: float | None = None
    error: str | None = None
    static_override: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return C.RUNS_DIR / self.run_id


def _posix(p: Path | str) -> str:
    return Path(p).resolve().as_posix()


def write_scenario_toml(base_toml: Path, out_toml: Path, multipliers: dict[str, float], model_dir: Path,
                        start: str, end: str, static_override: Path | None = None,
                        static_map_overrides: dict[str, str] | None = None) -> None:
    doc = tomlkit.parse(base_toml.read_text(encoding="utf-8"))
    doc["dir_input"] = _posix(model_dir)
    doc["dir_output"] = _posix(out_toml.parent / "output")
    doc["time"]["starttime"] = start
    doc["time"]["endtime"] = end
    doc["model"]["cold_start__flag"] = True
    if static_override:
        doc["input"]["path_static"] = _posix(static_override)
    static = doc["input"]["static"]
    for std_name, var in (static_map_overrides or {}).items():
        # read this parameter from a NetCDF variable (e.g. a constant turned into a spatial map)
        if std_name not in static:
            raise ValueError(f"{std_name!r} is not an [input.static] parameter of this model")
        if not static_override:
            raise ValueError("static_map_overrides needs an edited staticmaps file (staticmaps_path)")
        static[std_name] = str(var)
    for key, mult in multipliers.items():
        p = PARAMETERS[key]
        entry = static.get(p.std_name)
        if p.kind == "map":
            # Wflow 1.x: [input.static.<std_name>] netcdf_variable_name = "...", scale = x, offset = y
            var = entry if isinstance(entry, str) else entry["netcdf_variable_name"]
            t = tomlkit.table()
            t.add("netcdf_variable_name", str(var))
            t.add("scale", float(mult))
            t.add("offset", 0.0)
            static[p.std_name] = t
        else:
            static[p.std_name]["value"] = float(static[p.std_name]["value"]) * float(mult)
    out_toml.write_text(tomlkit.dumps(doc), encoding="utf-8")


def _save_info(info: RunInfo) -> None:
    (info.path / "run.json").write_text(json.dumps(asdict(info), indent=2))


def load_run(run_id: str) -> RunInfo:
    f = C.RUNS_DIR / run_id / "run.json"
    if not f.exists():
        raise FileNotFoundError(f"No run {run_id!r}. Existing: {[r.run_id for r in list_runs()][-10:]}")
    return RunInfo(**json.loads(f.read_text()))


def list_runs() -> list[RunInfo]:
    if not C.RUNS_DIR.exists():
        return []
    runs = [RunInfo(**json.loads(f.read_text())) for f in C.RUNS_DIR.glob("*/run.json")]
    return sorted(runs, key=lambda r: r.created)


def create_run(multipliers: dict[str, float] | None = None, label: str = "", start: str = C.DEFAULT_START,
               end: str = C.DEFAULT_END, model_dir: Path | str = C.BASE_MODEL_DIR,
               static_override: Path | str | None = None, run_id: str | None = None,
               static_map_overrides: dict[str, str] | None = None) -> RunInfo:
    mult = validate_multipliers(multipliers or {})
    model_dir = Path(model_dir)
    if not base_model_ready(model_dir):
        raise RuntimeError(f"Model not built in {model_dir}. Run `cli.py prepare` first.")
    C.ensure_dirs()
    slug = re.sub(r"[^A-Za-z0-9]+", "-", label or "run").strip("-").lower()[:30] or "run"
    run_id = run_id or f"{datetime.now():%m%d-%H%M%S}-{slug}-{uuid.uuid4().hex[:4]}"
    info = RunInfo(run_id=run_id, label=label or slug, multipliers=mult, model_dir=_posix(model_dir), start=start,
                   end=end, created=datetime.now().isoformat(timespec="seconds"),
                   static_override=_posix(static_override) if static_override else None,
                   extra={"static_map_overrides": static_map_overrides} if static_map_overrides else {})
    info.path.mkdir(parents=True, exist_ok=True)
    write_scenario_toml(model_dir / "wflow_sbm.toml", info.path / "wflow_sbm.toml", mult, model_dir, start, end,
                        Path(static_override) if static_override else None, static_map_overrides)
    _save_info(info)
    return info


def execute_run(run_id: str, mode: str | None = None) -> RunInfo:
    info = load_run(run_id)
    if info.status == "done" and (info.path / "output" / "output.csv").exists():
        return info
    info.status = "running"
    _save_info(info)
    try:
        info.runtime_s = round(run_wflow_toml(info.path / "wflow_sbm.toml", mode=mode), 1)
        info.status, info.error = "done", None
    except Exception as e:  # report errors back to the agent instead of crashing
        info.status, info.error = "failed", str(e)[:2000]
    _save_info(info)
    return info


def run_scenario(multipliers: dict[str, float] | None = None, label: str = "", **kw) -> RunInfo:
    info = create_run(multipliers, label, **kw)
    return execute_run(info.run_id)


def ensure_baseline() -> RunInfo:
    """A shared, cached baseline run (no multipliers) that every scenario is compared to."""
    try:
        info = load_run("baseline")
        if info.status == "done":
            return info
    except FileNotFoundError:
        info = create_run({}, "baseline", run_id="baseline")
    return execute_run(info.run_id)


# --------------------------------------------------------------------------------------
# Results, metrics, plots
# --------------------------------------------------------------------------------------


def read_discharge(run: str | Path, gauge: str = C.DEFAULT_GAUGE) -> pd.Series:
    """Daily discharge [m3/s] at a GRDC gauge. ``run`` is a run_id or a path to an output.csv."""
    csv = Path(run) if str(run).endswith(".csv") else C.RUNS_DIR / str(run) / "output" / "output.csv"
    if not csv.exists():
        raise FileNotFoundError(f"{csv} not found (run not executed or failed?)")
    df = pd.read_csv(csv, index_col=0, parse_dates=True)
    col = f"river_q_{gauge}"
    if col not in df:
        raise KeyError(f"Gauge {gauge} not in output. Available: {list(C.GAUGES)}")
    s = df[col]
    s.name = str(run)
    return s


def _eval_period(s: pd.Series) -> pd.Series:
    return s[s.index > pd.Timestamp(C.SPINUP_END)]


def flow_stats(q: pd.Series) -> dict:
    q = _eval_period(q)
    return {"mean": float(q.mean()), "max": float(q.max()), "q95_high": float(q.quantile(0.95)),
            "q10_low": float(q.quantile(0.10)), "date_of_max": str(q.idxmax().date())}


def skill(sim: pd.Series, obs: pd.Series) -> dict:
    """KGE, NSE and percent bias of ``sim`` against ``obs`` (after spin-up)."""
    s, o = _eval_period(sim).align(_eval_period(obs), join="inner")
    r = float(np.corrcoef(s, o)[0, 1])
    alpha = float(s.std() / o.std())
    beta = float(s.mean() / o.mean())
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    nse = 1 - float(((s - o) ** 2).sum() / ((o - o.mean()) ** 2).sum())
    return {"kge": round(float(kge), 4), "nse": round(nse, 4), "pbias_pct": round(100 * (beta - 1), 2),
            "r": round(r, 4), "alpha": round(alpha, 4), "beta": round(beta, 4)}


def summarise_run(run_id: str, gauge: str = C.DEFAULT_GAUGE, reference: str = "baseline") -> dict:
    """Flow statistics of a run and their % change vs a reference run (default: baseline)."""
    info = load_run(run_id)
    out: dict = {"run_id": run_id, "label": info.label, "multipliers": info.multipliers, "status": info.status,
                 "gauge": f"{gauge} - {C.GAUGES.get(gauge, '')}"}
    if info.status != "done":
        out["error"] = info.error
        return out
    q = read_discharge(run_id, gauge)
    st = flow_stats(q)
    out["stats_m3s"] = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in st.items()}
    if reference and reference != run_id:
        ref_info = ensure_baseline() if reference == "baseline" else load_run(reference)
        ref = flow_stats(read_discharge(ref_info.run_id, gauge))
        out[f"pct_change_vs_{reference}"] = {k: round(100 * (st[k] / ref[k] - 1), 1)
                                             for k in ("mean", "max", "q95_high", "q10_low") if ref[k]}
    return out


def plot_runs(run_ids: list[str], gauge: str = C.DEFAULT_GAUGE, obs: pd.Series | None = None,
              title: str | None = None, out_path: Path | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    if obs is not None:
        ax.plot(obs.index, obs.values, color="black", lw=1.4, label="observed (twin truth)")
    for rid in run_ids:
        try:
            q = read_discharge(rid, gauge)
        except FileNotFoundError:
            continue
        lbl = load_run(rid).label
        ax.plot(q.index, q.values, lw=1.0, label=lbl, alpha=0.9)
    ax.set_ylabel("Q [m$^3$/s]")
    ax.set_title(title or f"Discharge at {C.GAUGES.get(gauge, gauge)}")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    C.ensure_dirs()
    out_path = out_path or C.OUTPUT_DIR / "plots" / f"hydrograph_{datetime.now():%H%M%S}_{uuid.uuid4().hex[:4]}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
