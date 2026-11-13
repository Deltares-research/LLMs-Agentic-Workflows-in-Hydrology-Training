"""LangChain tools: thin, JSON-returning wrappers around ``wflow_tools`` for LangGraph agents.

Tool design matters more than architecture: clear names, docstrings that say *when* to use the
tool, bounded inputs, and errors returned as text so the agent can recover.
"""

from __future__ import annotations

import json
from contextvars import ContextVar

from langchain_core.tools import tool

from . import config as C
from . import twin
from . import wflow_tools as W

# Collected per session so the UI can show the runs/plots an agent produced.
ARTIFACTS: ContextVar[dict | None] = ContextVar("ARTIFACTS", default=None)


def _record(kind: str, value: str) -> None:
    a = ARTIFACTS.get()
    if a is not None and value not in a.setdefault(kind, []):
        a[kind].append(value)


def _j(x) -> str:
    return json.dumps(x, indent=1, default=str)


@tool
def list_parameters() -> str:
    """List the Wflow parameters you may change, with their allowed multiplier range and meaning.
    Call this first if you are unsure which parameter key represents a physical change."""
    return _j(W.list_parameters())


@tool
def list_gauges() -> str:
    """List the discharge gauges (GRDC ids) available in model output. Default gauge is Epinal (outlet)."""
    return _j(C.GAUGES)


@tool
def run_scenario(multipliers: dict[str, float], label: str, model_variant: str = "") -> str:
    """Create AND run a Wflow scenario for the upper Moselle (2019-2020, daily).

    multipliers: {parameter_key: multiplier}, e.g. {"ksat_vertical": 0.7} = 30% less permeable soil.
                 Use {} for an unchanged run. Keys and ranges come from list_parameters.
    label: short human-readable name, e.g. "ksat -30%".
    model_variant: optional name of a model built with build_model_variant ("" = base model).
    Takes ~15-80 s. Returns run_id, flow statistics at Epinal and % change vs the baseline run."""
    try:
        W.ensure_baseline()
        model_dir = C.VARIANTS_DIR / model_variant if model_variant else C.BASE_MODEL_DIR
        info = W.run_scenario(multipliers, label, model_dir=model_dir)
        _record("run_ids", info.run_id)
        return _j(W.summarise_run(info.run_id))
    except Exception as e:
        return f"ERROR: {e}"


@tool
def get_run_summary(run_id: str, gauge: str = C.DEFAULT_GAUGE) -> str:
    """Flow statistics (mean, max, high/low flow quantiles) of an existing run at a gauge, and % change vs baseline."""
    try:
        return _j(W.summarise_run(run_id, gauge))
    except Exception as e:
        return f"ERROR: {e}"


@tool
def compare_runs(run_ids: list[str], gauge: str = C.DEFAULT_GAUGE) -> str:
    """Compare several runs side by side (flow statistics at one gauge). Include "baseline" to see the reference."""
    rows = []
    for rid in run_ids:
        try:
            s = W.summarise_run(rid, gauge, reference="")
            rows.append({"run_id": rid, "label": s["label"], "multipliers": s["multipliers"], **s.get("stats_m3s", {})})
        except Exception as e:
            rows.append({"run_id": rid, "error": str(e)})
    return _j(rows)


@tool
def plot_hydrographs(run_ids: list[str], gauge: str = C.DEFAULT_GAUGE, include_observed: bool = False) -> str:
    """Plot discharge time series of runs (and optionally the observed series) to a PNG. Returns the file path."""
    try:
        obs = twin.ensure_truth() if include_observed else None
        if obs is not None and gauge != C.DEFAULT_GAUGE:
            obs = twin.load_observed(gauge)
        p = W.plot_runs(run_ids, gauge, obs=obs)
        _record("plots", str(p))
        return str(p)
    except Exception as e:
        return f"ERROR: {e}"


@tool
def score_against_observed(run_id: str, gauge: str = C.DEFAULT_GAUGE) -> str:
    """Skill of a run against observed discharge: KGE (1 = perfect), NSE, percent bias, and the KGE components
    r (timing), alpha (variability), beta (volume). Use this when calibrating."""
    try:
        return _j(twin.score_run(run_id, gauge))
    except Exception as e:
        return f"ERROR: {e}"


@tool
def build_model_variant(name: str, ptf_ksatver: str = "brakensiek", river_upa: int = 30,
                        rivdph_method: str = "powlaw") -> str:
    """Rebuild the Wflow model with HydroMT using a modified build YAML (slow: ~1-2 min).

    ptf_ksatver: pedotransfer function for KsatVer, "brakensiek" (default) or "cosby".
    river_upa: upstream-area threshold [km2] for river cells: 10, 30 (default) or 100.
    rivdph_method: river depth method "powlaw" (default) or "manning".
    Use the returned name as model_variant in run_scenario."""
    try:
        W.build_model_variant(name, {"ptf_ksatver": ptf_ksatver, "river_upa": river_upa,
                                     "rivdph_method": rivdph_method})
        return f"Model variant {name!r} built. Use run_scenario(..., model_variant={name!r})."
    except Exception as e:
        return f"ERROR: {e}"


TOOLSETS = {
    "parameters": [list_parameters, list_gauges],
    "simulation": [run_scenario],
    "analysis": [get_run_summary, compare_runs, plot_hydrographs],
    "calibration": [score_against_observed],
    "hydromt": [build_model_variant],
}
ALL_TOOLS = {t.name: t for ts in TOOLSETS.values() for t in ts}


def get_tools(*sets: str) -> list:
    out = []
    for s in sets or TOOLSETS:
        out += [t for t in TOOLSETS[s] if t not in out]
    return out
