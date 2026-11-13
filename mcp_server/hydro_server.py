"""Small MCP server (MCPServer, mcp 2.x) for the hydrology data-analysis exercise (stdio).

Maps to hydro_data_analysis_example.pptx:
  Step 1 (input validation): compare_precip, check_discharge
  Step 5 (performance):      skill_scores, plot_hydrograph, water_balance
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from mcp.server.mcpserver import MCPServer

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("HYDRO_DATA_DIR", ROOT / "hydromt_data"))
EXERCISE_DIR = ROOT / "exercise" / "data"
OUT_DIR = Path(os.environ.get("HYDRO_OUT_DIR", ROOT / "exercise" / "output"))

PRECIP_SOURCES = {
    "era5": ("era5/precip.nc", "precip"),
    "chirps": ("chirps_global.nc", "precip"),
    "eobs": ("eobs_v31.0e.nc", "precip"),
}

mcp = MCPServer("hydro-analysis")


def _resolve(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    for base in (EXERCISE_DIR, ROOT):
        if (base / p).exists():
            return base / p
    raise FileNotFoundError(f"{path} not found in {EXERCISE_DIR} or {ROOT}")


def _read_series(path: str, column: str) -> pd.Series:
    df = pd.read_csv(_resolve(path), index_col=0, parse_dates=True)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not in {path}. Available: {list(df.columns)}")
    return df[column].astype(float)


def nse(obs: np.ndarray, sim: np.ndarray) -> float:
    return float(1 - np.sum((sim - obs) ** 2) / np.sum((obs - obs.mean()) ** 2))


def kge(obs: np.ndarray, sim: np.ndarray) -> float:
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = sim.std() / obs.std()
    beta = sim.mean() / obs.mean()
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def _aligned(obs: pd.Series, sim: pd.Series) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    both = pd.concat([obs.rename("obs"), sim.rename("sim")], axis=1).dropna()
    if len(both) < 10:
        raise ValueError("Fewer than 10 overlapping time steps between obs and sim")
    return both["obs"].to_numpy(), both["sim"].to_numpy(), both.index


@mcp.tool()
def compare_precip(start: str = "1990-01-01", end: str = "2020-12-31") -> dict:
    """Step 1: compare spatially averaged daily precipitation (mm/day) of ERA5,
    CHIRPS and E-OBS over a period. Returns annual mean totals, daily max and
    the correlation of monthly sums between sources."""
    series = {}
    for name, (rel, var) in PRECIP_SOURCES.items():
        da = xr.open_dataset(DATA_DIR / rel)[var].sel(time=slice(start, end))
        series[name] = da.mean(dim=[d for d in da.dims if d != "time"]).to_series()
    df = pd.DataFrame(series).dropna()
    annual = df.resample("YS").sum()
    monthly = df.resample("MS").sum()
    return {
        "period": [str(df.index[0].date()), str(df.index[-1].date())],
        "units": "mm/day (daily), mm (sums)",
        "mean_annual_total_mm": annual.mean().round(1).to_dict(),
        "daily_max_mm": df.max().round(1).to_dict(),
        "negative_values": {k: int((df[k] < 0).sum()) for k in df},
        "monthly_sum_correlation": monthly.corr().round(3).to_dict(),
    }


@mcp.tool()
def check_discharge(csv_path: str, column: str) -> dict:
    """Step 1: basic quality check of a discharge time series (m3/s): missing
    days, negative values, longest flat run and high outliers (> median + 10*IQR)."""
    q = _read_series(csv_path, column)
    full = pd.date_range(q.index.min(), q.index.max(), freq="D")
    flat = (q.diff() == 0).astype(int)
    run = flat.groupby((flat == 0).cumsum()).cumsum()
    iqr = q.quantile(0.75) - q.quantile(0.25)
    return {
        "start": str(q.index.min().date()),
        "end": str(q.index.max().date()),
        "missing_days": int(len(full) - q.dropna().shape[0]),
        "negative_values": int((q < 0).sum()),
        "longest_flat_run_days": int(run.max()),
        "outliers_above_median_plus_10_iqr": int((q > q.median() + 10 * iqr).sum()),
        "min": round(float(q.min()), 3),
        "mean": round(float(q.mean()), 3),
        "max": round(float(q.max()), 3),
    }


@mcp.tool()
def skill_scores(obs_csv: str, obs_column: str, sim_csv: str, sim_column: str) -> dict:
    """Step 5: NSE, log-NSE, KGE, RMSE, R2 and relative volume error (bias) of
    simulated vs observed discharge over their overlapping period."""
    obs, sim, idx = _aligned(_read_series(obs_csv, obs_column), _read_series(sim_csv, sim_column))
    eps = 0.01 * obs.mean()
    return {
        "n_timesteps": len(idx),
        "period": [str(idx[0].date()), str(idx[-1].date())],
        "NSE": round(nse(obs, sim), 3),
        "logNSE": round(nse(np.log(obs + eps), np.log(sim + eps)), 3),
        "KGE": round(kge(obs, sim), 3),
        "RMSE_m3s": round(float(np.sqrt(np.mean((sim - obs) ** 2))), 3),
        "R2": round(float(np.corrcoef(obs, sim)[0, 1] ** 2), 3),
        "relative_volume_error": round(float((sim.sum() - obs.sum()) / obs.sum()), 3),
    }


@mcp.tool()
def plot_hydrograph(obs_csv: str, obs_column: str, sim_csv: str, sim_column: str) -> str:
    """Step 5: save a PNG of observed vs simulated discharge (linear and log axes)
    and return its path."""
    obs, sim = _read_series(obs_csv, obs_column), _read_series(sim_csv, sim_column)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, log in zip(axes, (False, True)):
        ax.plot(obs.index, obs, label="observed", lw=1)
        ax.plot(sim.index, sim, label="simulated", lw=1)
        ax.set_ylabel("Q [m3/s]" + (" (log)" if log else ""))
        if log:
            ax.set_yscale("log")
    axes[0].legend()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "hydrograph.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return str(out)


@mcp.tool()
def water_balance(sim_csv: str, q_column: str, precip_column: str, area_km2: float) -> dict:
    """Step 5: long-term water balance for a gauge. Converts discharge to mm over
    the upstream area and reports P, Q, runoff coefficient and the remainder
    P - Q (= ET + change in storage)."""
    q = _read_series(sim_csv, q_column)
    p = _read_series(sim_csv, precip_column)
    both = pd.concat([q.rename("q"), p.rename("p")], axis=1).dropna()
    n_years = len(both) / 365.25
    q_mm = both["q"].sum() * 86400 / (area_km2 * 1e6) * 1000 / n_years
    p_mm = both["p"].sum() / n_years
    return {
        "years": round(n_years, 2),
        "precip_mm_per_year": round(float(p_mm), 1),
        "discharge_mm_per_year": round(float(q_mm), 1),
        "runoff_coefficient": round(float(q_mm / p_mm), 3),
        "P_minus_Q_mm_per_year": round(float(p_mm - q_mm), 1),
    }


@mcp.prompt()
def validate_inputs() -> str:
    """Step 1 workflow from the deck."""
    return (
        "Validate the model input data: compare ERA5, CHIRPS and E-OBS precipitation "
        "with compare_precip, check the discharge series with check_discharge, and "
        "summarise any issue that would make the observed data unreliable."
    )


@mcp.prompt()
def assess_performance() -> str:
    """Step 5 workflow from the deck."""
    return (
        "Assess model performance: compute skill_scores, plot the hydrograph, and "
        "check the water_balance. Combine several indicators and state which flow "
        "regime (peaks, low flows, volume) the model reproduces best."
    )


if __name__ == "__main__":
    mcp.run()
