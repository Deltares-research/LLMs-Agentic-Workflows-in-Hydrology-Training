"""Plot all precipitation and discharge time series used in the exercise."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from mcp_server.hydro_server import DATA_DIR, EXERCISE_DIR, OUT_DIR, PRECIP_SOURCES

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Gridded precipitation, spatial mean, 1990-2020
series = {}
for name, (rel, var) in PRECIP_SOURCES.items():
    da = xr.open_dataset(DATA_DIR / rel)[var].sel(time=slice("1990", "2020"))
    series[name] = da.mean(dim=[d for d in da.dims if d != "time"]).to_series()
p = pd.DataFrame(series).dropna()

fig, axes = plt.subplots(2, 1, figsize=(12, 7))
p.resample("MS").sum().plot(ax=axes[0])
axes[0].set_ylabel("Monthly precip (mm)")
axes[0].set_title("ERA5 / CHIRPS / E-OBS, basin mean")
p.resample("YS").sum().plot(ax=axes[1], marker="o")
axes[1].set_ylabel("Annual precip (mm)")
fig.tight_layout()
fig.savefig(OUT_DIR / "precip_timeseries.png", dpi=120)
plt.close(fig)

# Wflow output + synthetic observations
w = pd.read_csv(EXERCISE_DIR / "wflow_output_sample.csv", index_col=0, parse_dates=True)
obs = pd.read_csv(EXERCISE_DIR / "observed_synthetic.csv", index_col=0, parse_dates=True)
q_cols = [c for c in w if c.startswith("river_q_")]
p_cols = [c for c in w if c.startswith("precip_")]

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
w[q_cols].plot(ax=axes[0], lw=0.8)
axes[0].set_ylabel("Simulated Q (m3/s)")
axes[0].set_yscale("symlog", linthresh=1)
w[p_cols].plot(ax=axes[1], lw=0.8)
axes[1].set_ylabel("Precip (mm/day)")
w["river_q_6136200"].plot(ax=axes[2], label="sim river_q_6136200")
obs["q_obs"].plot(ax=axes[2], label="q_obs (synthetic)", alpha=0.7)
axes[2].set_ylabel("Q at Epinal (m3/s)")
axes[2].legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "wflow_timeseries.png", dpi=120)
plt.close(fig)
print(OUT_DIR / "precip_timeseries.png", OUT_DIR / "wflow_timeseries.png")
