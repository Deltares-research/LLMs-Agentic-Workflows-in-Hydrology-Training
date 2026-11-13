"""Temperature validation: ERA5 vs E-OBS temperature and runoff at Epinal (2019-2020)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from mcp_server.hydro_server import DATA_DIR, EXERCISE_DIR, OUT_DIR


def spatial_mean(path, var):
    da = xr.open_dataset(path)[var]
    return da.mean(dim=[d for d in da.dims if d != "time"]).to_series()


era5 = spatial_mean(DATA_DIR / "era5" / "temp.nc", "temp") - 273.15  # K -> degC
eobs = spatial_mean(DATA_DIR / "eobs_v31.0e.nc", "temp")
t = pd.DataFrame({"ERA5": era5, "E-OBS": eobs}).dropna()
diff = t["ERA5"] - t["E-OBS"]
print(f"1990-2020 mean temp: ERA5 {t['ERA5'].mean():.2f} C, E-OBS {t['E-OBS'].mean():.2f} C, "
      f"bias {diff.mean():.2f} C, daily corr {t['ERA5'].corr(t['E-OBS']):.3f}")
print("Monthly bias (ERA5 - E-OBS, C):")
print(diff.groupby(diff.index.month).mean().round(2).to_string())

q_sim = pd.read_csv(EXERCISE_DIR / "wflow_output_sample.csv", index_col=0, parse_dates=True)["river_q_6136200"]
q_obs = pd.read_csv(EXERCISE_DIR / "observed_synthetic.csv", index_col=0, parse_dates=True)["q_obs"]
period = slice(q_sim.index.min(), q_sim.index.max())
tp = t.loc[period]

fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
ax[0].plot(tp.index, tp["E-OBS"], color="tab:blue", lw=0.9, label="E-OBS")
ax[0].plot(tp.index, tp["ERA5"], color="tab:green", lw=0.9, label="ERA5")
ax[0].axhline(0, color="grey", lw=0.6)
ax[0].set_ylabel("Temperature (°C)")
ax[0].legend(loc="upper right")
d = diff.loc[period]
ax[1].plot(d.index, d, color="tab:purple", lw=0.6, alpha=0.6, label="daily")
ax[1].plot(d.index, d.rolling(30, center=True).mean(), color="k", lw=1.5, label="30-day mean")
ax[1].axhline(0, color="grey", lw=0.6)
ax[1].set_ylabel("ERA5 - E-OBS (°C)")
ax[1].legend(loc="upper right")
ax[2].plot(q_sim.index, q_sim, color="tab:red", ls="--", lw=1, label="Simulated")
ax[2].plot(q_obs.index, q_obs.where(q_obs < 500), color="k", ls="--", lw=0.8, label="Observed (synthetic, spike removed)")
ax[2].set_ylabel("Runoff (m3/s)")
ax[2].legend(loc="upper right")
ax[2].set_title("Station Epinal (river_q_6136200)", y=-0.35)
for a in ax:
    a.grid(alpha=0.4)
fig.suptitle("Validation of input data: Temperature", fontsize=14)
fig.tight_layout()
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "temperature_validation.png", dpi=130)
print(OUT_DIR / "temperature_validation.png")
