"""Compare ERA5 and CHIRPS against E-OBS (station-based proxy): magnitude, min/max, monthly and yearly sums."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from mcp_server.hydro_server import DATA_DIR, OUT_DIR, PRECIP_SOURCES

REF = "eobs"
series = {}
for name, (rel, var) in PRECIP_SOURCES.items():
    da = xr.open_dataset(DATA_DIR / rel)[var].sel(time=slice("1990", "2020"))
    series[name] = da.mean(dim=[d for d in da.dims if d != "time"]).to_series()
df = pd.DataFrame(series).dropna()
monthly = df.resample("MS").sum()
yearly = df.resample("YS").sum()

rows = {}
for n in df:
    rows[n] = {
        "mean_daily_mm": df[n].mean(),
        "wet_day_frac(>1mm)": (df[n] > 1).mean(),
        "daily_min_mm": df[n].min(),
        "daily_max_mm": df[n].max(),
        "monthly_min_mm": monthly[n].min(),
        "monthly_max_mm": monthly[n].max(),
        "yearly_min_mm": yearly[n].min(),
        "yearly_max_mm": yearly[n].max(),
        "yearly_mean_mm": yearly[n].mean(),
        "bias_vs_ref_%": 100 * (yearly[n].mean() / yearly[REF].mean() - 1),
        "daily_corr_vs_ref": df[n].corr(df[REF]),
        "monthly_corr_vs_ref": monthly[n].corr(monthly[REF]),
        "yearly_corr_vs_ref": yearly[n].corr(yearly[REF]),
        "monthly_RMSE_vs_ref_mm": ((monthly[n] - monthly[REF]) ** 2).mean() ** 0.5,
        "yearly_RMSE_vs_ref_mm": ((yearly[n] - yearly[REF]) ** 2).mean() ** 0.5,
    }
stats = pd.DataFrame(rows).round(2)
print(stats.to_string())
OUT_DIR.mkdir(parents=True, exist_ok=True)
stats.to_csv(OUT_DIR / "precip_vs_eobs_stats.csv")

clim = monthly.groupby(monthly.index.month).mean()
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
monthly.plot(ax=ax[0, 0], lw=0.8)
ax[0, 0].set_title("Monthly sums")
ax[0, 0].set_ylabel("mm")
yearly.plot(ax=ax[0, 1], marker="o")
ax[0, 1].set_title("Yearly sums")
ax[0, 1].set_ylabel("mm")
clim.plot(ax=ax[1, 0], marker="o")
ax[1, 0].set_title("Mean monthly climatology")
ax[1, 0].set_xlabel("Month")
ax[1, 0].set_ylabel("mm")
for n, c in (("era5", "tab:blue"), ("chirps", "tab:orange")):
    ax[1, 1].scatter(monthly[REF], monthly[n], s=8, alpha=0.5, c=c, label=n)
lim = [0, monthly.values.max() * 1.05]
ax[1, 1].plot(lim, lim, "k--", lw=1)
ax[1, 1].set_xlabel("E-OBS monthly sum (mm)")
ax[1, 1].set_ylabel("Source monthly sum (mm)")
ax[1, 1].set_title("Monthly sums vs E-OBS (1:1 dashed)")
ax[1, 1].legend()
fig.suptitle("ERA5 and CHIRPS vs E-OBS (station-based proxy), 1990-2020")
fig.tight_layout()
fig.savefig(OUT_DIR / "precip_vs_eobs.png", dpi=120)
print(OUT_DIR / "precip_vs_eobs.png")
