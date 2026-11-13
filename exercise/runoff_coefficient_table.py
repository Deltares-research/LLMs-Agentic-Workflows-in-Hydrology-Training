"""Runoff-coefficient table per gauge and year (slide: Validation of input data: Discharge).

RC = annual runoff depth / annual precipitation. Precipitation is the single gauge-cell forcing
column from the Wflow output, so values are indicative only.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from mcp_server.hydro_server import EXERCISE_DIR, OUT_DIR, ROOT

w = pd.read_csv(EXERCISE_DIR / "wflow_output_sample.csv", index_col=0, parse_dates=True)
d = xr.open_dataset(ROOT / "wflow_moselle_epinal" / "staticmaps.nc")
g, up = d["gauges_grdc"], d["meta_upstream_area"]
area = {}
for iy, ix in zip(*np.where(g.notnull().values)):
    area[int(g.values[iy, ix])] = float(up.values[iy, ix])  # km2

ids = [i for i in sorted(area, key=area.get) if f"river_q_{i}" in w and f"precip_{i}" in w]
years = [y for y in sorted(set(w.index.year)) if (w.index.year == y).sum() > 300]
rc = pd.DataFrame(index=years, columns=ids, dtype=float)
for i in ids:
    q = w[f"river_q_{i}"]
    for y in years:
        sel = w.index.year == y
        runoff_mm = q[sel].sum() * 86400 / (area[i] * 1e6) * 1000
        rc.loc[y, i] = runoff_mm / w[f"precip_{i}"][sel].sum()
print(pd.Series(area).loc[ids].round(2).to_string())
print(rc.round(2).to_string())
rc.round(3).to_csv(OUT_DIR / "runoff_coefficients.csv")

fig, ax = plt.subplots(figsize=(11, 2.6 + 0.35 * len(years)))
ax.axis("off")
cell = [["Subcatch"] + [str(i) for i in ids],
        ["Area [km2]"] + [f"{area[i]:.2f}" for i in ids],
        [""] * (len(ids) + 1),
        [""] + ["Runoff coefficient"] * len(ids)]
cell += [[str(y)] + [f"{rc.loc[y, i]:.2f}" for i in ids] for y in years]
t = ax.table(cellText=cell, loc="center", cellLoc="center")
t.auto_set_font_size(False)
t.set_fontsize(9)
t.scale(1, 1.4)
for r in range(len(cell)):
    for c in range(len(ids) + 1):
        if r >= 4 and c > 0 and not 0.2 <= rc.loc[years[r - 4], ids[c - 1]] <= 0.8:
            t[r, c].set_facecolor("#f8cecc")  # outside plausible range
ax.set_title("Runoff coefficient per gauge sub-catchment (red = outside 0.2-0.8)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT_DIR / "runoff_coefficient_table.png", dpi=130)
print(OUT_DIR / "runoff_coefficient_table.png")
