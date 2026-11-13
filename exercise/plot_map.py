"""Map of the Wflow model area: basin, rivers, GRDC gauges and E-OBS grid-cell centres."""
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from mcp_server.hydro_server import DATA_DIR, OUT_DIR, ROOT

geoms = ROOT / "wflow_moselle_epinal" / "staticgeoms"
basin = gpd.read_file(geoms / "basins.geojson")
rivers = gpd.read_file(geoms / "rivers.geojson")
gauges = gpd.read_file(geoms / "gauges_grdc.geojson")

eobs = xr.open_dataset(DATA_DIR / "eobs_v31.0e.nc")
lon, lat = np.meshgrid(eobs.longitude.values, eobs.latitude.values)

fig, ax = plt.subplots(figsize=(9, 8))
basin.plot(ax=ax, color="#e8f0e8", edgecolor="k")
if "strord" in rivers:
    rivers.plot(ax=ax, column="strord", cmap="Blues", linewidth=rivers["strord"] * 0.4 + 0.3)
else:
    rivers.plot(ax=ax, color="tab:blue", linewidth=0.8)
ax.scatter(lon.ravel(), lat.ravel(), s=12, c="tab:red", marker="+", label="E-OBS cell centres (0.1°)")
gauges.plot(ax=ax, color="k", marker="^", markersize=50, label="GRDC gauges", zorder=5)
ax.set_xlim(*basin.total_bounds[[0, 2]] + np.array([-0.1, 0.1]))
ax.set_ylim(*basin.total_bounds[[1, 3]] + np.array([-0.1, 0.1]))
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Moselle at Epinal: basin, rivers, gauges, E-OBS grid")
ax.legend(loc="lower right")
fig.tight_layout()
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "model_map.png", dpi=120)
print(OUT_DIR / "model_map.png")
