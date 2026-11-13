"""Inspect static model data (land use, soil, terrain): maps, no-data zones and value ranges."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from mcp_server.hydro_server import OUT_DIR, ROOT

d = xr.open_dataset(ROOT / "wflow_moselle_epinal" / "staticmaps.nc")
inside = d["land_elevation"].notnull()
print(f"Cells: {inside.size}, inside basin: {int(inside.sum())} ({inside.mean().item():.1%}); rest is outside the basin mask (expected NaN)")

# no-data inside the basin mask for every 2-D variable
for k, v in d.data_vars.items():
    if v.dims == ("latitude", "longitude") and k not in ("river_mask", "outlets", "gauges_grdc") and not k.startswith("river_"):
        n = int((v.isnull() & inside).sum())
        if n:
            print(f"WARNING {k}: {n} no-data cells inside the basin")

lu = d["meta_landuse"].where(inside)
codes, counts = np.unique(lu.values[~np.isnan(lu.values)], return_counts=True)
print("Land use codes (GlobCover) and area share:")
for c, n in zip(codes, counts):
    print(f"  {int(c):4d}: {n / counts.sum():6.1%}")
tex = d["meta_soil_texture"].where(inside)
codes, counts = np.unique(tex.values[~np.isnan(tex.values)], return_counts=True)
print("Soil texture (USDA class) share:", {int(c): f"{n / counts.sum():.0%}" for c, n in zip(codes, counts)})
st = d["soil_thickness"].where(inside)
print(f"Soil thickness at max (2000 mm cap): {(st == st.max()).sum().item() / inside.sum().item():.0%} of cells")

panels = [
    ("Elevation (m)", d["land_elevation"], "terrain"),
    ("Land use (GlobCover code)", lu, "tab20"),
    ("Soil texture (USDA class)", tex, "tab10"),
    ("Soil thickness (mm)", st, "viridis"),
    ("Ksat vertical (mm/day)", d["soil_ksat_vertical"].where(inside), "viridis"),
    ("Mean LAI (-)", d["vegetation_leaf_area_index"].mean("time").where(inside), "YlGn"),
]
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, (title, da, cmap) in zip(axes.ravel(), panels):
    da.plot(ax=ax, cmap=cmap)
    ax.set_facecolor("#dddddd")  # grey = outside basin / no data
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
fig.suptitle("Static data check: grey = no data (outside basin mask)", fontsize=13)
fig.tight_layout()
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "static_data_check.png", dpi=120)
print(OUT_DIR / "static_data_check.png")
