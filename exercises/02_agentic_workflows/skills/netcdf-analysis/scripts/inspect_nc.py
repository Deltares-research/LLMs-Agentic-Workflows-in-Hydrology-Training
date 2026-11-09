"""Print a compact summary of a NetCDF file: dims, coords range, and per-variable dims/dtype/min/max."""
import sys

import xarray as xr

ds = xr.open_dataset(sys.argv[1])
print("dims:", dict(ds.sizes))
for c in ds.coords:
    v = ds[c].values
    print(f"coord {c}: {v.min()} .. {v.max()} (n={v.size}, first={v[0]})" if v.size else f"coord {c}")
for name, da in ds.data_vars.items():
    try:
        lo, hi = float(da.min()), float(da.max())
        rng = f"min={lo:.4g} max={hi:.4g}"
    except Exception:
        rng = ""
    print(f"{name:45s} {str(da.dims):35s} {str(da.dtype):8s} {rng} {da.attrs.get('units', '')}")
