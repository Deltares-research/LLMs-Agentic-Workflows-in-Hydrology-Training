---
name: netcdf-analysis
description: Read, analyse and modify Wflow NetCDF files (staticmaps.nc parameters, output.nc gridded discharge) with xarray, including spatially varying parameter changes per sub-catchment. Use when a task needs custom analysis or edits that the fixed multiplier tools cannot express.
---

# Wflow NetCDF with xarray

Run Python with the interpreter given in your instructions (the project venv). Write scripts to
files in your workspace and run them. Do not use long `python -c` one-liners.

Start every task by inspecting the file:

```bash
<python> .claude/skills/netcdf-analysis/scripts/inspect_nc.py <file.nc>
```

## staticmaps.nc (model parameters, read-only base)

- Dims `(latitude, longitude)`. Latitude is **descending**. Some variables have a `layer` dim.
- Key variables:

| Variable | Meaning | Unit |
|---|---|---|
| `soil_ksat_vertical` | KsatVer | mm/day |
| `soil_thickness` | | mm |
| `soil_f` | | |
| `river_manning_n` | | |
| `land_elevation` | | |
| `meta_landuse` | | |
| `subcatchment_grdc` | sub-catchment id = GRDC gauge id | |

- Gauge ids:

| GRDC id | Gauge |
|---|---|
| 9116044 | Epinal (outlet) |
| 9116043 | Noir Gueux |
| 9116040 | Rupt-sur-Moselle |
| 9116039 | Fresse-sur-Moselle |
| 9116042 | Cleurie |

## Spatially varying parameter change (pattern)

```python
import xarray as xr, numpy as np
src = "<BASE_MODEL_DIR>/staticmaps.nc"
ds = xr.open_dataset(src).load(); ds.close()
mask = ds["subcatchment_grdc"] == 9116039          # or e.g. ds["land_elevation"] > 700
ds["soil_ksat_vertical"] = ds["soil_ksat_vertical"].where(~mask, ds["soil_ksat_vertical"] * 0.5)
ds.to_netcdf("staticmaps_modified.nc")             # inside your workspace
```

Then call the MCP tool `run_wflow_scenario` with `staticmaps_path` = absolute path of the new file.
Keep `encoding` defaults. Do not drop variables. Wflow needs all of them.

## Changing the forcing (rainfall, temperature, PET)

`inmaps-eobs-2019-2020.nc`: `precip` (mm/day), `pet` (mm/day), `temp` (degC), dims `(time, latitude, longitude)`,
daily 2019-01-01 .. 2020-12-31, same grid as staticmaps. Edit a copy and keep grid, time axis and all three variables.

```python
import xarray as xr
src = "<BASE_MODEL_DIR>/inmaps-eobs-2019-2020.nc"
ds = xr.open_dataset(src).load(); ds.close()
ds["precip"] = ds["precip"] * 1.2                              # +20% rain everywhere, all days
# synthetic storm: add 60 mm spread over 3 days on top of what fell
storm = slice("2020-02-01", "2020-02-03")
ds["precip"].loc[dict(time=storm)] = ds["precip"].sel(time=storm) + 20.0
# warming: ds["temp"] = ds["temp"] + 2.0
ds["precip"] = ds["precip"].clip(min=0)                       # never negative rain
ds.to_netcdf("forcing_modified.nc")                          # inside your workspace
```

Then call `run_wflow_scenario` with `forcing_path` = absolute path of the new file (it can be combined with
`multipliers` and `staticmaps_path`). Events placed in the spin-up (before 2019-04-01) are ignored by the metrics.

## output.nc (gridded results of a run)

- `river_q (time, lat, lon)` in m3/s. Here latitude is **ascending** and named `lat`/`lon`, unlike staticmaps.
- Gauge time series are easier from `output/output.csv` (columns `river_q_<gauge id>`).
- Typical analyses:
  - annual maxima: `q.resample(time="YS").max()`
  - flow-duration curve: sort descending
  - peak timing: `idxmax`
  - difference maps between two runs: `(a - b).mean("time")`

## Reporting

Save figures as PNG in the workspace and mention their paths. Give numbers with units. State which run_ids you compared.
