---
name: wflow-toml
description: How Wflow.jl 1.x TOML files are structured and how to change parameters safely (scale/offset on NetCDF maps, constant values, cold start, CSV output). Use when creating or editing a wflow_sbm.toml, applying parameter multipliers, or debugging a TOML that Wflow refuses to parse.
---

# Wflow 1.x TOML (wflow_sbm)

Wflow is the **Julia** package `Wflow` (1.x). Never `pip install wflow` or `import wflow` in Python.
In this exercise you do not call Julia yourself: use the MCP tool `run_wflow_scenario`, which
writes a TOML from the base model, applies multipliers and runs it on a warm Julia server.

## Structure you will see

```toml
dir_input = "<abs path to base model>"     # paths in [input] are relative to this
dir_output = "<run folder>/output"

[time]
starttime = "2019-01-01T00:00:00"
endtime = "2020-12-31T00:00:00"
timestepsecs = 86400

[input]
path_forcing = "inmaps-eobs-2019-2020.nc"
path_static  = "staticmaps.nc"              # replace with an absolute path to use edited maps

[input.static]                              # standard_name = "netcdf variable"
soil_surface_water__vertical_saturated_hydraulic_conductivity = "soil_ksat_vertical"
soil__thickness = "soil_thickness"

[input.static.snowpack__degree_day_coefficient]
value = 3.75653                             # constant parameter

[output.csv]                                # gauge time series -> output.csv, columns river_q_<gauge id>
```

## Changing parameters

| Want | TOML |
|---|---|
| Uniform multiplier on a map | `[input.static.<std_name>]` with `netcdf_variable_name = "<var>"`, `scale = 0.7`, `offset = 0.0` |
| Change a constant | edit `value` in `[input.static.<std_name>]` |
| Spatially varying change of a map | write a new staticmaps NetCDF (see the `netcdf-analysis` skill) and pass it as `staticmaps_path` |
| Turn a constant into a spatial map | add a new variable (e.g. `snow_ddf_map`) to your staticmaps copy, then call `run_wflow_scenario` with `staticmaps_path` and `static_overrides={"snowpack__degree_day_coefficient": "snow_ddf_map"}` |
| More/less rain, a synthetic storm, warmer climate | edit a copy of `inmaps-eobs-2019-2020.nc` (see the `netcdf-analysis` skill) and pass it as `forcing_path` |

The key is `netcdf_variable_name` (flat). A nested `netcdf.variable.name` table fails with
`Couldn't parse the section under [input]`.

Wflow logs every applied scaling in `output/log.txt`:
`NetCDF parameter soil_ksat_vertical is modified with scale 0.7 ...`. Check this line to prove a change was applied.

## Parameter keys used by the exercise tools

`ksat_vertical`, `ksat_decay_f`, `ksat_horizontal_ratio`, `soil_thickness`, `theta_s`, `root_depth`,
`crop_factor`, `canopy_evap_ratio`, `compacted_infiltration`, `degree_day`, `river_manning_n`, `land_manning_n`.
Call the `list_parameters` MCP tool for their meaning and allowed multiplier range.

## Gotchas

- Runs are cold starts (`cold_start__flag = true`). Ignore the first 3 months (spin-up) in statistics.
- Never edit files in the base model folder `wflow_moselle_epinal/`. Work in your workspace.
