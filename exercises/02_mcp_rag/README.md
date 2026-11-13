# Exercise 2: Natural language over knowledge and data

Two methods for interacting with hydrology material in **natural language**:

- **Knowledge (text) — RAG:** retrieve from documents, then generate an answer. Here: the Wflow_sbm GMD paper, keyword vs semantic search.
- **Data (model inputs/outputs) — MCP:** call tools on numbers, maps and time series.

> Always read retrieved chunks and tool output yourself: LLMs can misread or over-interpret them.

## Part A: RAG tutorial

Index the Wflow_sbm GMD paper, compare keyword vs semantic search, and generate answers.

Notebook: [`RAG_tutorial.ipynb`](RAG_tutorial.ipynb)

### Setup

From the **repo root**:

1. Course environment verify is green (`python .agents/skills/hydrology-env/scripts/verify_env.py`).
2. Put Foundry settings in the repo-root `.env` (never paste keys into the notebook):

   ```
   FOUNDRY_ENDPOINT=https://<resource>.services.ai.azure.com/anthropic/v1/messages
   FOUNDRY_KEY=...
   DEPLOYMENT_OPUS_55=claude-opus-5-5
   DEPLOYMENT_MISTRAL_LARGE_4=mistral-large-4
   DEPLOYMENT_TEXT_EMBEDDINGS_3_LARGE=text-embedding-3-large
   ```

   Claude can also use `ANTHROPIC_API_KEY` instead of Foundry. Switch `PROVIDER` in the notebook to `"mistral"` if you want Mistral for generation. Embeddings always use the Foundry OpenAI-compatible endpoint.

3. Jupyter: select the project `.venv` kernel. 

## Part B: Inspect a Wflow model through MCP

Ask a coding agent questions in plain language about **Wflow inputs and outputs** for the upper Moselle (Épinal). The agent calls tools on a small MCP server.

### Setup

From the **repo root**:

1. Install and verify the course environment (hydrology-env skill, or `uv sync`). Finish with:

   ```bash
   python .agents/skills/hydrology-env/scripts/verify_env.py
   ```

   Every row must be `OK`, including `mcp + pandas + xarray + geopandas + matplotlib`.

2. The base model `wflow_moselle_epinal/` must already exist (HydroMT-built, gitignored). Then generate the baseline CSV and twin “observations”:

   ```bash
   uv run python exercises/02_mcp_rag/prepare.py
   ```

   Twin observations use the same hidden multipliers as Exercise 3 (`ksat_vertical=0.4`, `soil_thickness=1.6`, `ksat_horizontal_ratio=3.0`). They are **not** real gauges. Quality checks on `observed.csv` should look clean; skill scores against the baseline will not be perfect.

### Connect the MCP server in your coding agent

Open **this repo as the workspace** (so the relative `exercises/02_mcp_rag/...` path works). Then add a small JSON file — the filename and the top-level key differ per product.

#### VS Code Copilot

1. Create `.vscode/mcp.json` at the **repo root** (create the `.vscode` folder if needed). VS Code uses the key `servers`, not `mcpServers`.
2. Put this in the file:

   ```json
   {
     "servers": {
       "hydro-analysis": {
         "type": "stdio",
         "command": "uv",
         "args": ["run", "python", "exercises/02_mcp_rag/mcp_server/hydro_server.py"]
       }
     }
   }
   ```

3. Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`) → **MCP: List Servers**. Start `hydro-analysis` if it is stopped.
4. State should be **Running**. Open the server in that list, or Copilot Chat → tools icon. You should see the six tools below.

#### Copilot CLI

There is no project `mcp.json` for Copilot CLI. From this repo root, add the server once:

```text
/mcp add hydro-analysis -- uv run python exercises/02_mcp_rag/mcp_server/hydro_server.py
```

Then `/mcp list` should show `hydro-analysis`.

#### Claude Code

1. Create `.mcp.json` at the **repo root**.
2. Put this in the file:

   ```json
   {
     "mcpServers": {
       "hydro-analysis": {
         "command": "uv",
         "args": ["run", "python", "exercises/02_mcp_rag/mcp_server/hydro_server.py"]
       }
     }
   }
   ```

3. Start Claude Code from the repo root. Type `/mcp` (or run `claude mcp list`).
4. `hydro-analysis` should be listed and connected. `/mcp` also lists its tools.

#### Cursor

1. Create `.cursor/mcp.json` at the **repo root** (create the `.cursor` folder if needed).
2. Put this in the file:

   ```json
   {
     "mcpServers": {
       "hydro-analysis": {
         "type": "stdio",
         "command": "uv",
         "args": ["run", "python", "exercises/02_mcp_rag/mcp_server/hydro_server.py"]
       }
     }
   }
   ```

3. Cursor Settings → **MCP**. Enable `hydro-analysis`. Reload the window if it does not appear.
4. Connected / green, not an error. Expand the server: you should see the six tools below.

### Check the tools

Ask the agent: *“List the hydro-analysis MCP tools.”* Expect `compare_precip`, `check_discharge`, `plot_staticmaps`, `plot_basin_map`, `skill_scores`, and `plot_hydrograph`. If none appear, reload MCP / the window and confirm `uv` is on PATH.

### Tools

| Tool | What it does |
|---|---|
| `compare_precip` | ERA5 / CHIRPS / E-OBS basin-mean precipitation (default 1990–2020) |
| `check_discharge` | Missing days, negatives, flat runs, outliers in a Q series |
| `plot_staticmaps` | 6-panel PNG of `staticmaps.nc` (elevation, land use, soil, Ksat, LAI) |
| `plot_basin_map` | Basin, rivers, GRDC gauges and E-OBS cell centres (`output/model_map.png`) |
| `skill_scores` | NSE, log-NSE, KGE, RMSE, R2, volume error |
| `plot_hydrograph` | Observed vs simulated Q (linear and log), PNG under `output/` |

Relative CSV paths resolve under [`data/`](data/). Gauge for sim/obs at Épinal: **9116044** (`river_q_9116044` in the Wflow CSV, `q_obs` in `observed.csv`).

The server also exposes two prompt templates: `validate_inputs` and `assess_performance`.

### Prompt examples

Paste these into the agent chat after MCP is connected.

**Forcing products**

```
Compare ERA5, CHIRPS and E-OBS precipitation for 1990-2020.
```

**Static maps and catchment**

```
Plot the static maps and the catchment map.
```

**Discharge QC**

```
Check discharge quality of observed.csv, column q_obs.
```

**Performance at Épinal**

```
Compute skill scores of observed.csv / q_obs against wflow_output_sample.csv / river_q_9116044, and plot the hydrograph.
```

**Use the bundled prompts**

```
validate inputs.
```

```
assess performance.
```

## Files

| Path | Role |
|---|---|
| [`RAG_tutorial.ipynb`](RAG_tutorial.ipynb) | Part A: keyword vs semantic RAG |
| [`data/pdfs/gmd-17-3199-2024.pdf`](data/pdfs/gmd-17-3199-2024.pdf) | Wflow_sbm GMD paper (CC BY 4.0) |
| [`mcp_server/hydro_server.py`](mcp_server/hydro_server.py) | Part B: MCP server (stdio) |
| [`prepare.py`](prepare.py) | Part B: baseline Wflow run + twin observations |
| `data/` | also `wflow_output_sample.csv`, `observed.csv` (generated, gitignored) |
| `output/` | Hydrograph and map PNGs (gitignored except `.gitkeep`) |
