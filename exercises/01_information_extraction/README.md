# Exercise 1: Retention-basin information extraction

Turn an unstructured hydrology PDF into structured JSON with [DLLMForge](https://github.com/Deltares-research/DLLMForge) (Docling preprocessor + Azure GPT).

Use either the notebook or the script; they follow the same steps. Docling emits tagged markdown (`[PAGE N]`, `[SECTION]`, `[TABLE]`); `source_page` is recovered from `source_quote` in code.

## Setup

From the **repo root**:

1. Install and verify the course environment (hydrology-env skill, or `uv sync`). Finish with:

   ```bash
   python .agents/skills/hydrology-env/scripts/verify_env.py
   ```

   Every row must be `OK`, including `dllmforge + dotenv + jupyter` (Docling is part of `dllmforge[api,docling]`).

2. Put Azure OpenAI settings in the repo-root `.env` (never paste keys into the notebook or script):

   ```
   AZURE_OPENAI_API_KEY=...
   AZURE_OPENAI_ENDPOINT=...
   AZURE_OPENAI_DEPLOYMENT_NAME=...
   AZURE_OPENAI_API_VERSION=...
   ```

3. Jupyter: select the project `.venv` kernel (`uv run python` is 3.13.18). From the repo root you can start Jupyter with `uv run jupyter lab`.

## Run

Notebook (from this folder, or open the `.ipynb` in the IDE):

- [`01_retention_basin_extraction.ipynb`](01_retention_basin_extraction.ipynb)

Script:

```bash
uv run python exercises/01_information_extraction/01_retention_basin_extraction.py
```

Optional:

```bash
uv run python exercises/01_information_extraction/01_retention_basin_extraction.py --pdf path/to/other.pdf --output-dir exercises/01_information_extraction/output
```

## Files

| Path | Role |
|---|---|
| [`schema.py`](schema.py) | Hand-written Pydantic schema (identity + engineering + `source_quote`; `source_page` is filled by DLLMForge) |
| [`data/pdfs/nhess-15-365-2015.pdf`](data/pdfs/nhess-15-365-2015.pdf) | Example paper (Copernicus CC BY; see [`data/README.md`](data/README.md)) |
| `output/` | Extracted JSON (gitignored except `.gitkeep`) |

Document-level file facts (`source_file`, `file_type`, `text_length`) are appended by DLLMForge as `metadata`. They are not schema fields. Both the notebook and the script extract **one PDF** with the Docling text path.
