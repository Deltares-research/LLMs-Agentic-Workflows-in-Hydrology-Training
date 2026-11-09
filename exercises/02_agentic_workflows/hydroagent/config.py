"""Paths, environment loading and model settings shared by every entry point.

Keys are never written here. They are read from a ``.env`` file:

1. ``AGENT_ENV_FILE`` (explicit path), else
2. the first ``.env`` found walking up from the current working directory, else
3. the first ``.env`` found walking up from this exercise folder
   (so a ``.env`` placed *next to* the repo folder also works).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

EXERCISE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXERCISE_DIR.parents[1]

# HydroMT inputs (from the repo) and the base Wflow model (built once, shared, read-only).
HYDROMT_BUILD_YML = REPO_ROOT / "hydromt" / "wflow_build.yml"
HYDROMT_FORCING_YML = REPO_ROOT / "hydromt" / "update_forcing.yml"
DATA_CATALOG = REPO_ROOT / "hydromt_data" / "data_catalog.yml"
BASE_MODEL_DIR = REPO_ROOT / "wflow_moselle_epinal"
BASE_TOML = BASE_MODEL_DIR / "wflow_sbm.toml"

# Everything an agent produces lives under output/ (gitignored).
OUTPUT_DIR = EXERCISE_DIR / "output"
RUNS_DIR = OUTPUT_DIR / "runs"
HARNESS_DIR = OUTPUT_DIR / "harness"
VARIANTS_DIR = OUTPUT_DIR / "model_variants"
TWIN_DIR = OUTPUT_DIR / "_twin"  # hidden "truth" for calibration (agents never get this path)
GRAPHS_DIR = OUTPUT_DIR / "graphs"

SKILLS_DIRS = [EXERCISE_DIR / "skills", REPO_ROOT / ".agents" / "skills"]

# Gauges written by the base model (GRDC ids). Epinal is the basin outlet station.
GAUGES = {
    "9116044": "Moselle at Epinal (outlet, 1220 km2)",
    "9116043": "Moselle at Noir Gueux (621 km2)",
    "9116040": "Moselle at Rupt-sur-Moselle (153 km2)",
    "9116039": "Moselle at Fresse-sur-Moselle (72 km2)",
    "9116042": "Cleurie at Cleurie (68 km2)",
}
DEFAULT_GAUGE = "9116044"

# Simulation window (E-OBS forcing covers 2019-01-01 .. 2020-12-31).
DEFAULT_START = "2019-01-01T00:00:00"
DEFAULT_END = "2020-12-31T00:00:00"
SPINUP_END = "2019-03-31"  # metrics ignore the cold-start spin-up period

_ENV_LOADED: Path | None = None


def load_env() -> Path | None:
    """Load the course ``.env`` once and return its path (or None)."""
    global _ENV_LOADED
    if _ENV_LOADED is not None:
        return _ENV_LOADED
    candidates: list[Path] = []
    if os.getenv("AGENT_ENV_FILE"):
        candidates.append(Path(os.environ["AGENT_ENV_FILE"]))
    found = find_dotenv(usecwd=True)
    if found:
        candidates.append(Path(found))
    for parent in [EXERCISE_DIR, *EXERCISE_DIR.parents]:
        candidates.append(parent / ".env")
    for c in candidates:
        if c.is_file():
            load_dotenv(c, override=False)
            _ENV_LOADED = c
            break
    # Exercise 1 uses AZURE_OPENAI_ENDPOINT; the Deltares .env uses AZURE_OPENAI_API_BASE.
    if not os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_BASE"):
        os.environ["AZURE_OPENAI_ENDPOINT"] = os.environ["AZURE_OPENAI_API_BASE"]
    return _ENV_LOADED


@dataclass
class Settings:
    """Model/provider settings. Override with env vars or by passing arguments."""

    provider: str = field(default_factory=lambda: os.getenv("AGENT_PROVIDER", "anthropic"))
    anthropic_model: str = field(
        default_factory=lambda: os.getenv("AGENT_ANTHROPIC_MODEL")
        or os.getenv("ANTHROPIC_DEPLOYMENT_SONNET55")
        or "claude-sonnet-5-5"
    )
    azure_deployment: str = field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        or os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT41")
        or "gpt-4.1"
    )
    temperature: float = 0.0
    # Claude Agent SDK auth: "auto" (API key if set), "api_key", or "login" (Claude Code login/subscription).
    harness_auth: str = field(default_factory=lambda: os.getenv("HARNESS_AUTH", "auto"))


def get_settings(**overrides) -> Settings:
    load_env()
    s = Settings()
    for k, v in overrides.items():
        if v is not None:
            setattr(s, k, v)
    return s


def ensure_dirs() -> None:
    for d in (OUTPUT_DIR, RUNS_DIR, HARNESS_DIR, VARIANTS_DIR, TWIN_DIR, GRAPHS_DIR):
        d.mkdir(parents=True, exist_ok=True)
