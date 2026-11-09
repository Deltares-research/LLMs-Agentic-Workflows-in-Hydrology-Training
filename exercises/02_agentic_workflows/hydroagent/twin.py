"""Twin experiment: a hidden "truth" run that plays the role of observed discharge.

The repo has no observed discharge series, so calibration exercises use a synthetic truth made
with secret multipliers. Agents only ever see the resulting discharge, never the multipliers.
Instructors can change the truth with ``TWIN_SEED`` (random) or by passing ``truth=``.
"""

from __future__ import annotations

import json
import os
import random
import shutil

import pandas as pd

from . import config as C
from . import wflow_tools as W

DEFAULT_TRUTH = {"ksat_vertical": 0.4, "soil_thickness": 1.6, "ksat_horizontal_ratio": 3.0}
CALIBRATION_PARAMS = ["ksat_vertical", "soil_thickness", "ksat_horizontal_ratio"]


def _random_truth(seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    return {"ksat_vertical": round(rng.uniform(0.25, 3.0), 2), "soil_thickness": round(rng.uniform(0.6, 2.0), 2),
            "ksat_horizontal_ratio": round(rng.uniform(0.3, 5.0), 2)}


def ensure_truth(truth: dict[str, float] | None = None, force: bool = False) -> pd.Series:
    """Create (once) and return the observed discharge series at the default gauge."""
    obs_csv = C.TWIN_DIR / "observed.csv"
    if obs_csv.exists() and not force:
        return load_observed()
    if truth is None:
        truth = _random_truth(int(os.environ["TWIN_SEED"])) if os.getenv("TWIN_SEED") else DEFAULT_TRUTH
    C.ensure_dirs()
    work = C.TWIN_DIR / "work"
    work.mkdir(parents=True, exist_ok=True)
    W.write_scenario_toml(C.BASE_TOML, work / "wflow_sbm.toml", W.validate_multipliers(truth), C.BASE_MODEL_DIR,
                          C.DEFAULT_START, C.DEFAULT_END)
    W.run_wflow_toml(work / "wflow_sbm.toml")
    df = pd.read_csv(work / "output" / "output.csv", index_col=0, parse_dates=True)
    obs = df.filter(like="river_q_").rename(columns=lambda c: c.replace("river_q_", ""))
    obs.to_csv(obs_csv)
    (C.TWIN_DIR / "truth.secret.json").write_text(json.dumps(truth))
    shutil.rmtree(work, ignore_errors=True)  # do not leave the truth TOML lying around
    return load_observed()


def load_observed(gauge: str = C.DEFAULT_GAUGE) -> pd.Series:
    df = pd.read_csv(C.TWIN_DIR / "observed.csv", index_col=0, parse_dates=True)
    s = df[gauge]
    s.name = "observed"
    return s


def reveal_truth() -> dict:
    """For the debrief only."""
    return json.loads((C.TWIN_DIR / "truth.secret.json").read_text())


def score_run(run_id: str, gauge: str = C.DEFAULT_GAUGE) -> dict:
    """Skill of a run against the observed (twin truth) discharge."""
    ensure_truth()
    return W.skill(W.read_discharge(run_id, gauge), load_observed(gauge))
