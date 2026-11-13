import numpy as np
import pytest

from mcp_server.hydro_server import check_discharge, kge, nse, skill_scores, water_balance


def test_perfect_scores():
    q = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
    assert nse(q, q) == pytest.approx(1.0)
    assert kge(q, q) == pytest.approx(1.0)


def test_synthetic_obs_scores_and_defects():
    scores = skill_scores(
        "observed_synthetic.csv", "q_obs", "wflow_output_sample.csv", "river_q_6136200"
    )
    assert scores["relative_volume_error"] < 0  # sim is below the biased obs
    assert scores["logNSE"] > 0.8

    qc = check_discharge("observed_synthetic.csv", "q_obs")
    assert qc["missing_days"] == 10
    assert qc["longest_flat_run_days"] >= 20


def test_water_balance_epinal():
    # Epinal upstream area (GRDC 6136200): 1220 km2
    wb = water_balance("wflow_output_sample.csv", "river_q_6136200", "precip_6136200", 1220.0)
    assert 0.5 < wb["runoff_coefficient"] < 1.2
