"""Fast tests (no LLM, no Wflow run). Run: uv run pytest exercises/02_agentic_workflows/tests -q"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import tomlkit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hydroagent import config as C  # noqa: E402
from hydroagent import wflow_tools as W  # noqa: E402
from hydroagent.architectures import registry as R  # noqa: E402
from hydroagent.builder.compile import compile_spec, validate  # noqa: E402
from hydroagent.builder.templates import TEMPLATES  # noqa: E402

needs_model = pytest.mark.skipif(not C.BASE_TOML.exists(), reason="base model not built (cli.py prepare)")


def test_multiplier_validation():
    assert W.validate_multipliers({"ksat_vertical": 0.5}) == {"ksat_vertical": 0.5}
    with pytest.raises(ValueError):
        W.validate_multipliers({"ksat_vertical": 50})
    with pytest.raises(ValueError):
        W.validate_multipliers({"not_a_param": 1.0})


@needs_model
def test_scenario_toml(tmp_path):
    out = tmp_path / "wflow_sbm.toml"
    W.write_scenario_toml(C.BASE_TOML, out, {"ksat_vertical": 0.7, "degree_day": 2.0}, C.BASE_MODEL_DIR,
                          "2019-01-01T00:00:00", "2019-12-31T00:00:00")
    doc = tomlkit.parse(out.read_text())
    ks = doc["input"]["static"]["soil_surface_water__vertical_saturated_hydraulic_conductivity"]
    assert ks["netcdf_variable_name"] == "soil_ksat_vertical" and ks["scale"] == 0.7
    base = tomlkit.parse(C.BASE_TOML.read_text())["input"]["static"]["snowpack__degree_day_coefficient"]["value"]
    assert doc["input"]["static"]["snowpack__degree_day_coefficient"]["value"] == pytest.approx(2 * base)
    assert doc["dir_input"] == C.BASE_MODEL_DIR.resolve().as_posix()


@needs_model
def test_static_map_override_requires_file(tmp_path):
    with pytest.raises(ValueError):
        W.write_scenario_toml(C.BASE_TOML, tmp_path / "x.toml", {}, C.BASE_MODEL_DIR, "2019-01-01T00:00:00",
                              "2019-12-31T00:00:00", static_map_overrides={"snowpack__degree_day_coefficient": "x"})


def test_skill_metrics_perfect_and_biased():
    idx = pd.date_range("2019-01-01", "2020-12-31")
    obs = pd.Series(10 + np.sin(np.arange(len(idx)) / 20), idx)
    assert W.skill(obs, obs)["kge"] == pytest.approx(1.0)
    s = W.skill(obs * 1.2, obs)
    assert s["beta"] == pytest.approx(1.2, rel=1e-3) and s["kge"] < 1


@pytest.mark.parametrize("name", list(TEMPLATES))
def test_templates_valid_and_compile(name):
    assert validate(TEMPLATES[name]) == []
    compile_spec(TEMPLATES[name], MagicMock())


def test_validate_catches_errors():
    bad = {"nodes": [{"id": "s", "type": "start"}, {"id": "c", "type": "critic"}],
           "edges": [{"source": "s", "target": "c"}]}
    errs = validate(bad)
    assert any("End" in e for e in errs) and any("Critic" in e for e in errs)


@pytest.mark.parametrize("key", list(R.MODULES))
def test_architecture_graphs_build(key):
    nodes, edges = R.graph_structure(key)
    assert "__start__" in nodes and "__end__" in nodes and edges


def test_recorder_cost_and_time_split():
    from hydroagent.tracing import Recorder

    rec = Recorder(model="anthropic:claude-sonnet-5-5")
    rec.add_llm("n1", input_tokens=1_000_000, output_tokens=100_000, seconds=2.0)
    rec.add_llm("h", cache_read=1_000_000, model="anthropic:claude-sonnet-5-5")
    rec.add_model_time(5.0)
    s = rec.summary()
    assert s["cost_usd"] == pytest.approx(2.0 + 1.0 + 0.2)
    assert s["llm_s"] == 2.0 and s["model_s"] == 5.0 and s["wflow_runs"] == 1
    unpriced = Recorder(model="azure:unknown-model")
    unpriced.add_llm("n", input_tokens=10)
    assert unpriced.cost_estimate()[0] is None


def test_scoreboard_roundtrip(tmp_path, monkeypatch):
    from hydroagent import scoreboard
    from hydroagent.runtime import Outcome

    monkeypatch.setattr(scoreboard, "FILE", tmp_path / "sb.jsonl")
    u = {"model": "m", "total_tokens": 100, "llm_calls": 2, "wall_time_s": 3.0, "cost_usd": 0.01}
    a = scoreboard.record("a0", "q", Outcome("done", answer="x", usage=u))
    b = scoreboard.record("a1", "q", Outcome("done", answer="y", usage={**u, "total_tokens": 300}))
    scoreboard.rate(a, 3)
    scoreboard.rate(a, 5)  # latest rating wins
    df = scoreboard.load().set_index("id")
    assert df.loc[a, "quality"] == 5
    assert pd.isna(df.loc[b, "quality"])  # not rated yet
    df = df.reset_index()
    summ = scoreboard.summary(df, "q")
    assert set(summ["arch"]) == {"a0", "a1"}
