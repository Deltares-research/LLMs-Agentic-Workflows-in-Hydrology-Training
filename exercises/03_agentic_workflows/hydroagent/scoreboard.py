"""Scoreboard: every finished agent run is logged with its cost, time and (optionally) a quality rating.

The point of the exercise is to find the architecture that strikes the right balance between
*answer quality* and *cost / time / predictability*. Tokens and time are measured automatically;
quality is judged by you (1-5) because only a hydrologist can say whether the answer is right.

File: ``output/scoreboard.jsonl`` (one JSON record per run; ratings are appended as separate records).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime

import pandas as pd

from . import config as C

FILE = C.OUTPUT_DIR / "scoreboard.jsonl"
METRICS = ["llm_calls", "tool_calls", "wflow_runs", "input_tokens", "output_tokens", "cache_read_tokens",
           "total_tokens", "wall_time_s", "llm_s", "model_s", "other_s", "cost_usd"]


def record(arch: str, question: str, outcome, label: str = "") -> str:
    """Append a finished run; returns its entry id (use it to add a rating)."""
    C.ensure_dirs()
    entry_id = uuid.uuid4().hex[:8]
    u = dict(outcome.usage or {})
    rec = {"id": entry_id, "type": "run", "time": datetime.now().isoformat(timespec="seconds"), "arch": arch,
           "label": label or arch, "model": u.get("model", ""), "question": question, "status": outcome.status,
           "n_scenarios": len(outcome.run_ids), "answer": (outcome.answer or outcome.error or "")[:1500],
           **{k: u.get(k) for k in METRICS}, "cost_source": u.get("cost_source"), "by_node": u.get("by_node", {})}
    with open(FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    return entry_id


def rate(entry_id: str, quality: int, note: str = "") -> None:
    """Rate an answer 1 (wrong/useless) .. 5 (correct, complete, well explained)."""
    if not 1 <= int(quality) <= 5:
        raise ValueError("quality must be 1..5")
    with open(FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": entry_id, "type": "rating", "quality": int(quality), "note": note,
                            "time": datetime.now().isoformat(timespec="seconds")}) + "\n")


def load() -> pd.DataFrame:
    """All runs, with the latest rating merged in (column ``quality``)."""
    if not FILE.exists():
        return pd.DataFrame(columns=["id", "arch", "question", "status", *METRICS, "quality"])
    rows = [json.loads(line) for line in FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    runs = pd.DataFrame([r for r in rows if r.get("type") == "run"])
    ratings = pd.DataFrame([r for r in rows if r.get("type") == "rating"])
    if runs.empty:
        return runs
    if not ratings.empty:
        last = ratings.groupby("id").last()[["quality", "note"]]
        runs = runs.merge(last, left_on="id", right_index=True, how="left")
    else:
        runs["quality"], runs["note"] = None, None
    return runs


def summary(df: pd.DataFrame | None = None, question: str | None = None) -> pd.DataFrame:
    """Mean metrics per architecture (+ model). Filter on one question for a fair comparison."""
    df = load() if df is None else df
    if df.empty:
        return df
    if question:
        df = df[df["question"] == question]
    agg = {"n": ("id", "count"), "success_rate": ("status", lambda s: round((s == "done").mean(), 2)),
           "quality": ("quality", "mean"), "cost_usd": ("cost_usd", "mean"), "total_tokens": ("total_tokens", "mean"),
           "llm_calls": ("llm_calls", "mean"), "wall_time_s": ("wall_time_s", "mean"), "llm_s": ("llm_s", "mean"),
           "model_s": ("model_s", "mean"), "wflow_runs": ("wflow_runs", "mean")}
    out = df.groupby(["arch", "model"]).agg(**agg).reset_index()
    # spread: how predictable is the architecture? (std of tokens between repeated runs)
    out["tokens_std"] = df.groupby(["arch", "model"])["total_tokens"].std().values
    return out.round({"quality": 2, "cost_usd": 4, "total_tokens": 0, "llm_calls": 1, "wall_time_s": 1, "llm_s": 1,
                      "model_s": 1, "wflow_runs": 1, "tokens_std": 0})
