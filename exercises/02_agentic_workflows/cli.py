"""Exercise 2 CLI: run, compare and build agentic architectures for HydroMT/Wflow.

Run from the repo root, e.g.:
    uv run python exercises/02_agentic_workflows/cli.py prepare
    uv run python exercises/02_agentic_workflows/cli.py list
    uv run python exercises/02_agentic_workflows/cli.py run a1 "What if soils were 30% less permeable?"
    uv run python exercises/02_agentic_workflows/cli.py run a2 "Calibrate ..." -o max_iter=3 -o critic=rule
    uv run python exercises/02_agentic_workflows/cli.py compare a0 a1 a3 "What if ..."
    uv run python exercises/02_agentic_workflows/cli.py build --graph my_graph.json "question"
    uv run python exercises/02_agentic_workflows/cli.py harness "Halve ksat in sub-catchment 9116039 ..."
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

EXERCISE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXERCISE_DIR))
try:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles
except Exception:
    pass

from hydroagent import config as C  # noqa: E402

app = typer.Typer(add_completion=False, help=__doc__.split("\n")[0])

COLOURS = {"model": "bright_black", "llm": "cyan", "tool_call": "yellow", "tool_result": "white", "node": "magenta", "human": "green",
           "error": "red", "final": "green", "harness": "blue", "info": "white"}


def _printer(verbose: bool):
    def on_event(ev):
        if ev.kind == "tool_result" and not verbose:
            return
        detail = ev.detail if verbose else ev.detail[:240].replace("\n", " ")
        typer.secho(f"[{ev.kind}] {ev.title}", fg=COLOURS.get(ev.kind, "white"), bold=ev.kind in ("final", "error"),
                    nl=False)
        typer.echo(f"  {detail}" if detail and ev.kind != "final" else "")
    return on_event


def _opts(opt: list[str]) -> dict:
    out = {}
    for o in opt or []:
        k, _, v = o.partition("=")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def _ask(q):
    if isinstance(q, dict):
        typer.secho(f"\n[agent asks] {q.get('question')}", fg="green", bold=True)
        typer.echo(json.dumps({k: v for k, v in q.items() if k != "question"}, indent=1, default=str))
    else:
        typer.secho(f"\n[agent asks] {q}", fg="green", bold=True)
    return typer.prompt(">")


def _metrics(u: dict) -> None:
    cost = u.get("cost_usd")
    typer.secho("\n=== COST / TIME ===", bold=True, fg="cyan")
    typer.echo(f"LLM calls {u.get('llm_calls')} | tool calls {u.get('tool_calls')} | Wflow runs {u.get('wflow_runs')}")
    typer.echo(f"tokens: input {u.get('input_tokens', 0):,}  output {u.get('output_tokens', 0):,}  "
               f"cache read {u.get('cache_read_tokens', 0):,}  cache write {u.get('cache_write_tokens', 0):,}  "
               f"-> total {u.get('total_tokens', 0):,}")
    typer.echo(f"time:   {u.get('wall_time_s')} s = LLM {u.get('llm_s')} + Wflow/HydroMT {u.get('model_s')} + other "
               f"{u.get('other_s')}   (Julia start-up {u.get('julia_startup_s', 0)} s not counted)")
    typer.echo(f"cost:   {'$%.4f' % cost if isinstance(cost, (int, float)) else 'n/a'}  ({u.get('cost_source')})")
    for node, v in (u.get("by_node") or {}).items():
        typer.echo(f"   {node:22s} {v['llm_calls']:3d} calls {v['input_tokens'] + v['output_tokens']:8,d} tok "
                   f"{v['llm_s']:6.1f} s")


def _report(out):
    typer.secho("\n=== ANSWER ===", bold=True)
    typer.echo(out.answer or out.error)
    _metrics(out.usage)
    if getattr(out, "entry_id", ""):
        typer.secho(f"scoreboard entry {out.entry_id}: rate it with  cli.py rate {out.entry_id} <1-5>", fg="green")
    if out.run_ids:
        typer.echo(f"runs: {out.run_ids}")
    for p in out.plots:
        typer.echo(f"plot: {p}")


@app.command()
def prepare(force: bool = False):
    """Build the Moselle model with HydroMT (once), run the baseline, and create the twin 'observations'."""
    from hydroagent import twin
    from hydroagent import wflow_tools as W

    W.LOG_SINK.set(lambda src, line: typer.secho(f"[{src}] {line}", fg="bright_black"))  # stream model output
    typer.echo("HydroMT build + forcing update ..." if force or not W.base_model_ready() else "Base model present.")
    W.prepare_base_model(force=force)
    typer.echo("Baseline run (first Julia start ~1-1.5 min) ...")
    b = W.ensure_baseline()
    typer.echo(f"baseline: {b.status} {b.error or ''}")
    twin.ensure_truth()
    typer.echo(json.dumps(W.summarise_run("baseline", reference=""), indent=1))


@app.command("list")
def list_():
    """List architectures, parameters and skills."""
    from hydroagent import wflow_tools as W
    from hydroagent.architectures.registry import INFOS
    from hydroagent.harness.claude_runner import available_skills

    for k, i in INFOS.items():
        typer.secho(f"{k}  {i.name}", bold=True)
        typer.echo(f"    pattern : {i.pattern}\n    use when: {i.use_when}\n    example : {i.example_query}")
    typer.secho("\nParameters (multiplier ranges):", bold=True)
    for p in W.list_parameters():
        typer.echo(f"  {p['key']:24s} x{p['allowed_multiplier']}  {p['description'][:70]}")
    typer.secho("\nSkills:", bold=True)
    for s in available_skills():
        typer.echo(f"  {s['name']:18s} {s['description'][:90]}")


@app.command()
def run(arch: str, query: str = typer.Argument(None), provider: str = typer.Option(None, "--provider", "-p"),
        opt: list[str] = typer.Option(None, "--opt", "-o", help="architecture option key=value (repeatable)"),
        verbose: bool = typer.Option(False, "--verbose", "-v")):
    """Run one architecture (a0..a5). Without QUERY its example question is used."""
    from hydroagent.architectures.registry import INFOS, create_session
    from hydroagent.runtime import run_interactive

    sess = create_session(arch, provider=provider, on_event=_printer(verbose), **_opts(opt))
    q = query or INFOS[arch].example_query
    out = run_interactive(sess, q, ask=_ask) if arch != "a4" else sess.start(q)
    _report(out)


@app.command()
def compare(archs: list[str] = typer.Argument(..., help="architecture keys followed by the question"),
            provider: str = typer.Option(None, "--provider", "-p"),
            auto_answer: str = typer.Option("yes", help="answer(s) for human questions, comma-separated")):
    """Run the same question on several architectures: compare cost, time and answers."""
    from hydroagent import wflow_tools as W
    from hydroagent.architectures.registry import INFOS, create_session

    *keys, query = archs
    answers = [a.strip() for a in auto_answer.split(",")]
    rows = []
    for k in keys:
        typer.secho(f"\n##### {INFOS[k].name}", bold=True)
        sess = create_session(k, provider=provider, on_event=_printer(False))
        out, i = sess.start(query), 0
        while out.status == "interrupted" and i < 4:
            out = sess.resume(answers[min(i, len(answers) - 1)])
            i += 1
        _report(out)
        rows.append((k, out))
    typer.secho("\n===== SUMMARY =====", bold=True)
    typer.echo(f"{'arch':5s} {'status':8s} {'llm':>4s} {'tools':>5s} {'tokens':>9s} {'time_s':>7s} {'llm_s':>6s} "
               f"{'wflow_s':>7s} {'cost':>8s} entry")
    for k, o in rows:
        u = o.usage
        cost = u.get("cost_usd")
        typer.echo(f"{k:5s} {o.status:8s} {u.get('llm_calls', 0):4d} {u.get('tool_calls', 0):5d} "
                   f"{u.get('total_tokens', 0):9,d} {u.get('wall_time_s', 0):7.1f} {u.get('llm_s', 0):6.1f} "
                   f"{u.get('model_s', 0):7.1f} {('$%.4f' % cost) if isinstance(cost, (int, float)) else 'n/a':>8s} "
                   f"{o.entry_id}")
    typer.echo("Rate the answers: cli.py rate <entry> <1-5>; overview: cli.py scoreboard")
    runs = [r for _, o in rows for r in o.run_ids]
    if runs:
        typer.echo(f"plot: {W.plot_runs(['baseline', *runs], title='compare')}")


@app.command()
def build(query: str, graph: Path = typer.Option(..., "--graph", "-g", help="graph JSON (from the Streamlit builder)"),
          provider: str = typer.Option(None, "--provider", "-p"), verbose: bool = False):
    """Run a custom graph (JSON spec) built in the Streamlit canvas or by hand."""
    from hydroagent.builder.compile import load_spec, make_custom_session, validate
    from hydroagent.runtime import run_interactive

    spec = load_spec(graph)
    errs = validate(spec)
    if errs:
        typer.secho("Invalid graph:\n- " + "\n- ".join(errs), fg="red")
        raise typer.Exit(1)
    out = run_interactive(make_custom_session(spec, provider=provider, on_event=_printer(verbose)), query, ask=_ask)
    _report(out)


@app.command()
def templates(out_dir: Path = typer.Option(C.GRAPHS_DIR, help="where to write the JSON files")):
    """Write the builder templates as JSON files to edit by hand."""
    from hydroagent.builder.templates import TEMPLATES

    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in TEMPLATES.values():
        p = out_dir / f"{spec['name']}.json"
        p.write_text(json.dumps(spec, indent=1))
        typer.echo(p)


@app.command()
def harness(task: str, skills: list[str] = typer.Option(None, "--skill", "-s", help="limit to these skills"),
            max_turns: int = 40, verbose: bool = False):
    """Run the Claude Agent SDK harness (A4) on a task, with skills."""
    from hydroagent.harness.claude_runner import run_harness

    from hydroagent import scoreboard
    from hydroagent.runtime import Outcome

    r = run_harness(task, skills=skills or None, max_turns=max_turns, on_event=_printer(verbose))
    typer.secho("\n=== ANSWER ===", bold=True)
    typer.echo(r.answer or r.error)
    typer.echo(f"\nskills loaded: {r.skills_loaded}\nworkspace: {r.workspace}\nruns: {r.run_ids}")
    _metrics(r.usage)
    out = Outcome("done" if r.status == "done" else "error", answer=r.answer, run_ids=r.run_ids, usage=r.usage)
    eid = scoreboard.record("a4-cli", task, out, label="skills:" + ",".join(skills or ["all"]))
    typer.secho(f"scoreboard entry {eid}: rate it with  cli.py rate {eid} <1-5>", fg="green")


@app.command()
def rate(entry: str, quality: int, note: str = ""):
    """Rate an answer on the scoreboard: 1 (wrong/useless) .. 5 (correct, complete, well explained)."""
    from hydroagent import scoreboard

    scoreboard.rate(entry, quality, note)
    typer.echo(f"rated {entry}: {quality}/5")


@app.command("scoreboard")
def scoreboard_(question: str = typer.Option(None, "--question", "-q", help="only runs of this exact question"),
                runs_: bool = typer.Option(False, "--runs", help="also list individual runs")):
    """Compare architectures on quality (your ratings), tokens, time and cost."""
    import pandas as pd

    from hydroagent import scoreboard

    df = scoreboard.load()
    if df.empty:
        typer.echo("Scoreboard is empty: run some architectures first.")
        raise typer.Exit()
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    typer.echo(scoreboard.summary(df, question).to_string(index=False))
    if runs_:
        cols = ["id", "time", "arch", "status", "quality", "total_tokens", "wall_time_s", "cost_usd", "question"]
        typer.echo("\n" + df[cols].sort_values("time").to_string(index=False, max_colwidth=50))


@app.command()
def runs():
    """List scenario runs made so far."""
    from hydroagent import wflow_tools as W

    for r in W.list_runs():
        typer.echo(f"{r.run_id:45s} {r.status:7s} {json.dumps(r.multipliers)} {r.label}")


@app.command()
def reveal_truth():
    """(Instructor) show the hidden multipliers of the twin experiment."""
    from hydroagent import twin

    typer.echo(json.dumps(twin.reveal_truth()))


if __name__ == "__main__":
    app()
