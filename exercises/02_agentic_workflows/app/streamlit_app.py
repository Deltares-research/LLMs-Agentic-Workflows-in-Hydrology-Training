"""Exercise 2 - Agentic architectures for HydroMT/Wflow (Streamlit level).

Run from the repo root:
    uv run streamlit run exercises/02_agentic_workflows/app/streamlit_app.py
"""

from __future__ import annotations

import json
import queue
import re
import sys
import threading
import time
from pathlib import Path

import streamlit as st

EXERCISE_DIR = Path(__file__).resolve().parents[1]
if str(EXERCISE_DIR) not in sys.path:
    sys.path.insert(0, str(EXERCISE_DIR))

from streamlit_flow import streamlit_flow  # noqa: E402
from streamlit_flow.elements import StreamlitFlowEdge, StreamlitFlowNode  # noqa: E402
from streamlit_flow.state import StreamlitFlowState  # noqa: E402

from hydroagent import config as C  # noqa: E402
from hydroagent import scoreboard  # noqa: E402
from hydroagent import wflow_tools as W  # noqa: E402
from hydroagent.architectures import registry as R  # noqa: E402
from hydroagent.builder import edit as GE  # noqa: E402
from hydroagent.builder.compile import make_custom_session, validate  # noqa: E402
from hydroagent.builder.nodes import NODE_TYPES, TOOL_ACTIONS  # noqa: E402
from hydroagent.builder.templates import TEMPLATES  # noqa: E402
from hydroagent.harness.claude_runner import available_skills, run_harness  # noqa: E402
from hydroagent.tracing import Event, Recorder  # noqa: E402

st.set_page_config(page_title="Agentic hydrology - Exercise 2", layout="wide")
C.load_env()

KIND_COLOUR = {"code": "#f5f5f5", "llm": "#dae8fc", "agent": "#d5e8d4", "human": "#ffe6cc", "harness": "#d0cee2",
               "io": "#ffffff"}
NODE_KIND = {"simulate": "code", "plan": "code", "finalize": "code", "blocked": "code", "tools": "code",
             "clarify": "human", "approve": "human", "human_check": "human",
             "claude_harness": "harness", "open_ended_code": "harness",
             "model": "agent", "modeller": "agent", "analyst": "agent", "parameter_scenario": "agent",
             "__start__": "io", "__end__": "io"}
EVENT_ICON = {"llm": "🧠", "tool_call": "🔧", "tool_result": "📄", "node": "▶️", "human": "🙋", "error": "⛔",
              "harness": "🤖", "final": "✅", "info": "ℹ️", "model": "🌊"}


# ----------------------------------------------------------------------------------------------- helpers
def sidebar() -> dict:
    st.sidebar.header("Settings")
    providers = ["anthropic", "mistral", "azure"]
    default = C.get_settings().provider
    provider = st.sidebar.selectbox("LLM provider (LangGraph architectures)", providers,
                                    index=providers.index(default) if default in providers else 0,
                                    format_func={"anthropic": "Claude (anthropic)", "mistral": "Mistral (Foundry)",
                                                 "azure": "GPT (Azure OpenAI)"}.get)
    s = C.get_settings(provider=provider)
    model = {"anthropic": s.anthropic_model, "mistral": s.mistral_deployment, "azure": s.azure_deployment}[provider]
    st.sidebar.caption(f"Model: `{model}`. The A4 harness always uses Claude.")
    auth_modes = ["auto", "foundry", "api_key", "login"]
    auth = st.sidebar.selectbox("Claude harness auth", auth_modes,
                                index=auth_modes.index(s.harness_auth) if s.harness_auth in auth_modes else 0,
                                help="'auto' uses Azure Foundry if FOUNDRY_ENDPOINT/FOUNDRY_KEY are set, else the "
                                     "API key, else the Claude Code login on this machine.")
    import os

    os.environ["HARNESS_AUTH"] = auth
    st.sidebar.divider()
    st.sidebar.subheader("Model status")
    ready = W.base_model_ready()
    st.sidebar.write(("✅" if ready else "❌") + " base model built")
    base = (C.RUNS_DIR / "baseline" / "output" / "output.csv").exists()
    st.sidebar.write(("✅" if base else "⏳") + " baseline run")
    srv = W._SERVER is not None and W._SERVER.proc is not None and W._SERVER.proc.poll() is None
    st.sidebar.write(("✅" if srv else "💤") + " Wflow (Julia) warm" + (f" v{W._SERVER.version}" if srv else ""))
    st.sidebar.toggle("Stream HydroMT / Wflow console output", True, key="show_model_log")

    def with_model_log(fn):
        def go(on_ev):
            W.LOG_SINK.set(lambda src, line: on_ev(Event("model", src, line)))
            return fn()
        return go

    if not ready and st.sidebar.button("Build base model (HydroMT, ~2 min)"):
        box = st.status("hydromt build + update ...", expanded=True)
        run_live(with_model_log(W.prepare_base_model), box)
        st.rerun()
    if ready and (not base or not srv) and st.sidebar.button("Warm up Wflow (baseline run)"):
        box = st.status("Starting Julia and running the baseline (first time ~1.5 min) ...", expanded=True)
        run_live(with_model_log(lambda: W.get_server().run(W.ensure_baseline().path / "wflow_sbm.toml")
                                if base else W.ensure_baseline()), box)
        st.rerun()
    return {"provider": provider}


@st.cache_data(show_spinner=False)
def arch_dot(key: str) -> str:
    """Graphviz DOT for a pre-built architecture (static image: no rerun loops)."""
    nodes, edges = R.graph_structure(key)
    lines = ['digraph G { rankdir=TB; bgcolor="transparent"; size="5,5"; nodesep=0.3; ranksep=0.35; node [shape=box, style="rounded,filled", '
             'fontname="Helvetica", fontsize=10, height=0.3, margin="0.12,0.04", color="#888888"]; edge [color="#555555", fontsize=9];']
    for n in nodes:
        kind = NODE_KIND.get(n, "llm")
        label = {"__start__": "START", "__end__": "END"}.get(n, n)
        shape = ', shape=ellipse' if kind == "io" else ""
        lines.append(f'"{n}" [label="{label}", fillcolor="{KIND_COLOUR[kind]}"{shape}];')
    for s, t, lab in edges:
        style = 'style=dashed' if lab == "cond" else ""
        text = f'label="{lab}"' if lab and lab != "cond" else ""
        lines.append(f'"{s}" -> "{t}" [{", ".join(x for x in (style, text) if x)}];')
    lines.append("}")
    return "\n".join(lines)


def legend():
    st.caption(" ".join(f"<span style='background:{KIND_COLOUR[k]};border:1px solid #888;padding:1px 6px;"
                        f"border-radius:4px'>{v}</span>" for k, v in
                        [("code", "deterministic code"), ("llm", "LLM call"), ("agent", "agent (LLM+tools)"),
                         ("harness", "harness agent"), ("human", "human")]) + " &nbsp; dashed = conditional",
               unsafe_allow_html=True)


def run_live(fn, container) -> object:
    """Run ``fn(on_event)`` in a worker thread and stream its events into ``container``."""
    q: queue.Queue = queue.Queue()
    box: dict = {}

    def target():
        try:
            box["out"] = fn(q.put)
        except Exception as e:  # noqa: BLE001
            box["err"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    show_model = st.session_state.get("show_model_log", True)
    console, lines = None, []
    with container:
        while t.is_alive() or not q.empty():
            try:
                ev = q.get(timeout=0.3)
            except queue.Empty:
                continue
            if ev.kind == "model":
                # consecutive HydroMT/Wflow lines go into one scrolling console block
                if not show_model:
                    continue
                if console is None:
                    console, lines = st.empty(), []
                lines.append(f"[{ev.title}] {ev.detail}")
                console.code("\n".join(lines[-20:]), language=None)
                continue
            console = None
            render_event(ev)
    if "err" in box:
        raise box["err"]
    return box["out"]


def start_job(key: str, fn) -> None:
    """Run ``fn(on_event)`` as a background job that survives page reruns.

    Events are collected in ``st.session_state[key]["events"]`` and shown by ``job_feed``. Unlike ``run_live``,
    a rerun of the page (a click, the canvas sending its state) does not cut off the feed or lose the answer.
    """
    job = {"events": [], "out": None, "err": None, "done": False}

    def target():
        try:
            job["out"] = fn(job["events"].append)
        except Exception as e:  # noqa: BLE001
            job["err"] = e
        finally:
            job["done"] = True

    job["thread"] = threading.Thread(target=target, daemon=True)
    st.session_state[key] = job
    job["thread"].start()


def job_running(key: str) -> bool:
    job = st.session_state.get(key)
    return bool(job) and not job["done"]


def job_feed(key: str, title: str) -> None:
    """Live feed of a background job; refreshes every second while it runs, then shows the outcome once."""
    running = job_running(key)

    @st.fragment(run_every=1.0 if running else None)
    def feed():
        job = st.session_state.get(key)
        if not job:
            return
        state = "running" if not job["done"] else "error" if job["err"] or (
            job["out"] is not None and job["out"].status == "error") else "complete"
        label = f"{title}: running ... ({len(job['events'])} events)" if state == "running" else \
            f"{title}: {job['out'].status if job['out'] is not None else 'error'}"
        show_model = st.session_state.get("show_model_log", True)
        with st.status(label, state=state, expanded=state != "complete"):
            lines: list[str] = []
            for ev in list(job["events"]):
                if ev.kind == "model":
                    if show_model:
                        lines.append(f"[{ev.title}] {ev.detail}")
                    continue
                if lines:  # consecutive HydroMT/Wflow lines go into one console block
                    st.code("\n".join(lines[-20:]), language=None)
                    lines = []
                render_event(ev)
            if lines:
                st.code("\n".join(lines[-20:]), language=None)
        if running and job["done"]:
            st.rerun()  # full rerun: stop polling and show the answer below

    feed()


def render_event(ev):
    icon = EVENT_ICON.get(ev.kind, "•")
    if ev.kind in ("llm", "final") and ev.detail:
        st.markdown(f"{icon} **{ev.title}**")
        st.markdown(ev.detail[:2000])
    elif ev.detail:
        st.markdown(f"{icon} **{ev.title}** <span style='color:#666'>{_esc(ev.detail[:300])}</span>",
                    unsafe_allow_html=True)
    else:
        st.markdown(f"{icon} **{ev.title}**")


def _esc(s: str) -> str:
    return s.replace("<", "&lt;").replace(">", "&gt;").replace("\n", " ")


def fmt_cost(u: dict) -> str:
    c = u.get("cost_usd")
    return f"${c:.4f}" if isinstance(c, (int, float)) else "n/a"


def show_metrics(u: dict, key: str):
    """Cost / time / token panel for one run (same numbers that go to the scoreboard)."""
    c = st.columns(6)
    c[0].metric("LLM calls", u.get("llm_calls", "-"))
    c[1].metric("tool calls", u.get("tool_calls", "-"))
    c[2].metric("Wflow runs", u.get("wflow_runs", "-"))
    c[3].metric("tokens (total)", f"{u.get('total_tokens', 0):,}",
                help=f"input {u.get('input_tokens', 0):,} | output {u.get('output_tokens', 0):,} | "
                     f"cache read {u.get('cache_read_tokens', 0):,} | cache write {u.get('cache_write_tokens', 0):,}")
    c[4].metric("time [s]", u.get("wall_time_s", "-"),
                help="Wall time excluding human think-time and the one-off Julia start-up "
                     f"({u.get('julia_startup_s', 0)} s).")
    c[5].metric("cost (est.)", fmt_cost(u), help=u.get("cost_source", ""))
    wall = u.get("wall_time_s") or 0
    if wall:
        parts = {"LLM": u.get("llm_s", 0), "Wflow / HydroMT": u.get("model_s", 0), "other": u.get("other_s", 0)}
        st.caption("Where did the time go?  " + "  |  ".join(
            f"**{k}** {v:.1f} s ({100 * v / max(wall, 1e-9):.0f}%)" for k, v in parts.items()))
    if u.get("by_node"):
        with st.expander("Tokens and LLM time per step (node)"):
            st.dataframe([{"node": n, **v} for n, v in u["by_node"].items()], width="stretch", hide_index=True)


def rate_widget(entry_id: str, key: str):
    """Let the participant judge answer quality (stored on the scoreboard)."""
    if not entry_id:
        return
    st.markdown("**How good is this answer?** (1 = wrong/useless, 5 = correct, complete, well explained)")
    stars = st.feedback("stars", key=f"rate_{key}_{entry_id}")
    if stars is not None and st.session_state.get(f"rated_{entry_id}") != stars:
        scoreboard.rate(entry_id, stars + 1)
        st.session_state[f"rated_{entry_id}"] = stars
        st.toast(f"Saved rating {stars + 1}/5 to the scoreboard")


def show_outcome(out, key: str):
    show_metrics(out.usage or {}, key)
    if out.status == "error":
        st.error(out.error)
    if out.answer:
        st.markdown("#### Answer")
        st.markdown(out.answer)
    hist = (out.state.get("data") or {}).get("history") if isinstance(out.state, dict) else None
    if hist:
        st.markdown("#### Calibration history")
        st.dataframe([{"iter": h["iter"], **h["multipliers"], **h["score"]} for h in hist], width="stretch")
    for p in out.plots:
        if Path(p).exists():
            st.image(p)
    if out.run_ids and not out.plots:
        try:
            st.image(str(W.plot_runs(["baseline", *out.run_ids])))
        except Exception:
            pass
    if out.run_ids:
        st.caption("Runs: " + ", ".join(f"`{r}`" for r in out.run_ids) + f" (in `{C.RUNS_DIR}`)")
    rate_widget(getattr(out, "entry_id", ""), key)


# ----------------------------------------------------------------------------------------------- page 1
def page_architectures(cfg):
    st.title("1 · Pre-built agent architectures")
    st.markdown("Start simple and add structure only when you hit a clear limit. Run the **same question** on "
                "several architectures, then look at the trace: how many LLM calls, which tools, what guarantees?")
    keys = list(R.INFOS)
    key = st.radio("Architecture", keys, format_func=lambda k: R.INFOS[k].name, horizontal=False,
                   key="arch_choice")
    info = R.INFOS[key]
    left, right = st.columns([1, 1])
    with left:
        st.markdown(f"**Pattern (Agentic Overview):** {info.pattern}")
        st.markdown(info.summary)
        st.markdown(f"**Use when:** {info.use_when}")
        st.markdown(f"**Trade-off:** {info.trade_off}")
        if info.claude_only:
            st.info("Claude-only: runs through the Claude Agent SDK harness regardless of the provider setting.")
        if info.hitl:
            st.info("Human-in-the-loop: the agent may pause and ask you something below.")
        if key == "a4m":
            from hydroagent.harness.vibe_runner import vibe_executable

            st.info("Runs Mistral Vibe with your Mistral deployment, regardless of the provider setting. "
                    "Same workspace, skills and Wflow tools as A4: ask both the same question and compare.")
            if not vibe_executable():
                st.error("Mistral Vibe is not installed. Install it once with `uv tool install mistral-vibe`.")
    with right:
        st.graphviz_chart(arch_dot(key))  # natural size; "stretch" blows tall graphs up
        legend()

    opts = {}
    with st.expander("Options"):
        if key == "a2":
            opts["max_iter"] = st.slider("max iterations", 1, 10, 4)
            opts["target_kge"] = st.slider("target KGE", 0.5, 0.99, 0.95)
            opts["critic"] = st.radio("critic", ["llm", "rule"], horizontal=True,
                                      help="LLM critic vs a deterministic rule: which is better/cheaper?")
        elif key == "a3":
            opts["require_approval"] = st.checkbox("require human plan approval", True)
        elif key in ("a4", "a4m"):
            names = [s["name"] for s in available_skills()]
            opts["skills"] = st.multiselect("skills available to the agent", names, default=names)
            opts["max_turns"] = st.slider("max turns", 5, 60, 40)
        elif key == "a5":
            opts["confidence_threshold"] = st.slider("human check if router confidence below", 0.0, 1.0, 0.7)
        else:
            st.caption("No options. Edit the code in `hydroagent/architectures/` to change behaviour.")

    query = st.text_area("Your question", info.example_query, key=f"q_{key}", height=90)
    ss = st.session_state
    if st.button("▶ Run", type="primary", key=f"run_{key}"):
        try:
            sess = R.create_session(key, provider=cfg["provider"], **opts)
        except Exception as e:
            st.error(f"Could not create the agent: {e}")
            return
        ss["active"] = {"key": key, "session": sess, "query": query}
        box = st.status(f"Running {info.name} ...", expanded=True)
        out = run_live(lambda on_ev: (setattr(sess.rec, "on_event", on_ev), sess.start(query))[1], box)
        box.update(state="complete" if out.status != "error" else "error",
                   label=f"{info.name}: {out.status}", expanded=out.status != "done")
        ss["active"]["outcome"] = out
    act = ss.get("active")
    if act and act["key"] == key and "outcome" in act:
        out = act["outcome"]
        if out.status == "interrupted":
            q = out.question
            st.warning("The agent is waiting for you")
            if isinstance(q, dict):
                st.markdown(f"**{q.get('question', '')}**")
                st.json({k: v for k, v in q.items() if k != "question"})
            else:
                st.markdown(f"**{q}**")
            with st.form(f"hitl_{key}"):
                ans = st.text_input("Your answer", "yes")
                if st.form_submit_button("Send"):
                    sess = act["session"]
                    box = st.status("Resuming ...", expanded=True)
                    out = run_live(lambda on_ev: (setattr(sess.rec, "on_event", on_ev), sess.resume(ans))[1], box)
                    box.update(state="complete", label=f"{info.name}: {out.status}")
                    act["outcome"] = out
                    st.rerun()
        else:
            show_outcome(out, key)
            with st.expander("Full trace"):
                evs = act["session"].rec.events
                model_lines = [f"[{e.title}] {e.detail}" for e in evs if e.kind == "model"]
                for ev in evs:
                    if ev.kind != "model":
                        render_event(ev)
                if model_lines:
                    st.markdown(f"🌊 **HydroMT / Wflow console** ({len(model_lines)} lines)")
                    st.code("\n".join(model_lines[-400:]), language=None)


# ----------------------------------------------------------------------------------------------- page 2
def page_compare(cfg):
    st.title("2 · Compare architectures on the same question")
    st.markdown("Same question, different designs. Is the extra structure worth the extra calls, tokens and time?")
    keys = st.multiselect("Architectures", list(R.INFOS), default=["a0", "a1", "a3"],
                          format_func=lambda k: R.INFOS[k].name)
    query = st.text_area("Question", "What happens to peak and low flows at Epinal if soils become 30% "
                                     "less permeable?", height=80)
    auto = st.text_input("Automatic answer to human-in-the-loop questions", "yes",
                         help="A3 may ask for a magnitude (try '-30') and approval ('yes').")
    if st.button("▶ Run all", type="primary"):
        results = {}
        for k in keys:
            box = st.status(f"{R.INFOS[k].name}", expanded=False)
            try:
                sess = R.create_session(k, provider=cfg["provider"])
            except Exception as e:
                box.update(state="error", label=f"{k}: {e}")
                continue
            answers = [a.strip() for a in auto.split(",")]

            def go(on_ev, sess=sess):
                sess.rec.on_event = on_ev
                out = sess.start(query)
                i = 0
                while out.status == "interrupted" and i < 4:
                    out = sess.resume(answers[min(i, len(answers) - 1)])
                    i += 1
                return out

            out = run_live(go, box)
            box.update(state="complete" if out.status == "done" else "error", label=f"{R.INFOS[k].name}: {out.status}")
            results[k] = out
        st.session_state["compare"] = results
    res = st.session_state.get("compare") or {}
    if res:
        cols_keep = ["llm_calls", "tool_calls", "wflow_runs", "total_tokens", "wall_time_s", "llm_s", "model_s",
                     "other_s", "cost_usd"]
        st.dataframe([{"architecture": R.INFOS[k].name, "status": o.status,
                       **{c: o.usage.get(c) for c in cols_keep}} for k, o in res.items()],
                     width="stretch", hide_index=True)
        st.markdown("**Where did the time go?** (seconds)")
        st.bar_chart({k.upper(): {"LLM": o.usage.get("llm_s", 0), "Wflow/HydroMT": o.usage.get("model_s", 0),
                                  "other": o.usage.get("other_s", 0)} for k, o in res.items()},
                     horizontal=True, height=220)
        cols = st.columns(len(res))
        for c, (k, o) in zip(cols, res.items()):
            with c:
                st.markdown(f"**{R.INFOS[k].name}**")
                st.markdown(o.answer or o.error)
                rate_widget(o.entry_id, f"cmp_{k}")
        all_runs = [r for o in res.values() for r in o.run_ids]
        if all_runs:
            st.image(str(W.plot_runs(["baseline", *all_runs], title="All runs from this comparison")))
        st.markdown("**Discuss:** Which answer would you trust? Which was cheapest? Which gave guarantees "
                    "(validated inputs, approval, audit trail)? When would you pick each?")


# ----------------------------------------------------------------------------------------------- page 3
def _spec_to_flow(spec) -> StreamlitFlowState:
    nodes = []
    for n in spec["nodes"]:
        nt = NODE_TYPES[n["type"]]
        x, y = n.get("pos", (0, 0))
        nodes.append(StreamlitFlowNode(
            n["id"], (x, y), {"content": f"**{n['id']}**  \n{nt.label}", "ntype": n["type"],
                              "config": {**nt.defaults, **(n.get("config") or {})}},
            "input" if n["type"] == "start" else "output" if n["type"] == "end" else "default",
            connectable=True, deletable=n["type"] not in ("start",), selectable=True,
            style={"background": nt.colour, "border": "1px solid #888", "width": 190}))
    edges = [StreamlitFlowEdge(f"e{i}-{e['source']}-{e['target']}", e["source"], e["target"], label=e.get("label", ""),
                               marker_end={"type": "arrowclosed"}, deletable=True, label_show_bg=True)
             for i, e in enumerate(spec["edges"])]
    return StreamlitFlowState(nodes, edges)


def _flow_to_spec(state: StreamlitFlowState, name: str) -> dict:
    return {"name": name,
            "nodes": [{"id": n.id, "type": n.data.get("ntype", "plain_llm"), "pos": [n.position["x"], n.position["y"]],
                       "config": n.data.get("config", {})} for n in state.nodes],
            "edges": [{"source": e.source, "target": e.target, "label": e.label or ""} for e in state.edges]}


def _edit_flow(fn, *args) -> None:
    """Button callback: apply a graph edit (from hydroagent.builder.edit) to the canvas state.

    All builder buttons use callbacks: they run before the page script, so a rerun sent by the canvas component
    (Firefox does this whenever the canvas is redrawn) cannot swallow the click.
    """
    ss = st.session_state
    ss["flow"] = _spec_to_flow(fn(_flow_to_spec(ss["flow"], ss["flow_name"]), *args))


def _save_node(nid: str, widget_keys: dict[str, str]) -> None:
    ss = st.session_state
    spec = _flow_to_spec(ss["flow"], ss["flow_name"])
    for sn in spec["nodes"]:
        if sn["id"] == nid:
            sn["config"] = {k: ss[wk] for k, wk in widget_keys.items()}
    ss["flow"] = _spec_to_flow(spec)  # new state -> the canvas reloads with the new config
    ss["form_ver"] = ss.get("form_ver", 0) + 1  # fresh widgets that show the saved config


def _save_edge_label(edge_id: str) -> None:
    ss = st.session_state
    spec = _flow_to_spec(ss["flow"], ss["flow_name"])
    for se, fe in zip(spec["edges"], ss["flow"].edges):
        if fe.id == edge_id:
            se["label"] = ss[f"edge_label_{edge_id}_{ss.get('form_ver', 0)}"]
    ss["flow"] = _spec_to_flow(spec)
    ss["form_ver"] = ss.get("form_ver", 0) + 1


def _save_graph() -> None:
    ss = st.session_state
    spec = _flow_to_spec(ss["flow"], ss["flow_name"])
    C.ensure_dirs()
    p = C.GRAPHS_DIR / f"{spec['name']}.json"
    p.write_text(json.dumps(spec, indent=1))
    ss["graph_saved_to"] = str(p)


def _fix_buttons(spec: dict, ss) -> None:
    """One-click fixes for steps with too many or no next steps."""
    ids = [n["id"] for n in spec["nodes"] if n["type"] != "start"]
    for k, pr in enumerate(GE.problems(spec)):
        nid = pr["node"]
        cols = st.columns([2, *([1] * max(len(pr["targets"]), 2))])
        if pr["kind"] == "too_many":
            cols[0].markdown(f"**{nid}** goes to several steps. Keep only:")
            for c, tgt in zip(cols[1:], dict.fromkeys(pr["targets"])):
                c.button(f"→ {tgt}", key=f"fix_keep_{k}_{tgt}", on_click=_edit_flow,
                         args=(GE.keep_only, nid, tgt))
        else:
            cols[0].markdown(f"**{nid}** has no next step. Connect to:")
            tgt = cols[1].selectbox("target", [i for i in ids if i != nid], key=f"fix_tgt_{k}",
                                    label_visibility="collapsed")
            cols[2].button("Connect", key=f"fix_conn_{k}", on_click=_edit_flow, args=(GE.connect, nid, tgt))


def _start_custom_run(provider: str) -> None:
    """Button callback: compile the graph on the canvas and start it as a background job."""
    ss = st.session_state
    spec = _flow_to_spec(ss["flow"], ss["flow_name"])
    query = ss.get("custom_query", "")
    sess = make_custom_session(spec, provider=provider)
    ss["custom_session"] = sess
    start_job("custom_job", lambda on_ev: (setattr(sess.rec, "on_event", on_ev), sess.start(query))[1])


def _resume_custom_run() -> None:
    """Form callback: answer the agent's human-in-the-loop question and continue in the background."""
    ss = st.session_state
    sess, ans = ss["custom_session"], ss.get("custom_answer", "yes")
    start_job("custom_job", lambda on_ev: (setattr(sess.rec, "on_event", on_ev), sess.resume(ans))[1])


def page_builder(cfg):
    st.title("3 · Build your own agent")
    st.markdown("Start from a template. To add a step, use **Add a node** in the sidebar and choose the connection "
                "to insert it on: it is wired in for you (a Critic also gets its *revise* loop). Click a node to edit "
                "its settings. Branch labels are filled in automatically; click an edge to change one. "
                "You can still drag from a node's bottom handle to another node's top to connect by hand, and "
                "right-click a node or edge to delete it.")
    ss = st.session_state
    c1, c2 = st.columns([2, 1])
    with c1:
        tpl = st.selectbox("Start from template", list(TEMPLATES))
    with c2:
        st.write("")
        if st.button("Load template") or "flow" not in ss:
            ss["flow"] = _spec_to_flow(TEMPLATES[tpl])
            ss["flow_name"] = TEMPLATES[tpl]["name"]
    with st.sidebar:
        st.divider()
        st.subheader("Add a node")
        ntype = st.selectbox("Node type", [t for t in NODE_TYPES if t != "start"],
                             format_func=lambda t: NODE_TYPES[t].label)
        st.caption(NODE_TYPES[ntype].description)
        nid = st.text_input("Node id", f"{ntype}_{len(ss['flow'].nodes)}")
        cur = _flow_to_spec(ss["flow"], ss["flow_name"])
        conns = [(e["source"], e["target"]) for e in cur["edges"]]
        into_end = [i for i, (s, tg) in enumerate(conns)
                    if any(n["id"] == tg and n["type"] == "end" for n in cur["nodes"])]
        where = st.selectbox("Insert on connection", [*range(len(conns)), None],
                             index=into_end[0] if into_end else len(conns),
                             format_func=lambda i: "not connected (wire it yourself)" if i is None
                             else f"{conns[i][0]} → {conns[i][1]}")
        if st.button("➕ Add node"):
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,30}", nid) or nid in {n.id for n in ss["flow"].nodes}:
                st.error("Id must be unique, start with a letter, and contain only letters, digits and _.")
            else:
                new = {"id": nid, "type": ntype}
                if where is None:
                    cur["nodes"].append({**new, "pos": [300, 60 * len(cur["nodes"])]})
                else:
                    cur = GE.insert_on_edge(cur, new, *conns[where])
                ss["flow"] = _spec_to_flow(cur)
                st.rerun()

    left_col, run_col = st.columns([3, 2], gap="large")
    with left_col:
        ss["flow"] = streamlit_flow("builder", ss["flow"], height=560, fit_view=True, allow_new_edges=True,
                                    animate_new_edges=False, get_node_on_click=True, get_edge_on_click=True,
                                    enable_node_menu=True, enable_edge_menu=True, show_minimap=False)
        sel = ss["flow"].selected_id
        nodes = {n.id: n for n in ss["flow"].nodes}
        edges = {e.id: e for e in ss["flow"].edges}
        if sel in nodes and nodes[sel].data.get("ntype") not in ("start", "end"):
            n = nodes[sel]
            nt = NODE_TYPES[n.data["ntype"]]
            with st.form(f"edit_{sel}"):
                st.markdown(f"**Edit node `{sel}`**: {nt.label}. {nt.description}")
                cfgd = dict(n.data.get("config", {}))
                ver = ss.get("form_ver", 0)
                wkeys = {k: f"cfg_{sel}_{k}_{ver}" for k in cfgd}
                for k, v in cfgd.items():
                    wkey = wkeys[k]
                    if k == "toolsets":
                        from hydroagent.tools import TOOLSETS

                        st.multiselect("tool sets", list(TOOLSETS), default=v, key=wkey,
                                       help=", ".join(f"{s}: {[t.name for t in ts]}" for s, ts in TOOLSETS.items()))
                    elif k == "action":
                        st.selectbox("action", list(TOOL_ACTIONS), index=list(TOOL_ACTIONS).index(v), key=wkey,
                                     format_func=lambda a: f"{a}: {TOOL_ACTIONS[a]}")
                    elif isinstance(v, int):
                        st.number_input(k, value=v, step=1, key=wkey)
                    else:
                        st.text_area(k, v, height=100, key=wkey)
                st.form_submit_button("Save node", on_click=_save_node, args=(sel, wkeys))
        elif sel in edges:
            e = edges[sel]
            with st.form(f"edge_{sel}"):
                src_type = nodes[e.source].data.get("ntype") if e.source in nodes else ""
                hint = {"critic": "'approved' or 'revise'", "human": "'approved' or 'rejected'",
                        "router": "a short description of when to take this branch"}.get(src_type, "(no label needed)")
                st.text_input(f"Label for edge {e.source} → {e.target} ({hint})", e.label or "",
                              key=f"edge_label_{sel}_{ss.get('form_ver', 0)}")
                st.form_submit_button("Save edge label", on_click=_save_edge_label, args=(sel,))

        spec = _flow_to_spec(ss["flow"], ss["flow_name"])
        labelled, notes = GE.auto_label(spec)
        sig = json.dumps(spec["edges"], sort_keys=True)
        if notes and ss.get("_auto_labelled") != sig:  # show the filled-in labels on the canvas (once per graph)
            ss["_auto_labelled"] = sig
            ss["flow"] = _spec_to_flow(labelled)
            ss["label_notes"] = notes
            st.rerun()
        for note in ss.pop("label_notes", []):
            st.info(note)
        errs = validate(spec)
        if errs:
            st.error("Graph not valid:\n\n- " + "\n- ".join(errs))
            _fix_buttons(spec, ss)
        else:
            st.success("Graph is valid ✔")
        cB, cC = st.columns(2)
        with cB:
            st.download_button("⬇ Download graph JSON", json.dumps(spec, indent=1), f"{spec['name']}.json")
            st.caption("Run it from the CLI: `cli.py build --graph my.json \"question\"`")
        with cC:
            st.button("💾 Save to output/graphs", on_click=_save_graph)
            if ss.get("graph_saved_to"):
                st.caption(f"saved {ss.pop('graph_saved_to')}")


    with run_col:
        st.subheader("Run")
        st.text_area("Question for your agent", "How sensitive is the Epinal peak flow to doubling soil thickness?",
                     key="custom_query")
        busy = job_running("custom_job")
        if errs:
            st.caption("Fix the graph (left) to enable Run.")
        if busy:
            st.caption("Your agent is running; the feed below updates every second.")
        # on_click callbacks run before the page script, so a rerun sent by the canvas component (Firefox does
        # this when the canvas is redrawn) cannot swallow the click.
        st.button("▶ Run my agent", type="primary", disabled=bool(errs) or busy, width="stretch",
                  on_click=_start_custom_run, args=(cfg["provider"],))
        job_feed("custom_job", "custom agent")
        job = ss.get("custom_job")
        if job and job["done"]:
            out = job["out"]
            if job["err"] is not None:
                st.error(f"{type(job['err']).__name__}: {job['err']}")
            elif out.status == "interrupted":
                st.warning(f"Your agent asks: {out.question}")
                with st.form("custom_hitl"):
                    st.text_input("Answer", "yes", key="custom_answer")
                    st.form_submit_button("Send", on_click=_resume_custom_run)
            else:
                show_outcome(out, "custom")


# ----------------------------------------------------------------------------------------------- page 4
def page_skills(cfg):
    st.title("4 · Skills lab (Claude Agent SDK)")
    st.markdown("A **skill** is a folder with a `SKILL.md`. The agent only sees each skill's *description* until "
                "it decides to load it (progressive disclosure). Write a skill, then check whether the harness "
                "picks it up and whether it helps.")
    skills = available_skills()
    st.dataframe([{k: s[k] for k in ("name", "description")} for s in skills], width="stretch")
    pick = st.selectbox("View a skill", [s["name"] for s in skills])
    s = next(x for x in skills if x["name"] == pick)
    with st.expander(f"{s['path']}"):
        st.code(Path(s["path"]).read_text(encoding="utf-8"), language="markdown")

    st.subheader("Write a new skill")
    template = (C.EXERCISE_DIR / "skills" / "_template" / "SKILL.md").read_text(encoding="utf-8")
    name = st.text_input("Skill name (lowercase, hyphens)", "flow-duration-curve")
    body = st.text_area("SKILL.md", template.replace("my-skill-name", name), height=320)
    if st.button("💾 Save skill"):
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,40}", name):
            st.error("Use lowercase letters, digits and hyphens.")
        else:
            d = C.EXERCISE_DIR / "skills" / name
            d.mkdir(parents=True, exist_ok=True)
            (d / "SKILL.md").write_text(body, encoding="utf-8")
            st.success(f"Saved {d / 'SKILL.md'}")
            st.rerun()

    st.subheader("Test with the harness")
    names = [s["name"] for s in available_skills()]
    use = st.multiselect("Skills available in this run", names, default=names)
    task = st.text_area("Task", "Make a flow-duration curve for Epinal for the baseline run and the most recent "
                                "scenario run, and report Q10, Q50 and Q90 for both.", height=80)
    if st.button("▶ Run harness", type="primary"):
        rec = Recorder()
        box = st.status("Claude harness running ...", expanded=True)
        r = run_live(lambda on_ev: (setattr(rec, "on_event", on_ev), run_harness(task, skills=use, recorder=rec))[1], box)
        box.update(state="complete" if r.status == "done" else "error", label=f"harness: {r.status}")
        st.markdown(f"**Skills loaded:** {', '.join(r.skills_loaded) or 'none'}  |  workspace `{r.workspace}`")
        from hydroagent.runtime import Outcome

        out = Outcome("done" if r.status == "done" else "error", answer=r.answer, run_ids=r.run_ids,
                      usage=r.usage, error=r.error)
        label = "a4 skills:" + (",".join(sorted(use)) or "none")
        out.entry_id = scoreboard.record("a4-skills-lab", task, out, label=label)
        show_metrics(r.usage, "skills")
        rate_widget(out.entry_id, "skills")
        if r.error:
            st.error(r.error)
        st.markdown(r.answer)
        for p in r.plots:
            st.image(p)


# ----------------------------------------------------------------------------------------------- page 5
def page_scoreboard(cfg):
    st.title("5 · Scoreboard: which architecture strikes the right balance?")
    st.markdown("Every run is logged automatically with its **tokens, time and estimated cost**. You add the "
                "**quality** (stars under each answer, or edit the table below). Compare architectures on the "
                "*same question*, ideally with 2-3 repeats each: agents are not deterministic.")
    df = scoreboard.load()
    if df.empty:
        st.info("No runs yet. Run some architectures on pages 1-3 first.")
        return
    questions = ["(all questions)"] + sorted(df["question"].dropna().unique().tolist())
    q = st.selectbox("Question", questions, index=0)
    sub = df if q == questions[0] else df[df["question"] == q]
    summ = scoreboard.summary(sub)
    summ["tokens_per_quality_pt"] = (summ["total_tokens"] / summ["quality"]).round(0)
    summ["cost_per_quality_pt"] = (summ["cost_usd"] / summ["quality"]).round(5)
    st.subheader("Per architecture (means)")
    st.dataframe(summ, width="stretch", hide_index=True,
                 column_config={"quality": st.column_config.NumberColumn("quality (1-5)", format="%.1f"),
                                "cost_usd": st.column_config.NumberColumn("cost $", format="$%.4f"),
                                "tokens_std": st.column_config.NumberColumn("token spread (std)",
                                                                            help="Higher = less predictable")})
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Quality vs tokens**: top-left is the sweet spot")
        rated = sub.dropna(subset=["quality"])
        if rated.empty:
            st.caption("Rate some answers to see this chart.")
        else:
            st.scatter_chart(rated, x="total_tokens", y="quality", color="arch", size="wall_time_s", height=320)
    with c2:
        st.markdown("**Where does the time go?** (mean seconds)")
        st.bar_chart(summ.set_index("arch")[["llm_s", "model_s"]].rename(
            columns={"llm_s": "LLM", "model_s": "Wflow/HydroMT"}), horizontal=True, height=320)

    st.subheader("All runs (edit the quality column to rate)")
    cols = ["id", "time", "arch", "model", "status", "quality", "total_tokens", "llm_calls", "wflow_runs",
            "wall_time_s", "cost_usd", "question", "answer"]
    view = sub[[c for c in cols if c in sub.columns]].sort_values("time", ascending=False).reset_index(drop=True)
    view["quality"] = view["quality"].astype("float")
    edited = st.data_editor(view, width="stretch", hide_index=True, key="sb_editor",
                            disabled=[c for c in view.columns if c != "quality"],
                            column_config={"quality": st.column_config.NumberColumn(min_value=1, max_value=5, step=1)})
    changed = edited[edited["quality"].fillna(-1) != view["quality"].fillna(-1)]
    if not changed.empty and st.button(f"Save {len(changed)} rating(s)"):
        for _, r in changed.iterrows():
            if r["quality"] == r["quality"]:  # not NaN
                scoreboard.rate(r["id"], int(r["quality"]))
        st.rerun()
    st.download_button("Download scoreboard (CSV)", sub.drop(columns=["by_node"], errors="ignore").to_csv(index=False),
                       "scoreboard.csv")
    st.markdown("""**Questions for the debrief**
- Which architecture gives the best quality *per token* (or per dollar)? Is the most expensive one the best?
- How much of the time is the LLM vs. the hydrological model? Would a faster/cheaper LLM change the ranking?
- Which architectures are *predictable* (small token spread, always the same steps)? Does that matter operationally?
- For which questions is the simple design (A0/A1) good enough, and where do you need A3/A4?""")


# ----------------------------------------------------------------------------------------------- main
cfg = sidebar()
pages = {"1 · Architectures": page_architectures, "2 · Compare": page_compare, "3 · Build your own": page_builder,
         "4 · Skills lab": page_skills, "5 · Scoreboard": page_scoreboard}
choice = st.sidebar.radio("Page", list(pages), index=0, key="page")
st.sidebar.caption(f"Runs: `{C.RUNS_DIR}`")
pages[choice](cfg)
