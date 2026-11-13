"""Notebook helper: answer an agent's human-in-the-loop questions with buttons instead of typed code.

    from hydroagent.notebook_ui import run_with_buttons
    run_with_buttons(session, "What if soils become less permeable?", report)

The session starts in the cell. Whenever the agent pauses to ask something, the cell shows:
* **Approve / Reject** buttons for yes/no questions (e.g. plan approval), plus a box for a comment;
* a text box + **Send** for open questions (e.g. "by how many percent?").
Clicking resumes the agent; its trace streams live under the buttons, followed by the result.

Only core ipywidgets controls (Button, Text, HTML) are used. The ``Output`` widget is avoided on purpose: VS Code's
widget renderer does not ship it ("No version of module @jupyter-widgets/output is registered"). Output produced
after a click (prints and ``display`` calls) is captured and shown in an HTML widget instead.
The latest outcome is kept in ``state["out"]`` of the returned dict.
"""

from __future__ import annotations

import base64
import html
import io
import sys
import time
from typing import Any, Callable

import ipywidgets as w
from IPython.display import display
from IPython.utils.capture import capture_output


def _question_text(q: Any) -> str:
    return str(q.get("question", q)) if isinstance(q, dict) else str(q)


def _is_yes_no(q: Any) -> bool:
    t = _question_text(q).lower()
    return any(k in t for k in ("yes / no", "yes/no", "approve", "accept"))


def _md_to_html(text: str) -> str:
    try:
        import mistune

        return mistune.html(text)
    except Exception:  # noqa: BLE001 - fall back to plain text
        return f"<pre style='white-space:pre-wrap'>{html.escape(text)}</pre>"


def _rich_to_html(data: dict) -> str:
    """One captured display() output (mime bundle) as HTML."""
    if "image/png" in data:
        png = data["image/png"]
        png = png if isinstance(png, str) else base64.b64encode(png).decode()
        return f"<img src='data:image/png;base64,{png}' style='max-width:100%'/>"
    if "text/html" in data:
        return data["text/html"]
    if "text/markdown" in data:
        return _md_to_html(data["text/markdown"])
    return f"<pre style='white-space:pre-wrap'>{html.escape(data.get('text/plain', ''))}</pre>"


class _LiveText(io.TextIOBase):
    """stdout replacement that mirrors printed text into an HTML widget (throttled), so the trace is live."""

    def __init__(self, widget: w.HTML, keep: int = 60):
        self.widget, self.keep, self.lines, self._last = widget, keep, [""], 0.0

    def write(self, s: str) -> int:
        parts = s.split("\n")
        self.lines[-1] += parts[0]
        self.lines += parts[1:]
        if time.time() - self._last > 0.3:
            self.flush()
        return len(s)

    def flush(self) -> None:
        self._last = time.time()
        tail = "\n".join(self.lines[-self.keep:])
        self.widget.value = ("<pre style='white-space:pre-wrap;font-size:11px;max-height:320px;overflow:auto;"
                             f"background:#f7f7f7;padding:6px'>{html.escape(tail)}</pre>")


def run_with_buttons(session, query: str, report: Callable[[Any], None]) -> dict:
    """Start ``session`` on ``query`` and handle every pause with buttons. Returns ``{"out": latest outcome}``."""
    state: dict = {}
    out = session.start(query)
    state["out"] = out
    report(out)  # first part runs in the cell itself: normal output
    if out.status == "interrupted":
        display(_panel(session, out, report, state))
    return state


def _panel(session, out, report: Callable[[Any], None], state: dict) -> w.VBox:
    """Buttons for one question + an area that receives everything that happens after the click."""
    status = w.HTML()
    trace = w.HTML()
    result = w.HTML()
    follow_up = w.VBox()  # the panel for the next question, if the agent asks again

    def submit(answer: str, controls: list) -> None:
        for c in controls:
            c.disabled = True
        status.value = f"<b>You answered:</b> {html.escape(answer)} &nbsp;⏳ <i>running ... (trace below)</i>"
        live = _LiveText(trace)
        old_stdout = sys.stdout
        try:
            with capture_output(stdout=False, stderr=False, display=True) as cap:
                sys.stdout = live
                try:
                    new = session.resume(answer)
                    state["out"] = new
                    report(new)
                finally:
                    sys.stdout = old_stdout
                    live.flush()
        except Exception as e:  # noqa: BLE001 - show the error in the cell instead of losing it
            status.value = f"<b>You answered:</b> {html.escape(answer)} &nbsp;⛔ {html.escape(f'{type(e).__name__}: {e}')}"
            return
        result.value = "".join(_rich_to_html(o.data) for o in cap.outputs)
        status.value = f"<b>You answered:</b> {html.escape(answer)} &nbsp;✅ <i>{new.status}</i>"
        if new.status == "interrupted":
            follow_up.children = [_panel(session, new, report, state)]

    if _is_yes_no(out.question):
        comment = w.Text(placeholder="optional comment (sent with 'Reject')", layout=w.Layout(width="420px"))
        yes = w.Button(description="Approve", button_style="success", icon="check")
        no = w.Button(description="Reject", button_style="danger", icon="times")
        controls = [yes, no, comment]
        yes.on_click(lambda _b: submit("yes", controls))
        no.on_click(lambda _b: submit(("no: " + comment.value.strip()) if comment.value.strip() else "no", controls))
        row = w.HBox([yes, no, comment])
    else:
        box = w.Text(placeholder="type your answer, e.g. -30", layout=w.Layout(width="420px"))
        send = w.Button(description="Send", button_style="primary", icon="paper-plane")
        controls = [box, send]
        send.on_click(lambda _b: box.value.strip() and submit(box.value.strip(), controls))
        row = w.HBox([box, send])
    return w.VBox([row, status, trace, result, follow_up])
