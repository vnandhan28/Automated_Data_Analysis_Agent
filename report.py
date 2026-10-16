"""Export the whole analysis (questions, answers, charts, agent steps) as one HTML file.

Plotly and Altair charts stay interactive (their JavaScript is loaded from a CDN, so
opening the report needs internet); Matplotlib charts are embedded as PNG images.
"""

import base64
import html
import io
import time
from typing import Any, Dict, List

import altair as alt
import markdown
from plotly.offline import get_plotlyjs_version

# JS versions match the installed Python libraries, so charts render the same as in the app.
PLOTLY_JS = f"https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"
VEGA_JS = [f"https://cdn.jsdelivr.net/npm/vega@{alt.VEGA_VERSION}",
           f"https://cdn.jsdelivr.net/npm/vega-lite@{alt.VEGALITE_VERSION}",
           f"https://cdn.jsdelivr.net/npm/vega-embed@{alt.VEGAEMBED_VERSION}"]

CSS = """
:root { --bg:#f7f7f5; --card:#ffffff; --text:#1f2328; --muted:#656d76; --line:#e3e3df;
        --accent:#2f6fde; --user:#eef3fd; --ok:#1a7f37; --warn:#9a6700; --code:#f3f3f0; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#16181c; --card:#1e2127; --text:#e6e6e6; --muted:#9aa0a6; --line:#30343b;
          --accent:#7aa7ff; --user:#1f2a3d; --ok:#4ac26b; --warn:#d4a72c; --code:#262a31; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--text);
       font:16px/1.6 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
main { max-width: 920px; margin: 0 auto; padding: 32px 16px 64px; }
header h1 { margin: 0 0 4px; font-size: 1.7rem; }
header p { margin: 0; color: var(--muted); }
.meta { display:flex; flex-wrap:wrap; gap:8px 20px; margin:16px 0 28px; padding:12px 16px;
        background:var(--card); border:1px solid var(--line); border-radius:10px; font-size:.92rem; }
.meta b { color: var(--muted); font-weight: 600; }
.turn { background:var(--card); border:1px solid var(--line); border-radius:12px;
        padding:20px 22px; margin-bottom:22px; }
.question { background:var(--user); border-radius:8px; padding:10px 14px; font-weight:600; margin-bottom:14px; }
.question span { color: var(--muted); font-weight: 500; margin-right: 6px; }
.answer table { border-collapse: collapse; margin: 10px 0; font-size: .92rem; display:block; overflow-x:auto; }
.answer th, .answer td { border: 1px solid var(--line); padding: 5px 10px; text-align: left; }
.answer th { background: var(--code); }
.note { color: var(--ok); font-weight: 600; }
.chart { margin: 16px 0 4px; border-top: 1px solid var(--line); padding-top: 12px; }
.chart img { max-width: 100%; height: auto; }
.chart .caption { color: var(--muted); font-size: .9rem; }
details { margin-top: 14px; border-top: 1px solid var(--line); padding-top: 10px; }
summary { cursor: pointer; color: var(--muted); font-size: .92rem; }
.step { margin: 12px 0; }
.step .name { font-weight: 600; font-size: .92rem; }
.step.bad .name { color: var(--warn); }
pre { background: var(--code); padding: 10px 12px; border-radius: 8px; overflow-x: auto;
      font-size: .82rem; line-height: 1.45; white-space: pre-wrap; word-break: break-word; }
footer { color: var(--muted); font-size: .85rem; text-align: center; margin-top: 30px; }
"""


def _md(text: str) -> str:
    # Escape raw HTML from the model, then render markdown (tables, lists, bold...).
    return markdown.markdown(text.replace("<", "&lt;"), extensions=["tables", "sane_lists"])


def _chart_html(chart, idx: str) -> str:
    caption = f'<div class="caption">{html.escape(chart.explanation or "")}</div>'
    if chart.kind == "plotly":
        body = chart.obj.to_html(full_html=False, include_plotlyjs=False, div_id=f"chart-{idx}",
                                 config={"displaylogo": False, "responsive": True})
    elif chart.kind == "matplotlib":
        buf = io.BytesIO()
        chart.obj.savefig(buf, format="png", bbox_inches="tight", dpi=130)
        body = f'<img alt="chart" src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'
    elif chart.kind == "altair":
        spec = chart.obj.to_json().replace("</", r"<\/")  # can't close the <script> tag early
        body = (f'<div id="chart-{idx}"></div><script>vegaEmbed("#chart-{idx}", {spec}, '
                '{"actions": false});</script>')
    else:
        return ""
    return f'<div class="chart">{body}{caption}</div>'


def _steps_html(steps) -> str:
    if not steps:
        return ""
    parts = []
    for i, s in enumerate(steps, 1):
        label = {"critic": "🧐 critic", "memory": "🧠 memory"}.get(s.tool, s.tool)
        code = f"<pre>{html.escape(s.code)}</pre>" if s.code and s.tool not in ("inspect_data",) else ""
        parts.append(f'<div class="step{"" if s.ok else " bad"}"><div class="name">{i}. '
                     f'{"✅" if s.ok else "⚠️"} {html.escape(label)}</div>{code}'
                     f"<pre>{html.escape(s.output[:3000])}</pre></div>")
    fixed = sum(1 for s in steps if not s.ok)
    return (f"<details><summary>Agent steps ({len(steps)}{f', {fixed} fixed' if fixed else ''})</summary>"
            + "".join(parts) + "</details>")


def build_report(messages: List[Dict[str, Any]], info: Dict[str, str]) -> str:
    """messages: the app's chat_messages (role, content, charts, steps, data_note).
    info: shown at the top, e.g. {"Dataset": ..., "Model": ...}."""
    turns, question, n = [], None, 0
    kinds = set()
    for msg in messages:
        if msg["role"] == "user":
            question = msg["content"]
            continue
        n += 1
        charts = msg.get("charts", [])
        kinds.update(c.kind for c in charts)
        note = f'<p class="note">{html.escape(msg["data_note"])}</p>' if msg.get("data_note") else ""
        turns.append(
            '<section class="turn">'
            + (f'<div class="question"><span>Q{n}</span>{html.escape(question)}</div>' if question else "")
            + f'<div class="answer">{_md(msg["content"])}</div>{note}'
            + "".join(_chart_html(c, f"{n}-{j}") for j, c in enumerate(charts))
            + _steps_html(msg.get("steps"))
            + "</section>")
        question = None

    scripts = ""
    if "plotly" in kinds:
        scripts += f'<script src="{PLOTLY_JS}" charset="utf-8"></script>'
    if "altair" in kinds:
        scripts += "".join(f'<script src="{u}"></script>' for u in VEGA_JS)
    meta = "".join(f"<div><b>{html.escape(k)}:</b> {html.escape(str(v))}</div>" for k, v in info.items())
    created = time.strftime("%Y-%m-%d %H:%M")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Data Analysis Report</title>
<style>{CSS}</style>{scripts}</head>
<body><main>
<header><h1>📊 Data Analysis Report</h1><p>Generated by the AI Data Analysis Agent · {created}</p></header>
<div class="meta">{meta}<div><b>Questions:</b> {n}</div></div>
{"".join(turns) or "<p>No questions asked yet.</p>"}
<footer>Charts are interactive (needs internet to load). Expand “Agent steps” to see the code behind each answer.</footer>
</main></body></html>"""
