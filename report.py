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

import theme as t

# JS versions match the installed Python libraries, so charts render the same as in the app.
PLOTLY_JS = f"https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"
VEGA_JS = [f"https://cdn.jsdelivr.net/npm/vega@{alt.VEGA_VERSION}",
           f"https://cdn.jsdelivr.net/npm/vega-lite@{alt.VEGALITE_VERSION}",
           f"https://cdn.jsdelivr.net/npm/vega-embed@{alt.VEGAEMBED_VERSION}"]

CSS = f"""
:root {{ --bg:{t.BG}; --card:{t.SURFACE}; --raised:{t.RAISED}; --text:{t.TEXT}; --muted:{t.MUTED};
        --line:{t.LINE}; --accent:{t.ACCENT}; --amber:{t.AMBER}; --ok:{t.SUCCESS}; --warn:{t.WARNING}; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text); font:16px/1.6 {t.FONT}; }}
main {{ max-width: 920px; margin: 0 auto; padding: 48px 16px 64px; }}
header h1 {{ margin: 0 0 6px; font-size: 2.25rem; font-weight: 600; letter-spacing: -0.02em; }}
header p {{ margin: 0; color: var(--muted); }}
a {{ color: var(--accent); }}
.meta {{ display:flex; flex-wrap:wrap; gap:8px 24px; margin:20px 0 32px; color: var(--muted); font-size:.92rem; }}
.meta b {{ color: var(--text); font-weight: 500; }}
.turn {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:22px 24px; margin-bottom:24px; }}
.question {{ font-weight:600; font-size:1.05rem; margin-bottom:12px; }}
.question span {{ color: var(--muted); font-weight: 500; margin-right: 8px; }}
.answer table {{ border-collapse: collapse; margin: 10px 0; font-size: .92rem; display:block; overflow-x:auto;
                font-variant-numeric: tabular-nums; }}
.answer th, .answer td {{ border-bottom: 1px solid var(--line); padding: 6px 12px; text-align: left; }}
.answer th {{ color: var(--muted); font-weight: 500; }}
mark {{ color: var(--amber); background: rgba(245, 196, 81, 0.15); border-radius: 4px; padding: 0 3px; }}
.note {{ color: var(--ok); }}
.chart {{ margin: 18px 0 4px; }}
.chart img {{ max-width: 100%; height: auto; }}
.chart .caption {{ color: var(--muted); font-size: .9rem; }}
details {{ margin-top: 16px; border-top: 1px solid var(--line); padding-top: 12px; }}
summary {{ cursor: pointer; color: var(--muted); font-size: .92rem; }}
.timeline {{ border-left: 1px solid var(--line); margin: 12px 0 0 6px; padding-left: 16px; }}
.step {{ margin: 12px 0; }}
.step .name {{ font-weight: 500; font-size: .95rem; }}
.step.bad .name {{ color: var(--warn); }}
pre {{ background: var(--bg); border: 1px solid var(--line); padding: 10px 12px; border-radius: 8px; overflow-x: auto;
      font: .82rem/1.5 {t.MONO}; white-space: pre-wrap; word-break: break-word; }}
footer {{ color: var(--muted); font-size: .85rem; margin-top: 32px; }}
"""


def _md(text: str) -> str:
    # Escape raw HTML from the model, mark ==key findings== in amber, then render markdown.
    return markdown.markdown(t.highlight_html(text.replace("<", "&lt;")), extensions=["tables", "sane_lists"])


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
    for s in steps:
        icon, label = t.step_meta(s)
        code = (f"<pre>{html.escape(s.code)}</pre>"
                if s.code and s.tool not in ("inspect_data",) + t.NOTE_STEPS else "")
        parts.append(f'<div class="step{"" if s.ok else " bad"}"><div class="name">{icon} {html.escape(label)}</div>'
                     f"{code}<pre>{html.escape(s.output[:3000])}</pre></div>")
    n = len([s for s in steps if s.tool not in t.NOTE_STEPS])
    return (f"<details><summary>How it got this · {n} step{'s' if n != 1 else ''}</summary>"
            f'<div class="timeline">{"".join(parts)}</div></details>')


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
<title>Analysis report</title>
<link rel="stylesheet" href="{t.FONTS_CSS_URL}">
<style>{CSS}</style>{scripts}</head>
<body><main>
<header><h1>Analysis report</h1><p>Questions, answers, charts and the steps behind them · {created}</p></header>
<div class="meta">{meta}<div><b>Questions:</b> {n}</div></div>
{"".join(turns) or "<p>No questions yet. Ask one in the app, then export again.</p>"}
<footer>Charts are interactive and load from the internet. Open “How it got this” under an answer to see its code.</footer>
</main></body></html>"""
