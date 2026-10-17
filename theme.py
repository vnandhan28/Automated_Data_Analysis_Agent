"""Design tokens for the "dark analyst workspace" look, used by the app, the charts and the HTML export.

Most of the app styling lives in .streamlit/config.toml (colours, fonts, sizes). This module holds
what config.toml can't do: the shared chart theme, the amber highlighter and a small CSS block.
"""

import re

# ---------------------------------------------------------------- colours (keep in sync with config.toml)

BG = "#121821"
SURFACE = "#1A222D"
RAISED = "#222C39"
LINE = "#2A3542"
TEXT = "#E6EAF0"
MUTED = "#8C97A6"
ACCENT = "#7C8CFF"
AMBER = "#F5C451"
SUCCESS = "#3FB27F"
WARNING = "#E0A43A"
ERROR = "#EF6A5B"
CHART_COLORS = ["#7C8CFF", "#3FB27F", "#F5C451", "#EF6A5B", "#4FC3D9", "#C084FC"]

FONT = "Geist, -apple-system, 'Segoe UI', sans-serif"
MONO = "'Geist Mono', ui-monospace, Consolas, monospace"
FONTS_CSS_URL = ("https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600"
                 "&family=Geist+Mono:wght@400;500&display=swap")


# Agent steps: tool -> (icon, label in the timeline, live label while it works)
STEPS = {
    "inspect_data": ("🔍", "Looked at the data", "Looking at the data"),
    "run_analysis": ("🧮", "Ran the analysis", "Running the analysis"),
    "make_chart": ("📊", "Drew a chart", "Drawing a chart"),
    "update_dataset": ("🧹", "Updated the data", "Updating the data"),
    "explain_change": ("🧭", "Found what drove the change", "Finding what drove the change"),
    "stats_check": ("📈", "Ran statistics", "Running statistics"),
    "ask_user": ("❓", "Asked you a question", "Asking you a question"),
    "plan": ("🗺️", "Made a plan", "Making a plan"),
    "memory": ("📌", "Used your saved rules", "Using your saved rules"),
    "critic": ("✅", "Checked the answer", "Checking the answer"),
}
NOTE_STEPS = ("plan", "memory", "ask_user", "critic")  # steps that carry text instead of code


def step_meta(step) -> tuple:
    """(icon, label) for a finished step, with the warning icon when it failed or was revised."""
    icon, done, _ = STEPS.get(step.tool, ("•", step.tool, step.tool))
    if step.tool == "critic" and not step.ok:
        return "⚠️", "Found problems, revised the answer"
    if not step.ok:
        return "⚠️", f"{done} · failed, retried"
    return icon, done


# ---------------------------------------------------------------- charts

PLOTLY_TEMPLATE = "plotly_dark+analyst"

def apply_chart_theme():
    """One look for every chart: transparent background, Geist, muted grid, the palette above.
    Called once at import time by AgentClass, so charts the agent builds pick it up."""
    import altair as alt
    import matplotlib as mpl
    import plotly.graph_objects as go
    import plotly.io as pio

    axis = dict(gridcolor=LINE, linecolor=LINE, zerolinecolor=LINE, tickcolor=LINE,
                title=dict(font=dict(color=MUTED)), tickfont=dict(color=MUTED))
    pio.templates["analyst"] = go.layout.Template(layout=dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=TEXT, size=13), colorway=CHART_COLORS,
        title=dict(font=dict(size=16, color=TEXT), x=0, xanchor="left"),
        xaxis=axis, yaxis=axis, legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED)),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=LINE, font=dict(family=FONT, color=TEXT)),
        margin=dict(l=8, r=8, t=48, b=8)))
    pio.templates.default = PLOTLY_TEMPLATE

    mpl.rcParams.update({
        "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
        "text.color": TEXT, "axes.labelcolor": MUTED, "axes.edgecolor": LINE, "axes.titlecolor": TEXT,
        "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": LINE,
        "axes.prop_cycle": mpl.cycler(color=CHART_COLORS), "axes.spines.top": False, "axes.spines.right": False,
    })

    @alt.theme.register("analyst", enable=True)
    def _altair_theme():
        return alt.theme.ThemeConfig({"config": {
            "background": "transparent", "font": FONT, "view": {"stroke": "transparent"},
            "title": {"color": TEXT, "fontSize": 16, "anchor": "start"},
            "axis": {"gridColor": LINE, "domainColor": LINE, "tickColor": LINE,
                     "labelColor": MUTED, "titleColor": MUTED},
            "legend": {"labelColor": MUTED, "titleColor": MUTED},
            "range": {"category": CHART_COLORS},
        }})


# ---------------------------------------------------------------- amber highlighter
# The agent marks key numbers/findings as ==text==. Code spans are left alone (`a == b`).

_HIGHLIGHT = re.compile(r"==(?=\S)([^=\n]{1,120}?)(?<=\S)==")  # "a == b == c" is not a mark
_CODE_SPAN = re.compile(r"(`[^`\n]*`)")


def _outside_code(text: str, replace) -> str:
    parts = _CODE_SPAN.split(text)
    return "".join(p if i % 2 else _HIGHLIGHT.sub(replace, p) for i, p in enumerate(parts))


def highlight_markdown(text: str) -> str:
    """==6,131 movies== -> Streamlit's coloured-text markdown (no raw HTML, so model output stays safe)."""
    def repl(m):
        inner = m.group(1).strip().replace("[", r"\[").replace("]", r"\]")
        return f":yellow-background[:yellow[{inner}]]"
    return _outside_code(text, repl)


def highlight_html(escaped_html: str) -> str:
    """Same marks for the HTML export; the input must already be HTML-escaped."""
    return _outside_code(escaped_html, lambda m: f"<mark>{m.group(1).strip()}</mark>")


def strip_highlights(text: str) -> str:
    return _outside_code(text, lambda m: m.group(1))


# ---------------------------------------------------------------- the only custom CSS
# Everything config.toml can't express. Keyed containers get a "st-key-<key>" class.

APP_CSS = f"""
<style>
.stMainBlockContainer {{ max-width: 1100px; padding-top: 4rem; }}
h1 {{ letter-spacing: -0.02em; }}
/* primary buttons: dark text on periwinkle for AA contrast */
[data-testid="stBaseButton-primary"] {{ color: {BG}; }}
[data-testid="stBaseButton-primary"] p {{ color: {BG}; }}
/* dataset cards: 12px radius, accent border on hover */
[class*="st-key-card-"] {{ border-radius: 12px; transition: border-color .15s ease; }}
[class*="st-key-card-"]:hover {{ border-color: {ACCENT} !important; }}
/* agent step timeline: one thin line down the left */
[class*="st-key-timeline-"] {{ border-left: 1px solid {LINE}; padding-left: 1rem; margin-left: .35rem; }}
/* metric numbers in Geist Mono; table cells keep Geist but with fixed-width (tabular) digits */
[data-testid="stMetricValue"] {{ font-family: {MONO}; }}
[data-testid="stMarkdownContainer"] td {{ font-variant-numeric: tabular-nums; }}
[data-testid="stMarkdownContainer"] :is(table, th, td) {{ border-color: {LINE}; }}
[data-testid="stMarkdownContainer"] th {{ color: {MUTED}; font-weight: 500; }}
[data-testid="stMetricLabel"] p {{ color: {MUTED}; }}
</style>
"""
