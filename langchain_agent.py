"""The data analysis agent.

The LLM works in a loop with four tools:

  inspect_data    look at columns, types, nulls and sample values
  run_analysis    run pandas / SQL code and read the printed result
  make_chart      build a chart (checked to make sure it actually has data)
  update_dataset  clean or transform the data (later steps see the new data)
  explain_change  root-cause finder: which segments drove a change between two groups

Every tool returns plain text. When code fails, the error goes back to the
LLM so it can fix the code and try again, up to MAX_FAILED_RUNS times.

Optional critic: a second LLM call reviews the final answer against the tool
outputs. If it finds problems, the agent gets one chance to fix them.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

from AgentClass import RESERVED_NAMES, ChartResult, describe_datasets, join_hints, run_code
from rootcause import format_root_cause, root_cause
from stats import run_stats
from theme import strip_highlights

# LLM providers (all OpenAI-compatible). "key" is the secrets.toml entry holding the API key;
# "models" are tool-calling models, the first one is the default.
PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key": "GROQ_key",
        "models": ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"],
    },
    "hf": {
        "base_url": "https://router.huggingface.co/v1",
        "key": "HF_key",
        "models": [
            "meta-llama/Llama-3.3-70B-Instruct",
            "meta-llama/Llama-3.3-70B-Instruct:together",  # ":provider" pins a provider
            "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "moonshotai/Kimi-K2-Instruct-0905",
        ],
    },
    # Gemini uses Google's own client: Gemini 3 tool calls need "thought signatures",
    # which the OpenAI-compatible endpoint loses.
    "gemini": {
        "key": "GEMINI_key",
        "models": ["gemini-3.5-flash", "gemini-3.8-flash", "gemini-flash-latest", "gemini-3.1-pro-preview"],
    },
}
DEFAULT_PROVIDER = "groq"
MAX_FAILED_RUNS = 4      # code errors allowed per question before the agent must stop
MAX_AGENT_STEPS = 40     # hard cap on loop length (LangGraph recursion limit); auto-reports need ~25
MAX_HISTORY_MESSAGES = 10
CRITIC_OUTPUT_BUDGET = 12000  # characters of tool output the critic sees in total
# Free tiers reject big requests (Groq: 8k tokens per request and per minute). The system prompt and
# tool schemas are ~7k chars; the conversation sent to the model is kept under this many chars.
REQUEST_BUDGET_CHARS = 18000
KEEP_FULL_TOOL_OUTPUTS = 2    # the most recent tool outputs are never shortened

SYSTEM_PROMPT = """You are a data analysis agent answering questions about the user's tables
(pandas DataFrames, available in code by name). The user's message lists each table's columns.

How to work:
1. The columns are listed; call inspect_data only for sample rows or a column's values/statistics.
2. Compute with run_analysis (print what you need) before answering or charting. Use make_chart when
   asked or when a chart helps.
3. Change data with update_dataset only when asked (clean, filter, rename, join). When asked, really
   change it with update_dataset; don't just compute what the result would be.
4. "Why did X change between A and B?" -> explain_change. Correlations, outliers, trends, forecasts ->
   stats_check; never invent trends or forecasts. To chart their tables in make_chart code:
   root_cause(df, metric, split_column, group_a, group_b, agg) or
   run_stats(df, kind, columns, metric, date_column, agg, freq, periods).
5. If you would have to choose between columns or definitions that give different numbers (e.g. "region"
   when there are Pickup and Drop Location), call ask_user with the choices, unless a saved rule or an
   earlier answer settles it.
6. On a tool error, fix the code; don't repeat it.
7. Answer briefly in plain words with the key numbers. No code, no image links (charts show
   automatically). Inside your sentences, wrap the 1-3 key numbers or short findings in double equals
   signs, e.g. "Lung has the highest average risk score at ==0.50==."

Code: plain Python, no imports (pd, np, px, go, plt, alt are ready). SQL: sql.query('SELECT ... FROM <table>')
(DuckDB; quote names with spaces: "Vehicle Type"). No files, OS or network; no names starting with _.
Join with pd.merge(a, b, on=..., how="left") and check the row count.
Each tool call runs fresh code: variables from earlier calls are gone. make_chart code must be
self-contained (prepare the data and draw the chart in the same code), e.g.
  df = data.groupby("Vehicle Type", as_index=False)["Booking Value"].mean()
  fig = px.bar(df, x="Vehicle Type", y="Booking Value", title="Average booking value by vehicle type")
  result = as_plotly(fig, "Go Sedan has the highest average value.")
make_chart code ends with result = as_plotly(fig, "one-sentence explanation") (or as_matplotlib /
as_altair); prefer plotly with a clear title and axis labels.
update_dataset code ends with result = as_data_prep({"<table>": new_df}, "what changed"). Cleaning
updates the table under its OWN name; use a new name only for a join or when the user wants to keep
the original, e.g. {"rides_priced": merged}.
"""


AUTO_REPORT_LABEL = "Analyze this dataset"
AUTO_REPORT_PROMPT = """Analyze this dataset on your own and write a short report.
Don't change the dataset (no update_dataset); only read it. Don't ask the user questions; pick sensible defaults.

1. The columns are listed in this message; call inspect_data only if you need sample rows.
2. Use run_analysis for the key facts: size, missing values, main numeric columns, top categories.
   Call stats_check with kind='correlations' and kind='outliers' (and kind='trend' if there is a date column).
3. Make 2 charts with make_chart (one chart per call) that show the most important patterns.
   Keep chart code simple (one plotly express call) so it works the first time.
4. Pick the most important numeric metric and a natural comparison: the last two months/periods if
   there is a date column (split_column = the date column, groups = prefixes like '2024-11'),
   otherwise the two main groups of a key category. Don't compare groups that are defined by the
   metric itself (e.g. risk score by risk level). Call explain_change to find what drives the difference.
   Report only what explain_change says (segment, contribution, unexpected); don't guess reasons.
5. Finish with this markdown report. Use only numbers from the tool outputs:
   ## Overview
   2-3 sentences: what the data is, rows/columns, time range.
   ## Key insights
   3-5 bullets, each with a number.
   ## What drives the change
   What you compared, the total change, and the top 2-3 driver segments from explain_change.
   ## Data quality
   Missing values or other issues worth knowing.
"""

PLANNER_PROMPT = """You plan the work of a data analysis agent before it starts.
The agent's tools: inspect_data, run_analysis (pandas/SQL), make_chart, update_dataset,
explain_change (root cause of a change between two groups/periods),
stats_check (correlations, outliers, trend, forecast), ask_user.

Write a short numbered plan of 2-5 steps for the user's request. One line per step, plain words,
name the tool and the columns where you can. No code, no explanations, nothing else.
"""

# Words that suggest a request has several parts and is worth planning.
PLAN_KEYWORDS = ("compare", "analy", "report", "overview", "explore", "breakdown", "break down",
                 "why", "trend", "and then", "then ", "also", "versus", " vs", "insight")


def needs_plan(question: str) -> bool:
    """Cheap check (no LLM call) so simple questions don't pay for planning."""
    q = question.lower()
    return (len(q.split()) >= 18 or any(k in q for k in PLAN_KEYWORDS)
            or q.count(" and ") >= 2 or q.count("?") >= 2)


CRITIC_PROMPT = """You review a data analysis agent's work before the user sees it.
You get the user's question, every tool step (code + output) and the agent's final answer.

Check only these things:
1. The code answers the question that was asked (right columns, filters, grouping, aggregation).
2. Every number in the final answer appears in, or follows directly from, the tool outputs. Rounding is fine.
3. No overclaiming: no causes, trends or certainty that the outputs don't support.
4. If a chart was asked for, a successful make_chart step exists. A chart doesn't replace the answer: if the
   question asks for numbers (e.g. "what percentage"), the answer text must state the key ones.
5. If the question lists saved rules from the user, the code and answer follow the ones that apply.
6. If the question includes a plan, every step of the plan was carried out or the answer says why not.
7. If the user asked to change the data (drop, remove, clean, filter, rename, join...), a successful
   update_dataset step exists; computing the result without changing the data is a problem.

Charts from make_chart are shown to the user automatically next to the answer, so the answer
text should NOT contain the chart itself. A short description of the chart is correct.
Only report real problems that would make the answer wrong or misleading. Don't comment on style.
Reply with JSON only, no other text:
{"ok": true, "problems": []}
or
{"ok": false, "problems": ["short, specific description of each problem"]}
"""


def _text(content: Any) -> str:
    """Message content as plain text (some models return a list of content blocks)."""
    if isinstance(content, list):
        return " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return content or ""


# ---------------------------------------------------------------- data summary sent with each question
# A compact column list in the first message lets the agent skip an inspect_data round trip,
# which on free tiers saves a whole model call plus ~1k tokens re-sent with every later call.

_summary_cache: Dict[Any, str] = {}


def _table_summary(name: str, df: pd.DataFrame) -> str:
    key = (name, id(df), df.shape)
    if key not in _summary_cache:
        cols = []
        for c in df.columns:
            s = df[c]
            kind = ("bool" if pd.api.types.is_bool_dtype(s) else "int" if pd.api.types.is_integer_dtype(s)
                    else "float" if pd.api.types.is_float_dtype(s) else "date" if pd.api.types.is_datetime64_any_dtype(s)
                    else "text")
            nulls = int(s.isna().sum())
            info = kind + (f", {nulls:,} missing" if nulls else "")
            if kind == "text" and 0 < s.nunique() <= 8:  # small categories: list the values
                # "…" marks a shortened value, so the model looks it up instead of filtering on a cut-off string
                info += ": " + " | ".join(v if len(v) <= 40 else v[:38] + "…"
                                          for v in map(str, s.dropna().unique()[:8]))
            cols.append(f"{c} ({info})")
        _summary_cache[key] = f"- {name} ({len(df):,} rows): " + "; ".join(cols)
    return _summary_cache[key]


def data_summary(datasets: Dict[str, Any]) -> str:
    lines = ["Tables (use these exact names):"] + [_table_summary(n, d) for n, d in datasets.items()
                                                     if isinstance(d, pd.DataFrame)]
    if len(datasets) > 1 and (hints := join_hints(datasets)):
        lines.append("Possible joins:\n" + "\n".join(hints))
    return "\n".join(lines)


# ---------------------------------------------------------------- quick number check (before the critic)

_SCALE = {"k": 1e3, "thousand": 1e3, "lakh": 1e5, "lakhs": 1e5, "m": 1e6, "million": 1e6,
          "crore": 1e7, "crores": 1e7, "b": 1e9, "billion": 1e9}
_NUMBER = re.compile(r"(\d+(?:\.\d+)?)(?:\s*(k|thousand|lakhs?|m|million|crores?|b|billion)\b)?", re.I)


def numbers_in(text: str) -> List[float]:
    """Numbers in free text: handles 1,234 / 1 234 / 4.5 lakh / 3.2M / unicode minus signs."""
    t = re.sub(r"[‐-―−]", "-", strip_highlights(text))
    t = re.sub(r"[    ]", " ", t)
    t = re.sub(r"(?<=\d)[, ](?=\d{3}(?!\d))", "", t)
    return [float(n) * (_SCALE[u.lower()] if u else 1) for n, u in _NUMBER.findall(t)]


# Requests to change the data: the quick number check can't tell whether the change was really made.
CHANGE_REQUEST = re.compile(r"\b(drop|remove|delete|clean|filter|rename|replace|fill|impute|dedupe|deduplicate|"
                            r"join|merge|convert|add (a )?column|keep only)\b", re.I)


# Clear commands to modify the data (not "filter"/"join", which are usually analysis, e.g. "filter to 2024
# and show the average"). If one of these is asked and no update_dataset ran, the agent must apply it.
MUST_CHANGE = re.compile(r"\b(drop|remove|delete|clean( it| them)? up|clean|deduplicate|dedupe|rename|"
                         r"fill (in )?(the )?missing|impute)\b", re.I)


def unverified_numbers(answer: str, sources: str) -> Tuple[int, List[float]]:
    """(numbers checked, numbers in the answer that appear nowhere in the sources, allowing for rounding
    and for fractions written as percentages). Small whole numbers (0-10) are skipped."""
    pool = [abs(v) for v in numbers_in(sources)]
    pool += [v * 100 for v in pool if v <= 1]
    checked, missing = 0, []
    for a in (abs(v) for v in numbers_in(answer)):
        if a <= 10 and a == int(a):
            continue
        checked += 1
        if not any(abs(a - s) <= max(0.005 * s, 0.006 if s < 1 else 0) for s in pool):
            missing.append(a)
    return checked, missing


def _size(msg) -> int:
    return len(str(msg.content)) + len(str(getattr(msg, "tool_calls", "") or ""))


def compact_messages(messages: List[Any], budget: int = REQUEST_BUDGET_CHARS) -> List[Any]:
    """Shorten the OLDEST tool outputs until the conversation fits the budget. The agent already read
    them (its later messages build on them); the newest outputs stay complete."""
    total = sum(_size(m) for m in messages)
    if total <= budget:
        return messages
    out = list(messages)
    tool_idx = [i for i, m in enumerate(out) if isinstance(m, ToolMessage)][:-KEEP_FULL_TOOL_OUTPUTS or None]
    for i in tool_idx:
        text = str(out[i].content)
        if len(text) > 400:
            short = text[:400] + "\n... (older output shortened to save space; you already read it)"
            total -= len(text) - len(short)
            out[i] = out[i].model_copy(update={"content": short})
        if total <= budget:
            break
    return out


MALFORMED_TOOL_CALL = ("tool_use_failed", "Failed to parse tool call", "output_parse_failed")
MALFORMED_RETRIES = 2


@wrap_model_call
def compact_context(request, handler):
    """Runs around every model call:
    - shortens old tool outputs in what is sent (the stored conversation stays complete);
    - retries when the model writes a broken tool call (Groq's gpt-oss sometimes emits raw code
      or loose text instead of JSON arguments; the API rejects it with tool_use_failed /
      output_parse_failed)."""
    request = request.override(messages=compact_messages(request.messages))
    for attempt in range(MALFORMED_RETRIES + 1):
        try:
            return handler(request)
        except Exception as e:
            if attempt == MALFORMED_RETRIES or not any(k in str(e) for k in MALFORMED_TOOL_CALL):
                raise


@dataclass
class Step:
    tool: str
    code: str
    ok: bool
    output: str


@dataclass
class AgentReply:
    answer: str
    charts: List[ChartResult] = field(default_factory=list)
    steps: List[Step] = field(default_factory=list)
    datasets: Dict[str, Any] = field(default_factory=dict)
    data_changed: bool = False
    clarification: Optional[Dict[str, Any]] = None  # {"question", "options"} if the agent asked the user


def _chart_problem(result: ChartResult) -> Optional[str]:
    """Quick self-check: return a problem description if the chart looks empty."""
    try:
        if result.kind == "plotly":
            traces = list(result.obj.data)
            if not traces:
                return "The figure has no traces."
            sizes = []
            for t in traces:
                for attr in ("x", "y", "values", "z", "r"):
                    v = getattr(t, attr, None)
                    if v is not None:
                        sizes.append(len(v))
                        break
            if sizes and max(sizes) == 0:
                return "All traces are empty (0 points). Check your filters."
        elif result.kind == "matplotlib":
            if not result.obj.get_axes():
                return "The figure has no axes."
            ax = result.obj.get_axes()[0]
            if not (ax.lines or ax.patches or ax.collections or ax.images or ax.texts or ax.tables):
                return "The axes are empty."
    except Exception:
        return None  # don't block on a check we can't run
    return None


def _chart_summary(result: ChartResult) -> str:
    try:
        if result.kind == "plotly":
            parts = []
            for t in result.obj.data:
                n = next((len(getattr(t, a)) for a in ("x", "y", "values", "z", "r")
                          if getattr(t, a, None) is not None), "?")
                parts.append(f"{t.type} ({n} points)")
            title = result.obj.layout.title.text or "no title"
            return f"title: {title}; traces: {', '.join(parts)}"
        if result.kind == "matplotlib":
            return f"{len(result.obj.get_axes())} axes; title: {result.obj.get_axes()[0].get_title() or 'no title'}"
        if result.kind == "altair":
            return f"mark: {getattr(result.obj, 'mark', '?')}"
    except Exception:
        pass
    return ""


class DataAgent:
    def __init__(self, api_key: str, model: str = "", provider: str = DEFAULT_PROVIDER):
        model = model or PROVIDERS[provider]["models"][0]
        self.provider, self.model = provider, model
        if provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            # Few retries + timeout so a busy model fails fast instead of hanging for minutes.
            self.llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0.2,
                                              max_retries=2, timeout=60)
        else:
            self.llm = ChatOpenAI(
                base_url=PROVIDERS[provider]["base_url"],
                api_key=api_key,
                model=model,
                temperature=0.2,
                # Free tiers have tokens-per-minute limits; retries wait the time the API asks for.
                max_retries=6,
                timeout=60,
            )
        self.history: List[Any] = []   # HumanMessage / AIMessage pairs
        self._agent = create_agent(model=self.llm, tools=self._make_tools(), system_prompt=SYSTEM_PROMPT,
                                   middleware=[compact_context])
        self._reset_turn({})

    # ------------------------------------------------------------ per-turn state

    def _reset_turn(self, datasets: Dict[str, Any], on_step: Optional[Callable[[Step], None]] = None):
        self.datasets: Dict[str, Any] = dict(datasets)
        self.charts: List[ChartResult] = []
        self.steps: List[Step] = []
        self.failed_runs = 0
        self.last_error = ""
        self.data_changed = False
        self.clarification: Optional[Dict[str, Any]] = None
        self.on_step = on_step

    def _log(self, tool: str, code: str, ok: bool, output: str) -> str:
        step = Step(tool=tool, code=code, ok=ok, output=output)
        self.steps.append(step)
        if self.on_step:
            self.on_step(step)  # e.g. the UI shows the step as it happens
        return output

    def _failed(self, tool: str, code: str, error: str) -> str:
        self.failed_runs += 1
        left = MAX_FAILED_RUNS - self.failed_runs
        attempts = f"{left} attempt{'s' if left != 1 else ''} left"
        # Same error as the previous failure: the model is going in circles. Compare the exception line
        # (e.g. "NameError: name 'plot_df' is not defined"), not the code line, which shifts between attempts.
        lines = error.strip().splitlines()
        key = next((l for l in lines if re.match(r"[\w.]+(Error|Exception)\b", l)), lines[-1] if lines else "")
        repeated = key == self.last_error
        self.last_error = key
        if left <= 0:
            msg = (f"ERROR:\n{error}\n\nToo many failed attempts. Stop calling tools and tell the user "
                   "briefly what went wrong and what they could try instead.")
        elif repeated:
            msg = (f"ERROR:\n{error}\n\nThis is the SAME error as your last attempt, so repeating the approach won't "
                   "work. Change it: e.g. put all the code in one tool call, use different columns, or a simpler "
                   f"method ({attempts}).")
        else:
            msg = f"ERROR:\n{error}\n\nFix the code and try again ({attempts})."
        return self._log(tool, code, False, msg)

    def _out_of_attempts(self, tool: str, code: str) -> Optional[str]:
        if self.failed_runs >= MAX_FAILED_RUNS:
            return self._log(tool, code, False, "No attempts left. Answer the user now without more tools.")
        return None

    # ------------------------------------------------------------ tools

    def inspect_data(self, column: str = "") -> str:
        """Show datasets: columns, types, null counts and sample rows. Pass a column name to see its value counts or statistics."""
        out = describe_datasets(self.datasets, column.strip() or None)
        return self._log("inspect_data", column, True, out)

    def run_analysis(self, code: str) -> str:
        """Run pandas or SQL code to compute something. Use print() to show results; you will see the printed output."""
        if (stop := self._out_of_attempts("run_analysis", code)):
            return stop
        res = run_code(code, self.datasets)
        if not res.ok:
            return self._failed("run_analysis", code, res.error)
        if res.result is not None and res.result.kind == "data_prep":
            # Nothing from run_analysis is kept, so saving data here would silently do nothing.
            return self._failed("run_analysis", code,
                                "run_analysis doesn't save data. Put all the code (data prep + chart) in make_chart, "
                                "or use update_dataset to save a new table."
                                + (f"\nPrinted output:\n{res.output.strip()}" if res.output.strip() else ""))
        out = res.output.strip() or "(nothing printed - use print() to see results)"
        return self._log("run_analysis", code, True, out)

    def make_chart(self, code: str) -> str:
        """Build a chart, shown to the user. The code must be self-contained: prepare the data AND draw the chart
        here (variables from earlier tool calls don't exist). End with result = as_plotly(fig, ...) /
        as_matplotlib(fig, ...) / as_altair(chart, ...)."""
        if (stop := self._out_of_attempts("make_chart", code)):
            return stop
        res = run_code(code, self.datasets)
        if not res.ok:
            return self._failed("make_chart", code, res.error)
        r = res.result
        if r is None or r.kind == "data_prep":
            return self._failed("make_chart", code,
                                "No chart was made. Set result = as_plotly(fig, ...) (or as_matplotlib / as_altair).")
        if (problem := _chart_problem(r)):
            return self._failed("make_chart", code, f"Chart check failed: {problem}")
        self.charts.append(r)
        summary = _chart_summary(r)
        return self._log("make_chart", code, True,
                         f"Chart created and shown to the user ({r.kind}). {summary}\nExplanation: {r.explanation}")

    def update_dataset(self, code: str) -> str:
        """Clean or transform a table (drop nulls, rename, filter, add columns, join). Code must set
        result = as_data_prep({"<table>": new_df}, "what changed"). Use the table's SAME name to replace it
        (cleaning, filtering); a new name only for a joined table or when the user wants to keep the original."""
        if (stop := self._out_of_attempts("update_dataset", code)):
            return stop
        res = run_code(code, self.datasets)
        if not res.ok:
            return self._failed("update_dataset", code, res.error)
        r = res.result
        if r is None or r.kind != "data_prep" or not r.modified_datasets:
            return self._failed("update_dataset", code,
                                'Data was not updated. Set result = as_data_prep({"<dataset name>": new_df}, "what changed").')
        changes = []
        for name, new_df in r.modified_datasets.items():
            if not isinstance(new_df, pd.DataFrame):
                return self._failed("update_dataset", code, f"'{name}' must be a pandas DataFrame.")
            if name not in self.datasets:  # a new dataset, e.g. a joined table
                if not str(name).isidentifier() or name in RESERVED_NAMES:
                    return self._failed("update_dataset", code,
                                        f"'{name}' can't be a dataset name. Use letters, digits and _ "
                                        f"(not {sorted(RESERVED_NAMES)}).")
                changes.append(f"new dataset {name}: {new_df.shape[0]}x{new_df.shape[1]}, "
                               f"columns: {list(new_df.columns)}")
            else:
                old = self.datasets[name]
                changes.append(f"{name}: {old.shape[0]}x{old.shape[1]} -> {new_df.shape[0]}x{new_df.shape[1]}, "
                               f"columns: {list(new_df.columns)}")
            self.datasets[name] = new_df
        self.data_changed = True
        return self._log("update_dataset", code, True,
                         "Dataset updated. " + "; ".join(changes) + f"\nExplanation: {r.explanation}")

    def explain_change(self, metric: str, split_column: str, group_a: str, group_b: str,
                       agg: str = "sum", dataset: str = "") -> str:
        """Why did a metric change between two groups or periods? Compares agg ('sum', 'mean' or 'count') of
        metric where split_column is group_a vs group_b, and ranks the segments of every other column that
        drove it. Periods: split_column='Date', group_a='2024-03', group_b='2024-04' (prefixes).
        Leave dataset empty when only one table is loaded."""
        call = (f"explain_change(metric={metric!r}, split_column={split_column!r}, "
                f"group_a={group_a!r}, group_b={group_b!r}, agg={agg!r})")
        if (stop := self._out_of_attempts("explain_change", call)):
            return stop
        if not dataset and len(self.datasets) == 1:
            dataset = next(iter(self.datasets))
        if dataset not in self.datasets:
            return self._failed("explain_change", call, f"Unknown or missing dataset '{dataset}'. "
                                                        f"Pass dataset= one of: {list(self.datasets)}")
        try:
            table, summary = root_cause(self.datasets[dataset], metric, split_column, group_a, group_b, agg)
        except ValueError as e:
            return self._failed("explain_change", call, str(e))
        return self._log("explain_change", call, True, format_root_cause(table, summary))

    def stats_check(self, kind: str, columns: str = "", metric: str = "", date_column: str = "",
                    agg: str = "sum", freq: str = "M", periods: int = 3, dataset: str = "") -> str:
        """Reliable statistics. kind: 'correlations' or 'outliers' (optional comma-separated columns);
        'trend' (is metric rising or falling over date_column per freq D/W/M/Q/Y, with significance);
        'forecast' (next `periods` periods with a 95% range and a backtest). agg: sum, mean or count.
        Leave dataset empty when only one table is loaded."""
        call = (f"stats_check(kind={kind!r}" + (f", columns={columns!r}" if columns else "")
                + (f", metric={metric!r}, date_column={date_column!r}, agg={agg!r}, freq={freq!r}"
                   if date_column else "") + (f", periods={periods}" if kind.startswith("forecast") else "") + ")")
        if (stop := self._out_of_attempts("stats_check", call)):
            return stop
        if not dataset and len(self.datasets) == 1:
            dataset = next(iter(self.datasets))
        if dataset not in self.datasets:
            return self._failed("stats_check", call, f"Unknown or missing dataset '{dataset}'. "
                                                     f"Pass dataset= one of: {list(self.datasets)}")
        cols = [c.strip() for c in columns.split(",") if c.strip()] or None
        try:
            _, text = run_stats(self.datasets[dataset], kind, cols, metric, date_column, agg, freq, periods)
        except ValueError as e:
            return self._failed("stats_check", call, str(e))
        return self._log("stats_check", call, True, text)

    def ask_user(self, question: str, options: Optional[List[str]] = None) -> str:
        """Ask the user one short clarifying question, then stop and wait. Only when choosing between
        columns or definitions would change the answer. Give 2-4 short options."""
        options = [str(o).strip() for o in (options or []) if str(o).strip()][:4]
        self.clarification = {"question": question.strip(), "options": options}
        self._log("ask_user", "", True, question.strip()
                  + "".join(f"\n- {o}" for o in options))
        return question.strip()

    def _make_tools(self):
        tools = [StructuredTool.from_function(func=f) for f in
                 (self.inspect_data, self.run_analysis, self.make_chart, self.update_dataset,
                  self.explain_change, self.stats_check)]
        # return_direct: the agent loop ends right after asking, so the user can answer.
        return tools + [StructuredTool.from_function(func=self.ask_user, return_direct=True)]

    # ------------------------------------------------------------ main entry

    def _run(self, messages: List[Any]):
        """One agent loop. Returns (answer, all messages), or (None, None) if the step limit was hit."""
        try:
            response = self._agent.invoke({"messages": messages},
                                          config={"recursion_limit": MAX_AGENT_STEPS})
        except Exception as e:  # e.g. step limit reached or API error
            if type(e).__name__ == "GraphRecursionError":
                return None, None
            raise
        return _text(response["messages"][-1].content), response["messages"]

    # ------------------------------------------------------------ critic

    def _critique(self, question: str, answer: str, steps: List[Step]) -> Dict[str, Any]:
        """Second LLM call that reviews the answer. Returns {"ok": bool, "problems": [str]}."""
        # The critic must see the outputs the answer's numbers come from, so successful steps get
        # most of the budget (~6k tokens in total, to stay under free-tier per-minute limits).
        ok_steps = [s for s in steps if s.ok]
        per_step = max(800, CRITIC_OUTPUT_BUDGET // max(len(ok_steps), 1))
        parts = []
        for i, s in enumerate(steps, 1):
            if s.ok:
                parts.append(f"[{i}] {s.tool} (ok)\ncode:\n{s.code[:1500]}\noutput:\n{s.output[:per_step]}")
            else:
                parts.append(f"[{i}] {s.tool} (failed, the agent retried): {' '.join(s.output.strip().splitlines()[1:2])[:200]}")
        work = "\n\n".join(parts) or "(no tools used)"
        prompt = f"QUESTION:\n{question}\n\nTOOL STEPS:\n{work}\n\nFINAL ANSWER:\n{answer}"
        try:
            reply = _text(self.llm.invoke([SystemMessage(CRITIC_PROMPT), HumanMessage(prompt)]).content)
            verdict = json.loads(re.search(r"\{.*\}", reply, re.S).group(0))
            problems = [str(p) for p in verdict.get("problems", []) if str(p).strip()]
            return {"ok": bool(verdict.get("ok", not problems)) and not problems, "problems": problems}
        except Exception as e:  # never block the answer on a broken review
            return {"ok": True, "problems": [], "note": f"critic could not run: {str(e)[:150]}"}

    # ------------------------------------------------------------ main entry

    def _numbers_verified(self, question: str, answer: str, planned: bool) -> bool:
        """True when the full LLM critic can be skipped: the answer has numbers, all of them appear in the
        question or tool outputs, and the work wasn't interpretive (plans, root causes, statistics and data
        changes always get the full review, since overclaiming is the risk there)."""
        done = [s for s in self.steps if s.ok and s.tool not in ("critic", "plan", "memory", "ask_user")]
        if planned or self.data_changed or not done or any(s.tool in ("explain_change", "stats_check") for s in done):
            return False
        if CHANGE_REQUEST.search(question):  # the action itself must be checked, not just the numbers
            return False
        sources = question + "\n" + "\n".join(f"{s.code}\n{s.output}" for s in done)
        checked, missing = unverified_numbers(answer, sources)
        return checked > 0 and not missing

    def _make_plan(self, question: str) -> Optional[str]:
        """One short LLM call that writes a numbered plan. None if it fails or looks wrong."""
        tables = "\n".join(f"- {name} ({df.shape[0]:,} rows): {', '.join(map(str, df.columns))[:600]}"
                           for name, df in self.datasets.items())
        try:
            reply = _text(self.llm.invoke([SystemMessage(PLANNER_PROMPT),
                                           HumanMessage(f"Datasets:\n{tables}\n\nRequest:\n{question}")]).content)
        except Exception:
            return None  # planning is optional; never block the answer
        steps = [line.strip() for line in reply.splitlines() if re.match(r"\s*\d+[.)]", line)]
        return "\n".join(steps[:5]) if len(steps) >= 2 else None

    def ask(self, question: str, datasets: Dict[str, Any],
            on_step: Optional[Callable[[Step], None]] = None, critic: bool = False,
            rules: Optional[List[str]] = None, plan: bool = False) -> AgentReply:
        """Answer one question. on_step(step) is called after every tool call.
        With critic=True a second LLM call reviews the answer; if it finds problems
        the agent gets one chance to fix them. `rules` are the user's saved rules
        relevant to this question (from memory.py); the agent must follow them.
        With plan=True, bigger requests (see needs_plan) first get a short written plan."""
        self._reset_turn(datasets, on_step)
        full_question = question
        if rules:
            self._log("memory", "", True, "Using saved rules:\n" + "\n".join(f"- {r}" for r in rules))
            full_question = ("The user saved these rules earlier. Follow them where they apply:\n"
                             + "\n".join(f"- {r}" for r in rules) + f"\n\nQuestion: {question}")
        planned = False
        if plan and needs_plan(question) and (steps := self._make_plan(full_question)):
            planned = True
            self._log("plan", "", True, steps)
            full_question += (f"\n\nFollow this plan step by step:\n{steps}\n"
                              "The user can already see the plan, so in the final answer don't narrate the "
                              "steps; just answer every part of the request concisely.")
        messages = self.history[-MAX_HISTORY_MESSAGES:] + [
            HumanMessage(f"{data_summary(self.datasets)}\n\n{full_question}")]

        answer, convo = self._run(messages)
        # Exact check (runs even with the critic off): a clear request to change the data must
        # actually change it; models sometimes only compute what the result would be.
        change_missing = bool(MUST_CHANGE.search(question)) and not any(
            s.tool == "update_dataset" and s.ok for s in self.steps)
        if self.clarification:  # the agent is waiting for the user: nothing to review yet
            answer = self.clarification["question"]
        elif answer is not None and critic and not change_missing and \
                self._numbers_verified(full_question, answer, planned):
            # Cheap path: every number is in the tool outputs and nothing needs interpretation.
            self._log("critic", "", True, "Every number in the answer matches the tool outputs "
                                          "(quick check, no extra AI call needed).")
        elif answer is not None and (critic or change_missing):
            verdict = ({"ok": False, "problems": ["The user asked to change the data, but no update_dataset step "
                                                  "ran. Apply the change with update_dataset, then answer."]}
                       if change_missing else self._critique(full_question, answer, self.steps))
            if verdict["ok"]:
                self._log("critic", "", True, verdict.get("note", "Looks correct: the code matches the question "
                                                                  "and the numbers match the tool outputs."))
            else:
                problems = "\n".join(f"- {p}" for p in verdict["problems"])
                self._log("critic", "", False, f"Problems found, asking the agent to fix them:\n{problems}")
                charts_before = len(self.charts)
                fixed, _ = self._run(convo + [HumanMessage(
                    f"A reviewer checked your answer and found these problems:\n{problems}\n\n"
                    "Fix them. Use the tools again if you need to check something. "
                    "Then give the corrected final answer to the original question.")])
                if fixed is not None:
                    answer = fixed
                    if len(self.charts) > charts_before:  # new charts replace the first attempt's
                        self.charts = self.charts[charts_before:]
        if answer is None:
            answer = "I took too many steps on this one and stopped. Try asking a simpler or more specific question."

        if not answer.strip():
            answer = "Done." if (self.charts or self.data_changed) else "I couldn't produce an answer."

        self.history += [HumanMessage(question), AIMessage(answer)]
        return AgentReply(answer=answer, charts=self.charts, steps=self.steps,
                          datasets=self.datasets, data_changed=self.data_changed,
                          clarification=self.clarification)

    def clear(self):
        self.history = []


# Old name, kept so existing imports still work.
LangChainVizAgent = DataAgent
