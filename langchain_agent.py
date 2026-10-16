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
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

from AgentClass import RESERVED_NAMES, ChartResult, describe_datasets, run_code
from rootcause import format_root_cause, root_cause

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
CRITIC_OUTPUT_BUDGET = 16000  # characters of tool output the critic sees in total

SYSTEM_PROMPT = """You are a data analysis agent. The user has loaded tabular data and asks questions about it.

Datasets are pandas DataFrames available in your code by their names (inspect_data lists them).
If there are several datasets, inspect_data also suggests join columns. Combine them with
pd.merge(a, b, on=..., how="left") or SQL JOINs; check the row count after a join.

How to work:
1. If you are not sure about column names, types or values, call inspect_data first.
2. Use run_analysis to compute numbers and check them before you chart or answer. Print what you need to see.
3. Use make_chart to draw a chart when it helps or when the user asks for one.
4. Use update_dataset only when the user asks to clean, filter, rename or otherwise change the data.
   Use explain_change when the user asks WHY a metric differs between two periods or groups
   (e.g. "why did revenue drop in April?"). To chart its breakdown, in make_chart code use
   table, summary = root_cause(<dataset>, metric, split_column, group_a, group_b, agg)
   (table columns: column, segment, A, B, contribution, pct_of_change, expected, unexpected).
5. If a tool returns an error, read it, fix the code and try again. Do not repeat the same code.
   Ask instead of guessing: after inspecting the data, if you would have to CHOOSE between two or more
   columns or definitions that give different numbers, call ask_user with one short question and the
   choices as options. Example: the user says "region" but the data has "Pickup Location" and
   "Drop Location" -> ask which one. Don't ask when only one reading makes sense, when a saved rule
   settles it, or when the user already answered in this conversation.
6. Finish with a short answer in simple words: what you found and the key numbers. Don't paste code in the answer.
   Charts are shown to the user automatically; don't add image links or markdown images.

Code rules (all tools that take code):
- Plain Python only. No imports: pd, np, px (plotly.express), go (plotly.graph_objects), plt, alt (altair) are ready.
- SQL: df = sql.query('SELECT ... FROM <dataset> ...') (DuckDB, read-only; quote column names with spaces: "Vehicle Type").
- No file, OS or network access. No names starting with underscore.
- make_chart code must end with one of:
    result = as_plotly(fig, "one-sentence explanation")
    result = as_matplotlib(fig, "one-sentence explanation")
    result = as_altair(chart, "one-sentence explanation")
  Prefer plotly. Give the chart a clear title and axis labels.
- update_dataset code must end with:
    result = as_data_prep({"<dataset name>": new_df}, "what changed")
  To keep a joined or derived table for later questions, use a NEW name, e.g.
    result = as_data_prep({"rides_with_prices": merged}, "joined rides with vehicle prices")
"""


AUTO_REPORT_LABEL = "📋 Analyze this dataset"
AUTO_REPORT_PROMPT = """Analyze this dataset on your own and write a short report.
Don't change the dataset (no update_dataset); only read it. Don't ask the user questions; pick sensible defaults.

1. Call inspect_data to understand the columns.
2. Use run_analysis for the key facts: size, missing values, main numeric columns, top categories.
3. Make 3 charts with make_chart (one chart per call) that show the most important patterns.
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

CRITIC_PROMPT = """You review a data analysis agent's work before the user sees it.
You get the user's question, every tool step (code + output) and the agent's final answer.

Check only these things:
1. The code answers the question that was asked (right columns, filters, grouping, aggregation).
2. Every number in the final answer appears in, or follows directly from, the tool outputs. Rounding is fine.
3. No overclaiming: no causes, trends or certainty that the outputs don't support.
4. If a chart was asked for, a successful make_chart step exists.
5. If the question lists saved rules from the user, the code and answer follow the ones that apply.

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
        self._agent = create_agent(model=self.llm, tools=self._make_tools(), system_prompt=SYSTEM_PROMPT)
        self._reset_turn({})

    # ------------------------------------------------------------ per-turn state

    def _reset_turn(self, datasets: Dict[str, Any], on_step: Optional[Callable[[Step], None]] = None):
        self.datasets: Dict[str, Any] = dict(datasets)
        self.charts: List[ChartResult] = []
        self.steps: List[Step] = []
        self.failed_runs = 0
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
        if left <= 0:
            msg = (f"ERROR:\n{error}\n\nToo many failed attempts. Stop calling tools and tell the user "
                   "briefly what went wrong and what they could try instead.")
        else:
            msg = f"ERROR:\n{error}\n\nFix the code and try again ({left} attempts left)."
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
        out = res.output.strip() or "(nothing printed - use print() to see results)"
        return self._log("run_analysis", code, True, out)

    def make_chart(self, code: str) -> str:
        """Build a chart. Code must set result = as_plotly(fig, ...) / as_matplotlib(fig, ...) / as_altair(chart, ...). The chart is shown to the user."""
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
        """Clean or transform data (drop nulls, rename, filter, add columns, join...). Code must set result = as_data_prep({"<dataset name>": new_df}, "what changed"). A new name creates a new dataset (e.g. a joined table). Later steps use the new data."""
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
        """Root-cause finder: why did a metric change between two groups or periods?
        Compares agg ('sum', 'mean' or 'count') of `metric` for split_column == group_a vs group_b,
        breaks the change down by every other column and ranks the segments that caused most of it.
        For periods, pass the date column itself with prefixes: split_column='Date',
        group_a='2024-03', group_b='2024-04' (no need to add a month column). For row counts use agg='count'.
        `dataset` is the dataset name; it can be left empty when only one dataset is loaded."""
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

    def ask_user(self, question: str, options: Optional[List[str]] = None) -> str:
        """Ask the user ONE short clarifying question instead of guessing, then stop and wait.
        Only use it when the request is ambiguous in a way that changes the answer and the data
        can't settle it (e.g. a column the user names doesn't exist, or a term like 'revenue' or
        'active customer' has several plausible meanings). Give 2-4 short suggested answers in
        `options` when you can (e.g. real column names)."""
        options = [str(o).strip() for o in (options or []) if str(o).strip()][:4]
        self.clarification = {"question": question.strip(), "options": options}
        self._log("ask_user", "", True, question.strip()
                  + "".join(f"\n- {o}" for o in options))
        return question.strip()

    def _make_tools(self):
        tools = [StructuredTool.from_function(func=f) for f in
                 (self.inspect_data, self.run_analysis, self.make_chart, self.update_dataset,
                  self.explain_change)]
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

    def ask(self, question: str, datasets: Dict[str, Any],
            on_step: Optional[Callable[[Step], None]] = None, critic: bool = False,
            rules: Optional[List[str]] = None) -> AgentReply:
        """Answer one question. on_step(step) is called after every tool call.
        With critic=True a second LLM call reviews the answer; if it finds problems
        the agent gets one chance to fix them. `rules` are the user's saved rules
        relevant to this question (from memory.py); the agent must follow them."""
        self._reset_turn(datasets, on_step)
        full_question = question
        if rules:
            self._log("memory", "", True, "Using saved rules:\n" + "\n".join(f"- {r}" for r in rules))
            full_question = ("The user saved these rules earlier. Follow them where they apply:\n"
                             + "\n".join(f"- {r}" for r in rules) + f"\n\nQuestion: {question}")
        messages = self.history[-MAX_HISTORY_MESSAGES:] + [HumanMessage(full_question)]

        answer, convo = self._run(messages)
        if self.clarification:  # the agent is waiting for the user: nothing to review yet
            answer = self.clarification["question"]
        elif answer is not None and critic:
            verdict = self._critique(full_question, answer, self.steps)
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
