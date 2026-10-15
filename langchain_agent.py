"""The data analysis agent.

The LLM works in a loop with four tools:

  inspect_data    look at columns, types, nulls and sample values
  run_analysis    run pandas / SQL code and read the printed result
  make_chart      build a chart (checked to make sure it actually has data)
  update_dataset  clean or transform the data (later steps see the new data)

Every tool returns plain text. When code fails, the error goes back to the
LLM so it can fix the code and try again, up to MAX_FAILED_RUNS times.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

from AgentClass import ChartResult, describe_datasets, run_code

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
MAX_AGENT_STEPS = 30     # hard cap on loop length (LangGraph recursion limit)
MAX_HISTORY_MESSAGES = 10

SYSTEM_PROMPT = """You are a data analysis agent. The user has loaded tabular data and asks questions about it.

Datasets are pandas DataFrames available in your code by name (usually `data`).

How to work:
1. If you are not sure about column names, types or values, call inspect_data first.
2. Use run_analysis to compute numbers and check them before you chart or answer. Print what you need to see.
3. Use make_chart to draw a chart when it helps or when the user asks for one.
4. Use update_dataset only when the user asks to clean, filter, rename or otherwise change the data.
5. If a tool returns an error, read it, fix the code and try again. Do not repeat the same code.
6. Finish with a short answer in simple words: what you found and the key numbers. Don't paste code in the answer.

Code rules (all tools that take code):
- Plain Python only. No imports: pd, np, px (plotly.express), go (plotly.graph_objects), plt, alt (altair) are ready.
- SQL: df = sql.query("SELECT ... FROM data ...") (DuckDB, read-only).
- No file, OS or network access. No names starting with underscore.
- make_chart code must end with one of:
    result = as_plotly(fig, "one-sentence explanation")
    result = as_matplotlib(fig, "one-sentence explanation")
    result = as_altair(chart, "one-sentence explanation")
  Prefer plotly. Give the chart a clear title and axis labels.
- update_dataset code must end with:
    result = as_data_prep({"data": new_df}, "what changed")
"""


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
            )
        self.history: List[Any] = []   # HumanMessage / AIMessage pairs
        self._agent = create_agent(model=self.llm, tools=self._make_tools(), system_prompt=SYSTEM_PROMPT)
        self._reset_turn({})

    # ------------------------------------------------------------ per-turn state

    def _reset_turn(self, datasets: Dict[str, Any]):
        self.datasets: Dict[str, Any] = dict(datasets)
        self.charts: List[ChartResult] = []
        self.steps: List[Step] = []
        self.failed_runs = 0
        self.data_changed = False

    def _log(self, tool: str, code: str, ok: bool, output: str) -> str:
        self.steps.append(Step(tool=tool, code=code, ok=ok, output=output))
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
        """Clean or transform data (drop nulls, rename, filter, add columns...). Code must set result = as_data_prep({"data": new_df}, "what changed"). Later steps use the new data."""
        if (stop := self._out_of_attempts("update_dataset", code)):
            return stop
        res = run_code(code, self.datasets)
        if not res.ok:
            return self._failed("update_dataset", code, res.error)
        r = res.result
        if r is None or r.kind != "data_prep" or not r.modified_datasets:
            return self._failed("update_dataset", code,
                                'Data was not updated. Set result = as_data_prep({"data": new_df}, "what changed").')
        changes = []
        for name, new_df in r.modified_datasets.items():
            if name not in self.datasets:
                return self._failed("update_dataset", code,
                                    f"Unknown dataset '{name}'. Use one of: {list(self.datasets)}")
            if not isinstance(new_df, pd.DataFrame):
                return self._failed("update_dataset", code, f"'{name}' must be a pandas DataFrame.")
            old = self.datasets[name]
            changes.append(f"{name}: {old.shape[0]}x{old.shape[1]} -> {new_df.shape[0]}x{new_df.shape[1]}, "
                           f"columns: {list(new_df.columns)}")
            self.datasets[name] = new_df
        self.data_changed = True
        return self._log("update_dataset", code, True,
                         "Dataset updated. " + "; ".join(changes) + f"\nExplanation: {r.explanation}")

    def _make_tools(self):
        return [StructuredTool.from_function(func=f) for f in
                (self.inspect_data, self.run_analysis, self.make_chart, self.update_dataset)]

    # ------------------------------------------------------------ main entry

    def ask(self, question: str, datasets: Dict[str, Any]) -> AgentReply:
        self._reset_turn(datasets)
        messages = self.history[-MAX_HISTORY_MESSAGES:] + [HumanMessage(question)]

        try:
            response = self._agent.invoke({"messages": messages},
                                          config={"recursion_limit": MAX_AGENT_STEPS})
            answer = response["messages"][-1].content or ""
            if isinstance(answer, list):  # some models return content blocks
                answer = " ".join(b.get("text", "") for b in answer if isinstance(b, dict))
        except Exception as e:  # e.g. step limit reached or API error
            if type(e).__name__ == "GraphRecursionError":
                answer = "I took too many steps on this one and stopped. Try asking a simpler or more specific question."
            else:
                raise

        if not answer.strip():
            answer = "Done." if (self.charts or self.data_changed) else "I couldn't produce an answer."

        self.history += [HumanMessage(question), AIMessage(answer)]
        return AgentReply(answer=answer, charts=self.charts, steps=self.steps,
                          datasets=self.datasets, data_changed=self.data_changed)

    def clear(self):
        self.history = []


# Old name, kept so existing imports still work.
LangChainVizAgent = DataAgent
