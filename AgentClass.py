"""Code execution layer for the data agent.

The agent (see langchain_agent.py) writes small pieces of pandas / DuckDB code.
This module checks that code, runs it against the user's datasets and returns
what happened: printed output, a chart or cleaned data, or an error message the
agent can read and fix.

Note: the checks below block the obvious ways to escape (imports, file access,
dunder tricks), but in-process Python can never be a perfect sandbox. If the
app is public, also run it inside a container with no secrets on disk.
"""

import ast
import io
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import altair as alt
import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

from rootcause import root_cause  # noqa: E402

MAX_OUTPUT_CHARS = 4000


# ---------------------------------------------------------------- results

@dataclass
class ChartResult:
    kind: str  # "plotly" | "matplotlib" | "altair" | "data_prep"
    obj: Any
    explanation: str
    modified_datasets: Optional[Dict[str, Any]] = None


def as_plotly(fig, explanation: str) -> ChartResult:
    return ChartResult(kind="plotly", obj=fig, explanation=explanation)


def as_matplotlib(fig, explanation: str) -> ChartResult:
    return ChartResult(kind="matplotlib", obj=fig, explanation=explanation)


def as_altair(chart, explanation: str) -> ChartResult:
    return ChartResult(kind="altair", obj=chart, explanation=explanation)


def as_data_prep(datasets: Dict[str, Any], explanation: str) -> ChartResult:
    return ChartResult(kind="data_prep", obj=None, explanation=explanation,
                       modified_datasets=datasets)


@dataclass
class RunOutcome:
    ok: bool
    output: str = ""
    result: Optional[ChartResult] = None
    error: Optional[str] = None


# ---------------------------------------------------------------- safety

SAFE_BUILTINS = {
    name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
    for name in [
        "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float",
        "format", "int", "isinstance", "len", "list", "map", "max", "min", "print",
        "range", "repr", "reversed", "round", "set", "slice", "sorted", "str", "sum",
        "tuple", "type", "zip", "Exception", "ValueError", "KeyError", "TypeError",
        "True", "False", "None",
    ]
}

BLOCKED_NAMES = {
    "eval", "exec", "compile", "open", "input", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "__import__", "breakpoint", "help", "memoryview",
}

# Attributes that reach the file system, OS or interpreter internals.
BLOCKED_ATTRS = {
    "os", "sys", "subprocess", "system", "popen", "builtins", "importlib",
    "load", "loads", "fromfile", "tofile", "savefig", "save", "write_html",
    "connect", "execute", "eval", "format", "format_map", "lib",
    "environ", "getenv", "secrets",  # API keys live in env vars / st.secrets
}

# Imports of libraries that are already loaded are removed instead of rejected,
# so the model doesn't waste a retry on "import pandas as pd".
PRELOADED_IMPORTS = {
    ("pandas", "pd"), ("numpy", "np"), ("plotly.express", "px"),
    ("plotly.graph_objects", "go"), ("matplotlib.pyplot", "plt"), ("altair", "alt"),
}


def strip_preloaded_imports(code: str) -> str:
    """Drop top-level lines like `import pandas as pd`; leave everything else for check_code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code  # check_code reports it
    drop = set()
    for node in tree.body:
        if isinstance(node, ast.Import) and all((a.name, a.asname) in PRELOADED_IMPORTS for a in node.names):
            drop.update(range(node.lineno, node.end_lineno + 1))
    if not drop:
        return code
    return "\n".join(line for i, line in enumerate(code.splitlines(), 1) if i not in drop)
BLOCKED_ATTR_PREFIXES = ("read_", "write_", "to_csv", "to_excel", "to_parquet", "to_feather",
                         "to_hdf", "to_sql", "to_stata", "to_pickle", "to_clipboard")


class UnsafeCodeError(Exception):
    pass


def check_code(code: str) -> None:
    """Raise UnsafeCodeError if the code uses anything we don't allow."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise UnsafeCodeError(f"Syntax error: {e}") from e

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise UnsafeCodeError("Imports are not allowed. pd, np, px, go, plt, alt and sql are already available.")
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            raise UnsafeCodeError("global/nonlocal are not allowed.")
        if isinstance(node, ast.Name):
            if node.id in BLOCKED_NAMES or node.id.startswith("__"):
                raise UnsafeCodeError(f"Use of '{node.id}' is not allowed.")
        if isinstance(node, ast.Attribute):
            a = node.attr
            if a.startswith("_") or a in BLOCKED_ATTRS or a.startswith(BLOCKED_ATTR_PREFIXES):
                raise UnsafeCodeError(f"Use of '.{a}' is not allowed (no file, OS or internal access).")


# ---------------------------------------------------------------- SQL helper

class SQLProxy:
    """Read-only DuckDB over the in-memory datasets. File access is switched off."""

    def __init__(self, dfs: Dict[str, Any]):
        self._con = duckdb.connect(":memory:")
        for name, df in dfs.items():
            self._con.register(name, df)
        self._con.execute("SET enable_external_access = false")
        self._con.execute("SET lock_configuration = true")

    def query(self, sql_text: str) -> pd.DataFrame:
        first = sql_text.strip().lstrip("(").split(None, 1)[0].lower() if sql_text.strip() else ""
        if first not in ("select", "with", "describe", "summarize"):
            raise ValueError("Only read queries (SELECT / WITH / DESCRIBE / SUMMARIZE) are allowed.")
        return self._con.query(sql_text).to_df()

    def describe(self, table: str) -> pd.DataFrame:
        return self.query(f'DESCRIBE "{table}"')

    def close(self):
        self._con.close()


# ---------------------------------------------------------------- runner

def _short_error(exc: BaseException, code: str) -> str:
    """Error message plus the failing line of the agent's code."""
    msg = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    line_no = None
    for frame in traceback.extract_tb(exc.__traceback__):
        if frame.filename == "<agent_code>":
            line_no = frame.lineno
    if line_no:
        lines = code.splitlines()
        if 0 < line_no <= len(lines):
            return f"Line {line_no}: {lines[line_no - 1].strip()}\n{msg}"
    return msg


def run_code(code: str, datasets: Dict[str, Any]) -> RunOutcome:
    """Run agent-written code. Never raises: errors come back in RunOutcome."""
    code = strip_preloaded_imports(strip_code_fences(code))
    try:
        check_code(code)
    except UnsafeCodeError as e:
        return RunOutcome(ok=False, error=f"Code rejected: {e}")

    sql = SQLProxy(datasets)
    env: Dict[str, Any] = {
        "__builtins__": SAFE_BUILTINS,
        "pd": pd, "np": np, "px": px, "go": go, "plt": plt, "alt": alt,
        "sql": sql,
        "ChartResult": ChartResult,
        "as_plotly": as_plotly, "as_matplotlib": as_matplotlib,
        "as_altair": as_altair, "as_data_prep": as_data_prep,
        "root_cause": root_cause,
    }
    # Datasets are also available directly by name, as copies so failed code
    # can't half-modify the real data.
    for name, df in datasets.items():
        if name.isidentifier():
            env[name] = df.copy()

    buf = io.StringIO()
    figs_before = set(plt.get_fignums())
    try:
        with redirect_stdout(buf):
            exec(compile(code, "<agent_code>", "exec"), env)  # noqa: S102 - checked above
        output = buf.getvalue()
        if len(output) > MAX_OUTPUT_CHARS:
            output = output[:MAX_OUTPUT_CHARS] + "\n... (output cut off)"
        result = env.get("result")
        if result is not None and not isinstance(result, ChartResult):
            return RunOutcome(ok=False, output=output,
                              error="`result` must be made with as_plotly / as_matplotlib / as_altair / as_data_prep.")
        return RunOutcome(ok=True, output=output, result=result)
    except Exception as exc:
        return RunOutcome(ok=False, output=buf.getvalue()[:MAX_OUTPUT_CHARS], error=_short_error(exc, code))
    finally:
        sql.close()
        # Detach new figures from pyplot so they don't pile up in memory.
        # A closed Figure object can still be shown and saved.
        for num in set(plt.get_fignums()) - figs_before:
            plt.close(num)


# ---------------------------------------------------------------- data overview

def strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else ""
    if s.endswith("```"):
        s = s[: -3]
    return s.strip()


def describe_datasets(dfs: Dict[str, Any], column: Optional[str] = None, sample_rows: int = 5) -> str:
    """Text overview the agent reads before writing code."""
    parts: List[str] = []
    for name, df in dfs.items():
        if column:
            if column not in df.columns:
                parts.append(f"Dataset '{name}' has no column '{column}'. Columns: {list(df.columns)}")
                continue
            s = df[column]
            lines = [f"Column '{column}' in '{name}' ({s.dtype})",
                     f"nulls: {int(s.isna().sum())} of {len(s)}, unique values: {s.nunique()}"]
            if pd.api.types.is_numeric_dtype(s):
                lines.append(s.describe().to_string())
            else:
                lines.append("most common values:\n" + s.value_counts().head(15).to_string())
            parts.append("\n".join(lines))
            continue

        lines = [f"Dataset '{name}': {df.shape[0]} rows x {df.shape[1]} columns", "Columns:"]
        for col in df.columns:
            s = df[col]
            lines.append(f"  - {col} ({s.dtype}), nulls={int(s.isna().sum())}, unique={s.nunique()}")
        lines.append(f"First {sample_rows} rows:")
        lines.append(df.head(sample_rows).to_string(index=False, max_colwidth=40))
        parts.append("\n".join(lines))
    return "\n\n".join(parts)
