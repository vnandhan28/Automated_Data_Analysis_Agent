"""Run the test questions against one or more models, with the critic on and/or off.

Usage (from the project folder):
  python eval/run_eval.py                                   # default model, critic on + off
  python eval/run_eval.py --models groq:openai/gpt-oss-120b gemini:gemini-3.5-flash
  python eval/run_eval.py --critic off --only c1 c2 u4      # a few questions
  python eval/run_eval.py --summary                         # just rebuild the summary

Results are appended to eval/results/results.csv after every question, so a run that
stops (e.g. a free-tier daily limit) continues where it left off next time.
Questions that failed because of an API error are retried on the next run.
"""

import argparse
import csv
import re
import sys
import time
import tomllib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_agent import PROVIDERS, DataAgent  # noqa: E402

from questions import DATASETS, QUESTIONS  # noqa: E402

RESULTS = ROOT / "eval" / "results" / "results.csv"
SUMMARY = ROOT / "eval" / "results" / "summary.md"
DEFAULT_MODELS = ["groq:openai/gpt-oss-120b"]
DEFAULT_TOL = 0.01
FIELDS = ["timestamp", "provider", "model", "critic", "qid", "dataset", "kind", "status",
          "numbers_ok", "text_ok", "chart_ok", "shape_ok", "tool_ok", "expected", "time_s", "steps",
          "tool_errors", "critic_flagged", "tools_used", "answer", "error"]

# ---------------------------------------------------------------- grading

SCALE = {"k": 1e3, "thousand": 1e3, "lakh": 1e5, "lakhs": 1e5, "m": 1e6, "million": 1e6,
         "crore": 1e7, "crores": 1e7, "b": 1e9, "billion": 1e9}
NUMBER = re.compile(r"(\d+(?:\.\d+)?)(?:\s*(k|thousand|lakhs?|m|million|crores?|b|billion)\b)?", re.I)


def normalize(text: str) -> str:
    """LLMs often write non-breaking spaces/hyphens (e.g. 'TV‑MA' with U+2011); make them plain."""
    text = re.sub(r"[‐-―−]", "-", text)
    return re.sub(r"[    ]", " ", text)


def numbers_in(text: str) -> list:
    """All numbers in an answer: handles 1,234 / 1 234 / ₹4.5 lakh / 3.2M / unicode minus signs."""
    t = normalize(text)
    t = re.sub(r"(?<=\d)[, ](?=\d{3}(?!\d))", "", t)  # thousands separators
    out = []
    for num, unit in NUMBER.findall(t):
        out.append(float(num) * (SCALE[unit.lower()] if unit else 1))
    return out


def has_number(expected: float, found: list, tol: float) -> bool:
    """Sign-insensitive ("fell by 314,399" = -314,399), within a relative tolerance."""
    e = abs(float(expected))
    return any(abs(abs(f) - e) <= max(tol * e, 0.006 if e < 1 else 0) for f in found)


def grade(q: dict, df: pd.DataFrame, reply) -> dict:
    checks = {}
    if "numbers" in q:
        expected = [float(x) for x in q["numbers"](df)]
        found = numbers_in(reply.answer)
        checks["numbers_ok"] = all(has_number(e, found, q.get("tol", DEFAULT_TOL)) for e in expected)
        checks["expected"] = "; ".join(f"{e:,.0f}" if abs(e) >= 100 else f"{e:.4g}" for e in expected)
    if "text" in q:
        checks["text_ok"] = all(normalize(w).lower() in normalize(reply.answer).lower() for w in q["text"])
        checks["expected"] = (checks.get("expected", "") + " " + " ".join(q["text"])).strip()
    if q.get("chart"):
        checks["chart_ok"] = len(reply.charts) > 0
    if "shape" in q:
        want = q["shape"](df)
        checks["shape_ok"] = reply.data_changed and tuple(reply.datasets["data"].shape) == tuple(want)
    if "tool" in q:
        checks["tool_ok"] = any(s.tool == q["tool"] and s.ok for s in reply.steps)
    return checks


# ---------------------------------------------------------------- running

def load_done() -> set:
    if not RESULTS.exists():
        return set()
    done = pd.read_csv(RESULTS, dtype=str, keep_default_na=False)
    done = done[done["status"] != "api_error"]
    return set(zip(done["provider"], done["model"], done["critic"], done["qid"]))


def append(row: dict):
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    new = not RESULTS.exists()
    with RESULTS.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def is_daily_limit(err: str) -> bool:
    return any(s in err for s in ("per day", "TPD", "RESOURCE_EXHAUSTED", "no remaining credits", "quota"))


def run(models, critic_modes, only):
    secrets = tomllib.load(open(ROOT / ".streamlit" / "secrets.toml", "rb"))
    data = {name: pd.read_csv(ROOT / path) for name, path in DATASETS.items()}
    questions = [q for q in QUESTIONS if not only or q["id"] in only]
    done = load_done()

    for spec in models:
        provider, model = spec.split(":", 1)
        key = secrets.get(PROVIDERS[provider]["key"], "")
        if not key:
            print(f"!! no key for {provider}, skipping")
            continue
        stop_model = False
        for critic in critic_modes:
            for q in questions:
                if stop_model:
                    break
                tag = (provider, model, "on" if critic else "off", q["id"])
                if tag in done:
                    continue
                df = data[q["dataset"]]
                row = {"timestamp": time.strftime("%Y-%m-%d %H:%M"), "provider": provider, "model": model,
                       "critic": tag[2], "qid": q["id"], "dataset": q["dataset"], "kind": q["kind"]}
                t = time.time()
                try:
                    reply = DataAgent(key, model=model, provider=provider).ask(
                        q["question"], {"data": df}, critic=critic, rules=q.get("rules"))
                except Exception as e:
                    err = str(e)[:300]
                    append({**row, "status": "api_error", "error": err, "time_s": round(time.time() - t, 1)})
                    print(f"   {q['id']:4} API ERROR: {err[:120]}")
                    if is_daily_limit(err):
                        print(f"!! {provider}:{model} hit its usage limit; run again later to continue.")
                        stop_model = True
                    continue
                checks = grade(q, df, reply)
                passed = all(v for k, v in checks.items() if k.endswith("_ok"))
                row.update(checks)
                row.update(status="pass" if passed else "fail", time_s=round(time.time() - t, 1),
                           steps=len([s for s in reply.steps if s.tool not in ("critic", "memory")]),
                           tool_errors=sum(1 for s in reply.steps if not s.ok and s.tool not in ("critic",)),
                           critic_flagged=int(any(s.tool == "critic" and not s.ok for s in reply.steps)),
                           tools_used=" ".join(s.tool for s in reply.steps),
                           answer=reply.answer.replace("\n", " ")[:2000])
                append(row)
                print(f"   {q['id']:4} {'PASS' if passed else 'FAIL'}  {row['time_s']:>6}s  "
                      f"{provider}:{model} critic={tag[2]}  " + " ".join(f"{k}={v}" for k, v in checks.items()
                                                                         if k.endswith("_ok")))
            if stop_model:
                break


def regrade():
    """Re-check the text matches on saved answers (after a grader fix) without calling the API.
    Chart / shape / tool checks can't be redone from text, so they are kept as they were."""
    if not RESULTS.exists():
        return
    data = {name: pd.read_csv(ROOT / path) for name, path in DATASETS.items()}
    qs = {q["id"]: q for q in QUESTIONS}
    r = pd.read_csv(RESULTS, dtype=str, keep_default_na=False)
    changed = 0
    for i, row in r[r["status"].isin(["pass", "fail"])].iterrows():
        q = qs.get(row["qid"])
        if not q:
            continue
        reply = type("Reply", (), {"answer": row["answer"], "charts": [], "steps": [],
                                   "data_changed": False, "datasets": {}})()
        q_text_only = {k: v for k, v in q.items() if k == "text"}  # saved answers are cut, so numbers stay as graded
        checks = grade(q_text_only, data[q["dataset"]], reply)
        for k, v in checks.items():
            r.at[i, k] = str(v)
        oks = [r.at[i, c] for c in FIELDS if c.endswith("_ok") and r.at[i, c] != ""]
        status = "pass" if all(v == "True" for v in oks) else "fail"
        changed += status != row["status"]
        r.at[i, "status"] = status
    r.to_csv(RESULTS, index=False)
    print(f"Regraded saved answers: {changed} changed status.")


# ---------------------------------------------------------------- summary

def summarize():
    if not RESULTS.exists():
        print("No results yet.")
        return
    r = pd.read_csv(RESULTS, keep_default_na=False)
    api_errors = r[r["status"] == "api_error"]
    r = r[r["status"] != "api_error"].drop_duplicates(["provider", "model", "critic", "qid"], keep="last")
    r["passed"] = r["status"] == "pass"
    for col in ("time_s", "steps", "tool_errors", "critic_flagged"):
        r[col] = pd.to_numeric(r[col], errors="coerce")

    overall = (r.groupby(["model", "critic"])
                .agg(questions=("qid", "count"), accuracy=("passed", "mean"), avg_time_s=("time_s", "mean"),
                     avg_steps=("steps", "mean"), tool_errors_per_q=("tool_errors", "mean"),
                     critic_flagged=("critic_flagged", "sum"))
                .reset_index())
    overall["accuracy"] = (overall["accuracy"] * 100).round(1).astype(str) + "%"
    overall = overall.round(2)

    by_kind = r.pivot_table(index="kind", columns=["model", "critic"], values="passed", aggfunc="mean") * 100
    failed = r[~r["passed"]][["model", "critic", "qid", "expected", "answer"]]

    md = ["# Evaluation results", "",
          f"{len(QUESTIONS)} questions on 3 sample datasets; expected answers computed with pandas.", "",
          "## Overall", "", overall.to_markdown(index=False), "",
          "## Accuracy by question type (%)", "", by_kind.round(0).to_markdown(), "",
          "## Failed questions", "",
          failed.assign(answer=failed["answer"].str[:200]).to_markdown(index=False) if len(failed) else "None.", ""]
    if len(api_errors):
        md += [f"_{len(api_errors)} API errors (rate limits etc.) were excluded and will be retried._", ""]
    SUMMARY.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:8]))
    print(f"\nFull summary: {SUMMARY.relative_to(ROOT)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="provider:model, e.g. groq:openai/gpt-oss-120b")
    p.add_argument("--critic", choices=["on", "off", "both"], default="both")
    p.add_argument("--only", nargs="+", help="question ids to run, e.g. c1 u4")
    p.add_argument("--summary", action="store_true", help="only rebuild the summary from saved results")
    p.add_argument("--regrade", action="store_true", help="re-check saved answers after a grader fix (no API calls)")
    p.add_argument("--results", help="CSV to use instead of eval/results/results.csv (e.g. to compare versions); "
                                     "the summary is written next to it")
    a = p.parse_args()
    if a.results:  # module-level names, read by the functions above
        RESULTS = Path(a.results).resolve()
        SUMMARY = RESULTS.with_name(RESULTS.stem + "_summary.md")
    if a.regrade:
        regrade()
    elif not a.summary:
        run(a.models, {"on": [True], "off": [False], "both": [False, True]}[a.critic], a.only)
    summarize()
