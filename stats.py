"""Ready-made statistics for the agent: correlations, outliers, trend and a simple forecast.

Each check returns (table, text): a DataFrame the agent can chart and a short text
summary it reads. Plain pandas/numpy; p-values use an exact Student-t distribution.
"""

import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd

FREQS = {"D": "day", "W": "week", "M": "month", "Q": "quarter", "Y": "year"}
SEASON = {"D": 7, "W": 52, "M": 12, "Q": 4}  # periods per seasonal cycle


# ---------------------------------------------------------------- p-values

def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Numerical Recipes)."""
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > 1e-30 else 1e-30)
    h = d
    for m in range(1, 200):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)), -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1 + aa * d
            d = 1 / (d if abs(d) > 1e-30 else 1e-30)
            c = 1 + aa / c
            c = c if abs(c) > 1e-30 else 1e-30
            h *= d * c
        if abs(d * c - 1) < 1e-12:
            break
    return h


def t_pvalue(t: float, dof: float) -> float:
    """Two-sided p-value of a t statistic."""
    if dof <= 0 or not math.isfinite(t):
        return 0.0 if math.isinf(t) else 1.0
    if t == 0:
        return 1.0
    x = dof / (dof + t * t)
    a, b = dof / 2, 0.5
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    ibeta = bt * _betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1 - bt * _betacf(b, a, 1 - x) / b
    return min(max(ibeta, 0.0), 1.0)


def _pfmt(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


# ---------------------------------------------------------------- helpers

def _numeric_columns(df: pd.DataFrame, columns: Optional[list]) -> list:
    """Numeric columns worth analysing: not constant, not an ID (every value unique integers)."""
    cols = columns or [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
                       and not pd.api.types.is_bool_dtype(df[c])]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Unknown columns {missing}. Columns: {list(df.columns)}")
    out = []
    for c in cols:
        s = df[c].dropna()
        if not pd.api.types.is_numeric_dtype(s) or s.nunique() < 2:
            continue
        if s.nunique() == len(s) and len(s) > 50 and (s % 1 == 0).all():
            continue  # looks like an ID
        out.append(c)
    return out


def _series(df: pd.DataFrame, date_column: str, metric: str, agg: str, freq: str) -> pd.Series:
    """Metric aggregated per period, e.g. monthly sum of Booking Value."""
    freq = freq.upper()[:1]
    if freq not in FREQS:
        raise ValueError(f"freq must be one of {list(FREQS)} (day, week, month, quarter, year).")
    if date_column not in df.columns:
        raise ValueError(f"Unknown date_column '{date_column}'. Columns: {list(df.columns)}")
    if agg != "count" and metric not in df.columns:
        raise ValueError(f"Unknown metric '{metric}'. Columns: {list(df.columns)}")
    if pd.api.types.is_numeric_dtype(df[date_column]):
        raise ValueError(f"'{date_column}' is a number column, not a date column.")
    dates = pd.to_datetime(df[date_column], errors="coerce")
    if dates.notna().mean() < 0.5:
        raise ValueError(f"'{date_column}' doesn't look like a date column.")
    periods = dates.dt.to_period(freq)
    values = df[metric] if agg != "count" else pd.Series(1, index=df.index)
    g = values.groupby(periods)
    s = g.size() if agg == "count" else g.sum() if agg == "sum" else g.mean()
    s = s.sort_index()
    # fill missing periods so gaps count as 0 (sum/count) or stay missing (mean)
    full = pd.period_range(s.index.min(), s.index.max(), freq=freq)
    s = s.reindex(full, fill_value=0 if agg != "mean" else np.nan).dropna()
    # A first/last period with far fewer days of data (e.g. a week with 2 days) would look like
    # a crash in a sum, so it is dropped.
    days = dates.dt.normalize().groupby(periods).nunique().reindex(s.index, fill_value=0)
    typical = days.median()
    dropped = []
    while len(s) > 3 and days.iloc[-1] < 0.5 * typical:
        dropped.append(str(s.index[-1])); s, days = s.iloc[:-1], days.iloc[:-1]
    while len(s) > 3 and days.iloc[0] < 0.5 * typical:
        dropped.append(str(s.index[0])); s, days = s.iloc[1:], days.iloc[1:]
    if len(s) < 3:
        raise ValueError(f"Only {len(s)} {FREQS[freq]}(s) of data; need at least 3 for a trend.")
    s.attrs["dropped"] = dropped
    return s


def _num(x: float) -> str:
    return f"{x:,.0f}" if abs(x) >= 100 else f"{x:,.3g}"


def _dropped_note(s: pd.Series) -> str:
    d = s.attrs.get("dropped")
    return f"\nLeft out incomplete period(s) with much less data: {', '.join(d)}." if d else ""


def _linear_fit(y: np.ndarray):
    """Least-squares line y = a + b*t. Returns slope, intercept, r2, p-value of slope, residual std."""
    n = len(y)
    t = np.arange(n, dtype=float)
    b, a = np.polyfit(t, y, 1)
    resid = y - (a + b * t)
    sxx = ((t - t.mean()) ** 2).sum()
    s = math.sqrt((resid ** 2).sum() / (n - 2)) if n > 2 else 0.0
    se_b = s / math.sqrt(sxx) if sxx and s else 0.0
    p = t_pvalue(b / se_b, n - 2) if se_b else 0.0
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid ** 2).sum() / ss_tot if ss_tot else 1.0
    return b, a, r2, p, s, sxx


# ---------------------------------------------------------------- checks

def correlations(df: pd.DataFrame, columns: Optional[list] = None, top: int = 10) -> Tuple[pd.DataFrame, str]:
    cols = _numeric_columns(df, columns)
    if len(cols) < 2:
        raise ValueError("Need at least 2 numeric columns for correlations.")
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            pair = df[[a, b]].dropna()
            n = len(pair)
            if n < 5 or pair[a].nunique() < 2 or pair[b].nunique() < 2:
                continue
            r = float(np.corrcoef(pair[a], pair[b])[0, 1])
            t = r * math.sqrt((n - 2) / max(1 - r * r, 1e-12))
            strength = "strong" if abs(r) >= 0.6 else "moderate" if abs(r) >= 0.3 else "weak"
            rows.append({"column_a": a, "column_b": b, "r": round(r, 3), "strength": strength,
                         "direction": "positive" if r > 0 else "negative", "n": n, "p_value": t_pvalue(t, n - 2)})
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("No column pairs with enough data.")
    table = table.reindex(table["r"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    lines = [f"Pearson correlations between {len(cols)} numeric columns (strongest first):"]
    for _, r in table.head(top).iterrows():
        lines.append(f"- {r.column_a} vs {r.column_b}: r = {r.r:+.3f} ({r.strength} {r.direction}), "
                     f"n = {r.n:,}, {_pfmt(r.p_value)}")
    lines.append("|r| < 0.3 weak, 0.3-0.6 moderate, >= 0.6 strong. Correlation does not mean causation; "
                 "with many rows even tiny correlations get small p-values, so judge by r.")
    return table, "\n".join(lines)


def outliers(df: pd.DataFrame, columns: Optional[list] = None) -> Tuple[pd.DataFrame, str]:
    rows = []
    for c in _numeric_columns(df, columns):
        s = df[c].dropna()
        if s.nunique() <= 10:  # scores / flags: "outlier" makes little sense
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        out = s[(s < lo) | (s > hi)]
        rows.append({"column": c, "outliers": len(out), "pct": round(len(out) / len(s) * 100, 2),
                     "low_limit": lo, "high_limit": hi, "min": s.min(), "max": s.max(),
                     "examples": ", ".join(f"{v:,.4g}" for v in out.sort_values(key=abs, ascending=False)
                                           .drop_duplicates().head(3))})
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("No numeric columns with enough distinct values to check for outliers.")
    table = table.sort_values("pct", ascending=False).reset_index(drop=True)
    lines = ["Outliers by the IQR rule (below Q1 - 1.5*IQR or above Q3 + 1.5*IQR):"]
    for _, r in table.iterrows():
        lines.append(f"- {r.column}: {r.outliers:,} outliers ({r.pct}%), normal range "
                     f"{r.low_limit:,.4g} to {r.high_limit:,.4g}, data range {r['min']:,.4g} to {r['max']:,.4g}"
                     + (f"; most extreme: {r.examples}" if r.outliers else ""))
    return table, "\n".join(lines)


def trend(df: pd.DataFrame, date_column: str, metric: str, agg: str = "sum",
          freq: str = "M") -> Tuple[pd.DataFrame, str]:
    s = _series(df, date_column, metric, agg, freq)
    y = s.to_numpy(dtype=float)
    b, a, r2, p, _, _ = _linear_fit(y)
    unit = FREQS[freq.upper()[:1]]
    per_period_pct = b / y.mean() * 100 if y.mean() else 0.0
    if p >= 0.05:
        verdict = f"no clear trend ({_pfmt(p)}): the ups and downs look like noise"
    else:
        verdict = f"a significant {'upward' if b > 0 else 'downward'} trend ({_pfmt(p)})"
    label = f"{agg} of {metric}" if agg != "count" else "row count"
    table = pd.DataFrame({"period": s.index.astype(str), "value": y,
                          "trend_line": a + b * np.arange(len(y))})
    text = (f"{label} per {unit}, {s.index[0]} to {s.index[-1]} ({len(y)} {unit}s): {verdict}.\n"
            f"Trend line: {'+' if b >= 0 else '-'}{_num(abs(b))} per {unit} ({per_period_pct:+.2f}% of the "
            f"average per {unit}), R² = {r2:.2f}.\n"
            f"First {unit}: {_num(y[0])}; last {unit}: {_num(y[-1])}; average: {_num(y.mean())}; "
            f"highest: {s.idxmax()} ({_num(y.max())}); lowest: {s.idxmin()} ({_num(y.min())})."
            + _dropped_note(s))
    return table, text


def _fit_predict(y: np.ndarray, horizon: int, season: Optional[int]):
    """Least squares on [1, t] (+ one column per season position), fitted together so the
    seasonal swings don't bias the trend. Returns fitted, forecast, residual std."""
    n = len(y)
    t_all = np.arange(n + horizon, dtype=float)
    cols = [np.ones_like(t_all), t_all]
    if season:
        pos = t_all.astype(int) % season
        cols += [(pos == k).astype(float) for k in range(1, season)]  # position 0 is the baseline
    X = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(X[:n], y, rcond=None)
    pred = X @ coef
    fitted, forecast = pred[:n], pred[n:]
    dof = max(n - X.shape[1], 1)
    resid_std = float(math.sqrt(((y - fitted) ** 2).sum() / dof))
    return fitted, forecast, resid_std


def forecast(df: pd.DataFrame, date_column: str, metric: str, agg: str = "sum", freq: str = "M",
             periods: int = 3) -> Tuple[pd.DataFrame, str]:
    s = _series(df, date_column, metric, agg, freq)
    f = freq.upper()[:1]
    y = s.to_numpy(dtype=float)
    n, periods = len(y), max(1, min(int(periods), 24))
    season = SEASON.get(f) if SEASON.get(f) and n >= 2 * SEASON[f] else None
    _, pred, resid_std = _fit_predict(y, periods, season)
    t_new = np.arange(n, n + periods)
    sxx = ((np.arange(n) - (n - 1) / 2) ** 2).sum()
    width = 1.96 * resid_std * np.sqrt(1 + 1 / n + (t_new - (n - 1) / 2) ** 2 / sxx)
    future = pd.period_range(s.index[-1] + 1, periods=periods, freq=f)
    table = pd.DataFrame({"period": future.astype(str), "forecast": pred, "low": pred - width, "high": pred + width})

    # Honest accuracy check: refit without the last few periods and compare with what really happened.
    hold = min(periods, n // 4)
    accuracy = ""
    if hold >= 1 and n - hold >= 4:
        season_bt = season if season and n - hold >= 2 * season else None
        _, back, _ = _fit_predict(y[:-hold], hold, season_bt)
        actual = y[-hold:]
        if np.all(actual != 0):
            mape = float(np.mean(np.abs((actual - back) / actual)) * 100)
            accuracy = (f"\nBacktest: predicting the last {hold} known {FREQS[f]}(s) from the earlier data was off "
                        f"by {mape:.1f}% on average.")
    method = (f"linear trend + {season}-{FREQS[f]} seasonal pattern" if season
              else "linear trend (not enough history for a seasonal pattern)")
    label = f"{agg} of {metric}" if agg != "count" else "row count"
    lines = [f"Forecast of {label} per {FREQS[f]} for the next {periods} {FREQS[f]}(s), based on {n} {FREQS[f]}s "
             f"({s.index[0]} to {s.index[-1]}); method: {method}. Range = 95% interval."]
    for _, r in table.iterrows():
        lines.append(f"- {r.period}: {_num(r.forecast)} (range {_num(r.low)} to {_num(r.high)})")
    lines.append(accuracy.strip() or "Not enough history for a backtest.")
    if _dropped_note(s):
        lines.append(_dropped_note(s).strip())
    lines.append("This is a simple statistical projection, not a guarantee; it assumes the past pattern continues.")
    return table, "\n".join(lines)


# ---------------------------------------------------------------- entry point

def run_stats(df: pd.DataFrame, kind: str, columns: Optional[list] = None, metric: str = "",
              date_column: str = "", agg: str = "sum", freq: str = "M", periods: int = 3):
    """kind: 'correlations', 'outliers', 'trend' or 'forecast'. Returns (table, text)."""
    kind = kind.lower().strip()
    agg = (agg or "sum").lower()
    if agg not in ("sum", "mean", "count"):
        raise ValueError("agg must be 'sum', 'mean' or 'count'.")
    if kind.startswith("corr"):
        return correlations(df, columns)
    if kind.startswith("outlier"):
        return outliers(df, columns)
    if kind.startswith("trend"):
        return trend(df, date_column, metric, agg, freq)
    if kind.startswith("forecast"):
        return forecast(df, date_column, metric, agg, freq, periods)
    raise ValueError("kind must be 'correlations', 'outliers', 'trend' or 'forecast'.")
