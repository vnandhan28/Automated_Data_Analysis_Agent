"""Root-cause breakdown: why did a metric change between two groups or periods?

For a metric (sum, mean or row count) compared between group A and group B
(two values of one column, e.g. Date "2024-03" vs "2024-04"), every other
column is split into segments and each segment's share of the change is
computed. Segments are ranked by how much more (or less) they changed than
their size would predict, because a segment that is simply big always gets a
big share of any change.

Contributions add up exactly to the total change:
  sum / count:  contribution = value_B - value_A per segment
                expected     = segment's share of A * total change
  mean:         contribution = avg_share * (mean_B - mean_A)            (segment's own mean moved)
                             + (share_B - share_A) * (avg_mean - level)  (rows shifted towards it)
                expected     = avg_share * total change
"unexpected" = contribution - expected: what the segment's size alone doesn't explain.
"""

from typing import Any, Dict, Tuple

import pandas as pd

MAX_SEGMENTS = 30   # columns with more distinct values (IDs, free text) are skipped
NUMERIC_BINS = 4    # numeric columns are split into quartiles


def _match(series: pd.Series, value: str) -> pd.Series:
    """Rows where series equals value; if none, rows where it starts with value (e.g. '2024-03')."""
    s = series.astype(str)
    mask = s == str(value)
    if not mask.any():
        mask = s.str.startswith(str(value))
    return mask


def _segments(col: pd.Series) -> pd.Series:
    """Column as segment labels: quartile bins for numeric columns, '(missing)' for nulls."""
    if pd.api.types.is_numeric_dtype(col) and col.nunique() > 10:
        bins = pd.qcut(col, NUMERIC_BINS, duplicates="drop")
        return bins.astype(str).replace("nan", "(missing)")
    return col.astype(str).where(col.notna(), "(missing)")


def _per_segment(values: pd.Series, labels: pd.Series, agg: str) -> pd.DataFrame:
    """Per segment: the aggregate, and (for mean) the segment's share of rows with a value."""
    g = pd.DataFrame({"v": values, "seg": labels})
    if agg == "count":
        out = g.groupby("seg").size().to_frame("value")
    elif agg == "sum":
        out = g.groupby("seg")["v"].sum().to_frame("value")
    else:  # mean
        g = g.dropna(subset=["v"])
        out = g.groupby("seg")["v"].agg(value="mean", n="count")
        out["share"] = out["n"] / max(len(g), 1)
    return out


def _total(values: pd.Series, agg: str) -> float:
    if agg == "count":
        return float(len(values))
    return float(values.sum() if agg == "sum" else values.mean())


def root_cause(df: pd.DataFrame, metric: str, split_column: str, group_a: Any, group_b: Any,
               agg: str = "sum") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Break the change of `metric` from group A to group B down by every other column.

    Returns (table, summary). table has one row per (column, segment) with columns:
      column, segment, A, B, contribution, pct_of_change, expected, unexpected
    sorted by |unexpected| (the part of the change the segment's size doesn't explain).
    """
    agg = agg.lower()
    if agg not in ("sum", "mean", "count"):
        raise ValueError("agg must be 'sum', 'mean' or 'count'.")
    if split_column not in df.columns:
        raise ValueError(f"Unknown split_column '{split_column}'. Columns: {list(df.columns)}")
    if agg != "count" and metric not in df.columns:
        raise ValueError(f"Unknown metric '{metric}'. Columns: {list(df.columns)}")
    if agg != "count" and not pd.api.types.is_numeric_dtype(df[metric]):
        raise ValueError(f"Metric '{metric}' is not numeric; use agg='count' to compare row counts.")

    mask_a, mask_b = _match(df[split_column], group_a), _match(df[split_column], group_b)
    if not mask_a.any() or not mask_b.any():
        examples = df[split_column].astype(str).value_counts().head(8).index.tolist()
        raise ValueError(f"No rows for {'group_a' if not mask_a.any() else 'group_b'} in '{split_column}'. "
                         f"Example values: {examples}")

    if agg != "count":
        va, vb = df.loc[mask_a, metric].dropna(), df.loc[mask_b, metric].dropna()
        if len(va) >= 10 and len(vb) >= 10 and (va.max() < vb.min() or vb.max() < va.min()):
            raise ValueError(f"The {metric} values of the two groups don't overlap at all, so '{split_column}' "
                             f"looks like it is defined by {metric} itself (e.g. score bands). Comparing them "
                             f"explains nothing; pick groups that aren't derived from the metric.")

    values = df[metric] if agg != "count" else pd.Series(1, index=df.index)
    total_a, total_b = _total(values[mask_a], agg), _total(values[mask_b], agg)
    change = total_b - total_a

    rows = []
    for col in df.columns:
        if col in (split_column, metric) or df[col].nunique(dropna=False) < 2:
            continue
        labels = _segments(df[col])
        if labels.nunique() > MAX_SEGMENTS:
            continue
        a = _per_segment(values[mask_a], labels[mask_a], agg)
        b = _per_segment(values[mask_b], labels[mask_b], agg)
        t = a.join(b, how="outer", lsuffix="_a", rsuffix="_b").fillna(0)
        if agg == "mean":
            # Rate effect (segment mean moved) + mix effect (segment share moved, measured against
            # the average level so that shifting rows between equal segments counts as 0).
            # Adds up to the total change because the share changes sum to 0.
            avg_share = (t["share_a"] + t["share_b"]) / 2
            avg_mean = t["value_a"].where(t["share_a"] > 0, t["value_b"]) / 2 \
                + t["value_b"].where(t["share_b"] > 0, t["value_a"]) / 2
            level = (total_a + total_b) / 2
            contribution = (avg_share * (t["value_b"] - t["value_a"]).where((t["share_a"] > 0) & (t["share_b"] > 0), 0)
                            + (t["share_b"] - t["share_a"]) * (avg_mean - level))
            expected = avg_share * change  # what the segment's size alone predicts
        else:
            contribution = t["value_b"] - t["value_a"]
            expected = (t["value_a"] / total_a * change) if total_a else pd.Series(0.0, index=t.index)
        value_a, value_b = t["value_a"], t["value_b"]
        if agg == "mean":  # a segment with no rows has no mean (not a mean of 0)
            value_a, value_b = value_a.where(t["share_a"] > 0), value_b.where(t["share_b"] > 0)
        rows.append(pd.DataFrame({
            "column": col, "segment": t.index, "A": value_a.values, "B": value_b.values,
            "contribution": contribution.values, "expected": expected.values,
        }))

    table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["column", "segment", "A", "B", "contribution", "expected"])
    table["unexpected"] = table["contribution"] - table["expected"]
    table["pct_of_change"] = table["contribution"] / change * 100 if change else 0.0
    table = table.reindex(table["unexpected"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    table = table[["column", "segment", "A", "B", "contribution", "pct_of_change", "expected", "unexpected"]]

    summary = {"metric": metric, "agg": agg, "split_column": split_column,
               "group_a": str(group_a), "group_b": str(group_b),
               "rows_a": int(mask_a.sum()), "rows_b": int(mask_b.sum()),
               "total_a": total_a, "total_b": total_b, "change": change,
               "pct_change": (change / total_a * 100) if total_a else None}
    return table, summary


def _num(x: float, sign: bool = False) -> str:
    if pd.isna(x):
        return "no rows"
    fmt = ("+" if sign else "") + (",.0f" if abs(x) >= 100 else ",.3g")
    return format(x, fmt)


def format_root_cause(table: pd.DataFrame, summary: Dict[str, Any], top_columns: int = 5,
                      per_column: int = 3) -> str:
    """Short text report for the agent: total change, then the columns/segments that drive it."""
    s = summary
    label = f"{s['agg']} of {s['metric']}" if s["agg"] != "count" else "row count"
    pct = f" ({s['pct_change']:+.1f}%)" if s["pct_change"] is not None else ""
    lines = [f"{label}: {s['split_column']}={s['group_a']} (A, {s['rows_a']:,} rows) = {_num(s['total_a'])}  ->  "
             f"{s['split_column']}={s['group_b']} (B, {s['rows_b']:,} rows) = {_num(s['total_b'])}",
             f"Change: {_num(s['change'], sign=True)}{pct}"]
    if s["change"] == 0 or table.empty:
        lines.append("No change to explain." if s["change"] == 0 else "No columns to break the change down by.")
        return "\n".join(lines)

    unexpected_note = ("'unexpected' = part of the change NOT explained by the segment's size "
                       "(big positive/negative values point to the cause; near 0 = moved with everyone else).")
    lines += ["", f"Top driver columns (segments ranked by unexpected change). {unexpected_note}"]
    strength = table.groupby("column")["unexpected"].apply(lambda x: x.abs().max())
    for i, col in enumerate(strength.sort_values(ascending=False).head(top_columns).index, 1):
        lines.append(f"{i}. {col}")
        for _, r in table[table["column"] == col].head(per_column).iterrows():
            lines.append(f"   - {r['segment']}: A {_num(r['A'])} -> B {_num(r['B'])}, contribution "
                         f"{_num(r['contribution'], sign=True)} ({r['pct_of_change']:+.0f}% of change)"
                         + f", unexpected {_num(r['unexpected'], sign=True)}")
    return "\n".join(lines)
