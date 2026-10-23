"""
Turns a finished AutoDataPreparer run into the JSON the demo app shows:
diagnosis, a before/after preview with a reason for every changed cell, stats,
decision cards, an automatic summary and the files to download.
"""
import math

import numpy as np
import pandas as pd

from api import explain
from api.shopify import to_shopify_csv

STEP_ORDER = ["load", "clean", "duplicates", "impute", "encode", "scale", "report"]
FILL_ACTIONS = [("Filled using", "fill_numeric"), ("Most Frequent Value (binary)", "fill_binary"),
                ("Most Frequent Value", "fill_frequent"), ("Category", "fill_missing_category")]
STANDARDIZATION_ACTIONS = {"Merged Category Variants": "merge", "Text to Number": "number", "Boolean Text to 0/1": "boolean"}


def display(value):
    """A cell as the user should read it: None for missing, dates without '00:00:00', no trailing '.0'."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d") if value == value.normalize() else value.strftime("%Y-%m-%d %H:%M")
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if math.isfinite(value) and float(value).is_integer() and abs(value) < 1e15:
            return str(int(value))
        # Two decimals reads like money and measurements; tiny values keep 4 significant digits
        return f"{value:.2f}" if abs(value) >= 1 else f"{value:.4g}"
    return str(value)


ISO_DATE = r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?"


def _logs(preparer, step=None, decision=None):
    return [d for d in preparer.logger.logs
            if (step is None or d["step"] == step) and (decision is None or d["decision"] == decision)]


def collect_facts(preparer):
    """What the pipeline did, read from its logs, in a form the explanations can use."""
    raw, clean = preparer.raw_data_snapshot, preparer.human_readable_data
    facts = {
        "rows_in": len(raw), "cols_in": raw.shape[1], "rows_out": len(clean),
        "target": preparer.target_col, "task_type": preparer.task_type if preparer.target_col else None,
        "merges": [], "numbers": [], "booleans": [], "dates": [], "capped": [],
        "identifiers": [], "duplicates": 0, "missing_target": 0, "unreadable_target": 0,
        "standardization": {},  # column -> action
        "fill": {},  # column -> action
        "metric": "R²" if preparer.task_type == "regression" else "F1",
    }

    changes_by_column = {}
    for change in preparer.logger.changes:
        changes_by_column.setdefault((change["category"], change["column"]), change)

    for d in _logs(preparer, "Standardization"):
        column, action = d["details"]["column"], STANDARDIZATION_ACTIONS[d["decision"]]
        facts["standardization"][column] = action
        cells = changes_by_column.get(("Standardization", column), {}).get("affected_rows", 0)
        if action == "merge":
            mapping = d["details"]["mapping"]
            facts["merges"].append({"column": column, "cells": cells,
                                    "examples": [[original, canonical] for original, canonical in list(mapping.items())[:4]]})
        elif action == "number":
            texts = raw[column].dropna().astype(str)
            example = next((v for v in texts if not _is_plain_number(v)), texts.iloc[0])
            facts["numbers"].append({"column": column, "cells": cells,
                                     "example": [example, display(_parsed_example(example)) or ""]})
        else:
            facts["booleans"].append({"column": column, "cells": cells, "values": d["details"]["values"]})

    for d in _logs(preparer, "Feature Engineering", "Date Parsing"):
        column = d["details"]["column"]
        example = raw[column].dropna().astype(str).iloc[0] if column in raw.columns else ""
        facts["dates"].append({"column": column, "example": example})

    for change in preparer.logger.changes:
        if change["category"] == "Outlier Capping":
            facts["capped"].append({"column": change["column"], "count": change["affected_rows"]})
        if change["category"] == "Imputation":
            for prefix, action in FILL_ACTIONS:
                if change["after"].startswith(prefix):
                    facts["fill"][change["column"]] = action
                    break

    for d in _logs(preparer, "Feature Selection", "Excluded Identifiers"):
        facts["identifiers"] = list(d["details"]["columns"])

    for info in preparer.dropped_rows.values():
        key = {"duplicate": "duplicates"}.get(info["reason"], info["reason"])
        facts[key] += 1

    def group(action):
        columns = [c for c, a in facts["fill"].items() if a == action]
        return {"columns": columns} if columns else None

    facts["binary_fill"] = group("fill_binary")
    facts["frequent_fill"] = group("fill_frequent")
    facts["missing_category"] = group("fill_missing_category")

    numeric_filled = [c for c, a in facts["fill"].items() if a == "fill_numeric"]
    if numeric_filled and "imputation" in preparer.chosen_strategies:
        facts["imputation"] = _strategy_facts(preparer, "Imputation", preparer.chosen_strategies["imputation"], len(numeric_filled))
    if "scaling" in preparer.chosen_strategies:
        scaled = sum(1 for c in preparer.logger.changes if c["category"] == "Scaling")
        facts["scaling"] = _strategy_facts(preparer, "Scaling", preparer.chosen_strategies["scaling"], scaled)

    encoding = preparer.chosen_strategies.get("encoding", {})
    facts["encoding"] = {"onehot": [c for c, s in encoding.items() if s == "onehot"],
                         "label": [c for c, s in encoding.items() if s == "label"]}
    return facts


def _is_plain_number(text):
    try:
        float(text)
        return True
    except ValueError:
        return False


def _parsed_example(text):
    from data_preparation import _parse_numeric_text
    value = _parse_numeric_text(pd.Series([text]))[0]
    return None if pd.isna(value) else float(value)


def _strategy_facts(preparer, step, strategy, columns):
    winner = next((d for d in _logs(preparer, step) if "(Winner)" in d["decision"]), None)
    scores = winner["details"].get("scores", {}) if winner else {}
    return {"strategy": strategy, "columns": columns, "evaluated": winner is not None,
            "scores": {k: (None if v is None else float(v)) for k, v in scores.items()}}


# --- Cell-level comparison -------------------------------------------------------------

def compare_cells(preparer, facts):
    """
    For every raw cell: its value before and after, a kind for colouring and a reason key.
    Returns (before_display, after_display, kinds, reasons) as DataFrames aligned to the raw table.
    """
    raw, clean = preparer.raw_data_snapshot, preparer.human_readable_data
    removed = pd.Series(False, index=raw.index)
    removed[list(preparer.dropped_rows)] = True

    before = raw.apply(lambda col: col.map(display)).astype(object)
    after = pd.DataFrame(index=raw.index, columns=raw.columns, dtype=object)
    for column in raw.columns:
        if column in clean.columns:
            after[column] = clean[column].reindex(raw.index).map(display).astype(object)

    kinds = pd.DataFrame("ok", index=raw.index, columns=raw.columns)
    reasons = pd.DataFrame(None, index=raw.index, columns=raw.columns, dtype=object)
    dates = {d["column"] for d in facts["dates"]}

    for column in raw.columns:
        b, a = before[column], after[column]
        was_missing, is_missing = b.isna(), a.isna()
        changed = ~was_missing & ~is_missing & (b != a)
        if column in dates:
            # "2010-12-02 13:39:00" was already a proper date; only other formats count as fixed
            changed &= ~b.astype(str).str.fullmatch(ISO_DATE)
            a = a.where(changed | is_missing, b)
            after[column] = a
        filled = was_missing & ~is_missing

        if column in facts["identifiers"]:
            kinds.loc[:, column] = "note"
            reasons.loc[:, column] = "identifier"
            reasons.loc[was_missing, column] = "identifier_missing"

        if column in facts["standardization"]:
            change_reason = facts["standardization"][column]
        elif column in dates:
            change_reason = "date"
        else:
            change_reason = "changed"
        kinds.loc[changed, column] = "fixed"
        reasons.loc[changed, column] = change_reason
        kinds.loc[filled, column] = "fixed"
        reasons.loc[filled, column] = facts["fill"].get(column, "fill_frequent")

    kinds.loc[removed, :] = "removed"
    for idx, info in preparer.dropped_rows.items():
        reasons.loc[idx, :] = info["reason"]
    return before, after, kinds, reasons, removed


def pick_preview_rows(kinds, reasons, removed, dropped_rows, max_rows):
    """Greedy: rows that show the most different kinds of fixes, plus a duplicate with its original."""
    changed = kinds.isin(["fixed", "removed"])
    picked = []

    duplicates = [idx for idx, info in dropped_rows.items() if info["reason"] == "duplicate"]
    if duplicates:
        copy = duplicates[0]
        picked += [dropped_rows[copy]["of"], copy]
    other_removed = [idx for idx, info in dropped_rows.items() if info["reason"] != "duplicate"]
    if other_removed:
        picked.append(other_removed[0])

    covered = {f"{c}:{reasons.at[i, c]}" for i in picked for c in kinds.columns if changed.at[i, c]}
    candidates = [i for i in kinds.index[changed.any(axis=1) & ~removed] if i not in picked]
    row_keys = {i: {f"{c}:{reasons.at[i, c]}" for c in kinds.columns[changed.loc[i]]} for i in candidates[:5000]}
    while len(picked) < max_rows and row_keys:
        best = max(row_keys, key=lambda i: (len(row_keys[i] - covered), len(row_keys[i]), -i))
        if not row_keys[best] - covered:
            break
        picked.append(best)
        covered |= row_keys.pop(best)

    # Top up with the rows that changed most, then with the first rows for context
    for i in sorted(row_keys, key=lambda i: (-len(row_keys[i]), i)) + list(kinds.index):
        if len(picked) >= max_rows:
            break
        if i not in picked:
            picked.append(i)
    return sorted(dict.fromkeys(picked))


def pick_preview_columns(kinds, max_columns):
    if kinds.shape[1] <= max_columns:
        return list(kinds.columns)
    activity = kinds.isin(["fixed", "note"]).sum().sort_values(ascending=False)
    keep = set(activity.index[:max_columns - 1]) | {kinds.columns[0]}
    return [c for c in kinds.columns if c in keep][:max_columns]


def _json_value(value):
    """pandas stores missing display values as NaN, which JSON cannot represent."""
    return None if value is None or (not isinstance(value, str) and pd.isna(value)) else str(value)


def target_options(preparer, facts, max_categories=30):
    """Columns that make sensible prediction targets: numbers, or categories with few values."""
    clean = preparer.human_readable_data
    dates = {d["column"] for d in facts["dates"]}
    options = []
    for column in preparer.raw_data_snapshot.columns:
        if column in facts["identifiers"] or column in dates or column not in clean.columns:
            continue
        series = clean[column]
        if pd.api.types.is_numeric_dtype(series) or series.nunique() <= max_categories:
            options.append(str(column))
    return options


# --- Result ------------------------------------------------------------------------------

def build_result(preparer, name, ml_csv, report_json, vendor=None, max_rows=8, max_columns=10):
    """`name` is a string or {"en": ..., "de": ...}."""
    name = name if isinstance(name, dict) else {"en": name, "de": name}
    facts = collect_facts(preparer)
    before, after, kinds, reasons, removed = compare_cells(preparer, facts)
    raw, clean = preparer.raw_data_snapshot, preparer.human_readable_data

    # Stats: a cell "needs a fix" if it was missing, changed or in a removed row
    total_cells = raw.size
    needs_fix = before.isna() | (kinds == "fixed") | (kinds == "removed")
    kept = ~removed
    remaining_missing = after.loc[kept].isna().to_numpy().sum()
    stats = {
        "quality_before": float(1 - needs_fix.to_numpy().sum() / total_cells) if total_cells else 1.0,
        "quality_after": float(1 - remaining_missing / (kept.sum() * raw.shape[1])) if kept.sum() else 1.0,
        "standardised": int(((kinds == "fixed") & before.notna()).to_numpy().sum()),
        "filled": int(((kinds == "fixed") & before.isna()).to_numpy().sum()),
        "duplicates": facts["duplicates"],
    }

    issues = []

    def add_issue(issue_id, count, columns):
        if count:
            issues.append({"id": issue_id, "count": int(count), "columns": columns, "label": explain.ISSUES[issue_id]})

    add_issue("spelling", sum(m["cells"] for m in facts["merges"]), [m["column"] for m in facts["merges"]])
    add_issue("numbers_text", sum(n["cells"] for n in facts["numbers"]), [n["column"] for n in facts["numbers"]])
    add_issue("yes_no_text", sum(b["cells"] for b in facts["booleans"]), [b["column"] for b in facts["booleans"]])
    date_fixes = {d["column"]: int((kinds[d["column"]] == "fixed").sum()) for d in facts["dates"] if d["column"] in kinds}
    for date in facts["dates"]:
        date["fixed"] = date_fixes.get(date["column"], 0)
    add_issue("dates_text", sum(date_fixes.values()), [c for c, n in date_fixes.items() if n])
    missing_columns = [c for c in raw.columns if raw[c].isna().any()]
    add_issue("missing", int(raw.isna().to_numpy().sum()), missing_columns)
    add_issue("duplicates", facts["duplicates"], [])
    add_issue("missing_target", facts["missing_target"] + facts["unreadable_target"], [facts["target"]] if facts["target"] else [])
    add_issue("outliers", sum(c["count"] for c in facts["capped"]), [c["column"] for c in facts["capped"]])
    add_issue("identifiers", len(facts["identifiers"]), facts["identifiers"])

    rows = pick_preview_rows(kinds, reasons, removed, preparer.dropped_rows, max_rows)
    columns = pick_preview_columns(kinds, max_columns)
    reason_keys = {}
    preview_rows = []
    for idx in rows:
        cells = []
        for column in columns:
            reason = reasons.at[idx, column]
            key = None
            if isinstance(reason, str):  # cells without a reason hold None/NaN
                extra = preparer.dropped_rows.get(idx, {})
                key = f"{reason}:{extra.get('of')}" if reason == "duplicate" else reason
                reason_keys.setdefault(key, explain.reason_text(reason, facts, {"of": int(extra.get("of", 0) or 0)}))
            cells.append({"before": _json_value(before.at[idx, column]),
                          "after": None if removed[idx] else _json_value(after.at[idx, column]),
                          "kind": kinds.at[idx, column], "reason": key})
        preview_rows.append({"row": int(idx) + 1, "removed": bool(removed[idx]), "cells": cells})

    return {
        "dataset": {"name": name, "rows": facts["rows_in"], "columns": facts["cols_in"],
                    "rows_out": facts["rows_out"], "columns_ml": int(preparer._with_target().shape[1]),
                    "column_names": [str(c) for c in raw.columns], "target_options": target_options(preparer, facts)},
        "target": facts["target"], "task_type": facts["task_type"],
        "steps": [{"id": step, "seconds": round(float(preparer.step_seconds.get(step, 0.0)), 4), "label": explain.STEPS[step]}
                  for step in STEP_ORDER],
        "issues": issues,
        "stats": stats,
        "stat_labels": explain.STATS,
        "preview": {"columns": columns, "rows": preview_rows, "reasons": reason_keys},
        "decisions": explain.decision_cards(facts),
        "summary": explain.summary(facts, stats, name),
        "strategies": {"imputation": preparer.chosen_strategies.get("imputation"),
                       "scaling": preparer.chosen_strategies.get("scaling")},
        "downloads": {"clean_csv": clean.to_csv(index=False), "ml_csv": ml_csv, "report_json": report_json,
                      "shopify_csv": to_shopify_csv(clean, vendor)},
    }
