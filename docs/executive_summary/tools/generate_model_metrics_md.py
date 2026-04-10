#!/usr/bin/env python3
"""
Generate a markdown table of saved model metrics for the executive summary site.

Reads:  models/meta_*.json
Writes: docs/executive_summary/_includes/model_metrics.md
"""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import Any


def _parse_timestamp(ts: str) -> dt.datetime | None:
    try:
        return dt.datetime.strptime(ts, "%Y%m%d_%H%M%S")
    except Exception:
        return None


def _fmt_float(value: Any, ndigits: int = 4) -> str:
    if value is None:
        return "N/A"
    try:
        f = float(value)
    except Exception:
        return "N/A"
    if math.isnan(f) or math.isinf(f):
        return "N/A"
    return f"{f:.{ndigits}f}"


def _fmt_fraction(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        f = float(value)
    except Exception:
        return "N/A"
    if math.isnan(f) or math.isinf(f):
        return "N/A"
    # Keep as a fraction because that's what the training script accepts.
    return f"{f:.3f}".rstrip("0").rstrip(".")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]
    meta_paths = sorted((repo_root / "models").glob("meta_*.json"))

    rows: list[dict[str, Any]] = []
    for path in meta_paths:
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        ts_str = str(meta.get("timestamp") or "")
        ts = _parse_timestamp(ts_str)

        train_metrics = meta.get("train_metrics") or {}
        classes = meta.get("classes") or []
        details = meta.get("details") or {}

        rows.append(
            {
                "timestamp": ts_str,
                "timestamp_dt": ts or dt.datetime.min,
                "api": meta.get("api"),
                "target": meta.get("target_column"),
                "sample_fraction": meta.get("sample_fraction"),
                "threads": meta.get("train_threads"),
                "cv_primary": meta.get("metric_primary_cv"),
                "train_f1_macro": train_metrics.get("train_f1_macro"),
                "train_accuracy": train_metrics.get("train_accuracy"),
                "n_classes": len(classes) if isinstance(classes, list) else None,
                "best_params": details.get("best_params"),
                "meta_file": path.name,
            }
        )

    rows.sort(key=lambda r: r["timestamp_dt"], reverse=True)

    # Identify a "best" run by CV score (ignoring missing/invalid values).
    def _cv_score(row: dict[str, Any]) -> float | None:
        try:
            f = float(row.get("cv_primary"))
        except Exception:
            return None
        if math.isnan(f) or math.isinf(f):
            return None
        return f

    best = None
    for r in rows:
        s = _cv_score(r)
        if s is None:
            continue
        if best is None or s > best[0]:
            best = (s, r)

    out_path = repo_root / "docs" / "executive_summary" / "_includes" / "model_metrics.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    generated_at = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    lines: list[str] = []
    lines.append("<!-- AUTO-GENERATED: docs/executive_summary/tools/generate_model_metrics_md.py -->")
    lines.append(f"<!-- Generated: {generated_at} -->")
    lines.append("")

    if not rows:
        lines.append("_No saved model metadata found in `models/meta_*.json`._")
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return 0

    if best is not None:
        score, r = best
        lines.append(
            f"**Best CV macro-F1:** `{_fmt_float(score)}`"
            f" (`{r.get('api')}` / `{r.get('target')}` at `{r.get('timestamp')}`)"
        )
        lines.append("")

    lines.append("| timestamp | api | target | sample_fraction | threads | cv_macro_f1 | train_macro_f1 | train_accuracy | n_classes | meta_file |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---|")

    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{r.get('timestamp')}`" if r.get("timestamp") else "N/A",
                    f"`{r.get('api')}`" if r.get("api") else "N/A",
                    f"`{r.get('target')}`" if r.get("target") else "N/A",
                    _fmt_fraction(r.get("sample_fraction")),
                    str(r.get("threads") if r.get("threads") is not None else "N/A"),
                    _fmt_float(r.get("cv_primary")),
                    _fmt_float(r.get("train_f1_macro")),
                    _fmt_float(r.get("train_accuracy")),
                    str(r.get("n_classes") if r.get("n_classes") is not None else "N/A"),
                    f"`{r.get('meta_file')}`",
                ]
            )
            + " |"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

