from __future__ import annotations

from pathlib import Path
from typing import Any

from core.utils import write_text


def _markdown_cell(value: Any) -> str:
    if value is None:
        return "—"
    return str(value).replace("|", "\\|").replace("\n", " ").strip() or "—"


def _metric_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:.4f}" if isinstance(value, float) else str(value)
    return _markdown_cell(value)


def generate_phase1_report(
    report_path,
    source_summary: dict[str, Any],
    metrics: dict[str, Any],
    quality: dict[str, Any],
    freshness: dict[str, Any],
) -> None:
    """Write a baseline report grounded in the generated pipeline artifacts."""
    metric_rows = [
        ("Evaluation samples", metrics.get("samples")),
        ("Retrieval hit rate", metrics.get("retrieval_hit_rate")),
        ("Mean token F1", metrics.get("mean_token_f1")),
        ("Judge accuracy", metrics.get("judge_accuracy")),
        ("Mean judge score", metrics.get("mean_judge_score")),
    ]
    metric_table = "\n".join(
        f"| {_markdown_cell(name)} | {_metric_value(value)} |" for name, value in metric_rows
    )

    check_rows = quality.get("checks", [])
    check_table = "\n".join(
        f"| {_markdown_cell(check.get('name'))} | {_markdown_cell(check.get('success'))} |"
        for check in check_rows
    ) or "| No checks returned | false |"

    ragas = metrics.get("ragas")
    if isinstance(ragas, dict):
        ragas_summary = "; ".join(f"{key}: {value}" for key, value in ragas.items())
    else:
        ragas_summary = _markdown_cell(ragas)

    report = f"""# Phase 1 — Baseline Pipeline Report

## Run summary

| Property | Value |
| --- | --- |
| Source | {_markdown_cell(source_summary.get('source'))} |
| Ingestion mode | {_markdown_cell(source_summary.get('ingestion_mode'))} |
| Live refresh requested | {_markdown_cell(source_summary.get('refresh_source_requested'))} |
| Query | {_markdown_cell(source_summary.get('query'))} |
| Filter | {_markdown_cell(source_summary.get('filter'))} |
| Records received | {_markdown_cell(source_summary.get('records_received'))} |
| Records after cleaning | {_markdown_cell(source_summary.get('records_cleaned'))} |
| Run started at (UTC) | {_markdown_cell(source_summary.get('run_started_at'))} |
| Raw response | {_markdown_cell(source_summary.get('raw_response_path'))} |
| Parsed raw records | {_markdown_cell(source_summary.get('raw_records_path'))} |

## Baseline evaluation

| Metric | Value |
| --- | ---: |
{metric_table}

Ragas: {_markdown_cell(ragas_summary)}

## Data quality

- Great Expectations status: **{_markdown_cell(quality.get('success'))}**
- Rows checked: {_markdown_cell(quality.get('row_count'))}
- Expectations passed: {_markdown_cell(quality.get('successful_expectations'))}/{_markdown_cell(quality.get('expectations_count'))}

| Check | Passed |
| --- | --- |
{check_table}

## Freshness

| Signal | Value |
| --- | --- |
| Status | {_markdown_cell(freshness.get('status'))} |
| Fresh | {_markdown_cell(freshness.get('is_fresh'))} |
| Stale rows | {_markdown_cell(freshness.get('stale_rows'))}/{_markdown_cell(freshness.get('total_rows'))} |
| Stale ratio | {_metric_value(freshness.get('stale_ratio'))} |
| Stale threshold | {_markdown_cell(freshness.get('stale_threshold_days'))} days |
| Maximum stale ratio | {_metric_value(freshness.get('maximum_stale_ratio'))} |
| Latest publication date | {_markdown_cell(freshness.get('latest_published'))} |
| Oldest publication date | {_markdown_cell(freshness.get('oldest_published'))} |
| Alert reasons | {_markdown_cell(', '.join(freshness.get('alert_reasons', [])))} |

## Interpretation

This report records the measured baseline for the current clean dataset. Compare later corruption and repair runs using the same evaluation set. Conclusions about degradation or recovery must be based on the generated metric and quality artifacts.
"""
    write_text(Path(report_path), report)


def generate_corruption_report(
    report_path,
    baseline_metrics: dict[str, Any],
    corrupted_metrics: dict[str, Any],
    repaired_metrics: dict[str, Any],
    corrupted_quality: dict[str, Any],
    repaired_quality: dict[str, Any],
    corrupted_freshness: dict[str, Any],
    repaired_freshness: dict[str, Any],
) -> None:
    """TODO(student): viet markdown report so sanh baseline/corrupted/repaired."""
    raise NotImplementedError("Student task: implement corruption comparison report.")
