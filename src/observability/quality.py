from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import great_expectations as gx
import pandas as pd

from core.config import Settings
from core.utils import now_utc, safe_slug, write_json


MINIMUM_ROW_COUNT = 5
MAXIMUM_ROW_COUNT = 5000
MINIMUM_SUMMARY_LENGTH = 30
MAXIMUM_STALE_RATIO = 0.25


def _quality_report_path(settings: Settings, report_name: str) -> Path:
    normalized_name = safe_slug(report_name or "quality")
    if normalized_name == "baseline":
        return settings.paths.baseline_quality_report
    if normalized_name == "corrupted":
        return settings.paths.corrupted_quality_report
    return settings.paths.quality_dir / f"{normalized_name}_quality_report.json"


def _validation_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Make whitespace-only values count as missing without mutating caller data."""
    validation_df = df.copy(deep=True)
    columns_to_check = ("paper_id", "title", "text_for_embedding", "summary")
    for column in columns_to_check:
        if column not in validation_df.columns:
            continue
        blank_mask = validation_df[column].map(
            lambda value: isinstance(value, str) and not value.strip()
        )
        validation_df.loc[blank_mask, column] = pd.NA
    return validation_df


def _expectations() -> list[tuple[str, Any]]:
    """Create the required GX 1.x expectations for the paper dataset."""
    expectations = gx.expectations
    checks: list[tuple[str, Any]] = [
        (
            "row_count_between_5_and_5000",
            expectations.ExpectTableRowCountToBeBetween(
                min_value=MINIMUM_ROW_COUNT,
                max_value=MAXIMUM_ROW_COUNT,
            ),
        ),
        (
            "paper_id_not_null",
            expectations.ExpectColumnValuesToNotBeNull(column="paper_id"),
        ),
        (
            "title_not_null",
            expectations.ExpectColumnValuesToNotBeNull(column="title"),
        ),
        (
            "text_for_embedding_not_null",
            expectations.ExpectColumnValuesToNotBeNull(column="text_for_embedding"),
        ),
        (
            "paper_id_unique",
            expectations.ExpectColumnValuesToBeUnique(column="paper_id"),
        ),
        (
            "summary_not_null",
            expectations.ExpectColumnValuesToNotBeNull(column="summary"),
        ),
        (
            "summary_length_at_least_30",
            expectations.ExpectColumnValueLengthsToBeBetween(
                column="summary",
                min_value=MINIMUM_SUMMARY_LENGTH,
            ),
        ),
    ]
    return checks


def _quality_setup_failure(
    *,
    report_name: str,
    row_count: int,
    error: Exception,
) -> dict[str, Any]:
    return {
        "report_name": report_name,
        "success": False,
        "row_count": row_count,
        "engine": "great_expectations",
        "context_mode": "ephemeral",
        "error": f"{type(error).__name__}: {error}",
        "generated_at": now_utc().isoformat(),
        "checks": [],
    }


def run_data_quality_checks(df: pd.DataFrame, settings: Settings, report_name: str) -> dict[str, Any]:
    """Validate a clean-papers dataframe with Great Expectations 1.x.

    The GX context is ephemeral, so validations run in memory and do not create
    a local Great Expectations project. The JSON validation report is written
    to the baseline, corrupted, or report-name-specific artifact path.
    """
    report_path = _quality_report_path(settings, report_name)
    validation_df = _validation_dataframe(df)

    try:
        context = gx.get_context(mode="ephemeral")
        data_source = context.data_sources.add_pandas(name="papers_source")
        data_asset = data_source.add_dataframe_asset(name="papers_asset")
        batch_definition = data_asset.add_batch_definition_whole_dataframe("papers_batch")
        batch = batch_definition.get_batch(batch_parameters={"dataframe": validation_df})
    except Exception as exc:
        report = _quality_setup_failure(
            report_name=report_name,
            row_count=len(df),
            error=exc,
        )
        write_json(report_path, report)
        return report

    check_results: list[dict[str, Any]] = []
    for check_name, expectation in _expectations():
        try:
            result = batch.validate(expectation)
            result_payload = result.to_json_dict()
            check_results.append(
                {
                    "name": check_name,
                    **result_payload,
                }
            )
        except Exception as exc:
            # Keep the remaining checks running so one missing column or GX
            # execution issue does not hide the other validation outcomes.
            check_results.append(
                {
                    "name": check_name,
                    "success": False,
                    "exception_info": {
                        "raised_exception": True,
                        "exception_message": f"{type(exc).__name__}: {exc}",
                    },
                }
            )

    report = {
        "report_name": report_name,
        "success": bool(check_results) and all(check.get("success") is True for check in check_results),
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "engine": "great_expectations",
        "context_mode": "ephemeral",
        "expectations_count": len(check_results),
        "successful_expectations": sum(check.get("success") is True for check in check_results),
        "unsuccessful_expectations": sum(check.get("success") is not True for check in check_results),
        "generated_at": now_utc().isoformat(),
        "checks": check_results,
    }
    write_json(report_path, report)
    return report


def build_freshness_report(
    df: pd.DataFrame,
    settings: Settings,
    report_path: str | Path | None,
) -> dict[str, Any]:
    """Report the stale-paper share and alert when it exceeds 25 percent."""
    total_rows = int(len(df))
    threshold_days = int(settings.freshness_threshold_days)

    if "age_days" in df.columns:
        age_days = pd.to_numeric(df["age_days"], errors="coerce")
        known_age_rows = int(age_days.notna().sum())
        stale_rows = int((age_days > threshold_days).fillna(False).sum())
    else:
        known_age_rows = 0
        stale_rows = 0

    unknown_age_rows = total_rows - known_age_rows
    stale_ratio = stale_rows / total_rows if total_rows else 0.0
    stale_alert = stale_ratio > MAXIMUM_STALE_RATIO
    has_age_data = total_rows > 0 and known_age_rows == total_rows
    is_fresh = has_age_data and not stale_alert

    if total_rows == 0 or known_age_rows == 0:
        status = "unknown"
    elif stale_alert:
        status = "stale"
    elif unknown_age_rows:
        status = "unknown"
    else:
        status = "fresh"

    published_dates = pd.Series(dtype="datetime64[ns, UTC]")
    if "published" in df.columns:
        published_dates = pd.to_datetime(df["published"], errors="coerce", utc=True).dropna()

    latest_published = published_dates.max().date().isoformat() if not published_dates.empty else None
    oldest_published = published_dates.min().date().isoformat() if not published_dates.empty else None
    alert_reasons: list[str] = []
    if stale_alert:
        alert_reasons.append("stale_ratio_exceeds_25_percent")
    if unknown_age_rows:
        alert_reasons.append("age_days_missing_or_invalid")
    if total_rows == 0:
        alert_reasons.append("dataset_is_empty")

    report = {
        "success": is_fresh,
        "is_fresh": is_fresh,
        "status": status,
        "alert": bool(alert_reasons),
        "alert_reasons": alert_reasons,
        "latest_published": latest_published,
        "oldest_published": oldest_published,
        "stale_rows": stale_rows,
        "unknown_age_rows": unknown_age_rows,
        "total_rows": total_rows,
        "stale_ratio": stale_ratio,
        "stale_threshold_days": threshold_days,
        "maximum_stale_ratio": MAXIMUM_STALE_RATIO,
        "generated_at": datetime.now(UTC).isoformat(),
    }

    output_path = Path(report_path) if report_path is not None else settings.paths.freshness_report
    write_json(output_path, report)
    return report
