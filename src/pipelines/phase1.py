from __future__ import annotations

from core.config import load_settings
from core.utils import now_utc, write_csv, write_text
from evaluation.metrics import evaluate_pipeline
from evaluation.testset import load_or_create_test_set
from ingestion.cleaning import build_clean_dataframe
from ingestion.crossref import fetch_source_records
from observability.quality import build_freshness_report, run_data_quality_checks
from observability.reporting import generate_phase1_report
from retrieval.index import LocalEmbeddingIndex


def main() -> None:
    """Run ingestion, cleaning, observability, indexing, and baseline evaluation."""
    settings = load_settings()
    run_started_at = now_utc()

    records = fetch_source_records(settings)
    if not records:
        raise RuntimeError("No source records were returned; baseline pipeline cannot continue.")

    clean_df = build_clean_dataframe(records, run_started_at)
    if clean_df.empty:
        raise RuntimeError("Cleaning produced no usable papers; baseline pipeline cannot continue.")

    write_csv(clean_df, settings.paths.clean_csv)
    write_text(
        settings.paths.clean_json,
        clean_df.to_json(orient="records", indent=2, force_ascii=False, date_format="iso") + "\n",
    )

    quality = run_data_quality_checks(clean_df, settings, "baseline")
    freshness = build_freshness_report(clean_df, settings, settings.paths.freshness_report)
    if not quality["success"]:
        failed_checks = [check["name"] for check in quality.get("checks", []) if check.get("success") is not True]
        raise RuntimeError(
            "Baseline data quality gate failed; refusing to index invalid data. "
            f"Failed checks: {', '.join(failed_checks) or 'Great Expectations setup'}"
        )

    load_or_create_test_set(
        clean_df,
        output_path=settings.paths.eval_testset,
        force_refresh=settings.refresh_test_set,
    )

    index = LocalEmbeddingIndex.build(
        clean_df,
        settings,
        embeddings_output_path=settings.paths.embeddings_json,
    )
    evaluation = evaluate_pipeline(
        settings=settings,
        index=index,
        test_set_path=settings.paths.eval_testset,
        metrics_output_path=settings.paths.baseline_metrics,
        answers_output_path=settings.paths.baseline_answers,
    )

    source_summary = {
        "source": settings.source_api,
        "query": settings.source_query,
        "filter": settings.source_filter,
        "refresh_source_requested": settings.refresh_source,
        "ingestion_mode": (
            "live API requested with snapshot fallback enabled"
            if settings.refresh_source
            else "local raw snapshot"
        ),
        "records_received": len(records),
        "records_cleaned": len(clean_df),
        "raw_response_path": settings.paths.raw_api_response.relative_to(settings.paths.project_dir).as_posix(),
        "raw_records_path": settings.paths.raw_records_json.relative_to(settings.paths.project_dir).as_posix(),
        "run_started_at": run_started_at.isoformat(),
    }
    generate_phase1_report(
        report_path=settings.paths.baseline_report,
        source_summary=source_summary,
        metrics=evaluation.summary,
        quality=quality,
        freshness=freshness,
    )

    print(f"Baseline pipeline complete: {len(clean_df)} clean papers.")
    print(f"Quality check status = {quality['success']}")
    print(f"Freshness status = {freshness['status']}")
    print(f"Retrieval hit rate = {evaluation.summary['retrieval_hit_rate']:.4f}")
    print(f"Mean token F1 = {evaluation.summary['mean_token_f1']:.4f}")
    print(f"Baseline report = {settings.paths.baseline_report}")
