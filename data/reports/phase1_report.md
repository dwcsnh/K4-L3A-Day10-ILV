# Phase 1 — Baseline Pipeline Report

## Run summary

| Property | Value |
| --- | --- |
| Source | Crossref REST API |
| Ingestion mode | local raw snapshot |
| Live refresh requested | False |
| Query | agentic retrieval augmented generation large language model |
| Filter | from-pub-date:2026-03-29,has-abstract:true |
| Records received | 24 |
| Records after cleaning | 24 |
| Run started at (UTC) | 2026-09-25T09:19:14.567640+00:00 |
| Raw response | data/raw/crossref_response.json |
| Parsed raw records | data/raw/crossref_records.json |

## Baseline evaluation

| Metric | Value |
| --- | ---: |
| Evaluation samples | 10 |
| Retrieval hit rate | 1.0000 |
| Mean token F1 | 1.0000 |
| Judge accuracy | 1.0000 |
| Mean judge score | 5 |

Ragas: skipped: Set RUN_RAGAS=1 to enable the slower Ragas pass.

## Data quality

- Great Expectations status: **True**
- Rows checked: 24
- Expectations passed: 7/7

| Check | Passed |
| --- | --- |
| row_count_between_5_and_5000 | True |
| paper_id_not_null | True |
| title_not_null | True |
| text_for_embedding_not_null | True |
| paper_id_unique | True |
| summary_not_null | True |
| summary_length_at_least_30 | True |

## Freshness

| Signal | Value |
| --- | --- |
| Status | fresh |
| Fresh | True |
| Stale rows | 1/24 |
| Stale ratio | 0.0417 |
| Stale threshold | 180 days |
| Maximum stale ratio | 0.2500 |
| Latest publication date | 2026-07-22 |
| Oldest publication date | 2026-03-28 |
| Alert reasons | — |

## Interpretation

This report records the measured baseline for the current clean dataset. Compare later corruption and repair runs using the same evaluation set. Conclusions about degradation or recovery must be based on the generated metric and quality artifacts.
