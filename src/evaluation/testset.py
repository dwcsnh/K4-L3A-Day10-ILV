from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from core.utils import first_sentence, read_json, write_json


EXPECTED_QUESTION_TYPE_COUNTS = {
    "summary": 3,
    "authors": 3,
    "date": 2,
    "categories": 2,
}


class TestSet(list):
    """Container for benchmark samples, compatible with JSON list storage."""

    def __init__(self, samples: list[dict[str, Any]] | None = None):
        super().__init__(samples or [])

    @property
    def samples(self) -> list[dict[str, Any]]:
        return list(self)


def _generate_samples(df: pd.DataFrame) -> list[dict[str, Any]]:
    if len(df) < 10:
        raise ValueError(f"Clean DataFrame must have at least 10 documents to build test set, got {len(df)}.")

    rows = df.iloc[:10].to_dict(orient="records")
    question_plan = [
        ("summary", 0),
        ("authors", 1),
        ("date", 2),
        ("categories", 3),
        ("summary", 4),
        ("authors", 5),
        ("date", 6),
        ("categories", 7),
        ("summary", 8),
        ("authors", 9),
    ]

    samples: list[dict[str, Any]] = []
    for index, (question_type, row_index) in enumerate(question_plan, start=1):
        row = rows[row_index]
        title = str(row["title"])
        sample: dict[str, Any] = {
            "id": f"eval_{index:03d}",
            "question_type": question_type,
            "ground_truth_doc_ids": [str(row["paper_id"])],
        }

        if question_type == "summary":
            sample["question"] = f"What is the summary of the paper '{title}'?"
            sample["ground_truth"] = first_sentence(str(row["summary"]))
        elif question_type == "authors":
            sample["question"] = f"Who authored the paper '{title}'?"
            sample["ground_truth"] = str(row["authors_joined"])
        elif question_type == "date":
            sample["question"] = f"When was the paper '{title}' published?"
            sample["ground_truth"] = str(row["published"])
        else:
            sample["question"] = f"What categories does the paper '{title}' belong to?"
            sample["ground_truth"] = str(row["categories_joined"])

        samples.append(sample)

    return samples


def _is_valid_test_set(samples: Any) -> bool:
    if not isinstance(samples, list) or len(samples) != 10:
        return False

    counts = {question_type: 0 for question_type in EXPECTED_QUESTION_TYPE_COUNTS}
    required_fields = {"id", "question_type", "question", "ground_truth", "ground_truth_doc_ids"}
    for sample in samples:
        if not isinstance(sample, dict) or not required_fields.issubset(sample):
            return False
        question_type = sample["question_type"]
        if question_type not in counts or not sample["ground_truth_doc_ids"]:
            return False
        counts[question_type] += 1

    return counts == EXPECTED_QUESTION_TYPE_COUNTS


def load_or_create_test_set(
    df: pd.DataFrame,
    output_path: Path | str | None = None,
    force_refresh: bool = False,
) -> TestSet:
    """Reuse a valid fixed benchmark or create and persist the required 10 samples."""
    target_path = Path(output_path) if output_path else None

    if target_path and target_path.exists() and not force_refresh:
        raw_data = read_json(target_path)
        samples = raw_data.get("samples") if isinstance(raw_data, dict) else raw_data
        if _is_valid_test_set(samples):
            return TestSet(samples)

    test_set = TestSet(_generate_samples(df))
    if target_path:
        write_json(target_path, list(test_set))
    return test_set


def build_test_set(df: pd.DataFrame, output_path: Path | str | None = None) -> TestSet:
    """Build and save a fresh benchmark set."""
    return load_or_create_test_set(df, output_path, force_refresh=True)
