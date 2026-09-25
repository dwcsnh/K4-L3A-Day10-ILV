from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, date, datetime, time
import html
import re
from typing import Any

import pandas as pd

from core.utils import normalize_whitespace
from ingestion.crossref import PaperRecord


CLEAN_COLUMNS = [
    "paper_id",
    "title",
    "summary",
    "authors_joined",
    "categories_joined",
    "primary_category",
    "published",
    "updated",
    "age_days",
    "summary_chars",
    "text_for_embedding",
    "abs_url",
    "pdf_url",
    "comment",
]


def _field(record: PaperRecord | Mapping[str, Any], name: str, default: Any = "") -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = " ".join(str(item) for item in value if item is not None)
    elif isinstance(value, dict):
        value = value.get("text") or value.get("value") or ""

    text = html.unescape(str(value))
    # Crossref abstracts often contain JATS/XML fragments. Replacing tags with
    # spaces preserves word boundaries, including across paragraph elements.
    for _ in range(2):
        text = html.unescape(text)
        text = re.sub(r"(?is)<(script|style)\b[^>]*>.*?</\1\s*>", " ", text)
        text = re.sub(r"(?s)<[^>]*>", " ", text)
    return normalize_whitespace(html.unescape(text))


def _paper_id(value: Any) -> str:
    identifier = normalize_whitespace(str(value or ""))
    identifier = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", identifier, flags=re.IGNORECASE)
    identifier = re.sub(r"^doi\s*:\s*", "", identifier, flags=re.IGNORECASE)
    return identifier.lower()


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _parse_date(value: Any) -> tuple[str, datetime | None]:
    """Return a normalized ISO date and its UTC midnight timestamp."""
    if value is None or value == "":
        return "", None

    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(value, time.min, tzinfo=UTC)
    else:
        candidate = normalize_whitespace(str(value)).replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            # Permit year-only and year-month strings while producing a valid
            # ISO calendar date for downstream storage.
            match = re.fullmatch(r"(\d{4})(?:-(\d{1,2})(?:-(\d{1,2}))?)?", candidate)
            if not match:
                return "", None
            year, month, day = match.groups()
            try:
                parsed = datetime(int(year), int(month or 1), int(day or 1), tzinfo=UTC)
            except (TypeError, ValueError, OverflowError):
                return "", None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    parsed = parsed.astimezone(UTC)
    return parsed.date().isoformat(), parsed


def _utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _joined_text_values(value: Any) -> str:
    parts = [_clean_text(item) for item in _as_list(value)]
    return ", ".join(part for part in parts if part)


def build_clean_dataframe(records: list[PaperRecord], run_date: datetime) -> pd.DataFrame:
    """Normalize Crossref records and prepare deterministic embedding text.

    Records without a DOI or title are excluded because they cannot serve as
    stable, searchable paper documents. The first valid occurrence of a DOI is
    retained; missing summaries remain visible for downstream quality checks.
    """
    run_timestamp = _utc_datetime(run_date)
    rows: list[dict[str, Any]] = []
    seen_paper_ids: set[str] = set()

    for record in records:
        paper_id = _paper_id(_field(record, "paper_id", _field(record, "DOI", "")))
        title = _clean_text(_field(record, "title", ""))
        if not paper_id or not title or paper_id in seen_paper_ids:
            continue

        authors_joined = _joined_text_values(_field(record, "authors", []))
        categories_joined = _joined_text_values(_field(record, "categories", []))
        summary = _clean_text(_field(record, "summary", ""))
        published, published_timestamp = _parse_date(_field(record, "published", ""))
        updated, _ = _parse_date(_field(record, "updated", ""))
        age_days = (run_timestamp - published_timestamp).days if published_timestamp else None
        primary_category = _clean_text(_field(record, "primary_category", ""))
        if not primary_category and categories_joined:
            primary_category = categories_joined.split(", ", maxsplit=1)[0]

        text_for_embedding = "\n".join(
            (
                f"Title: {title}",
                f"Authors: {authors_joined}",
                f"Published: {published}",
                f"Categories: {categories_joined}",
                f"Summary: {summary}",
            )
        )

        rows.append(
            {
                "paper_id": paper_id,
                "title": title,
                "summary": summary,
                "authors_joined": authors_joined,
                "categories_joined": categories_joined,
                "primary_category": primary_category,
                "published": published,
                "updated": updated,
                "age_days": age_days,
                "summary_chars": len(summary),
                "text_for_embedding": text_for_embedding,
                "abs_url": _clean_text(_field(record, "abs_url", "")),
                "pdf_url": _clean_text(_field(record, "pdf_url", "")),
                "comment": _clean_text(_field(record, "comment", "")),
            }
        )
        seen_paper_ids.add(paper_id)

    if not rows:
        return pd.DataFrame(columns=CLEAN_COLUMNS)

    clean_df = pd.DataFrame(rows, columns=CLEAN_COLUMNS)
    # Keep artifact order stable across reruns while placing the newest papers
    # first for easier inspection and predictable corruption scenarios.
    clean_df = clean_df.sort_values(
        by=["published", "paper_id"],
        ascending=[False, True],
        kind="mergesort",
        na_position="last",
    )
    return clean_df.reset_index(drop=True)
