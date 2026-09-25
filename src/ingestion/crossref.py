from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
import html
from pathlib import Path
import re
import time
from typing import Any

import requests

from core.config import Settings
from core.utils import normalize_whitespace, read_json, write_json


CROSSREF_WORKS_URL = "https://api.crossref.org/works"
REQUEST_TIMEOUT_SECONDS = 20
MAX_REQUEST_ATTEMPTS = 3
MAX_RETRY_DELAY_SECONDS = 5


@dataclass(frozen=True)
class PaperRecord:
    paper_id: str
    title: str
    summary: str
    authors: list[str]
    categories: list[str]
    primary_category: str
    published: str
    updated: str
    abs_url: str
    pdf_url: str
    comment: str


class _RetryableCrossrefError(RuntimeError):
    """Raised after a transient API/network error exhausts its retries."""


def _as_text(value: Any) -> str:
    """Convert a Crossref text value to a whitespace-normalized string."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return normalize_whitespace(" ".join(_as_text(item) for item in value))
    if isinstance(value, dict):
        # Crossref abstracts can occasionally be represented as a structured
        # value rather than a plain string.
        value = value.get("text") or value.get("value") or ""
    return normalize_whitespace(str(value))


def _normalize_doi(value: Any) -> str:
    doi = _as_text(value)
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
    doi = re.sub(r"^doi\s*:\s*", "", doi, flags=re.IGNORECASE)
    return doi.strip().lower()


def _strip_markup(value: Any) -> str:
    """Remove HTML/JATS tags, decode entities, and normalize the remaining text."""
    text = _as_text(value)
    # Decode first so escaped markup such as ``&lt;jats:p&gt;`` is removed too.
    # Replace tags with spaces to avoid joining words across element boundaries.
    for _ in range(2):
        text = html.unescape(text)
        text = re.sub(r"(?is)<(script|style)\b[^>]*>.*?</\1\s*>", " ", text)
        text = re.sub(r"(?s)<[^>]*>", " ", text)
    return normalize_whitespace(html.unescape(text))


def _iso_date(value: Any) -> str:
    """Return Crossref date-parts or an ISO 8601 date as ``YYYY-MM-DD``."""
    if value is None:
        return ""

    if isinstance(value, dict):
        date_parts = value.get("date-parts") or value.get("date_parts")
        if date_parts:
            value = date_parts
        else:
            for key in ("date-time", "datetime", "date", "timestamp"):
                if value.get(key) is not None:
                    value = value[key]
                    break
            else:
                return ""

    if isinstance(value, (list, tuple)):
        parts: Any = value
        # Crossref date-parts are nested: [[year, month, day]].
        if parts and isinstance(parts[0], (list, tuple)):
            parts = parts[0]
        if not parts:
            return ""
        try:
            year = int(parts[0])
            month = int(parts[1]) if len(parts) > 1 else 1
            day = int(parts[2]) if len(parts) > 2 else 1
            return date(year, month, day).isoformat()
        except (TypeError, ValueError, OverflowError):
            return ""

    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()

    candidate = _as_text(value)
    if not candidate:
        return ""
    candidate = candidate.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(candidate).date().isoformat()
    except ValueError:
        # Accept Crossref's year-only and year-month values as ISO dates with
        # the missing components set to their earliest valid values.
        match = re.fullmatch(r"(\d{4})(?:-(\d{1,2})(?:-(\d{1,2}))?)?", candidate)
        if not match:
            return ""
        year, month, day = match.groups()
        try:
            return date(int(year), int(month or 1), int(day or 1)).isoformat()
        except (TypeError, ValueError, OverflowError):
            return ""


def _first_iso_date(item: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        parsed = _iso_date(item.get(key))
        if parsed:
            return parsed
    return ""


def _authors(item: dict[str, Any]) -> list[str]:
    raw_authors = item.get("author") or item.get("authors") or []
    if isinstance(raw_authors, (str, dict)):
        raw_authors = [raw_authors]

    authors: list[str] = []
    for author in raw_authors:
        if isinstance(author, dict):
            name = _as_text(author.get("name"))
            if not name:
                name = _as_text(
                    " ".join(
                        part
                        for part in (
                            _as_text(author.get("given")),
                            _as_text(author.get("family")),
                            _as_text(author.get("suffix")),
                        )
                        if part
                    )
                )
        else:
            name = _as_text(author)
        if name:
            authors.append(name)
    return authors


def _categories(item: dict[str, Any]) -> list[str]:
    raw_categories = item.get("subject") or item.get("categories") or []
    if isinstance(raw_categories, str):
        raw_categories = [raw_categories]
    if not isinstance(raw_categories, (list, tuple)):
        return []
    return [category for value in raw_categories if (category := _as_text(value))]


def _pdf_url(item: dict[str, Any]) -> str:
    links = item.get("link") or []
    if isinstance(links, dict):
        links = [links]
    if isinstance(links, list):
        for link in links:
            if not isinstance(link, dict):
                continue
            content_type = _as_text(link.get("content-type")).lower()
            url = _as_text(link.get("URL"))
            if url and ("pdf" in content_type or "pdf" in url.lower()):
                return url

    resource = item.get("resource")
    if isinstance(resource, dict):
        primary = resource.get("primary")
        if isinstance(primary, dict):
            return _as_text(primary.get("URL"))
    return ""


def _record_from_crossref_item(item: dict[str, Any]) -> PaperRecord | None:
    paper_id = _normalize_doi(item.get("DOI") or item.get("doi") or item.get("paper_id"))
    if not paper_id:
        # A record without a stable identifier cannot safely be indexed or
        # compared during repair.
        return None

    raw_title = item.get("title") or item.get("paper_title") or ""
    if isinstance(raw_title, (list, tuple)):
        raw_title = raw_title[0] if raw_title else ""
    title = _strip_markup(raw_title)
    summary = _strip_markup(item.get("abstract") or item.get("summary") or "")
    authors = _authors(item)
    categories = _categories(item)

    published = _first_iso_date(
        item,
        ("published", "published-print", "published-online", "issued", "published-other"),
    )
    updated = _first_iso_date(item, ("updated", "indexed", "deposited", "created"))
    abs_url = _as_text(item.get("URL") or item.get("url") or item.get("abs_url"))
    if not abs_url:
        abs_url = f"https://doi.org/{paper_id}"

    raw_comment = item.get("comment") or ""
    if isinstance(raw_comment, list):
        comment = "; ".join(filter(None, (_as_text(value) for value in raw_comment)))
    else:
        comment = _as_text(raw_comment)

    return PaperRecord(
        paper_id=paper_id,
        title=title,
        summary=summary,
        authors=authors,
        categories=categories,
        primary_category=_as_text(item.get("primary_category")) or (categories[0] if categories else ""),
        published=published,
        updated=updated,
        abs_url=abs_url,
        pdf_url=_as_text(item.get("pdf_url")) or _pdf_url(item),
        comment=comment,
    )


def parse_crossref_payload(payload: dict) -> list[PaperRecord]:
    """Parse a Crossref works response into normalized ``PaperRecord`` values.

    Records without a DOI are skipped because the DOI is the stable document
    key used by cleaning, retrieval, evaluation, and repair. Missing descriptive
    fields are retained as empty strings so that downstream quality checks can
    report them instead of hiding the issue during ingestion.
    """
    if not isinstance(payload, dict):
        raise ValueError("Crossref payload must be a JSON object.")

    message = payload.get("message", payload)
    if not isinstance(message, dict):
        raise ValueError("Crossref payload is missing a valid 'message' object.")
    items = message.get("items")
    if not isinstance(items, list):
        raise ValueError("Crossref payload is missing a valid 'message.items' list.")

    records: list[PaperRecord] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        record = _record_from_crossref_item(item)
        if record is not None:
            records.append(record)
    return records


def _record_from_saved_mapping(item: dict[str, Any]) -> PaperRecord | None:
    """Restore the project's serialized ``PaperRecord`` snapshot format."""
    record = _record_from_crossref_item(item)
    if record is None:
        return None
    # Serialized records store the cleaned abstract under ``summary`` and
    # already use ``paper_id``/``authors``/``categories`` field names.
    return record


def _load_records_file(path: Path) -> list[PaperRecord]:
    payload = read_json(path)
    if isinstance(payload, dict) and "message" in payload:
        return parse_crossref_payload(payload)
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        payload = payload["records"]
    if not isinstance(payload, list):
        raise ValueError(f"Raw records file must contain a JSON list: {path}")

    records: list[PaperRecord] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        record = _record_from_saved_mapping(item)
        if record is not None:
            records.append(record)
    return records


def load_raw_records(path: Path) -> list[PaperRecord]:
    """Load either a Crossref response snapshot or saved PaperRecord list."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Raw data snapshot does not exist: {path}")
    records = _load_records_file(path)
    if not records:
        raise ValueError(f"Raw data snapshot contains no records with a DOI: {path}")
    return records


def _snapshot_records(settings: Settings) -> list[PaperRecord]:
    """Try the original Crossref response first, then parsed raw records."""
    errors: list[str] = []
    saved_records: list[PaperRecord] | None = None
    if settings.paths.raw_records_json.is_file():
        try:
            saved_records = load_raw_records(settings.paths.raw_records_json)
        except (OSError, ValueError, TypeError) as exc:
            errors.append(f"{settings.paths.raw_records_json}: {exc}")

    if settings.paths.raw_api_response.is_file():
        try:
            response_records = load_raw_records(settings.paths.raw_api_response)
            if saved_records:
                saved_by_id = {record.paper_id: record for record in saved_records}
                response_records = [
                    replace(
                        record,
                        title=record.title or saved_by_id.get(record.paper_id, record).title,
                        summary=record.summary or saved_by_id.get(record.paper_id, record).summary,
                        authors=record.authors or saved_by_id.get(record.paper_id, record).authors,
                        categories=record.categories or saved_by_id.get(record.paper_id, record).categories,
                        primary_category=(
                            record.primary_category
                            or saved_by_id.get(record.paper_id, record).primary_category
                        ),
                        published=record.published or saved_by_id.get(record.paper_id, record).published,
                        updated=record.updated or saved_by_id.get(record.paper_id, record).updated,
                        abs_url=record.abs_url or saved_by_id.get(record.paper_id, record).abs_url,
                        pdf_url=record.pdf_url or saved_by_id.get(record.paper_id, record).pdf_url,
                        comment=record.comment or saved_by_id.get(record.paper_id, record).comment,
                    )
                    for record in response_records
                ]
            return response_records
        except (OSError, ValueError, TypeError) as exc:
            errors.append(f"{settings.paths.raw_api_response}: {exc}")

    if saved_records:
        return saved_records

    detail = "; ".join(errors) if errors else "no snapshot files were found"
    raise FileNotFoundError(
        "Crossref is unavailable and no usable offline snapshot could be loaded "
        f"({detail}). Expected data/raw/crossref_response.json or crossref_records.json."
    )


def _save_records(settings: Settings, records: list[PaperRecord]) -> None:
    write_json(settings.paths.raw_records_json, [asdict(record) for record in records])


def _save_records_if_missing(settings: Settings, records: list[PaperRecord]) -> None:
    """Materialize parsed records once without overwriting an existing raw snapshot."""
    if not settings.paths.raw_records_json.exists():
        _save_records(settings, records)


def _retry_delay(response: requests.Response | None, attempt: int) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After", "").strip()
        if retry_after:
            try:
                return min(max(float(retry_after), 0.0), MAX_RETRY_DELAY_SECONDS)
            except ValueError:
                try:
                    retry_at = parsedate_to_datetime(retry_after)
                    if retry_at.tzinfo is None:
                        retry_at = retry_at.replace(tzinfo=timezone.utc)
                    wait_seconds = (retry_at - datetime.now(timezone.utc)).total_seconds()
                    return min(max(wait_seconds, 0.0), MAX_RETRY_DELAY_SECONDS)
                except (TypeError, ValueError, OverflowError):
                    pass
    return min(2**attempt, MAX_RETRY_DELAY_SECONDS)


def _request_crossref_payload(settings: Settings) -> dict[str, Any]:
    params = {
        "query": settings.source_query,
        "filter": settings.source_filter,
        "rows": settings.max_results,
    }
    headers = {
        "User-Agent": "day10-data-observability-lab/0.1 (Crossref metadata ingestion)"
    }

    last_error: Exception | None = None
    for attempt in range(MAX_REQUEST_ATTEMPTS):
        try:
            response = requests.get(
                CROSSREF_WORKS_URL,
                params=params,
                headers=headers,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            last_error = exc
            if attempt + 1 < MAX_REQUEST_ATTEMPTS:
                time.sleep(_retry_delay(None, attempt))
                continue
            break

        if response.status_code == 429 or 500 <= response.status_code <= 599:
            last_error = RuntimeError(f"Crossref returned HTTP {response.status_code}.")
            if attempt + 1 < MAX_REQUEST_ATTEMPTS:
                time.sleep(_retry_delay(response, attempt))
                continue
            break

        # Non-retryable client errors (for example an invalid filter) should be
        # visible to the caller instead of being silently treated as network loss.
        response.raise_for_status()
        try:
            payload = response.json()
        except ValueError as exc:
            raise ValueError("Crossref returned a response that is not valid JSON.") from exc
        if not isinstance(payload, dict):
            raise ValueError("Crossref returned a JSON value that is not an object.")
        return payload

    raise _RetryableCrossrefError(
        f"Crossref request failed after {MAX_REQUEST_ATTEMPTS} attempts: {last_error}"
    ) from last_error


def fetch_source_records(settings: Settings) -> list[PaperRecord]:
    """Load Crossref records offline by default, or fetch live with snapshot fallback.

    Set ``REFRESH_SOURCE=1`` to request the live Crossref API. Network errors,
    HTTP 429, and HTTP 5xx responses are retried and then fall back to the local
    raw snapshot. Other HTTP 4xx responses are raised so configuration errors
    are not hidden.
    """
    if not settings.refresh_source:
        records = _snapshot_records(settings)
        _save_records_if_missing(settings, records)
        return records

    try:
        payload = _request_crossref_payload(settings)
    except _RetryableCrossrefError as exc:
        try:
            records = _snapshot_records(settings)
        except FileNotFoundError as snapshot_error:
            raise RuntimeError(
                f"{exc}. Automatic offline fallback also failed: {snapshot_error}"
            ) from exc
        _save_records_if_missing(settings, records)
        return records

    records = parse_crossref_payload(payload)
    if not records:
        raise ValueError("Crossref response contained no records with a DOI; raw snapshot was not replaced.")

    # Save the response exactly as received and the parsed records separately.
    write_json(settings.paths.raw_api_response, payload)
    _save_records(settings, records)
    return records
