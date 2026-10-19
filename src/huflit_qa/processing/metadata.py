"""Document metadata used for filtering and for telling regulation versions apart."""

import re

# "ngày 15 tháng 9 năm 2022" in a regulation's header ("Ban hành kèm theo Quyết định ... ngày ...").
RE_ISSUED = re.compile(r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})", re.IGNORECASE)
HEADER_CHARS = 3000


def parse_issued_date(content: str | None) -> str | None:
    """First 'ngày D tháng M năm YYYY' in the header, as YYYY-MM-DD (None if absent or invalid)."""
    if not content:
        return None
    m = RE_ISSUED.search(content[:HEADER_CHARS])
    if not m:
        return None
    day, month, year = (int(g) for g in m.groups())
    if not (1 <= month <= 12 and 1 <= day <= 31 and 1990 <= year <= 2100):
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"


def record_metadata(record: dict) -> dict:
    """Filterable fields copied onto every chunk of a record."""
    is_regulation = record["doc_type"] == "Quy chế"
    return {
        "doc_group": doc_group(record["id"]),
        "issuer_unit": record.get("issuer_unit"),
        "issued_date": parse_issued_date(record.get("content")) if is_regulation else None,
    }


def doc_group(record_id: str) -> str:
    """Attachments of one portal article ("3209_1", "3209_7") share a group ("3209")."""
    return str(record_id).split("_")[0]


def mark_newest_in_group(chunks: list[dict]) -> None:
    """Set ``newest_in_group`` on regulation chunks: True for the record with the latest issue
    date in its group. A group's records are often successive versions of one rule (e.g. language
    requirements 2017 -> 2024 -> 2025), but some are parallel rules for different cohorts, so this
    is a hint for ranking and answer generation, not a validity flag."""
    newest: dict[str, str] = {}
    for c in chunks:
        if c["doc_type"] == "quy_che" and c.get("issued_date"):
            group = c["doc_group"]
            newest[group] = max(newest.get(group, ""), c["issued_date"])
    for c in chunks:
        if c["doc_type"] == "quy_che":
            group_newest = newest.get(c["doc_group"])
            c["newest_in_group"] = bool(group_newest) and c.get("issued_date") == group_newest
