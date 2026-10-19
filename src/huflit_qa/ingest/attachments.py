"""Rename downloaded ``.bin`` attachments to their real type based on magic bytes."""

from dataclasses import dataclass
from pathlib import Path


def detect_kind(file_path: Path) -> str:
    data = file_path.read_bytes()[:512]
    trimmed = data.lstrip()
    if data.startswith(b"%PDF-"):
        return "pdf"
    if trimmed.startswith(b"<!DOCTYPE html") or trimmed.startswith(b"<html") or trimmed.startswith(b"<!--"):
        return "html"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if data.startswith(b"PK\x03\x04"):
        return "zip"
    if data.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        return "doc"
    return "unknown"


@dataclass
class NormalizeStats:
    converted: int = 0
    deleted_html: int = 0
    unknown: int = 0


def normalize_attachments(attachments_dir: Path, delete_html: bool = False) -> NormalizeStats:
    """Rename ``*.bin`` files; optionally delete ones that are actually HTML error pages."""
    stats = NormalizeStats()
    for file_path in attachments_dir.rglob("*.bin"):
        kind = detect_kind(file_path)
        if kind == "unknown":
            stats.unknown += 1
            continue
        if kind == "html":
            if delete_html:
                file_path.unlink(missing_ok=True)
                stats.deleted_html += 1
            continue
        new_path = file_path.with_suffix(f".{kind}")
        if new_path.exists():
            new_path.unlink()
        file_path.rename(new_path)
        stats.converted += 1
    return stats
