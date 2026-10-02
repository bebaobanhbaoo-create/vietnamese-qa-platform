#!/usr/bin/env python3
"""Normalize downloaded attachment extensions from .bin to real types."""

import argparse
from pathlib import Path


def detect_kind(file_path: Path) -> str:
    data = file_path.read_bytes()[:512]
    trimmed = data.lstrip()
    if data.startswith(b"%PDF-"):
        return "pdf"
    if trimmed.startswith(b"<!DOCTYPE html") or trimmed.startswith(b"<html") or trimmed.startswith(b"<!--"):
        return "html"
    if data.startswith(b"\xFF\xD8\xFF"):
        return "jpg"
    if data.startswith(b"PK\x03\x04"):
        return "zip"
    if data.startswith(b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"):
        return "doc"
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize .bin attachments to proper extension")
    parser.add_argument(
        "--attachments-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "attachments",
    )
    parser.add_argument(
        "--delete-html",
        action="store_true",
        help="Delete .bin files that are actually HTML error pages",
    )
    args = parser.parse_args()

    converted = 0
    deleted = 0
    unknown = 0

    for file_path in args.attachments_dir.rglob("*.bin"):
        kind = detect_kind(file_path)
        if kind == "unknown":
            unknown += 1
            continue
        if kind == "html":
            if args.delete_html:
                file_path.unlink(missing_ok=True)
                deleted += 1
            continue
        new_path = file_path.with_suffix(f".{kind}")
        if new_path.exists():
            new_path.unlink()
        file_path.rename(new_path)
        converted += 1

    print(
        f"converted={converted} deleted_html={deleted} unknown_bin={unknown} "
        f"attachments_dir={args.attachments_dir}"
    )


if __name__ == "__main__":
    main()
