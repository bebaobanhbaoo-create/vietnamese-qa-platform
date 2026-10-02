"""Update content trong quy_che_flat.json từ các file .txt đã chỉnh sửa thủ công.

Usage:
    python scripts/sync_content.py
    python scripts/sync_content.py --flat data/processed/by_doc_type/quy_che_flat.json
"""
import json
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from offline.scripts.utils import read_txt_content

BASE_DIR = Path(__file__).parent.parent
DEFAULT_FLAT = BASE_DIR / "data/processed/by_doc_type/quy_che_flat.json"


def sync_records(flat_path: Path) -> None:
    with open(flat_path, encoding="utf-8") as f:
        records = json.load(f)

    updated = 0
    for record in records:
        if not record.get("pdf_path") or record.get("content_source") == "image":
            continue

        txt_path = BASE_DIR / record["pdf_path"]
        new_content, new_status = read_txt_content(txt_path)

        old_status = record["content_status"]
        if new_content != record["content"] or new_status != old_status:
            record["content"] = new_content
            record["content_status"] = new_status
            updated += 1
            print(f"  Updated {record['id']}: {old_status} → {new_status}")

    with open(flat_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Sync complete: {updated} updated")


def main():
    parser = argparse.ArgumentParser(description="Sync .txt content into quy_che_flat.json")
    parser.add_argument("--flat", type=Path, default=DEFAULT_FLAT)
    args = parser.parse_args()

    if not args.flat.exists():
        print(f"Error: {args.flat} not found. Run build_records.py first.")
        raise SystemExit(1)

    print(f"Syncing content from .txt files into {args.flat}")
    sync_records(args.flat)


if __name__ == "__main__":
    main()
