"""Bước 2: Xử lý dữ liệu thô → quy_che_flat.json + thong_bao_flat.json.

Usage:
    python offline/run_process.py
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from offline.scripts.build_records import build_flat_records
from offline.scripts.flatten_thong_bao import flatten_all

BASE_DIR = Path(__file__).parent.parent
QUY_CHE_SRC = BASE_DIR / "data/processed/by_doc_type/quy_che.json"
QUY_CHE_FLAT = BASE_DIR / "data/processed/by_doc_type/quy_che_flat.json"
QUY_CHE_ATTACHMENTS = BASE_DIR / "data/attachments/quy_ch"
THONG_BAO_SRC = BASE_DIR / "data/processed/by_doc_type/thong_bao.json"
THONG_BAO_FLAT = BASE_DIR / "data/processed/by_doc_type/thong_bao_flat.json"


def main():
    print("=== Bước 2: Xử lý dữ liệu ===")

    print("1/2 Build quy_che_flat.json...")
    records = build_flat_records(QUY_CHE_SRC, QUY_CHE_ATTACHMENTS)
    QUY_CHE_FLAT.parent.mkdir(parents=True, exist_ok=True)
    with open(QUY_CHE_FLAT, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"  {len(records)} records → {QUY_CHE_FLAT}")

    print("2/2 Flatten thong_bao...")
    tb_records = flatten_all(THONG_BAO_SRC)
    THONG_BAO_FLAT.parent.mkdir(parents=True, exist_ok=True)
    with open(THONG_BAO_FLAT, "w", encoding="utf-8") as f:
        json.dump(tb_records, f, ensure_ascii=False, indent=2)
    print(f"  {len(tb_records)} records → {THONG_BAO_FLAT}")

    print("=== Xử lý hoàn tất ===")


if __name__ == "__main__":
    main()
