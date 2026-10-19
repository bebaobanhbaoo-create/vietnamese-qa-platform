# MỤC ĐÍCH: Cung cấp các hàm lưu và đọc dữ liệu (JSON + checkpoint).
#           Được dùng bởi main.py để lưu kết quả và resume crawl.
import json
from pathlib import Path


def save_json(data: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(article_ids: set, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_ids = sorted(article_ids)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted_ids, f)


def load_checkpoint(path: Path) -> set:
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return set(json.load(f))
