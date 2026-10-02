"""Flatten thong_bao.json thành thong_bao_flat.json cho IR pipeline."""
import json
from pathlib import Path


PLACEHOLDER_TEXTS = {"vui lòng xem tại đây", "bấm vào để xem", "click here"}


def is_placeholder(content):
    if not content:
        return True
    if not content.strip():
        return True
    if content.strip().lower() in PLACEHOLDER_TEXTS:
        return True
    return False


def flatten_record(doc):
    content = doc.get("content")
    if is_placeholder(content):
        content = None
    else:
        content = content.strip()

    if content is None:
        status = "empty"
    else:
        status = "ok"

    return {
        "id": str(doc["id"]),
        "doc_title": doc["title"],
        "doc_url": doc["url"],
        "date_posted": doc.get("date_posted"),
        "doc_type": doc["doc_type"],
        "content_source": "web",
        "content_status": status,
        "content": content,
    }


def flatten_all(source):
    with open(source, encoding="utf-8") as f:
        docs = json.load(f)
    records = []
    for doc in docs:
        records.append(flatten_record(doc))
    return records
