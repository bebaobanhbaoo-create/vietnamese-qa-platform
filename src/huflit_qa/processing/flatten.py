"""Flatten thong_bao.json thành thong_bao_flat.json cho IR pipeline."""

import json

PLACEHOLDER_TEXTS = {"vui lòng xem tại đây", "bấm vào để xem", "click here"}


def is_placeholder(content):
    if not content:
        return True
    if not content.strip():
        return True
    return content.strip().lower() in PLACEHOLDER_TEXTS


def flatten_record(doc):
    content = doc.get("content")
    content = None if is_placeholder(content) else content.strip()

    status = "empty" if content is None else "ok"

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
