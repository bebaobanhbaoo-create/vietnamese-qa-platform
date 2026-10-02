"""Build quy_che_flat.json từ quy_che.json + attachments folder."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from offline.scripts.utils import find_attachment_file, resolve_attachment_title, read_txt_content

BASE_DIR = Path(__file__).parent.parent


def build_web_record(doc):
    record = {}
    record["id"] = str(doc["id"])
    record["doc_id"] = doc["id"]
    record["doc_title"] = doc["title"]
    record["doc_url"] = doc["url"]
    record["date_posted"] = doc["date_posted"]
    record["date_updated"] = doc["date_updated"]
    record["doc_type"] = doc["doc_type"]
    record["category_name"] = doc["category_name"]
    record["issuer_unit"] = doc["issuer_unit"]
    record["attachment_index"] = None
    record["attachment_title"] = None
    record["attachment_url"] = None
    record["pdf_path"] = None
    record["content_source"] = "web"
    record["content_status"] = "ok"
    record["content"] = doc["content"]
    return record


def build_attachment_record(doc, attachment, index, attachments_base):
    att_title = resolve_attachment_title(attachment["text"], index, doc["content"])
    file_path, file_type = find_attachment_file(attachments_base, doc["id"], index)

    if file_type == "image":
        content_source = "image"
        content_status = "image"
        content = None
        pdf_path = file_path.relative_to(BASE_DIR).as_posix() if file_path else None
    elif file_type == "txt":
        content_source = "pdf"
        content, content_status = read_txt_content(file_path)
        pdf_path = file_path.relative_to(BASE_DIR).as_posix() if file_path else None
    else:
        content_source = "pdf"
        content_status = "empty"
        content = None
        pdf_path = None

    record = {}
    record["id"] = f"{doc['id']}_{index}"
    record["doc_id"] = doc["id"]
    record["doc_title"] = doc["title"]
    record["doc_url"] = doc["url"]
    record["date_posted"] = doc["date_posted"]
    record["date_updated"] = doc["date_updated"]
    record["doc_type"] = doc["doc_type"]
    record["category_name"] = doc["category_name"]
    record["issuer_unit"] = doc["issuer_unit"]
    record["attachment_index"] = index
    record["attachment_title"] = att_title
    record["attachment_url"] = attachment["url"]
    record["pdf_path"] = pdf_path
    record["content_source"] = content_source
    record["content_status"] = content_status
    record["content"] = content
    return record


def build_flat_records(source, attachments_base):
    with open(source, encoding="utf-8") as f:
        docs = json.load(f)

    records = []
    for doc in docs:
        attachments = doc.get("attachments", [])
        if not attachments:
            records.append(build_web_record(doc))
        else:
            for i, attachment in enumerate(attachments, start=1):
                records.append(build_attachment_record(doc, attachment, i, attachments_base))
    return records
