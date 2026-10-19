"""Re-read hand-corrected ``.txt`` attachments into ``quy_che_flat.json``."""

import json
import logging
from pathlib import Path

from huflit_qa.processing.attachments import read_txt_content

logger = logging.getLogger(__name__)


def sync_records(flat_path: Path, root_dir: Path) -> int:
    """Update content/status of records whose ``.txt`` changed. Returns the number updated."""
    with open(flat_path, encoding="utf-8") as f:
        records = json.load(f)

    updated = 0
    for record in records:
        if not record.get("pdf_path") or record.get("content_source") == "image":
            continue

        new_content, new_status = read_txt_content(root_dir / record["pdf_path"])
        old_status = record["content_status"]
        if new_content != record["content"] or new_status != old_status:
            record["content"] = new_content
            record["content_status"] = new_status
            updated += 1
            logger.info("Updated %s: %s -> %s", record["id"], old_status, new_status)

    with open(flat_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return updated
