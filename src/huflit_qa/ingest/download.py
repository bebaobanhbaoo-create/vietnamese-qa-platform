"""Download regulation attachments (PDF and scans) linked from portal articles.

Regulations are mostly published as SharePoint or Google Drive share links, not files on the
portal. Files land where the processing step looks for them:
``data/attachments/quy_ch/{doc_id}_{title}/{doc_id}_{n}__{slug}.{ext}``, with ``n`` the 1-based
position of the link in the article (the same numbering as the flat records).
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import requests

from huflit_qa.ingest.attachments import detect_kind
from huflit_qa.ingest.classify import slugify
from huflit_qa.ingest.sources import HEADERS, REQUEST_TIMEOUT

logger = logging.getLogger(__name__)

DOWNLOADABLE = {"sharepoint", "google_drive"}
RE_DRIVE_ID = re.compile(r"/file/d/([A-Za-z0-9_-]+)")
# Characters Windows does not allow in folder names.
RE_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def download_url(url: str, link_type: str) -> str | None:
    """Direct-download URL for a share link, or None if it is not a file we can fetch."""
    if link_type == "google_drive":
        m = RE_DRIVE_ID.search(url)
        return f"https://drive.google.com/uc?export=download&id={m.group(1)}" if m else None
    if link_type == "sharepoint":
        parts = urlparse(url)
        query = {k: v[0] for k, v in parse_qs(parts.query).items()}
        query["download"] = "1"
        return urlunparse(parts._replace(query=urlencode(query)))
    if link_type.startswith("file_"):
        return url
    return None


def folder_name(doc_id: int, title: str) -> str:
    """Matches the existing layout: id + title with spaces as underscores (diacritics kept)."""
    name = RE_UNSAFE.sub("", re.sub(r"\s+", "_", title.strip()))
    return f"{doc_id}_{name}"[:150].rstrip("._")


def has_file(attachments_base: Path, doc_id: int, index: int) -> bool:
    return any(attachments_base.glob(f"{doc_id}_*/{doc_id}_{index}__*"))


@dataclass
class DownloadStats:
    downloaded: list[str] = field(default_factory=list)
    skipped_existing: int = 0
    unsupported: int = 0
    failed: list[dict] = field(default_factory=list)


def download_attachments(
    quy_che_path: Path, attachments_base: Path, session: requests.Session | None = None
) -> DownloadStats:
    """Fetch every downloadable attachment of the regulation articles that is not on disk yet."""
    session = session or requests.Session()
    session.headers.update(HEADERS)
    docs = json.loads(quy_che_path.read_text(encoding="utf-8"))
    stats = DownloadStats()
    for doc in docs:
        for index, att in enumerate(doc.get("attachments", []), start=1):
            if has_file(attachments_base, doc["id"], index):
                stats.skipped_existing += 1
                continue
            url = download_url(att["url"], att.get("type", ""))
            if url is None or not (
                att.get("type") in DOWNLOADABLE or att.get("type", "").startswith("file_")
            ):
                stats.unsupported += 1
                continue
            name = f"{doc['id']}_{index}__{slugify(att.get('text') or 'tai_lieu')[:60]}"
            target_dir = attachments_base / folder_name(doc["id"], doc["title"])
            try:
                resp = session.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
                resp.raise_for_status()
            except requests.RequestException as exc:
                stats.failed.append({"doc_id": doc["id"], "index": index, "reason": str(exc)[:200]})
                continue
            target_dir.mkdir(parents=True, exist_ok=True)
            tmp = target_dir / f"{name}.bin"
            tmp.write_bytes(resp.content)
            kind = detect_kind(tmp)
            if kind in ("html", "unknown"):
                # A login or preview page instead of the file: the link is not public.
                tmp.unlink()
                stats.failed.append(
                    {"doc_id": doc["id"], "index": index, "reason": f"got {kind}, not a file"}
                )
                continue
            final = tmp.with_suffix(f".{kind}")
            tmp.replace(final)
            stats.downloaded.append(final.name)
            logger.info("Downloaded %s", final.name)
    return stats
