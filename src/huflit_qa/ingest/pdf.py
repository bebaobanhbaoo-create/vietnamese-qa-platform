"""Extract text from PDF attachments into sibling ``.txt`` files (PyMuPDF)."""

import logging
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def extract_pdf_text(path: Path) -> str:
    with fitz.open(path) as doc:
        pages = [page.get_text().strip() for page in doc]
    return "\n\n".join(p for p in pages if p)


def extract_missing_txt(attachments_dir: Path, overwrite: bool = False) -> int:
    """Write ``<name>.txt`` next to each PDF. Existing ``.txt`` files are kept unless ``overwrite``,
    because some of them were corrected by hand after extraction."""
    written = 0
    for pdf_path in sorted(attachments_dir.rglob("*.pdf")):
        txt_path = pdf_path.with_suffix(".txt")
        if txt_path.exists() and not overwrite:
            continue
        txt_path.write_text(extract_pdf_text(pdf_path), encoding="utf-8")
        written += 1
        logger.info("Extracted %s", txt_path.name)
    return written
