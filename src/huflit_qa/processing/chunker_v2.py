"""Chunker v2: structure-aware packing under a word budget.

Differences from v1:
- Sizes are in words, so the budget maps to the embedding model's token limit
  (v1 counted characters and let some chunks grow to 19K characters).
- Oversized parts fall back to paragraphs, then sentences, then a hard word split: a chunk never
  ends mid-sentence unless a single sentence exceeds the budget.
- Small consecutive parts (Khoản, paragraphs) are packed together instead of one chunk each.
- Khoản numbers ("1.", "2.") are kept; v1 dropped them when splitting.
- Every chunk of an Điều carries its heading, plus the Chương it belongs to, as ``context``.
"""

import re

from huflit_qa.config import ChunkingSettings
from huflit_qa.processing.chunker import DOC_TYPE_MAP
from huflit_qa.processing.metadata import record_metadata

RE_DIEU = re.compile(r"^Điều\s+\d+", re.MULTILINE)
RE_CHUONG_LINE = re.compile(r"^\s*Chương\s+[IVXLC\d]+\b.*$", re.IGNORECASE)
RE_KHOAN_START = re.compile(r"(?m)^(?=\s*\d+\.\s)")
RE_SENTENCE_END = re.compile(r"(?<=[.;:!?])\s+")


def n_words(text: str) -> int:
    return len(text.split())


def _split_to_fit(text: str, max_words: int) -> list[str]:
    """Split text into parts of at most ``max_words``, preferring Khoản, lines, then sentences."""
    if n_words(text) <= max_words:
        return [text]
    for splitter in (RE_KHOAN_START, re.compile(r"\n+"), RE_SENTENCE_END):
        parts = [p.strip() for p in splitter.split(text) if p.strip()]
        if len(parts) > 1:
            out: list[str] = []
            for p in parts:
                out.extend(_split_to_fit(p, max_words))
            return out
    words = text.split()
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def _pack(parts: list[str], max_words: int, min_words: int, sep: str = "\n") -> list[str]:
    """Greedily join consecutive parts up to ``max_words``; fold a short tail into its neighbour."""
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for part in parts:
        w = n_words(part)
        if current and size + w > max_words:
            chunks.append(sep.join(current))
            current, size = [], 0
        current.append(part)
        size += w
    if current:
        tail = sep.join(current)
        if chunks and size < min_words and n_words(chunks[-1]) + size <= max_words * 1.25:
            chunks[-1] = chunks[-1] + sep + tail
        else:
            chunks.append(tail)
    return chunks


def _sections(text: str) -> list[tuple[str | None, str | None, str]]:
    """Split a regulation into (chương, điều heading, body) sections; text before Điều 1 is a preamble."""
    sections: list[tuple[str | None, str | None, str]] = []
    chuong: str | None = None
    heading: str | None = None
    lines: list[str] = []

    def flush() -> None:
        body = "\n".join(lines).strip()
        if body:
            sections.append((chuong, heading, body))

    raw_lines = text.split("\n")
    i = 0
    while i < len(raw_lines):
        line = raw_lines[i]
        if RE_CHUONG_LINE.match(line):
            flush()
            lines, heading = [], None
            # "Chương II" is usually followed by its name on the next line.
            name = raw_lines[i + 1].strip() if i + 1 < len(raw_lines) else ""
            chuong = f"{line.strip()} {name}".strip() if name and not RE_DIEU.match(name) else line.strip()
            i += 2 if name and not RE_DIEU.match(name) else 1
            continue
        if RE_DIEU.match(line):
            flush()
            lines, heading = [line], line.strip()
        else:
            lines.append(line)
        i += 1
    flush()
    return sections


def _make_chunk(record: dict, index: int, text: str, title: str | None, context: str) -> dict:
    return {
        "chunk_id": f"{record['id']}_c{index}",
        "doc_id": record["id"],
        "doc_title": record["doc_title"],
        "chunk_title": title,
        "doc_type": DOC_TYPE_MAP.get(record["doc_type"], record["doc_type"]),
        "source_url": record["doc_url"],
        "date_posted": record.get("date_posted"),
        **record_metadata(record),
        "position": index,
        "context": context,
        "text": text,
    }


def chunk_quy_che(record: dict, cfg: ChunkingSettings) -> list[dict]:
    content = (record.get("content") or "").strip()
    if not content:
        return []
    if not RE_DIEU.search(content):
        return chunk_thong_bao(record, cfg)

    chunks: list[dict] = []
    for chuong, heading, body in _sections(content):
        parts = _split_to_fit(body, cfg.max_words)
        context = " > ".join(p for p in (record["doc_title"], chuong, heading) if p)
        for text in _pack(parts, cfg.max_words, cfg.min_words):
            chunks.append(_make_chunk(record, len(chunks), text, heading, context))
    return chunks


def chunk_thong_bao(record: dict, cfg: ChunkingSettings) -> list[dict]:
    content = (record.get("content") or "").strip()
    if not content:
        return []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
    parts = [piece for p in paragraphs for piece in _split_to_fit(p, cfg.max_words)]
    return [
        _make_chunk(record, i, text, None, record["doc_title"])
        for i, text in enumerate(_pack(parts, cfg.max_words, cfg.min_words, sep="\n\n"))
    ]
