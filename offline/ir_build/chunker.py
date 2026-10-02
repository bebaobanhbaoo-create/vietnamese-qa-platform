"""Chunker for quy_che and thong_bao flat records."""
import re
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # root = D:/scape_data
from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE

RE_DIEU = re.compile(r'^Điều\s+\d+', re.MULTILINE)
RE_KHOAN = re.compile(r'(?m)^\s*\d+\.\s')

DOC_TYPE_MAP = {
    'Quy chế': 'quy_che',
    'Thông báo': 'thong_bao',
}


def _split_large(text):
    """Split text > CHUNK_SIZE at Khoản boundaries, then paragraphs."""
    parts = RE_KHOAN.split(text)
    clean_parts = []
    for p in parts:
        p = p.strip()
        if p:
            clean_parts.append(p)
    if len(clean_parts) > 1:
        return clean_parts
    clean_parts = []
    for p in text.split('\n\n'):
        p = p.strip()
        if p:
            clean_parts.append(p)
    if len(clean_parts) > 1:
        return clean_parts
    return [text]


def _merge_short(chunks):
    """Merge chunks shorter than MIN_CHUNK_SIZE into a neighbor."""
    if not chunks:
        return []
    result = list(chunks)
    i = 0
    while i < len(result):
        if len(result[i]) < MIN_CHUNK_SIZE:
            if i + 1 < len(result):
                result[i + 1] = result[i] + '\n\n' + result[i + 1]
                result.pop(i)
            elif i > 0:
                result[i - 1] = result[i - 1] + '\n\n' + result[i]
                result.pop(i)
            else:
                i += 1
        else:
            i += 1
    return result


def _apply_overlap(segments):
    """Prepend last CHUNK_OVERLAP chars of previous segment to each segment."""
    if len(segments) <= 1 or CHUNK_OVERLAP <= 0:
        return segments
    result = [segments[0]]
    for i in range(1, len(segments)):
        overlap = segments[i - 1][-CHUNK_OVERLAP:]
        result.append(overlap + '\n\n' + segments[i])
    return result


def _make_chunk(record, chunk_id, text, chunk_title, position):
    return {
        'chunk_id': chunk_id,
        'doc_id': record['id'],
        'doc_title': record['doc_title'],
        'chunk_title': chunk_title,
        'doc_type': DOC_TYPE_MAP.get(record['doc_type'], record['doc_type']),
        'source_url': record['doc_url'],
        'date_posted': record.get('date_posted'),
        'position': position,
        'text': text,
    }


def _chunk_by_paragraph(record, prefix):
    """Split content by paragraphs. Used for thong_bao and quy_che fallback."""
    content = record.get('content')
    if not content or not content.strip():
        return []
    text = content.strip()
    doc_id = record['id']

    if len(text) <= CHUNK_SIZE:
        return [_make_chunk(record, f'{doc_id}_{prefix}0', text, None, 0)]

    paragraphs = []
    for p in text.split('\n\n'):
        p = p.strip()
        if p:
            paragraphs.append(p)
    paragraphs = _merge_short(paragraphs)
    paragraphs = _apply_overlap(paragraphs)

    chunks = []
    for i, para in enumerate(paragraphs):
        chunks.append(_make_chunk(record, f'{doc_id}_{prefix}{i}', para, None, i))
    return chunks


def chunk_quy_che(record):
    """Chunk a quy_che flat record by Điều structure."""
    content = record.get('content')
    if not content or not content.strip():
        return []
    text = content.strip()
    doc_id = record['id']

    matches = list(RE_DIEU.finditer(text))
    if not matches:
        return _chunk_by_paragraph(record, prefix='d')

    # Slice text into segments, one per Điều
    preamble = text[:matches[0].start()].strip()
    raw_segments = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        seg = text[start:end].strip()
        if i == 0 and preamble:
            seg = preamble + '\n\n' + seg
        raw_segments.append(seg)

    # Split large segments; merge short sub-splits within each large segment
    segments = []
    for seg in raw_segments:
        if len(seg) > CHUNK_SIZE:
            sub = _split_large(seg)
            sub = _merge_short(sub)
            segments.extend(sub)
        else:
            segments.append(seg)
    segments = _apply_overlap(segments)

    # Build chunk dicts
    result = []
    for i, seg in enumerate(segments):
        first_line = seg.split('\n')[0].strip()
        chunk_title = first_line if RE_DIEU.match(first_line) else None
        result.append(_make_chunk(record, f'{doc_id}_d{i}', seg, chunk_title, i))
    return result


def chunk_thong_bao(record):
    """Chunk a thong_bao flat record by paragraphs."""
    return _chunk_by_paragraph(record, prefix='p')


def chunk_all(quy_che_path, thong_bao_path):
    """Load both flat JSONs and return all chunks."""
    chunks = []

    quy_che = json.loads(quy_che_path.read_text(encoding='utf-8'))
    for rec in quy_che:
        if rec.get('content') is None or rec.get('content_status') in ('image', 'empty'):
            continue
        chunks.extend(chunk_quy_che(rec))

    thong_bao = json.loads(thong_bao_path.read_text(encoding='utf-8'))
    for rec in thong_bao:
        if rec.get('content') is None or rec.get('content_status') in ('image', 'empty'):
            continue
        chunks.extend(chunk_thong_bao(rec))

    return chunks
