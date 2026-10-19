"""Make relevance labels independent of the chunking.

Labels point at *label units*: the chunks of the original (v1) chunking, frozen in
``eval/label_units.json``. Any other chunking is evaluated by projecting each retrieved chunk onto
the units it overlaps (same source record, shared word 3-grams), then scoring the projected ranking
against the unchanged relevant unit set. Re-chunking therefore never invalidates labels, and every
configuration is scored against the same qrels.

A chunk that spans several units gets credit for all of them, which favours long chunks; the
evaluation report shows words per retrieved chunk so that trade-off stays visible.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

WORD_RE = re.compile(r"\w+", re.UNICODE)
# A chunk covers a unit if it contains at least this share of the unit's shingles,
# or if at least this share of the chunk's shingles come from the unit (the chunk is a piece of it).
MIN_CONTAINMENT = 0.5


def shingles(text: str, n: int = 3) -> set[tuple[str, ...]]:
    words = WORD_RE.findall(text.lower())
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def load_units(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_projection(chunks: list[dict], units: list[dict]) -> dict[str, list[str]]:
    """chunk_id -> ids of the label units it covers (empty if none)."""
    units_by_doc: dict[str, list[tuple[str, set]]] = defaultdict(list)
    unit_text = {u["unit_id"]: u["text"] for u in units}
    for u in units:
        units_by_doc[u["doc_id"]].append((u["unit_id"], shingles(u["text"])))

    projection: dict[str, list[str]] = {}
    for chunk in chunks:
        if unit_text.get(chunk["chunk_id"]) == chunk["text"]:
            projection[chunk["chunk_id"]] = [chunk["chunk_id"]]  # the label unit itself
            continue
        chunk_sh = shingles(chunk["text"])
        covered = []
        for unit_id, unit_sh in units_by_doc.get(chunk["doc_id"], []):
            if not unit_sh or not chunk_sh:
                continue
            common = len(chunk_sh & unit_sh)
            if common / len(unit_sh) >= MIN_CONTAINMENT or common / len(chunk_sh) >= MIN_CONTAINMENT:
                covered.append(unit_id)
        projection[chunk["chunk_id"]] = covered
    return projection


def project_ranking(chunk_ids: list[str], projection: dict[str, list[str]]) -> list[str]:
    """Ranked chunk ids -> ranked unit ids (first occurrence wins, order kept)."""
    seen: set[str] = set()
    ranked = []
    for chunk_id in chunk_ids:
        for unit_id in projection.get(chunk_id, []):
            if unit_id not in seen:
                seen.add(unit_id)
                ranked.append(unit_id)
    return ranked
