"""Metadata filters shared by BM25, FAISS and Qdrant, so every backend returns the same subset."""

from pydantic import BaseModel


class SearchFilters(BaseModel):
    doc_type: str | None = None  # "quy_che" | "thong_bao"
    issuer_unit: str | None = None  # e.g. "Phòng Đào tạo"
    year_from: int | None = None  # by date_posted
    # Drop regulation chunks that are not the newest version in their document group.
    newest_only: bool = False

    def is_empty(self) -> bool:
        return not (self.doc_type or self.issuer_unit or self.year_from or self.newest_only)

    def matches(self, chunk: dict) -> bool:
        if self.doc_type and chunk["doc_type"] != self.doc_type:
            return False
        if self.issuer_unit and chunk.get("issuer_unit") != self.issuer_unit:
            return False
        if self.year_from and chunk_year(chunk) < self.year_from:
            return False
        return not (self.newest_only and chunk["doc_type"] == "quy_che" and not chunk.get("newest_in_group"))


def chunk_year(chunk: dict) -> int:
    return int((chunk.get("date_posted") or "0")[:4] or 0)
