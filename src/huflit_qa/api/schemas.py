"""Request and response models of the HTTP API."""

from pydantic import BaseModel, Field

from huflit_qa.retrieval.filters import SearchFilters


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500, examples=["điều kiện xét tốt nghiệp"])
    top_k: int = Field(default=5, ge=1, le=20)
    filters: SearchFilters = SearchFilters()


class Source(BaseModel):
    chunk_id: str
    doc_title: str
    chunk_title: str | None = None
    text: str
    source_url: str
    date_posted: str | None = None
    issued_date: str | None = None
    issuer_unit: str | None = None
    newest_in_group: bool | None = None
    score: float | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[Source]
    took_ms: float


class AskRequest(BaseModel):
    question: str = Field(
        min_length=2, max_length=500, examples=["Mỗi học kỳ được đăng ký tối đa bao nhiêu tín chỉ?"]
    )
    filters: SearchFilters = SearchFilters()


class Citations(BaseModel):
    cited: list[int]
    invalid: list[int]
    refused: bool
    grounded: bool


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    citations: Citations
    model: str
    retrieval_ms: float
    generation_ms: float


class Health(BaseModel):
    status: str
    version: str


class Readiness(BaseModel):
    ready: bool
    index: dict
    llm_model: str
    llm_available: bool
    detail: str | None = None
