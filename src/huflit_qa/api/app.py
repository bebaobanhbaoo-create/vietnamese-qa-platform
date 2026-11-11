"""HTTP API: search and cited answers over the HUFLIT regulations.

    uvicorn huflit_qa.api.app:app --port 8080        # or: huflit-qa serve

Models and indexes are loaded once at startup. Search and generation are blocking (GPU / Ollama),
so the endpoints are plain ``def`` and run in Starlette's thread pool; a lock serialises GPU work so
concurrent requests queue instead of competing for 6 GB of VRAM.
"""

import json
import logging
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from huflit_qa import __version__
from huflit_qa.api.schemas import (
    AskRequest,
    AskResponse,
    Citations,
    Health,
    Readiness,
    SearchRequest,
    SearchResponse,
    Source,
)
from huflit_qa.config import Settings, get_settings
from huflit_qa.generation.citations import check_citations
from huflit_qa.generation.llm import LLM, LLMUnavailableError
from huflit_qa.generation.rag import RAGPipeline
from huflit_qa.retrieval.hybrid import HybridRetriever

logger = logging.getLogger("huflit_qa.api")
STATIC_DIR = Path(__file__).parent / "static"

REQUESTS = Counter("huflit_requests_total", "HTTP requests", ["route", "status"])
LATENCY = Histogram(
    "huflit_request_seconds",
    "Request latency",
    ["route"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60),
)


@dataclass
class Services:
    retriever: HybridRetriever
    pipeline: RAGPipeline
    llm: LLM
    gpu_lock: threading.Lock


def build_services(settings: Settings) -> Services:
    from huflit_qa.generation.llm import OllamaLLM

    gen = settings.generation
    retriever = HybridRetriever(settings)
    llm = OllamaLLM(gen.model, gen.ollama_url, gen.temperature, gen.max_tokens, gen.num_ctx, gen.timeout_s)
    return Services(retriever, RAGPipeline(retriever, llm, gen), llm, threading.Lock())


def create_app(services: Services | None = None, settings: Settings | None = None) -> FastAPI:
    """Build the app. Tests pass ready-made ``services``; otherwise they are built at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if services is None:
            logger.info("Loading indexes and models")
            app.state.services = build_services(settings or get_settings())
        else:
            app.state.services = services
        yield

    app = FastAPI(
        title="HUFLIT Regulation QA",
        version=__version__,
        description="Hybrid search and cited answers over HUFLIT academic regulations and announcements.",
        lifespan=lifespan,
    )

    def get_services(request: Request) -> Services:
        return request.app.state.services

    ServicesDep = Annotated[Services, Depends(get_services)]

    @app.middleware("http")
    async def observe(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        route = request.url.path
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("request_id=%s path=%s failed", request_id, route)
            response = JSONResponse(
                {"detail": "Internal server error", "request_id": request_id}, status_code=500
            )
        elapsed = time.perf_counter() - start
        response.headers["x-request-id"] = request_id
        if route != "/metrics":
            REQUESTS.labels(route, str(response.status_code)).inc()
            LATENCY.labels(route).observe(elapsed)
            logger.info(
                "request_id=%s method=%s path=%s status=%s ms=%.1f",
                request_id,
                request.method,
                route,
                response.status_code,
                elapsed * 1000,
            )
        return response

    @app.get("/health", response_model=Health, tags=["ops"])
    def health() -> Health:
        """Liveness: the process is up."""
        return Health(status="ok", version=__version__)

    @app.get("/ready", response_model=Readiness, tags=["ops"])
    def ready(svc: ServicesDep) -> Readiness:
        """Readiness: indexes are loaded and the LLM can answer."""
        llm_ok, detail = svc.llm.available()
        manifest = svc.retriever.manifest
        return Readiness(
            ready=llm_ok,
            index={
                "num_chunks": manifest.get("num_chunks"),
                "chunking": manifest.get("chunking", {}).get("strategy"),
                "embedding": manifest.get("embedding", {}).get("model"),
                "vector_store": manifest.get("index", {}).get("vector_store", "faiss"),
                "rerank": svc.retriever.rerank_cfg.model if svc.retriever.reranker else None,
            },
            llm_model=svc.llm.model,
            llm_available=llm_ok,
            detail=detail,
        )

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/search", response_model=SearchResponse, tags=["search"])
    def search(req: SearchRequest, svc: ServicesDep) -> SearchResponse:
        """Hybrid search (BM25 + dense, RRF, optional reranking) with metadata filters."""
        start = time.perf_counter()
        with svc.gpu_lock:
            results = svc.retriever.retrieve(req.query, top_k=req.top_k, filters=req.filters)
        return SearchResponse(
            query=req.query,
            results=[Source(**r) for r in results],
            took_ms=(time.perf_counter() - start) * 1000,
        )

    @app.post("/v1/ask", response_model=AskResponse, tags=["answer"])
    def ask(req: AskRequest, svc: ServicesDep) -> AskResponse:
        """Answer with numbered sources and [n] citations; refuses when the sources lack the answer."""
        try:
            with svc.gpu_lock:
                result = svc.pipeline.answer(req.question, req.filters)
        except LLMUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return AskResponse(
            question=result.question,
            answer=result.answer,
            sources=[Source(**_source(s)) for s in result.sources],
            citations=Citations(**_citations(result.citations)),
            model=svc.llm.model,
            retrieval_ms=result.retrieval_ms,
            generation_ms=result.generation_ms,
        )

    @app.post("/v1/ask/stream", tags=["answer"])
    def ask_stream(req: AskRequest, svc: ServicesDep) -> StreamingResponse:
        """SSE stream: ``sources``, then ``token`` events, then ``done`` with the citation check."""
        with svc.gpu_lock:
            sources, tokens = svc.pipeline.stream(req.question, req.filters)

        def events() -> Iterator[str]:
            yield _sse("sources", [Source(**_source(s)).model_dump() for s in sources])
            parts: list[str] = []
            try:
                for token in tokens:
                    parts.append(token)
                    yield _sse("token", token)
            except LLMUnavailableError as exc:
                yield _sse("error", str(exc))
                return
            check = check_citations("".join(parts), len(sources))
            yield _sse("done", _citations(check))

        return StreamingResponse(
            events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


def _source(chunk: dict) -> dict:
    return {k: chunk.get(k) for k in Source.model_fields}


def _citations(check) -> dict:
    return {
        "cited": check.cited,
        "invalid": check.invalid,
        "refused": check.refused,
        "grounded": check.grounded,
    }


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


app = create_app()
