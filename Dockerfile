# HUFLIT Regulation QA: API + chat page, CPU-only.
# The LLM runs in a separate Ollama container (see docker-compose.yml).
#
#   docker build -t huflit-qa .
#   docker run -p 8080:8080 -e HUFLIT_GENERATION__OLLAMA_URL=http://host.docker.internal:11434 huflit-qa

ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /src
# CPU build of torch first (the default wheel bundles ~2 GB of CUDA libraries).
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .
# Bake the embedding model of the shipped index into the image so the container starts offline.
ENV HF_HOME=/opt/hf
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

FROM python:${PYTHON_VERSION}-slim
RUN useradd --create-home --uid 1000 app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=app /opt/hf /opt/hf
ENV PATH=/opt/venv/bin:$PATH \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    HUFLIT_ROOT_DIR=/app \
    HUFLIT_GENERATION__OLLAMA_URL=http://ollama:11434
WORKDIR /app
COPY --chown=app data/index ./data/index
USER app
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=4)"
CMD ["uvicorn", "huflit_qa.api.app:app", "--host", "0.0.0.0", "--port", "8080"]
