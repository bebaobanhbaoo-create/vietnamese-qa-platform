import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging and force UTF-8 console output (Windows defaults to cp1252)."""
    for stream in (sys.stdout, sys.stderr):
        encoding = getattr(stream, "encoding", None)
        if encoding and encoding.lower() != "utf-8" and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # sentence-transformers / httpx are chatty at INFO.
    for noisy in ("sentence_transformers", "httpx", "urllib3", "faiss", "faiss.loader"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
