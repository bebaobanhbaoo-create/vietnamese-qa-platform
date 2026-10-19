"""Sentence-embedding encoders with per-model conventions (prefixes, word segmentation)."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EncoderSpec:
    query_prefix: str = ""
    passage_prefix: str = ""
    # PhoBERT-based models are trained on pyvi-segmented text ("tốt_nghiệp").
    segment_words: bool = False
    # Hub revision to load; None = main.
    revision: str | None = None


# Conventions from each model card. Unknown models get the defaults.
ENCODER_SPECS: dict[str, EncoderSpec] = {
    "paraphrase-multilingual-MiniLM-L12-v2": EncoderSpec(),
    "intfloat/multilingual-e5-base": EncoderSpec(query_prefix="query: ", passage_prefix="passage: "),
    "intfloat/multilingual-e5-large": EncoderSpec(query_prefix="query: ", passage_prefix="passage: "),
    "bkai-foundation-models/vietnamese-bi-encoder": EncoderSpec(segment_words=True),
    # main only ships pytorch_model.bin, which transformers refuses to torch.load with torch < 2.6
    # (CVE-2025-32434); refs/pr/130 is the Hub's safetensors conversion of the same weights.
    "BAAI/bge-m3": EncoderSpec(revision="refs/pr/130"),
}


class Encoder:
    def __init__(self, model_name: str, max_seq_length: int = 512, batch_size: int = 32):
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.spec = ENCODER_SPECS.get(model_name, EncoderSpec())
        device = "cuda" if torch.cuda.is_available() else "cpu"
        revision = self.spec.revision
        try:
            # Load from the local HF cache first so repeated runs work offline.
            self.model = SentenceTransformer(
                model_name, device=device, revision=revision, local_files_only=True
            )
        except Exception:
            self.model = SentenceTransformer(model_name, device=device, revision=revision)
        self.model.max_seq_length = min(max_seq_length, self.model.max_seq_length or max_seq_length)
        self.batch_size = batch_size

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension() or 0)

    def _prepare(self, texts: list[str], prefix: str) -> list[str]:
        if self.spec.segment_words:
            from pyvi import ViTokenizer

            texts = [ViTokenizer.tokenize(t) for t in texts]
        return [prefix + t for t in texts]

    def _encode(self, texts: list[str], show_progress: bool) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype="float32")

    def encode_passages(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        return self._encode(self._prepare(texts, self.spec.passage_prefix), show_progress)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(self._prepare(texts, self.spec.query_prefix), show_progress=False)
