"""Cross-encoder reranking of the fused candidates.

A bi-encoder embeds query and chunk separately; a cross-encoder reads them together, which is much
more accurate but too slow to run over the whole corpus. It is applied to the top ``depth``
candidates of hybrid search only, so R@depth of the first stage bounds what reranking can reach.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RerankerSpec:
    max_length: int = 512
    # PhoBERT-based models expect pyvi-segmented input ("tốt_nghiệp").
    segment_words: bool = False


RERANKER_SPECS: dict[str, RerankerSpec] = {
    "BAAI/bge-reranker-v2-m3": RerankerSpec(max_length=512),
    "itdainb/PhoRanker": RerankerSpec(max_length=256, segment_words=True),
}


class CrossEncoderReranker:
    def __init__(self, model_name: str, batch_size: int = 16):
        import torch
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.spec = RERANKER_SPECS.get(model_name, RerankerSpec())
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = CrossEncoder(
                model_name, max_length=self.spec.max_length, device=device, local_files_only=True
            )
        except Exception:
            self.model = CrossEncoder(model_name, max_length=self.spec.max_length, device=device)
        self.batch_size = batch_size

    def _prepare(self, text: str) -> str:
        if self.spec.segment_words:
            from pyvi import ViTokenizer

            return str(ViTokenizer.tokenize(text))
        return text

    def score(self, query: str, passages: list[str]) -> list[float]:
        q = self._prepare(query)
        pairs = [(q, self._prepare(p)) for p in passages]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return [float(s) for s in scores]
