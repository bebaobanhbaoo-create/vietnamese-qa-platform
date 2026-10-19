"""Tokenisers for BM25."""

import re
from collections.abc import Callable

WORD_RE = re.compile(r"\w+", re.UNICODE)


def whitespace_tokens(text: str) -> list[str]:
    """Original behaviour: punctuation stays attached ("nghiệp." != "nghiệp")."""
    return text.lower().split()


def word_tokens(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def pyvi_tokens(text: str) -> list[str]:
    """Vietnamese word segmentation: multi-syllable words become one token ("tốt_nghiệp")."""
    from pyvi import ViTokenizer

    return WORD_RE.findall(ViTokenizer.tokenize(text.lower()))


TOKENIZERS: dict[str, Callable[[str], list[str]]] = {
    "whitespace": whitespace_tokens,
    "words": word_tokens,
    "pyvi": pyvi_tokens,
}


def get_tokenizer(name: str) -> Callable[[str], list[str]]:
    return TOKENIZERS[name]
