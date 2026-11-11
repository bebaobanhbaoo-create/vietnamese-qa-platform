"""Parse and check the [n] citations in a generated answer."""

import re
from dataclasses import dataclass

from huflit_qa.generation.prompts import REFUSAL

# [1], [1, 3], [1-3], [1; 2]
RE_CITATION = re.compile(r"\[(\d+(?:\s*[,;–-]\s*\d+)*)\]")


@dataclass
class CitationCheck:
    cited: list[int]  # valid source numbers, in first-cited order
    invalid: list[int]  # numbers that do not match any source
    refused: bool
    has_citation: bool

    @property
    def grounded(self) -> bool:
        """An answer is grounded if it refuses, or cites at least one source and nothing invalid."""
        return self.refused or (self.has_citation and not self.invalid)


def _expand(group: str) -> list[int]:
    numbers: list[int] = []
    for part in re.split(r"\s*[,;]\s*", group):
        if re.fullmatch(r"\d+\s*[–-]\s*\d+", part):
            lo, hi = (int(x) for x in re.split(r"\s*[–-]\s*", part))
            numbers.extend(range(lo, hi + 1) if hi - lo < 20 else [lo, hi])
        elif part.strip().isdigit():
            numbers.append(int(part))
    return numbers


def is_refusal(answer: str) -> bool:
    normalized = " ".join(answer.split()).strip().rstrip(".").lower()
    return REFUSAL.rstrip(".").lower() in normalized


def check_citations(answer: str, n_sources: int) -> CitationCheck:
    cited: list[int] = []
    invalid: list[int] = []
    for match in RE_CITATION.finditer(answer):
        for n in _expand(match.group(1)):
            target = cited if 1 <= n <= n_sources else invalid
            if n not in target:
                target.append(n)
    return CitationCheck(cited=cited, invalid=invalid, refused=is_refusal(answer), has_citation=bool(cited))
