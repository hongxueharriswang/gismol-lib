# ============================================================
# FILE: gismol/nlp/similarity.py
# ============================================================
"""Similarity and distance measures, plus a simple nearest-neighbour index."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..core.embedding import cosine as _cosine, euclidean, norm

__all__ = [
    "cosine_similarity", "jaccard", "dice", "overlap", "levenshtein",
    "jaro_winkler", "manhattan", "SimilarityIndex",
]


def cosine_similarity(u: Sequence[float], v: Sequence[float]) -> float:
    return _cosine(u, v)


def jaccard(a: Iterable[Any], b: Iterable[Any]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


def dice(a: Iterable[Any], b: Iterable[Any]) -> float:
    sa, sb = set(a), set(b)
    denom = len(sa) + len(sb)
    return (2 * len(sa & sb) / denom) if denom else 1.0


def overlap(a: Iterable[Any], b: Iterable[Any]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / min(len(sa), len(sb)) if sa and sb else 0.0


def manhattan(u: Sequence[float], v: Sequence[float]) -> float:
    return float(sum(abs(x - y) for x, y in zip(u, v)))


def levenshtein(s1: str, s2: str) -> int:
    """Classic edit distance with O(min(n, m)) memory."""
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    previous = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1, start=1):
        current = [i]
        for j, c2 in enumerate(s2, start=1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (0 if c1 == c2 else 1),
            ))
        previous = current
    return previous[-1]


def jaro_winkler(s1: str, s2: str, prefix_weight: float = 0.1) -> float:
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    window = max(len(s1), len(s2)) // 2 - 1
    window = max(0, window)
    s1_matches = [False] * len(s1)
    s2_matches = [False] * len(s2)
    matches = 0
    for i, c1 in enumerate(s1):
        start = max(0, i - window)
        end = min(i + window + 1, len(s2))
        for j in range(start, end):
            if s2_matches[j] or c1 != s2[j]:
                continue
            s1_matches[i] = s2_matches[j] = True
            matches += 1
            break
    if matches == 0:
        return 0.0
    transpositions = 0
    k = 0
    for i, matched in enumerate(s1_matches):
        if not matched:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    transpositions //= 2
    jaro = (matches / len(s1) + matches / len(s2) +
            (matches - transpositions) / matches) / 3.0
    prefix = 0
    for c1, c2 in zip(s1, s2):
        if c1 != c2 or prefix == 4:
            break
        prefix += 1
    return jaro + prefix * prefix_weight * (1 - jaro)


# --------------------------------------------------------------------------
class SimilarityIndex:
    """Brute-force nearest-neighbour index over vectors or arbitrary keys."""

    def __init__(self, metric: Callable[[Sequence[float], Sequence[float]], float] = cosine_similarity,
                 higher_is_better: bool = True):
        self.metric = metric
        self.higher_is_better = higher_is_better
        self._items: List[Tuple[str, Sequence[float], Any]] = []

    def add(self, key: str, vector: Sequence[float], payload: Any = None):
        self._items.append((key, tuple(float(v) for v in vector), payload))
        return self

    def extend(self, entries: Iterable[Tuple[str, Sequence[float], Any]]):
        for key, vector, payload in entries:
            self.add(key, vector, payload)
        return self

    def query(self, vector: Sequence[float], k: int = 5) -> List[Tuple[str, float, Any]]:
        scored = [(key, self.metric(vector, vec), payload)
                  for key, vec, payload in self._items]
        scored.sort(key=lambda t: t[1], reverse=self.higher_is_better)
        return scored[:k]

    def __len__(self):
        return len(self._items)

    def __repr__(self):
        return f"SimilarityIndex(n={len(self._items)}, higher_is_better={self.higher_is_better})"
    