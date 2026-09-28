# ============================================================
# FILE: gismol/nlp/embedders.py
# ============================================================
"""Text embedding functions implementing the COH ``E`` interface."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from ..core.embedding import Embedder, Vector, cosine, zeros
from .normalizers import normalize
from .parsers import tokenize

__all__ = [
    "BagOfWordsEmbedder", "TfidfEmbedder", "HashingEmbedder", "CharNGramEmbedder",
]


class BagOfWordsEmbedder(Embedder):
    """Vocabulary-indexed bag-of-words embedding."""

    def __init__(self, vocabulary: Optional[Iterable[str]] = None,
                 dim: Optional[int] = None, binary: bool = False,
                 lowercase: bool = True, name: str = "bow"):
        self.vocabulary: List[str] = list(vocabulary or [])
        self._index: Dict[str, int] = {w: i for i, w in enumerate(self.vocabulary)}
        self.binary = binary
        self.lowercase = lowercase
        super().__init__(dim or len(self.vocabulary), name)

    def fit(self, documents: Sequence[str], min_count: int = 1, max_vocab: Optional[int] = None):
        counts: Counter = Counter()
        for doc in documents:
            counts.update(self._tokens(doc))
        items = [(w, c) for w, c in counts.items() if c >= min_count]
        items.sort(key=lambda kv: (-kv[1], kv[0]))
        if max_vocab:
            items = items[:max_vocab]
        self.vocabulary = [w for w, _ in items]
        self._index = {w: i for i, w in enumerate(self.vocabulary)}
        self.dim = len(self.vocabulary)
        self.space = self.space.__class__(self.dim, name=f"H[{self.name}]")
        return self

    def _tokens(self, text: str) -> List[str]:
        raw = normalize(text, lowercase=self.lowercase)
        return [t for t in tokenize(raw) if any(c.isalnum() for c in t)]

    def __call__(self, text: str) -> Vector:
        vec = [0.0] * max(1, self.dim or 1)
        for token in self._tokens(text):
            idx = self._index.get(token)
            if idx is None:
                continue
            vec[idx] = 1.0 if self.binary else vec[idx] + 1.0
        return tuple(vec)

    def __repr__(self):
        return f"BagOfWordsEmbedder(vocab={len(self.vocabulary)})"


class TfidfEmbedder(Embedder):
    """TF-IDF embedding with a fitted vocabulary and document frequencies."""

    def __init__(self, vocabulary: Optional[Iterable[str]] = None,
                 idf: Optional[Mapping[str, float]] = None, name: str = "tfidf"):
        self.vocabulary: List[str] = list(vocabulary or [])
        self._index: Dict[str, int] = {w: i for i, w in enumerate(self.vocabulary)}
        self.idf: Dict[str, float] = dict(idf or {})
        super().__init__(len(self.vocabulary), name)

    def fit(self, documents: Sequence[str], min_count: int = 1):
        counts: Counter = Counter()
        df: Counter = Counter()
        for doc in documents:
            tokens = tokenize(normalize(doc))
            counts.update(tokens)
            df.update(set(tokens))
        items = [(w, c) for w, c in counts.items() if c >= min_count]
        items.sort(key=lambda kv: (-kv[1], kv[0]))
        self.vocabulary = [w for w, _ in items]
        self._index = {w: i for i, w in enumerate(self.vocabulary)}
        n = max(1, len(documents))
        self.idf = {w: math.log((1 + n) / (1 + df[w])) + 1.0 for w in self.vocabulary}
        self.dim = len(self.vocabulary)
        return self

    def __call__(self, text: str) -> Vector:
        tokens = tokenize(normalize(text))
        counts = Counter(tokens)
        total = max(1, sum(counts.values()))
        vec = [0.0] * max(1, self.dim or 1)
        for token, c in counts.items():
            idx = self._index.get(token)
            if idx is None:
                continue
            vec[idx] = (c / total) * self.idf.get(token, 1.0)
        return tuple(vec)

    def __repr__(self):
        return f"TfidfEmbedder(vocab={len(self.vocabulary)})"


class HashingEmbedder(Embedder):
    """Feature-hashing embedder: no vocabulary needed, fixed dimension."""

    def __init__(self, dim: int = 128, lowercase: bool = True,
                 ngram: int = 1, name: str = "hash_text"):
        super().__init__(dim, name)
        self.lowercase = lowercase
        self.ngram = int(ngram)

    def _features(self, text: str) -> List[str]:
        tokens = tokenize(normalize(text, lowercase=self.lowercase))
        if self.ngram <= 1:
            return tokens
        return [" ".join(tokens[i:i + self.ngram])
                for i in range(max(0, len(tokens) - self.ngram + 1))] or tokens

    def __call__(self, text: str) -> Vector:
        vec = [0.0] * self.dim
        for feature in self._features(text):
            digest = hashlib.md5(feature.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return tuple(v / norm for v in vec)

    def __repr__(self):
        return f"HashingEmbedder(dim={self.dim}, ngram={self.ngram})"


class CharNGramEmbedder(Embedder):
    """Character n-gram embedding, robust to morphology and typos."""

    def __init__(self, n: int = 3, dim: int = 128, name: str = "char_ngram"):
        super().__init__(dim, name)
        self.n = int(n)

    def __call__(self, text: str) -> Vector:
        norm = normalize(text)
        grams = [norm[i:i + self.n] for i in range(max(0, len(norm) - self.n + 1))]
        vec = [0.0] * self.dim
        for gram in grams:
            digest = hashlib.sha1(gram.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            vec[idx] += 1.0
        total = math.sqrt(sum(v * v for v in vec)) or 1.0
        return tuple(v / total for v in vec)

    def __repr__(self):
        return f"CharNGramEmbedder(n={self.n}, dim={self.dim})"