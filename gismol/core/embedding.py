# ============================================================
# FILE: gismol/core/embedding.py
# ============================================================
"""Embedding functions ``E : O -> H`` into a Hilbert space (Definition 2.6).

The toolkit deliberately uses plain tuples of floats as vectors so that the
library stays dependency free.  :class:`HilbertSpace` bundles the inner product
and induced metric.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence, Tuple

from .exceptions import EmbeddingError

__all__ = [
    "Vector", "HilbertSpace", "Embedder", "FeatureEmbedder", "HashEmbedder",
    "CompositeEmbedder", "ConstantEmbedder", "vadd", "vsub", "vscale", "dot",
    "norm", "cosine", "euclidean", "mean_vector", "zeros",
]

Vector = Tuple[float, ...]


# --------------------------------------------------------------------------
# vector helpers
# --------------------------------------------------------------------------
def zeros(dim: int) -> Vector:
    return tuple(0.0 for _ in range(dim))


def vadd(a: Sequence[float], b: Sequence[float]) -> Vector:
    if len(a) != len(b):
        raise EmbeddingError(f"dimension mismatch: {len(a)} vs {len(b)}")
    return tuple(float(x) + float(y) for x, y in zip(a, b))


def vsub(a, b) -> Vector:
    return tuple(float(x) - float(y) for x, y in zip(a, b))


def vscale(a, s: float) -> Vector:
    return tuple(float(x) * float(s) for x in a)


def dot(a, b) -> float:
    return float(math.fsum(float(x) * float(y) for x, y in zip(a, b)))


def norm(a) -> float:
    return math.sqrt(max(0.0, dot(a, a)))


def cosine(a, b) -> float:
    na, nb = norm(a), norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot(a, b) / (na * nb)))


def euclidean(a, b) -> float:
    return norm(vsub(a, b))


def mean_vector(vectors: Iterable[Sequence[float]]) -> Vector:
    vs = [tuple(map(float, v)) for v in vectors]
    if not vs:
        return ()
    dim = len(vs[0])
    return tuple(math.fsum(v[i] for v in vs) / len(vs) for i in range(dim))


# --------------------------------------------------------------------------
# Hilbert space
# --------------------------------------------------------------------------
class HilbertSpace:
    """A finite dimensional real Hilbert space."""

    def __init__(self, dim: Optional[int] = None, name: str = "H",
                 kernel: Optional[Callable[[Any, Any], float]] = None):
        self.dim = dim
        self.name = name
        self.kernel = kernel

    def inner(self, u, v) -> float:
        if self.kernel is not None:
            return float(self.kernel(u, v))
        return dot(u, v)

    def norm(self, u) -> float:
        return math.sqrt(max(0.0, self.inner(u, u)))

    def distance(self, u, v) -> float:
        return self.norm(vsub(u, v)) if self.kernel is None else math.sqrt(
            max(0.0, self.inner(u, u) - 2 * self.inner(u, v) + self.inner(v, v))
        )

    def similarity(self, u, v) -> float:
        nu, nv = self.norm(u), self.norm(v)
        if nu == 0.0 or nv == 0.0:
            return 0.0
        return max(-1.0, min(1.0, self.inner(u, v) / (nu * nv)))

    def __repr__(self):
        return f"HilbertSpace(dim={self.dim}, name={self.name!r})"


# --------------------------------------------------------------------------
# embedder interface
# --------------------------------------------------------------------------
class Embedder:
    """Base class for embedding functions ``E : O -> H``."""

    dim: Optional[int] = None
    space: HilbertSpace

    def __init__(self, dim: Optional[int] = None, name: str = "embedder"):
        self.dim = dim
        self.name = name
        self.space = HilbertSpace(dim, name=f"H[{name}]")

    def __call__(self, obj) -> Vector:  # pragma: no cover - abstract
        raise NotImplementedError

    def embed(self, obj) -> Vector:
        return self(obj)

    def similarity(self, a, b) -> float:
        return self.space.similarity(self(a), self(b))

    def distance(self, a, b) -> float:
        return self.space.distance(self(a), self(b))

    def __repr__(self):
        return f"{type(self).__name__}(name={self.name!r}, dim={self.dim})"


class ConstantEmbedder(Embedder):
    """Trivial embedder mapping everything to the zero vector."""

    def __init__(self, dim: int = 1):
        super().__init__(dim, "constant")

    def __call__(self, obj) -> Vector:
        return zeros(self.dim)


# --------------------------------------------------------------------------
# feature embedder
# --------------------------------------------------------------------------
class FeatureEmbedder(Embedder):
    """Embed a state by extracting a fixed list of named attributes.

    Parameters
    ----------
    features:
        Names of attributes (dotted paths allowed).
    ranges:
        Optional ``{name: (low, high)}`` used for min-max normalisation.
    default:
        Value substituted for missing attributes.
    """

    def __init__(self, features: Sequence[str], ranges: Optional[Mapping[str, Tuple[float, float]]] = None,
                 default: float = 0.0, name: str = "features"):
        super().__init__(len(features), name)
        self.features = list(features)
        self.ranges = dict(ranges or {})
        self.default = float(default)

    def _extract(self, obj) -> float:
        from .expression import lookup, _MISSING
        if isinstance(obj, Mapping) or hasattr(obj, "get_path"):
            try:
                value = lookup(obj, self.name) if False else None
            except Exception:
                value = None
        return value

    def __call__(self, obj) -> Vector:
        from .expression import lookup
        out: List[float] = []
        for feat in self.features:
            try:
                raw = lookup(obj, feat)
            except KeyError:
                raw = self.default
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = float(hash(str(raw)) % 1000) / 1000.0
            if feat in self.ranges:
                low, high = self.ranges[feat]
                span = (high - low) or 1.0
                value = (value - low) / span
            out.append(value)
        return tuple(out)


# --------------------------------------------------------------------------
# hashing embedder
# --------------------------------------------------------------------------
def _hash_floats(key: str, dim: int, seed: int = 0) -> Vector:
    """Deterministic pseudo-random projection of *key* into R^dim."""
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    values = []
    for i in range(dim):
        chunk = digest[(2 * i) % len(digest): (2 * i) % len(digest) + 2] or b"\x00\x00"
        if len(chunk) < 2:
            chunk = chunk + b"\x00"
        raw = int.from_bytes(chunk, "big")
        values.append((raw / 65535.0) * 2.0 - 1.0)
    return tuple(values)


class HashEmbedder(Embedder):
    """Structure-hashing embedder: identical structures embed identically.

    Works on any object by hashing its canonical string representation, which
    makes it a drop-in default for :class:`COHObject`.
    """

    def __init__(self, dim: int = 64, seed: int = 0, name: str = "hash"):
        super().__init__(dim, name)
        self.seed = int(seed)

    def __call__(self, obj) -> Vector:
        key = self._canonical(obj)
        return _hash_floats(key, self.dim, self.seed)

    @staticmethod
    def _canonical(obj) -> str:
        if hasattr(obj, "canonical_form"):
            return obj.canonical_form()
        if isinstance(obj, Mapping):
            return "{" + ",".join(f"{k}:{HashEmbedder._canonical(v)}"
                                  for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))) + "}"
        if isinstance(obj, (list, tuple)):
            return "[" + ",".join(HashEmbedder._canonical(v) for v in obj) + "]"
        return f"{type(obj).__name__}:{obj}"


# --------------------------------------------------------------------------
# composite embedder
# --------------------------------------------------------------------------
class CompositeEmbedder(Embedder):
    """Embed a hierarchical object by combining its own features and its
    children's embeddings (``E(O) = E_self(O) (+) sum_c E(c)``)."""

    def __init__(self, self_embedder: Optional[Embedder] = None,
                 child_weight: float = 1.0, mode: str = "concat",
                 name: str = "composite"):
        super().__init__(None, name)
        self.self_embedder = self_embedder
        self.child_weight = float(child_weight)
        self.mode = mode

    def __call__(self, obj) -> Vector:
        own = self.self_embedder(obj) if self.self_embedder is not None else ()
        child_vecs = []
        children = getattr(obj, "children", None)
        if callable(children):
            for child in children().values():
                child_vecs.append(self(child))

        if not child_vecs:
            return tuple(own)

        if self.mode == "sum":
            total = tuple(own) if own else zeros(len(child_vecs[0]))
            for cv in child_vecs:
                if len(cv) != len(total):
                    total = tuple(list(total) + [0.0] * (len(cv) - len(total)))
                total = vadd(total, vscale(cv, self.child_weight))
            return total

        # concat mode
        flat: List[float] = list(own)
        for cv in child_vecs:
            flat.extend(vscale(cv, self.child_weight))
        return tuple(flat)
    