# ============================================================
# FILE: gismol/neural/embeddings.py
# ============================================================
"""Neural / learned embedding functions (component ``E`` of the COH tuple)."""

from __future__ import annotations

import math
import random
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from ..core.embedding import Embedder, Vector, cosine, euclidean, norm, vsub
from .nn import MLP

__all__ = ["NeuralEmbedder", "AutoencoderEmbedder", "SiameseEmbedder"]


class NeuralEmbedder(Embedder):
    """``E(x) = MLP(x)`` with an L2-normalised output."""

    def __init__(self, input_dim: int, dim: int = 16,
                 hidden: Sequence[int] = (32,), seed: Optional[int] = None,
                 name: str = "neural_embedder"):
        super().__init__(dim, name)
        sizes = [input_dim] + list(hidden) + [dim]
        self.mlp = MLP(sizes, activation="tanh", output_activation="linear", seed=seed)
        self.loss_history: List[float] = []

    def __call__(self, x) -> Vector:
        raw = self.mlp.forward(list(map(float, x)))
        n = math.sqrt(sum(v * v for v in raw)) or 1.0
        return tuple(v / n for v in raw)

    # -- training ----------------------------------------------------------
    def train(self, X: Sequence[Sequence[float]], epochs: int = 100, lr: float = 0.01,
              optimizer=None, verbose: bool = False) -> List[float]:
        """Train the embedder as an autoencoder-style identity map."""
        from .optimizers import Adam
        optimizer = optimizer or Adam(lr)
        history: List[float] = []
        for epoch in range(epochs):
            total = 0.0
            grad_acc = [0.0] * self.mlp.n_parameters
            for x in X:
                pred = self.mlp.forward(list(map(float, x)))
                # target: unit-normalised input padded/truncated to dim
                target = list(map(float, x))[:self.dim]
                target += [0.0] * (self.dim - len(target))
                n = math.sqrt(sum(v * v for v in target)) or 1.0
                target = [v / n for v in target]
                loss = sum((p - t) ** 2 for p, t in zip(pred, target)) / self.dim
                total += loss
                grad = [2.0 * (p - t) / self.dim for p, t in zip(pred, target)]
                self.mlp.backward(grad)
                g = self.mlp.get_gradients()
                for k, gv in enumerate(g):
                    grad_acc[k] += gv
            grad_acc = [g / max(1, len(X)) for g in grad_acc]
            self.mlp.set_parameters(optimizer.step(self.mlp.get_parameters(), grad_acc))
            history.append(total / max(1, len(X)))
            if verbose and epoch % max(1, epochs // 5) == 0:
                print(f"[{self.name}] epoch {epoch}: loss={history[-1]:.6f}")
        self.loss_history = history
        return history

    def __repr__(self):
        return f"NeuralEmbedder(dim={self.dim}, params={self.mlp.n_parameters})"


class AutoencoderEmbedder(Embedder):
    """Symmetric encoder/decoder pair; embedding is the encoder output."""

    def __init__(self, input_dim: int, dim: int = 8, hidden: int = 32,
                 seed: Optional[int] = None, name: str = "autoencoder"):
        super().__init__(dim, name)
        self.encoder = MLP([input_dim, hidden, dim], "tanh", "tanh", seed)
        self.decoder = MLP([dim, hidden, input_dim], "tanh", "linear", seed)

    def __call__(self, x) -> Vector:
        return tuple(self.encoder.forward(list(map(float, x))))

    def reconstruct(self, x):
        return self.decoder.forward(self(x))

    def reconstruction_error(self, x) -> float:
        rec = self.reconstruct(x)
        return float(sum((a - b) ** 2 for a, b in zip(rec, x)) / max(1, len(x)))

    def __repr__(self):
        return f"AutoencoderEmbedder(dim={self.dim})"


class SiameseEmbedder(Embedder):
    """Contrastive embedder trained to pull similar pairs together."""

    def __init__(self, input_dim: int, dim: int = 16, hidden: int = 32,
                 margin: float = 0.5, seed: Optional[int] = None,
                 name: str = "siamese"):
        super().__init__(dim, name)
        self.base = NeuralEmbedder(input_dim, dim, (hidden,), seed, name=f"{name}.base")
        self.margin = float(margin)
        self.space = self.base.space

    def __call__(self, x) -> Vector:
        return self.base(x)

    def contrastive_loss(self, x1, x2, similar: bool) -> float:
        d = euclidean(self(x1), self(x2))
        if similar:
            return d * d
        return max(0.0, self.margin - d) ** 2

    def train_pairs(self, pairs: Iterable[Tuple[Any, Any, bool]],
                    epochs: int = 50, lr: float = 0.01) -> List[float]:
        from .optimizers import Adam
        opt = Adam(lr)
        pairs = list(pairs)
        history = []
        for _ in range(epochs):
            grad_acc = [0.0] * self.base.mlp.n_parameters
            total = 0.0
            for x1, x2, similar in pairs:
                e1 = self.base(x1)
                e2 = self.base(x2)
                d = euclidean(e1, e2)
                loss = (d * d) if similar else max(0.0, self.margin - d) ** 2
                total += loss
                # numerical gradient (kept simple; small models only)
                params = self.base.mlp.get_parameters()
                eps = 1e-4
                for k in range(len(params)):
                    bumped = list(params)
                    bumped[k] += eps
                    self.base.mlp.set_parameters(bumped)
                    d2 = euclidean(self.base(x1), self.base(x2))
                    l2 = (d2 * d2) if similar else max(0.0, self.margin - d2) ** 2
                    grad_acc[k] += (l2 - loss) / eps
                self.base.mlp.set_parameters(params)
            self.base.mlp.set_parameters(opt.step(self.base.mlp.get_parameters(),
                                                  grad_acc))
            history.append(total / max(1, len(pairs)))
        return history
    