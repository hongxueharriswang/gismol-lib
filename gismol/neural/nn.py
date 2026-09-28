# ============================================================
# FILE: gismol/neural/nn.py
# ============================================================
"""A dependency-free multilayer perceptron with manual backpropagation.

Kept deliberately small: the point is to have *adaptive, parameterised
functions* (component ``N`` of the COH 9-tuple) available without pulling in a
heavy tensor framework.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "Dense", "Activation", "MLP", "mse_loss", "mse_grad",
    "cross_entropy_loss", "softmax", "relu", "sigmoid", "tanh", "linear",
]


# --------------------------------------------------------------------------
# activations
# --------------------------------------------------------------------------
def linear(x): return x
def relu(x): return x if x > 0.0 else 0.0
def sigmoid(x): return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x))))
def tanh(x): return math.tanh(x)


_ACT = {"linear": linear, "relu": relu, "sigmoid": sigmoid, "tanh": tanh}
_ACT_DERIV = {
    "linear": lambda x, y: 1.0,
    "relu": lambda x, y: 1.0 if x > 0.0 else 0.0,
    "sigmoid": lambda x, y: y * (1.0 - y),
    "tanh": lambda x, y: 1.0 - y * y,
}


def softmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    total = sum(exps) or 1.0
    return [e / total for e in exps]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


# --------------------------------------------------------------------------
# layers
# --------------------------------------------------------------------------
class Dense:
    """Fully connected layer ``a = phi(W x + b)``."""

    def __init__(self, n_in: int, n_out: int, activation: str = "tanh",
                 rng: Optional[random.Random] = None):
        if activation not in _ACT:
            raise ValueError(f"unknown activation {activation!r}")
        self.n_in, self.n_out = n_in, n_out
        self.activation = activation
        rng = rng or random.Random(0)
        scale = math.sqrt(2.0 / max(1, n_in)) if activation == "relu" else math.sqrt(1.0 / max(1, n_in))
        self.W: List[List[float]] = [[rng.gauss(0.0, scale) for _ in range(n_in)]
                                     for _ in range(n_out)]
        self.b: List[float] = [0.0] * n_out
        self.grad_W: List[List[float]] = [[0.0] * n_in for _ in range(n_out)]
        self.grad_b: List[float] = [0.0] * n_out
        self._cache: Optional[Tuple[List[float], List[float], List[float]]] = None

    def forward(self, x: Sequence[float]) -> List[float]:
        z = [_dot(w, x) + b for w, b in zip(self.W, self.b)]
        phi = _ACT[self.activation]
        a = [phi(v) for v in z]
        self._cache = (list(x), z, a)
        return a

    def backward(self, grad_a: Sequence[float]) -> List[float]:
        if self._cache is None:
            raise RuntimeError("backward called before forward")
        x, z, a = self._cache
        deriv = _ACT_DERIV[self.activation]
        grad_z = [g * deriv(zi, ai) for g, zi, ai in zip(grad_a, z, a)]
        self.grad_W = [[gz * xi for xi in x] for gz in grad_z]
        self.grad_b = list(grad_z)
        return [_dot([self.W[i][j] for i in range(self.n_out)], grad_z)
                for j in range(self.n_in)]

    # -- parameters --------------------------------------------------------
    def parameters(self) -> List[float]:
        flat = [v for row in self.W for v in row]
        flat.extend(self.b)
        return flat

    def gradients(self) -> List[float]:
        flat = [v for row in self.grad_W for v in row]
        flat.extend(self.grad_b)
        return flat

    def set_parameters(self, flat: Sequence[float]):
        idx = 0
        for i in range(self.n_out):
            for j in range(self.n_in):
                self.W[i][j] = float(flat[idx]); idx += 1
        for i in range(self.n_out):
            self.b[i] = float(flat[idx]); idx += 1
        return self

    @property
    def n_parameters(self) -> int:
        return self.n_in * self.n_out + self.n_out

    def __repr__(self):
        return f"Dense({self.n_in}->{self.n_out}, act={self.activation!r})"


class Activation:
    """Standalone elementwise activation layer."""

    def __init__(self, name: str = "tanh"):
        self.name = name
        self._cache = None

    def forward(self, x):
        y = [_ACT[self.name](v) for v in x]
        self._cache = (list(x), y)
        return y

    def backward(self, grad):
        x, y = self._cache
        deriv = _ACT_DERIV[self.name]
        return [g * deriv(xi, yi) for g, xi, yi in zip(grad, x, y)]

    def parameters(self): return []
    def gradients(self): return []
    def set_parameters(self, flat): return self

    @property
    def n_parameters(self): return 0

    def __repr__(self):
        return f"Activation({self.name!r})"


# --------------------------------------------------------------------------
# losses
# --------------------------------------------------------------------------
def mse_loss(prediction: Sequence[float], target: Sequence[float]) -> float:
    return float(sum((p - t) ** 2 for p, t in zip(prediction, target)) / max(1, len(target)))


def mse_grad(prediction: Sequence[float], target: Sequence[float]) -> List[float]:
    n = max(1, len(target))
    return [2.0 * (p - t) / n for p, t in zip(prediction, target)]


def cross_entropy_loss(probs: Sequence[float], target_index: int) -> float:
    p = max(1e-12, probs[target_index])
    return float(-math.log(p))


def cross_entropy_grad(probs: Sequence[float], target_index: int) -> List[float]:
    grad = list(probs)
    grad[target_index] -= 1.0
    return grad


# --------------------------------------------------------------------------
# MLP
# --------------------------------------------------------------------------
class MLP:
    """A stack of :class:`Dense` layers."""

    def __init__(self, sizes: Sequence[int], activation: str = "tanh",
                 output_activation: str = "linear", seed: Optional[int] = None):
        if len(sizes) < 2:
            raise ValueError("need at least input and output size")
        rng = random.Random(seed) if seed is not None else random.Random()
        self.layers: List[Dense] = []
        for i in range(len(sizes) - 1):
            act = output_activation if i == len(sizes) - 2 else activation
            self.layers.append(Dense(sizes[i], sizes[i + 1], act, rng))
        self.sizes = list(sizes)

    # -- forward / backward ------------------------------------------------
    def forward(self, x: Sequence[float]) -> List[float]:
        out = list(x)
        for layer in self.layers:
            out = layer.forward(out)
        return out

    __call__ = forward

    def backward(self, grad_out: Sequence[float]) -> List[float]:
        grad = list(grad_out)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    # -- parameters --------------------------------------------------------
    def get_parameters(self) -> List[float]:
        flat: List[float] = []
        for layer in self.layers:
            flat.extend(layer.parameters())
        return flat

    def set_parameters(self, flat: Sequence[float]):
        idx = 0
        for layer in self.layers:
            n = layer.n_parameters
            layer.set_parameters(flat[idx:idx + n])
            idx += n
        return self

    def get_gradients(self) -> List[float]:
        flat: List[float] = []
        for layer in self.layers:
            flat.extend(layer.gradients())
        return flat

    @property
    def n_parameters(self) -> int:
        return sum(layer.n_parameters for layer in self.layers)

    # -- training ----------------------------------------------------------
    def fit(self, X: Sequence[Sequence[float]], Y: Sequence[Sequence[float]],
            epochs: int = 100, lr: float = 0.05, batch_size: Optional[int] = None,
            optimizer=None, shuffle: bool = True, seed: int = 0,
            verbose: bool = False) -> List[float]:
        from .optimizers import SGD
        optimizer = optimizer or SGD(lr)
        rng = random.Random(seed)
        n = len(X)
        batch_size = batch_size or n
        history: List[float] = []
        for epoch in range(epochs):
            order = list(range(n))
            if shuffle:
                rng.shuffle(order)
            epoch_loss = 0.0
            batches = 0
            for start in range(0, n, batch_size):
                idxs = order[start:start + batch_size]
                grad_acc = [0.0] * self.n_parameters
                for i in idxs:
                    pred = self.forward(X[i])
                    loss = mse_loss(pred, Y[i])
                    epoch_loss += loss
                    grad = self.backward(mse_grad(pred, Y[i]))
                    grads = self.get_gradients()
                    for k, g in enumerate(grads):
                        grad_acc[k] += g
                grad_acc = [g / max(1, len(idxs)) for g in grad_acc]
                new_params = optimizer.step(self.get_parameters(), grad_acc)
                self.set_parameters(new_params)
                batches += 1
            avg = epoch_loss / max(1, n)
            history.append(avg)
            if verbose and epoch % max(1, epochs // 10) == 0:
                print(f"epoch {epoch:4d}  loss {avg:.6f}")
        return history

    def predict(self, X: Sequence[Sequence[float]]) -> List[List[float]]:
        return [self.forward(x) for x in X]

    def __repr__(self):
        return f"MLP(sizes={self.sizes}, params={self.n_parameters})"
    