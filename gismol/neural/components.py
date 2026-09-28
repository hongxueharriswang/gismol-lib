# ============================================================
# FILE: gismol/neural/components.py
# ============================================================
"""Neural components ``N`` — parameterised adaptive functions (Definition 2.5)."""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from .nn import MLP, softmax

__all__ = [
    "NeuralComponent", "MLPComponent", "NeuralComponentSet",
    "PerceptionComponent", "PredictorComponent", "PolicyComponent", "ValueComponent",
]


class NeuralComponent:
    """``f_theta : X -> Y`` with ``theta`` in a parameter space ``Theta``."""

    def __init__(self, name: str, fn: Optional[Callable] = None,
                 input_dim: Optional[int] = None, output_dim: Optional[int] = None,
                 description: str = ""):
        self.name = name
        self.fn = fn
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.description = description
        self.updates = 0

    def __call__(self, x):
        if self.fn is None:
            raise NotImplementedError(f"neural component {self.name!r} has no function")
        return self.fn(x)

    # -- parameter interface ----------------------------------------------
    @property
    def theta(self) -> List[float]:
        return []

    def set_theta(self, theta: Sequence[float]):
        return self

    def n_parameters(self) -> int:
        return len(self.theta)

    def update_parameters(self, new_theta: Sequence[float]):
        self.set_theta(new_theta)
        self.updates += 1
        return self

    def __repr__(self):
        return f"{type(self).__name__}({self.name!r})"


# --------------------------------------------------------------------------
class MLPComponent(NeuralComponent):
    """A neural component backed by a :class:`~gismol.neural.nn.MLP`."""

    def __init__(self, name: str, layer_sizes: Sequence[int], activation: str = "tanh",
                 output_activation: str = "linear", seed: Optional[int] = None,
                 description: str = ""):
        super().__init__(name, input_dim=layer_sizes[0], output_dim=layer_sizes[-1],
                         description=description)
        self.mlp = MLP(layer_sizes, activation, output_activation, seed)
        self.fn = self.mlp.forward

    @property
    def theta(self) -> List[float]:
        return self.mlp.get_parameters()

    def set_theta(self, theta):
        self.mlp.set_parameters(theta)
        return self

    def fit(self, X, Y, **kwargs) -> List[float]:
        return self.mlp.fit(X, Y, **kwargs)

    def predict(self, X):
        return self.mlp.predict(X)

    def __repr__(self):
        return f"MLPComponent({self.name!r}, sizes={self.mlp.sizes})"


class PerceptionComponent(MLPComponent):
    """Maps raw observations to a latent representation."""

    def __init__(self, name: str = "perception", input_dim: int = 8,
                 latent_dim: int = 4, seed: Optional[int] = None):
        super().__init__(name, [input_dim, max(2, input_dim), latent_dim],
                         activation="tanh", seed=seed,
                         description="perception / feature extraction")


class PredictorComponent(MLPComponent):
    """Predicts the next observation or attribute value."""

    def __init__(self, name: str = "predictor", input_dim: int = 8,
                 output_dim: int = 1, hidden: int = 16, seed: Optional[int] = None):
        super().__init__(name, [input_dim, hidden, output_dim],
                         activation="tanh", seed=seed,
                         description="predictive model  p(s_{t+1} | s_t, a_t)")


class ValueComponent(MLPComponent):
    """Approximates a value function ``V(s)``."""

    def __init__(self, name: str = "value", input_dim: int = 8,
                 hidden: int = 16, seed: Optional[int] = None):
        super().__init__(name, [input_dim, hidden, 1], activation="tanh",
                         seed=seed, description="value function V(s)")

    def value(self, x) -> float:
        return float(self.forward(x)[0])


class PolicyComponent(NeuralComponent):
    """A softmax policy over a discrete action set."""

    def __init__(self, name: str = "policy", input_dim: int = 8,
                 n_actions: int = 4, hidden: int = 16, seed: Optional[int] = None):
        super().__init__(name, input_dim=input_dim, output_dim=n_actions,
                         description="policy pi(a|s)")
        self.n_actions = n_actions
        self.mlp = MLP([input_dim, hidden, n_actions], "tanh", "linear", seed)
        self.fn = self._probs

    def _probs(self, x) -> List[float]:
        return softmax(self.mlp.forward(x))

    def sample(self, x, rng: Optional[random.Random] = None) -> int:
        rng = rng or random
        probs = self._probs(x)
        r = rng.random()
        acc = 0.0
        for i, p in enumerate(probs):
            acc += p
            if r <= acc:
                return i
        return len(probs) - 1

    def log_prob(self, x, action: int) -> float:
        import math
        return math.log(max(1e-12, self._probs(x)[action]))

    @property
    def theta(self):
        return self.mlp.get_parameters()

    def set_theta(self, theta):
        self.mlp.set_parameters(theta)
        return self

    def __repr__(self):
        return f"PolicyComponent({self.name!r}, actions={self.n_actions})"


# --------------------------------------------------------------------------
class NeuralComponentSet(dict):
    """Dictionary-like container for the ``N`` component of a COH object."""

    def add(self, component: NeuralComponent, name: Optional[str] = None):
        self[name or component.name] = component
        return component

    def theta(self) -> List[float]:
        flat: List[float] = []
        for c in self.values():
            flat.extend(c.theta)
        return flat

    def set_theta(self, theta: Sequence[float]):
        idx = 0
        for c in self.values():
            n = len(c.theta)
            c.set_theta(theta[idx:idx + n])
            idx += n
        return self

    def n_parameters(self) -> int:
        return sum(len(c.theta) for c in self.values())

    def trainable(self):
        return {k: v for k, v in self.items() if v.n_parameters()}

    def __repr__(self):
        return f"NeuralComponentSet({list(self)}, params={self.n_parameters()})"
    