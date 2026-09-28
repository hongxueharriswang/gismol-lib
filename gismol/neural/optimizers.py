# ============================================================
# FILE: gismol/neural/optimizers.py
# ============================================================
"""First-order optimisers operating on flat parameter vectors."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

__all__ = [
    "Optimizer", "SGD", "Momentum", "AdaGrad", "RMSProp", "Adam",
    "ClippedSGD", "LagrangianOptimizer",
]


class Optimizer:
    def __init__(self, lr: float = 0.01):
        self.lr = float(lr)
        self.steps = 0

    def step(self, params: Sequence[float], grads: Sequence[float]) -> List[float]:
        raise NotImplementedError

    def reset(self):
        self.steps = 0

    def __repr__(self):
        return f"{type(self).__name__}(lr={self.lr})"


class SGD(Optimizer):
    def step(self, params, grads):
        self.steps += 1
        return [p - self.lr * g for p, g in zip(params, grads)]


class ClippedSGD(SGD):
    def __init__(self, lr: float = 0.01, clip: float = 1.0):
        super().__init__(lr)
        self.clip = float(clip)

    def step(self, params, grads):
        clipped = [max(-self.clip, min(self.clip, g)) for g in grads]
        return super().step(params, clipped)


class Momentum(Optimizer):
    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        super().__init__(lr)
        self.momentum = float(momentum)
        self.velocity: Optional[List[float]] = None

    def step(self, params, grads):
        self.steps += 1
        if self.velocity is None:
            self.velocity = [0.0] * len(params)
        self.velocity = [self.momentum * v - self.lr * g
                         for v, g in zip(self.velocity, grads)]
        return [p + v for p, v in zip(params, self.velocity)]

    def reset(self):
        super().reset()
        self.velocity = None


class AdaGrad(Optimizer):
    def __init__(self, lr: float = 0.05, eps: float = 1e-8):
        super().__init__(lr)
        self.eps = float(eps)
        self.accum: Optional[List[float]] = None

    def step(self, params, grads):
        self.steps += 1
        if self.accum is None:
            self.accum = [0.0] * len(params)
        self.accum = [a + g * g for a, g in zip(self.accum, grads)]
        return [p - self.lr * g / (math.sqrt(a) + self.eps)
                for p, g, a in zip(params, grads, self.accum)]

    def reset(self):
        super().reset()
        self.accum = None


class RMSProp(Optimizer):
    def __init__(self, lr: float = 0.01, decay: float = 0.9, eps: float = 1e-8):
        super().__init__(lr)
        self.decay = float(decay)
        self.eps = float(eps)
        self.sq: Optional[List[float]] = None

    def step(self, params, grads):
        self.steps += 1
        if self.sq is None:
            self.sq = [0.0] * len(params)
        self.sq = [self.decay * s + (1 - self.decay) * g * g
                   for s, g in zip(self.sq, grads)]
        return [p - self.lr * g / (math.sqrt(s) + self.eps)
                for p, g, s in zip(params, grads, self.sq)]

    def reset(self):
        super().reset()
        self.sq = None


class Adam(Optimizer):
    def __init__(self, lr: float = 0.01, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8):
        super().__init__(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.m: Optional[List[float]] = None
        self.v: Optional[List[float]] = None

    def step(self, params, grads):
        self.steps += 1
        if self.m is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)
        self.m = [self.beta1 * m + (1 - self.beta1) * g for m, g in zip(self.m, grads)]
        self.v = [self.beta2 * v + (1 - self.beta2) * g * g for v, g in zip(self.v, grads)]
        m_hat = [m / (1 - self.beta1 ** self.steps) for m in self.m]
        v_hat = [v / (1 - self.beta2 ** self.steps) for v in self.v]
        return [p - self.lr * mh / (math.sqrt(vh) + self.eps)
                for p, mh, vh in zip(params, m_hat, v_hat)]

    def reset(self):
        super().reset()
        self.m = self.v = None


class LagrangianOptimizer(Optimizer):
    """Dual ascent on Lagrange multipliers with projection to ``lambda >= 0``.

    Used by :mod:`gismol.neural.constrained_rl` to solve the constrained policy
    optimisation problem of Definition 5.1.
    """

    def __init__(self, lr: float = 0.05, max_multiplier: float = 1e6):
        super().__init__(lr)
        self.max_multiplier = float(max_multiplier)

    def step(self, params, grads):
        """``params`` are multipliers, ``grads`` are the constraint violations."""
        self.steps += 1
        out = []
        for lam, violation in zip(params, grads):
            new = lam + self.lr * violation
            out.append(min(self.max_multiplier, max(0.0, new)))
        return out
    