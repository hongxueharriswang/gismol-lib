# ============================================================
# FILE: gismol/dynamical/stability.py
# ============================================================
"""Lyapunov-based stability analysis (Section 4.2)."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from ..core.dynamics import HybridDynamicalSystem
from ..core.expression import lookup

__all__ = [
    "LyapunovFunction", "StabilityResult", "LyapunovAnalysis",
    "verify_decrease_condition", "estimate_region_of_attraction",
    "quadratic_lyapunov",
]


# --------------------------------------------------------------------------
class LyapunovFunction:
    """A candidate Lyapunov function ``V : S -> R_{>=0}``."""

    def __init__(self, V: Callable[[Any], float], name: str = "V",
                 gradient: Optional[Callable[[Any], List[float]]] = None,
                 offset: float = 0.0):
        self.V = V
        self.name = name
        self.gradient = gradient
        self.offset = float(offset)

    def value(self, state) -> float:
        return float(self.V(state)) - self.offset

    def lie_derivative(self, system: HybridDynamicalSystem, state,
                       action=None, theta=None, eps: float = 1e-6) -> float:
        """Numerical ``dV/dt`` along the vector field of *system*."""
        if system.flow is None:
            return 0.0
        d = system.flow.derivative(state, action, theta)
        base = self.value(state)
        shifted = state.copy() if hasattr(state, "copy") else dict(state)
        for var, delta in zip(system.flow.variables, d):
            try:
                shifted[var] = float(shifted.get(var, 0.0)) + eps * delta
            except (TypeError, ValueError):
                return 0.0
        return (self.value(shifted) - base) / eps

    def __call__(self, state) -> float:
        return self.value(state)

    def __repr__(self):
        return f"LyapunovFunction({self.name!r})"


def quadratic_lyapunov(center: Mapping[str, float],
                       weights: Optional[Mapping[str, float]] = None,
                       name: str = "quadratic") -> LyapunovFunction:
    """``V(s) = sum_i w_i (s_i - c_i)^2``."""
    weights = dict(weights or {})

    def V(state):
        total = 0.0
        for key, target in center.items():
            try:
                value = float(lookup(state, key))
            except (KeyError, TypeError, ValueError):
                continue
            total += weights.get(key, 1.0) * (value - target) ** 2
        return total

    return LyapunovFunction(V, name=name)


# --------------------------------------------------------------------------
class StabilityResult:
    """Outcome of a Lyapunov analysis."""

    def __init__(self, stable: bool, asymptotic: bool = False, exponential: bool = False,
                 rate: float = 0.0, margin: float = 0.0, samples: int = 0,
                 worst: float = 0.0, notes: Optional[List[str]] = None,
                 admissible_fraction: float = 0.0):
        self.stable = stable
        self.asymptotic = asymptotic
        self.exponential = exponential
        self.rate = float(rate)
        self.margin = float(margin)
        self.samples = samples
        self.worst = float(worst)
        self.notes = list(notes or [])
        self.admissible_fraction = float(admissible_fraction)

    def to_dict(self):
        return self.__dict__.copy()

    def __repr__(self):
        kind = ("exponentially stable" if self.exponential else
                "asymptotically stable" if self.asymptotic else
                "stable" if self.stable else "unstable")
        return f"StabilityResult({kind}, rate={self.rate:.4f}, n={self.samples})"


# --------------------------------------------------------------------------
class LyapunovAnalysis:
    """Verify Theorem 4.2: stability inside the constraint-admissible region."""

    def __init__(self, system: HybridDynamicalSystem):
        self.system = system

    def admissible_region(self, states: Sequence[Any]) -> List[Any]:
        return [s for s in states if self.system.constraints.satisfies_identity(s)]

    def analyse(self, V: LyapunovFunction, states: Sequence[Any],
                action=None, theta=None, alpha: float = 0.0) -> StabilityResult:
        admissible = self.admissible_region(states)
        notes: List[str] = []
        if not admissible:
            return StabilityResult(False, notes=["empty admissible region"])

        worst = -float("inf")
        violations = 0
        for state in admissible:
            v = V.value(state)
            if v < -1e-9:
                violations += 1
                notes.append("V is negative inside the admissible region")
                break
            d = V.lie_derivative(self.system, state, action, theta)
            worst = max(worst, d)
            if d > 1e-9:
                violations += 1

        stable = violations == 0 and worst <= 1e-9
        asymptotic = violations == 0 and worst < -1e-9
        exponential = asymptotic and alpha > 0.0
        rate = 0.0
        if asymptotic:
            # estimate the decay rate from the sampled trajectories
            ratios = []
            for state in admissible:
                v = V.value(state)
                if v > 1e-9:
                    d = V.lie_derivative(self.system, state, action, theta)
                    ratios.append(-d / v)
            if ratios:
                rate = float(sum(ratios) / len(ratios))
        return StabilityResult(
            stable=stable, asymptotic=asymptotic, exponential=exponential,
            rate=rate, margin=-worst, samples=len(admissible), worst=worst,
            notes=notes, admissible_fraction=len(admissible) / max(1, len(states)),
        )

    def estimate_decay_rate(self, V: LyapunovFunction, states: Sequence[Any],
                            action=None, theta=None) -> float:
        ratios = []
        for state in self.admissible_region(states):
            v = V.value(state)
            if v > 1e-9:
                d = V.lie_derivative(self.system, state, action, theta)
                ratios.append(-d / v)
        return float(sum(ratios) / len(ratios)) if ratios else 0.0


# --------------------------------------------------------------------------
def verify_decrease_condition(system: HybridDynamicalSystem, V: LyapunovFunction,
                              states: Sequence[Any], alpha: float = 0.0,
                              action=None, theta=None) -> bool:
    """Check ``dV/dt <= -alpha V`` on every sampled admissible state."""
    for state in states:
        if not system.constraints.satisfies_identity(state):
            continue
        d = V.lie_derivative(system, state, action, theta)
        if d > -alpha * V.value(state) + 1e-9:
            return False
    return True


def estimate_region_of_attraction(system: HybridDynamicalSystem, V: LyapunovFunction,
                                  states: Sequence[Any], levels: int = 20,
                                  action=None, theta=None) -> float:
    """Return the largest sampled level set ``{V <= c}`` inside the admissible
    region on which ``dV/dt < 0``."""
    values = []
    for state in states:
        if not system.constraints.satisfies_identity(state):
            continue
        d = V.lie_derivative(system, state, action, theta)
        values.append((V.value(state), d))
    if not values:
        return 0.0
    values.sort()
    best = 0.0
    for v, d in values:
        if d < 0.0:
            best = v
        else:
            break
    return best
