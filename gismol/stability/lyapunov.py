# ============================================================
# FILE: gismol/stability/lyapunov.py
# ============================================================
"""Lyapunov certificates specialised to COH constraint semantics.

Theorem 4.2: if the goal constraints define a Lyapunov function ``V`` that
decreases along trajectories inside the admissible region ``R``, then the COH
system is asymptotically stable within ``R``.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from ..core.coh_object import COHObject
from ..dynamical.stability import LyapunovFunction, StabilityResult
from ..core.expression import lookup

__all__ = [
    "LyapunovFunction", "LyapunovCertificate", "ConstraintStabilityAnalyzer",
    "goal_lyapunov", "lyapunov_from_constraints",
]


# --------------------------------------------------------------------------
def goal_lyapunov(coh: COHObject, offset: Optional[float] = None,
                  name: str = "V_goal") -> LyapunovFunction:
    """Build ``V(s) = -sum_j w_j g_j(s) + offset`` from the goal constraints.

    The offset (defaulting to the maximum sampled value) shifts ``V`` so that
    it is non-negative on the admissible region.
    """
    def raw(state):
        return -coh.constraints.total_goal(state, None, 0.0)

    if offset is None:
        offset = 0.0
    return LyapunovFunction(lambda s: raw(s) + offset, name=name, offset=offset)


def lyapunov_from_constraints(coh: COHObject,
                              samples: Optional[Sequence[Any]] = None,
                              name: str = "V_constraints") -> LyapunovFunction:
    """``V(s) = sum_i violation_i(s)`` — zero exactly on the admissible region.

    This is the natural Lyapunov function for constraint restoration and the
    object that daemons drive to zero.
    """
    def V(state):
        return float(sum(c.violation(state) for c in coh.I))

    return LyapunovFunction(V, name=name)


# --------------------------------------------------------------------------
class LyapunovCertificate:
    """The result of a constraint-stability analysis."""

    def __init__(self, stable: bool, asymptotic: bool, rate: float = 0.0,
                 admissible_fraction: float = 0.0, samples: int = 0,
                 worst_derivative: float = 0.0, daemons_enforcing: bool = False,
                 notes: Optional[List[str]] = None):
        self.stable = stable
        self.asymptotic = asymptotic
        self.rate = float(rate)
        self.admissible_fraction = float(admissible_fraction)
        self.samples = samples
        self.worst_derivative = float(worst_derivative)
        self.daemons_enforcing = daemons_enforcing
        self.notes = list(notes or [])

    def to_dict(self):
        return self.__dict__.copy()

    def __repr__(self):
        verdict = "asymptotically stable" if self.asymptotic else (
            "stable" if self.stable else "unstable")
        return (f"LyapunovCertificate({verdict}, rate={self.rate:.4f}, "
                f"daemons={self.daemons_enforcing})")


# --------------------------------------------------------------------------
class ConstraintStabilityAnalyzer:
    """Applies Theorem 4.2 to a :class:`COHObject`."""

    def __init__(self, coh: COHObject):
        self.coh = coh

    def admissible_region(self, states: Sequence[Any]) -> List[Any]:
        return [s for s in states if self.coh.constraints.satisfies_identity(s)]

    def analyze(self, V: Optional[LyapunovFunction] = None,
                states: Optional[Sequence[Any]] = None,
                dt: float = 1e-3) -> LyapunovCertificate:
        states = list(states or [self.coh.initial_state()])
        if V is None:
            V = lyapunov_from_constraints(self.coh, states)

        admissible = self.admissible_region(states)
        notes: List[str] = []
        if not admissible:
            return LyapunovCertificate(False, False, notes=["empty admissible region"])

        system = self.coh.to_hybrid_system()
        worst = -float("inf")
        ratios: List[float] = []
        for state in admissible:
            v = V.value(state)
            d = V.lie_derivative(system, state)
            worst = max(worst, d)
            if v > 1e-9:
                ratios.append(-d / v)

        stable = worst <= 1e-9
        asymptotic = worst < -1e-9
        rate = float(sum(ratios) / len(ratios)) if ratios else 0.0
        daemons_enforcing = len(self.coh.D) > 0
        if daemons_enforcing and not asymptotic:
            notes.append("daemons will correct boundary violations, preserving "
                         "admissibility (Section 4.2)")
        if not stable:
            notes.append("V increases on some admissible state; check goal signs")

        return LyapunovCertificate(
            stable=stable, asymptotic=asymptotic, rate=rate,
            admissible_fraction=len(admissible) / max(1, len(states)),
            samples=len(admissible), worst_derivative=worst,
            daemons_enforcing=daemons_enforcing, notes=notes,
        )

    def exponential_bound(self, V: LyapunovFunction, states: Sequence[Any]) -> float:
        """Estimate the constant ``alpha`` such that ``dV/dt <= -alpha V``."""
        system = self.coh.to_hybrid_system()
        alphas = []
        for state in self.admissible_region(states):
            v = V.value(state)
            if v > 1e-9:
                d = V.lie_derivative(system, state)
                alphas.append(-d / v)
        return float(min(alphas)) if alphas else 0.0

    def __repr__(self):
        return f"ConstraintStabilityAnalyzer({self.coh.name!r})"
    