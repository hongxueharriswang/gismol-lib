# ============================================================
# FILE: gismol/core/daemon_impl.py
# ============================================================
"""Concrete, reusable daemon implementations.

These complement the abstract daemons in :mod:`gismol.core.daemons` with the
specific mechanisms referenced by the COH paper: homeostatic regulation,
projection onto the admissible region, control-barrier supervision and
precision-weighted prediction-error minimisation (the free-energy reading of
``D``).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from .daemons import CorrectiveAction, Daemon
from .expression import lookup

__all__ = [
    "HomeostasisDaemon",
    "ProjectionDaemon",
    "BarrierDaemon",
    "PrecisionWeightedDaemon",
    "EventDaemon",
    "RateLimitDaemon",
]


# --------------------------------------------------------------------------
class HomeostasisDaemon(Daemon):
    """Keeps a scalar attribute near a set-point via proportional correction.

    Mirrors biological homeostatic regulation (Sterling & Laughlin, 2015).
    """

    def __init__(self, attribute: str, setpoint: float, gain: float = 1.0,
                 tolerance: float = 1e-6, name: str = "homeostasis",
                 description: str = "set-point regulation"):
        super().__init__(name, description)
        self.attribute = attribute
        self.setpoint = float(setpoint)
        self.gain = float(gain)
        self.tolerance = float(tolerance)

    def error(self, state) -> float:
        try:
            return float(lookup(state, self.attribute)) - self.setpoint
        except (KeyError, TypeError, ValueError):
            return 0.0

    def observe(self, state, t: float = 0.0) -> Optional[CorrectiveAction]:
        if not self.enabled:
            return None
        err = self.error(state)
        if abs(err) <= self.tolerance:
            return None
        correction = -self.gain * err
        return self._emit(CorrectiveAction(
            "regulate", payload={self.attribute: correction},
            daemon=self.name, severity=abs(err),
            message=f"{self.attribute} off set-point by {err:+.4f}",
        ))


# --------------------------------------------------------------------------
class ProjectionDaemon(Daemon):
    """Projects a state onto the admissible region defined by identity
    constraints, using a caller-supplied projection map."""

    def __init__(self, projection: Callable[[Any], Any], constraints=None,
                 name: str = "projection",
                 description: str = "projection onto feasible region"):
        super().__init__(name, description)
        if constraints is None:
            self.constraints = []
        elif hasattr(constraints, "identity"):
            self.constraints = list(constraints.identity)
        else:
            self.constraints = list(constraints)
        self.projection = projection

    def observe(self, state, t: float = 0.0) -> Optional[CorrectiveAction]:
        if not self.enabled:
            return None
        broken = [c for c in self.constraints if not c.satisfied(state)]
        if not broken:
            return None
        try:
            projected = self.projection(state)
        except Exception as exc:  # pragma: no cover - defensive
            return self._emit(CorrectiveAction(
                "projection_failed", daemon=self.name, severity=float("inf"),
                message=str(exc)))
        return self._emit(CorrectiveAction(
            "project", payload=projected, daemon=self.name,
            severity=float(len(broken)),
            message="projected onto admissible region"))


# --------------------------------------------------------------------------
class BarrierDaemon(Daemon):
    """Control-barrier-function supervisor.

    Given a barrier ``h`` and a decay class ``alpha`` (a non-decreasing
    function with ``alpha(0) = 0``), a state is *safe* when ``h(state) >= 0``
    and the forward-invariance condition ``dh/dt + alpha(h) >= 0`` holds.
    """

    def __init__(self, barrier: Callable[[Any], float],
                 derivative: Optional[Callable[[Any], float]] = None,
                 alpha: Callable[[float], float] = lambda x: x,
                 name: str = "barrier", intervention: Any = "safe_stop",
                 description: str = "control barrier supervision"):
        super().__init__(name, description)
        self.barrier = barrier
        self.derivative = derivative
        self.alpha = alpha
        self.intervention = intervention

    def observe(self, state, t: float = 0.0) -> Optional[CorrectiveAction]:
        if not self.enabled:
            return None
        try:
            h = float(self.barrier(state))
        except Exception:
            return None
        if h < 0.0:
            return self._emit(CorrectiveAction(
                "intervene", payload=self.intervention, daemon=self.name,
                severity=-h, message=f"barrier violated (h={h:.4f})"))
        if self.derivative is not None:
            try:
                dh = float(self.derivative(state))
            except Exception:
                dh = 0.0
            slack = dh + self.alpha(h)
            if slack < 0.0:
                return self._emit(CorrectiveAction(
                    "intervene", payload=self.intervention, daemon=self.name,
                    severity=-slack,
                    message=f"forward-invariance slack {slack:.4f} < 0"))
        return None


# --------------------------------------------------------------------------
class PrecisionWeightedDaemon(Daemon):
    """Daemon implementing precision-weighted prediction-error minimisation.

    This is the free-energy reading of ``D``: given a prediction function and a
    precision (inverse variance) schedule, the daemon emits a correction
    proportional to ``precision * error``.
    """

    def __init__(self, predictor: Callable[[Any], Any], target: Callable[[Any], Any],
                 precision: float = 1.0, name: str = "precision",
                 description: str = "precision-weighted prediction error"):
        super().__init__(name, description)
        self.predictor = predictor
        self.target = target
        self.precision = float(precision)

    def prediction_error(self, state) -> float:
        try:
            pred = self.predictor(state)
            obs = self.target(state)
            if isinstance(pred, (list, tuple)):
                return math.sqrt(sum((float(p) - float(o)) ** 2
                                     for p, o in zip(pred, obs)) / max(1, len(pred)))
            return float(pred) - float(obs)
        except Exception:
            return 0.0

    def observe(self, state, t: float = 0.0) -> Optional[CorrectiveAction]:
        if not self.enabled:
            return None
        err = self.prediction_error(state)
        weighted = self.precision * err
        if abs(weighted) < 1e-9:
            return None
        return self._emit(CorrectiveAction(
            "minimise_free_energy", payload=weighted, daemon=self.name,
            severity=abs(weighted),
            message=f"precision-weighted error {weighted:+.5f}"))


# --------------------------------------------------------------------------
class EventDaemon(Daemon):
    """Fires a fixed action whenever a predicate becomes true (edge triggered)."""

    def __init__(self, predicate: Callable[[Any], bool], action: Any,
                 name: str = "event", description: str = "edge-triggered action"):
        super().__init__(name, description)
        self.predicate = predicate
        self.action = action
        self._was_true = False

    def observe(self, state, t: float = 0.0) -> Optional[CorrectiveAction]:
        if not self.enabled:
            return None
        now = bool(self.predicate(state))
        fired = now and not self._was_true
        self._was_true = now
        if fired:
            return self._emit(CorrectiveAction(
                "event", payload=self.action, daemon=self.name,
                severity=1.0, message="predicate became true"))
        return None

    def reset(self):
        super().reset()
        self._was_true = False


# --------------------------------------------------------------------------
class RateLimitDaemon(Daemon):
    """Alias of :class:`gismol.core.daemons.RateDaemon` with saturation."""

    def __init__(self, attribute: str, max_rate: float, name: str = "rate_limit",
                 description: str = "saturating rate limiter"):
        super().__init__(name, description)
        self.attribute = attribute
        self.max_rate = float(max_rate)
        self._prev = None
        self._prev_t = None

    def observe(self, state, t: float = 0.0) -> Optional[CorrectiveAction]:
        if not self.enabled:
            return None
        try:
            value = float(lookup(state, self.attribute))
        except (KeyError, TypeError, ValueError):
            return None
        if self._prev is None:
            self._prev, self._prev_t = value, t
            return None
        dt = (t - self._prev_t) or 1.0
        rate = (value - self._prev) / dt
        self._prev, self._prev_t = value, t
        if abs(rate) > self.max_rate:
            allowed = math.copysign(self.max_rate, rate)
            return self._emit(CorrectiveAction(
                "saturate",
                payload={self.attribute: self._prev + allowed * dt},
                daemon=self.name, severity=abs(rate) - self.max_rate,
                message=f"rate {rate:.3f} saturated to {allowed:.3f}"))
        return None