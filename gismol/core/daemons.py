# ============================================================
# FILE: gismol/core/daemons.py
# ============================================================
"""Constraint daemons ``D`` — continuous monitoring processes (Definition 2.10).

A daemon observes the state (or a whole state trajectory) and emits corrective
actions, keeping the system inside the constraint-admissible region.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from .constraints import ConstraintSet, IdentityConstraint

__all__ = [
    "Daemon", "InvariantDaemon", "BoundaryDaemon", "RateDaemon",
    "SafetyDaemon", "DaemonNetwork", "CorrectiveAction",
]


class CorrectiveAction:
    """An action emitted by a daemon."""

    def __init__(self, name: str, payload: Any = None, daemon: str = "",
                 severity: float = 0.0, message: str = ""):
        self.name = name
        self.payload = payload
        self.daemon = daemon
        self.severity = float(severity)
        self.message = message

    def __repr__(self):
        return (f"CorrectiveAction({self.name!r}, severity={self.severity:.3f}, "
                f"daemon={self.daemon!r})")

    def to_dict(self):
        return {"name": self.name, "daemon": self.daemon,
                "severity": self.severity, "message": self.message,
                "payload": self.payload}


# --------------------------------------------------------------------------
# base daemon
# --------------------------------------------------------------------------
class Daemon:
    """Base class for constraint daemons.

    Subclasses implement :meth:`observe`, which returns ``None`` or a
    :class:`CorrectiveAction`.
    """

    def __init__(self, name: str, description: str = "", enabled: bool = True):
        self.name = name
        self.description = description
        self.enabled = enabled
        self.history: List[CorrectiveAction] = []
        self._prev_state = None
        self._prev_time = None

    # -- monitoring --------------------------------------------------------
    def observe(self, state, t: float = 0.0) -> Optional[CorrectiveAction]:
        """Inspect the current state; return a corrective action if needed."""
        raise NotImplementedError

    def monitor(self, trajectory) -> List[CorrectiveAction]:
        """Monitor a whole trajectory (``S^T -> A_act``, Definition 2.10)."""
        actions: List[CorrectiveAction] = []
        for t, state in trajectory:
            act = self.observe(state, t)
            if act is not None:
                actions.append(act)
        return actions

    def reset(self):
        self.history.clear()
        self._prev_state = None
        self._prev_time = None

    # -- internal ----------------------------------------------------------
    def _emit(self, action: CorrectiveAction) -> CorrectiveAction:
        self.history.append(action)
        return action

    def __repr__(self):
        return f"{type(self).__name__}({self.name!r})"


# --------------------------------------------------------------------------
# invariant daemon
# --------------------------------------------------------------------------
class InvariantDaemon(Daemon):
    """Monitors a set of identity constraints and reports violations."""

    def __init__(self, name: str = "invariant", constraints=None,
                 on_violation: Optional[Callable[[Any, List[IdentityConstraint]], Any]] = None,
                 description: str = "monitors identity constraints"):
        super().__init__(name, description)
        if isinstance(constraints, ConstraintSet):
            constraints = constraints.identity
        self.constraints: List[IdentityConstraint] = list(constraints or [])
        self.on_violation = on_violation

    def attach(self, constraints):
        if isinstance(constraints, ConstraintSet):
            constraints = constraints.identity
        self.constraints.extend(constraints)
        return self

    def observe(self, state, t: float = 0.0) -> Optional[CorrectiveAction]:
        if not self.enabled or not self.constraints:
            return None
        broken = [c for c in self.constraints if not c.satisfied(state)]
        if not broken:
            return None
        severity = float(sum(c.violation(state) for c in broken))
        payload = None
        if self.on_violation is not None:
            payload = self.on_violation(state, broken)
        return self._emit(CorrectiveAction(
            "restore_invariants", payload=payload, daemon=self.name,
            severity=severity,
            message="violated: " + ", ".join(c.name for c in broken),
        ))


# --------------------------------------------------------------------------
# boundary daemon
# --------------------------------------------------------------------------
class BoundaryDaemon(Daemon):
    """Projects states back into a box-shaped admissible region.

    ``bounds`` maps attribute names to ``(low, high)`` pairs.  The daemon emits
    a ``"project"`` action whose payload is the clipped state.
    """

    def __init__(self, bounds: Dict[str, Sequence[float]], name: str = "boundary",
                 description: str = "keeps state inside admissible box"):
        super().__init__(name, description)
        self.bounds = {k: (float(v[0]), float(v[1])) for k, v in bounds.items()}

    def project(self, state):
        from .expression import lookup
        out = state.copy() if hasattr(state, "copy") else dict(state)
        for key, (low, high) in self.bounds.items():
            try:
                value = float(lookup(state, key))
            except KeyError:
                continue
            clipped = min(max(value, low), high)
            if clipped != value:
                if hasattr(out, "set_path"):
                    out.set_path(key, clipped)
                else:
                    out[key] = clipped
        return out

    def observe(self, state, t: float = 0.0) -> Optional[CorrectiveAction]:
        if not self.enabled:
            return None
        from .expression import lookup
        worst = 0.0
        for key, (low, high) in self.bounds.items():
            try:
                value = float(lookup(state, key))
            except (KeyError, TypeError, ValueError):
                continue
            if value < low:
                worst = max(worst, low - value)
            elif value > high:
                worst = max(worst, value - high)
        if worst <= 0.0:
            return None
        return self._emit(CorrectiveAction(
            "project", payload=self.project(state), daemon=self.name,
            severity=worst, message="state outside admissible box",
        ))


# --------------------------------------------------------------------------
# rate daemon
# --------------------------------------------------------------------------
class RateDaemon(Daemon):
    """Monitors the rate of change of a scalar attribute."""

    def __init__(self, attribute: str, max_rate: float, name: str = "rate",
                 description: str = "limits rate of change"):
        super().__init__(name, description)
        self.attribute = attribute
        self.max_rate = float(max_rate)

    def observe(self, state, t: float = 0.0) -> Optional[CorrectiveAction]:
        if not self.enabled:
            return None
        from .expression import lookup
        try:
            value = float(lookup(state, self.attribute))
        except (KeyError, TypeError, ValueError):
            return None
        if self._prev_state is not None and self._prev_time is not None:
            try:
                prev = float(lookup(self._prev_state, self.attribute))
            except (KeyError, TypeError, ValueError):
                prev = value
            dt = (t - self._prev_time) or 1.0
            rate = abs(value - prev) / abs(dt)
            self._prev_state, self._prev_time = state, t
            if rate > self.max_rate:
                return self._emit(CorrectiveAction(
                    "limit_rate", payload={"rate": rate, "limit": self.max_rate},
                    daemon=self.name, severity=rate - self.max_rate,
                    message=f"rate {rate:.3f} exceeds {self.max_rate:.3f}",
                ))
            return None
        self._prev_state, self._prev_time = state, t
        return None


# --------------------------------------------------------------------------
# safety daemon
# --------------------------------------------------------------------------
class SafetyDaemon(Daemon):
    """Monitors a hazard function ``h(state) >= 0`` (e.g. a control barrier
    function) and triggers an intervention when the margin becomes small."""

    def __init__(self, hazard: Callable[[Any], float], name: str = "safety",
                 threshold: float = 0.0, intervention: Any = "safe_stop",
                 description: str = "hazard monitoring"):
        super().__init__(name, description)
        self.hazard = hazard
        self.threshold = float(threshold)
        self.intervention = intervention

    def observe(self, state, t: float = 0.0) -> Optional[CorrectiveAction]:
        if not self.enabled:
            return None
        try:
            h = float(self.hazard(state))
        except Exception:
            return None
        if h < self.threshold:
            return self._emit(CorrectiveAction(
                "intervene", payload=self.intervention, daemon=self.name,
                severity=self.threshold - h,
                message=f"hazard margin {h:.4f} below threshold {self.threshold:.4f}",
            ))
        return None


# --------------------------------------------------------------------------
# daemon network
# --------------------------------------------------------------------------
class DaemonNetwork(Daemon):
    """Composition of several daemons; the highest-severity action wins."""

    def __init__(self, daemons: Iterable[Daemon] = (), name: str = "network",
                 description: str = "composite daemon"):
        super().__init__(name, description)
        self.daemons: List[Daemon] = list(daemons)

    def add(self, daemon: Daemon) -> Daemon:
        self.daemons.append(daemon)
        return daemon

    def remove(self, name: str):
        self.daemons = [d for d in self.daemons if d.name != name]

    def __iter__(self):
        return iter(self.daemons)

    def __len__(self):
        return len(self.daemons)

    def observe(self, state, t: float = 0.0) -> Optional[CorrectiveAction]:
        best: Optional[CorrectiveAction] = None
        for daemon in self.daemons:
            act = daemon.observe(state, t)
            if act is not None and (best is None or act.severity > best.severity):
                best = act
        if best is not None:
            self.history.append(best)
        return best

    def observe_all(self, state, t: float = 0.0) -> List[CorrectiveAction]:
        return [a for a in (d.observe(state, t) for d in self.daemons) if a is not None]

    def reset(self):
        super().reset()
        for d in self.daemons:
            d.reset()

    def by_name(self, name: str) -> Optional[Daemon]:
        for d in self.daemons:
            if d.name == name:
                return d
        return None
    