# ============================================================
# FILE: gismol/core/constraints.py
# ============================================================
"""Constraint types for Constrained Object Hierarchies.

Implements components ``I`` (identity), ``T`` (trigger) and ``G`` (goal) of the
COH 9-tuple together with the :class:`ConstraintSet` container that the
:class:`~gismol.core.coh_object.COHObject` uses.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from . import expression as E
from .exceptions import ConstraintViolationError, UnsatisfiableConstraintError

__all__ = [
    "Constraint", "IdentityConstraint", "TriggerConstraint", "GoalConstraint",
    "ConstraintSet", "as_predicate",
]


def as_predicate(spec) -> Callable[[Any], bool]:
    """Coerce *spec* into a predicate ``state -> bool``."""
    if isinstance(spec, E.Expr):
        return lambda s, _e=spec: bool(_e.evaluate(s))
    if callable(spec):
        return lambda s, _f=spec: bool(_f(s))
    raise TypeError(f"cannot use {spec!r} as a predicate")


# --------------------------------------------------------------------------
# base
# --------------------------------------------------------------------------
class Constraint:
    """Common behaviour for all COH constraints."""

    kind = "constraint"

    def __init__(self, name: str, description: str = "", weight: float = 1.0):
        self.name = name
        self.description = description
        self.weight = float(weight)

    # -- to be provided by subclasses --------------------------------------
    def satisfied(self, state, **kwargs) -> bool:
        raise NotImplementedError

    def violation(self, state, **kwargs) -> float:
        """Return 0.0 when satisfied, a positive magnitude otherwise."""
        return 0.0 if self.satisfied(state, **kwargs) else 1.0

    # -- helpers -----------------------------------------------------------
    def check(self, state, **kwargs) -> None:
        if not self.satisfied(state, **kwargs):
            raise ConstraintViolationError(self, state)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "name": self.name,
                "description": self.description, "weight": self.weight}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"


# --------------------------------------------------------------------------
# identity constraints (I)
# --------------------------------------------------------------------------
class IdentityConstraint(Constraint):
    """Invariant predicate ``phi : S -> B`` (Definition 2.7).

    ``predicate`` may be a callable, an :class:`~gismol.core.expression.Expr`,
    or a string that is parsed with :func:`expression.parse_expression`.
    ``margin`` optionally gives a scalar function returning the signed distance
    to the boundary, used by daemons for projection and by the violation score.
    """

    kind = "identity"

    def __init__(self, name, predicate, description="", weight=1.0,
                 margin: Optional[Callable[[Any], float]] = None):
        super().__init__(name, description, weight)
        if isinstance(predicate, str):
            predicate = E.parse_expression(predicate)
        self.expression = predicate if isinstance(predicate, E.Expr) else None
        self.predicate = as_predicate(predicate)
        self.margin = margin

    def satisfied(self, state, **kwargs) -> bool:
        return self.predicate(state)

    def violation(self, state, **kwargs) -> float:
        if self.predicate(state):
            return 0.0
        if self.margin is not None:
            try:
                return abs(float(self.margin(state))) * self.weight
            except Exception:
                pass
        return float(self.weight)

    def to_dict(self):
        data = super().to_dict()
        data["expression"] = self.expression.to_dict() if self.expression else None
        return data


# --------------------------------------------------------------------------
# trigger constraints (T)
# --------------------------------------------------------------------------
class TriggerConstraint(Constraint):
    """Event-condition-action rule ``(e, c, a)`` (Definition 2.8).

    Parameters
    ----------
    event:
        Event name this rule listens to, or ``"*"`` for any event.
    condition:
        Predicate over the state that must hold for the rule to fire.
    action:
        Either an action name (string), a callable ``state -> action`` or a
        callable ``state -> state`` for direct state rewrites.
    preserves:
        Optional predicate that the *resulting* state must satisfy.  When
        omitted the object's identity constraints are used (see
        :meth:`COHObject.soundness_report`).
    """

    kind = "trigger"

    def __init__(self, name, event, condition, action, description="",
                 weight=1.0, preserves=None, priority: int = 0):
        super().__init__(name, description, weight)
        self.event = event
        self.condition_expr = condition if isinstance(condition, E.Expr) else None
        self.condition = as_predicate(
            E.parse_expression(condition) if isinstance(condition, str) else condition
        )
        self.action = action
        self.preserves = None if preserves is None else as_predicate(preserves)
        self.priority = int(priority)

    # -- matching ----------------------------------------------------------
    def matches(self, event: Optional[str], state) -> bool:
        if self.event not in ("*", None) and event != self.event:
            return False
        return self.condition(state)

    def fire(self, state) -> Any:
        """Return the action to execute (a name or a value)."""
        if callable(self.action):
            return self.action(state)
        return self.action

    def satisfied(self, state, event=None, **kwargs) -> bool:
        """A trigger is 'satisfied' when it does not apply, or when it fires and
        the resulting condition is consistent."""
        return not self.matches(event, state)

    def violation(self, state, event=None, **kwargs) -> float:
        return 0.0 if self.satisfied(state, event=event) else float(self.weight)

    def to_dict(self):
        data = super().to_dict()
        data.update({
            "event": self.event,
            "condition": self.condition_expr.to_dict() if self.condition_expr else None,
            "action": self.action if isinstance(self.action, str) else repr(self.action),
            "priority": self.priority,
        })
        return data


# --------------------------------------------------------------------------
# goal constraints (G)
# --------------------------------------------------------------------------
class GoalConstraint(Constraint):
    """Optimisation objective ``g : S x A x T -> R`` (Definition 2.9).

    ``objective`` may be

    * a callable ``(state, action, t) -> float``,
    * a callable ``(state) -> float`` (auto-adapted),
    * an :class:`Expr` or a string parsed into one (evaluated on the state).

    ``direction`` is ``"max"`` or ``"min"``; internally everything is
    normalised so that *larger is better*.
    """

    kind = "goal"

    def __init__(self, name, objective, description="", weight=1.0,
                 direction: str = "max", target: Optional[float] = None):
        super().__init__(name, description, weight)
        if direction not in ("max", "min"):
            raise ValueError("direction must be 'max' or 'min'")
        self.direction = direction
        self.target = target

        if isinstance(objective, str):
            objective = E.parse_expression(objective)
        self.expression = objective if isinstance(objective, E.Expr) else None

        if isinstance(objective, E.Expr):
            self._raw = lambda s, a=None, t=0.0, _e=objective: float(_e.evaluate(s))
        elif callable(objective):
            self._raw = self._adapt(objective)
        else:
            self._raw = lambda s, a=None, t=0.0, _c=objective: float(_c)

    @staticmethod
    def _adapt(fn):
        import inspect
        try:
            n = len(inspect.signature(fn).parameters)
        except (TypeError, ValueError):
            n = 1
        if n >= 3:
            return lambda s, a=None, t=0.0, _f=fn: float(_f(s, a, t))
        if n == 2:
            return lambda s, a=None, t=0.0, _f=fn: float(_f(s, a))
        return lambda s, a=None, t=0.0, _f=fn: float(_f(s))

    # -- evaluation --------------------------------------------------------
    def value(self, state, action=None, t: float = 0.0) -> float:
        """Raw objective value (in the declared direction)."""
        try:
            return float(self._raw(state, action, t))
        except Exception:
            return 0.0

    def utility(self, state, action=None, t: float = 0.0) -> float:
        """Objective normalised so that larger is always better."""
        v = self.value(state, action, t)
        if self.direction == "min":
            v = -v
        if self.target is not None:
            v = -abs(v - (self.target if self.direction == "max" else -self.target))
        return self.weight * v

    # -- constraint interface ---------------------------------------------
    def satisfied(self, state, action=None, t: float = 0.0, **kwargs) -> bool:
        return True  # goals are soft; they are never "violated"

    def violation(self, state, action=None, t: float = 0.0, **kwargs) -> float:
        return 0.0

    def to_dict(self):
        data = super().to_dict()
        data.update({"direction": self.direction, "target": self.target,
                     "expression": self.expression.to_dict() if self.expression else None})
        return data


# --------------------------------------------------------------------------
# constraint set
# --------------------------------------------------------------------------
class ConstraintSet:
    """Container for the ``I``, ``T`` and ``G`` components of a COH object."""

    def __init__(self, identity=None, triggers=None, goals=None,
                 regularizer: Optional[Callable[[Any], float]] = None,
                 lam: float = 0.0):
        self.identity: List[IdentityConstraint] = list(identity or [])
        self.triggers: List[TriggerConstraint] = list(triggers or [])
        self.goals: List[GoalConstraint] = list(goals or [])
        self.regularizer = regularizer
        self.lam = float(lam)

    # -- mutation ----------------------------------------------------------
    def add_identity(self, c): self.identity.append(c); return c
    def add_trigger(self, c):  self.triggers.append(c); return c
    def add_goal(self, c):     self.goals.append(c); return c

    def __len__(self): return len(self.identity) + len(self.triggers) + len(self.goals)

    def __iter__(self):
        yield from self.identity
        yield from self.triggers
        yield from self.goals

    def __repr__(self):
        return (f"ConstraintSet(I={len(self.identity)}, T={len(self.triggers)}, "
                f"G={len(self.goals)})")

    # -- identity ----------------------------------------------------------
    def identity_violations(self, state) -> List[IdentityConstraint]:
        return [c for c in self.identity if not c.satisfied(state)]

    def satisfies_identity(self, state) -> bool:
        return all(c.satisfied(state) for c in self.identity)

    def violation_score(self, state) -> float:
        return float(sum(c.violation(state) for c in self.identity))

    def check_identity(self, state) -> None:
        for c in self.identity:
            c.check(state)

    def assert_satisfiable(self, witness) -> None:
        """Raise :class:`UnsatisfiableConstraintError` if *witness* fails."""
        bad = self.identity_violations(witness)
        if bad:
            raise UnsatisfiableConstraintError(bad)

    # -- triggers ----------------------------------------------------------
    def applicable_triggers(self, event, state) -> List[TriggerConstraint]:
        found = [t for t in self.triggers if t.matches(event, state)]
        found.sort(key=lambda t: -t.priority)
        return found

    # -- goals -------------------------------------------------------------
    def total_goal(self, state, action=None, t: float = 0.0,
                   parameters=None) -> float:
        """``G_total = sum_j w_j g_j + lambda * R(theta)`` (Definition 2.9)."""
        total = float(sum(g.utility(state, action, t) for g in self.goals))
        if self.regularizer is not None and self.lam:
            total -= self.lam * float(self.regularizer(parameters))
        return total

    def goal_vector(self, state, action=None, t: float = 0.0) -> List[float]:
        return [g.value(state, action, t) for g in self.goals]

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "identity": [c.to_dict() for c in self.identity],
            "triggers": [c.to_dict() for c in self.triggers],
            "goals": [c.to_dict() for c in self.goals],
            "lam": self.lam,
        }

    def summary(self) -> str:
        lines = ["ConstraintSet:"]
        for c in self.identity:
            lines.append(f"  I  {c.name}: {c.description or '(invariant)'}")
        for c in self.triggers:
            lines.append(f"  T  {c.name}: on {c.event!r} -> {c.action}")
        for c in self.goals:
            lines.append(f"  G  {c.name}: {c.direction} (w={c.weight})")
        return "\n".join(lines)
    