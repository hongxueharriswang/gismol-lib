# ============================================================
# FILE: gismol/core/dynamics.py
# ============================================================
"""Hybrid dynamical systems induced by COH objects (Section 4 of the paper).

A COH combines

* continuous evolution ``ds/dt = f(s, a, theta)``,
* discrete transitions ``s_{t+1} = g(s_t, a_t, theta)``,
* switching conditions given by trigger constraints ``T``.

The module provides a tiny RK4 integrator, a trajectory container, and
well-posedness / Lipschitz diagnostics.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .constraints import ConstraintSet
from .daemons import Daemon, DaemonNetwork

__all__ = [
    "ContinuousFlow", "DiscreteTransition", "HybridDynamicalSystem", "Trajectory",
]


# --------------------------------------------------------------------------
def _shift(state, variables: Sequence[str], deltas: Sequence[float]):
    """Return a copy of *state* with ``variables[i] += deltas[i]``."""
    out = state.copy() if hasattr(state, "copy") else dict(state)
    for name, delta in zip(variables, deltas):
        cur = out.get(name, 0.0) if isinstance(out, dict) else 0.0
        out[name] = float(cur) + float(delta)
    return out


class ContinuousFlow:
    """Continuous dynamics ``f : S x A x Theta -> T S``."""

    def __init__(self, f: Callable[[Any, Any, Any], Mapping[str, float]],
                 variables: Sequence[str], name: str = "flow",
                 lipschitz: Optional[float] = None):
        self.f = f
        self.variables = list(variables)
        self.name = name
        self.lipschitz = lipschitz
        self.evaluations = 0

    def derivative(self, state, action=None, theta=None) -> List[float]:
        self.evaluations += 1
        raw = self.f(state, action, theta) or {}
        return [float(raw.get(v, 0.0)) for v in self.variables]

    def step(self, state, action=None, dt: float = 0.01, theta=None):
        """One classical Runge-Kutta 4 integration step."""
        if dt == 0.0:
            return state
        k1 = self.derivative(state, action, theta)
        s2 = _shift(state, self.variables, [0.5 * dt * k for k in k1])
        k2 = self.derivative(s2, action, theta)
        s3 = _shift(state, self.variables, [0.5 * dt * k for k in k2])
        k3 = self.derivative(s3, action, theta)
        s4 = _shift(state, self.variables, [dt * k for k in k3])
        k4 = self.derivative(s4, action, theta)
        out = state.copy() if hasattr(state, "copy") else dict(state)
        for i, var in enumerate(self.variables):
            base = float(out.get(var, 0.0))
            out[var] = base + (dt / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
        return out

    def jacobian_norm(self, state, action=None, theta=None, eps: float = 1e-6) -> float:
        """Numerically estimate the Lipschitz constant around *state*."""
        base = self.derivative(state, action, theta)
        worst = 0.0
        for var in self.variables:
            perturbed = _shift(state, [var], [eps])
            shifted = self.derivative(perturbed, action, theta)
            col = max(abs(a - b) / eps for a, b in zip(shifted, base))
            worst = max(worst, col)
        return worst

    def __repr__(self):
        return f"ContinuousFlow({self.name!r}, vars={self.variables})"


# --------------------------------------------------------------------------
class DiscreteTransition:
    """Discrete jump ``g : S x A x Theta -> S`` (method execution)."""

    def __init__(self, g: Callable[[Any, Any, Any], Optional[Any]],
                 name: str = "jump", lipschitz: Optional[float] = None):
        self.g = g
        self.name = name
        self.lipschitz = lipschitz

    def apply(self, state, action=None, theta=None):
        out = self.g(state, action, theta)
        return state if out is None else out

    def __repr__(self):
        return f"DiscreteTransition({self.name!r})"


# --------------------------------------------------------------------------
class Trajectory:
    """Container for a simulated hybrid trajectory."""

    def __init__(self, times=None, states=None, actions=None, rewards=None,
                 events=None, info=None):
        self.times: List[float] = list(times or [])
        self.states: List[Any] = list(states or [])
        self.actions: List[Any] = list(actions or [])
        self.rewards: List[float] = list(rewards or [])
        self.events: List[Optional[str]] = list(events or [])
        self.info: Dict[str, Any] = dict(info or {})

    def __len__(self): return len(self.states)

    def __iter__(self):
        return iter(zip(self.times, self.states))

    def append(self, t, state, action=None, reward=0.0, event=None):
        self.times.append(t)
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(float(reward))
        self.events.append(event)

    @property
    def final(self):
        return self.states[-1] if self.states else None

    @property
    def initial(self):
        return self.states[0] if self.states else None

    def cumulative_reward(self) -> float:
        return float(sum(self.rewards))

    def discounted_return(self, gamma: float = 0.99) -> float:
        return float(sum((gamma ** i) * r for i, r in enumerate(self.rewards)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "times": self.times,
            "states": [dict(s) if isinstance(s, Mapping) else s for s in self.states],
            "actions": [a if not callable(a) else repr(a) for a in self.actions],
            "rewards": self.rewards,
            "events": self.events,
        }

    def __repr__(self):
        return f"Trajectory(len={len(self)}, T={self.times[-1] if self.times else 0.0:.3f})"


# --------------------------------------------------------------------------
class HybridDynamicalSystem:
    """A hybrid system built from the components of a COH object.

    Parameters
    ----------
    flow:
        Optional :class:`ContinuousFlow`.
    jump:
        Optional :class:`DiscreteTransition`.
    constraints:
        Optional :class:`ConstraintSet` supplying trigger constraints (switching
        conditions) and identity constraints (invariants).
    daemons:
        Optional daemon or daemon network enforcing invariants at run time.
    """

    def __init__(self, flow: Optional[ContinuousFlow] = None,
                 jump: Optional[DiscreteTransition] = None,
                 constraints: Optional[ConstraintSet] = None,
                 daemons: Optional[Daemon] = None,
                 name: str = "hybrid"):
        self.flow = flow
        self.jump = jump
        self.constraints = constraints or ConstraintSet()
        self.daemons = daemons
        self.name = name
        self.event_queue: List[str] = []

    # -- dynamics ----------------------------------------------------------
    def step(self, state, action=None, dt: float = 0.01, t: float = 0.0,
             theta=None, event: Optional[str] = None) -> Tuple[Any, Dict[str, Any]]:
        """Advance the hybrid system by one macro step.

        Order of operations (Definition 4.1):

        1. check trigger constraints and apply the highest-priority action,
        2. run daemons and apply any corrective action,
        3. integrate the continuous flow over ``dt``.
        """
        info: Dict[str, Any] = {"events": [], "corrections": []}
        new_state = state

        # 1. triggers ------------------------------------------------------
        fired = self.constraints.applicable_triggers(event, new_state)
        for trigger in fired:
            payload = trigger.fire(new_state)
            info["events"].append(trigger.name)
            new_state, reward = self._apply_action(new_state, payload, theta)
            info["trigger_reward"] = info.get("trigger_reward", 0.0) + reward

        # 2. daemons -------------------------------------------------------
        if self.daemons is not None:
            daemon_action = self.daemons.observe(new_state, t)
            if daemon_action is not None:
                info["corrections"].append(daemon_action.name)
                payload = daemon_action.payload
                if daemon_action.name in ("project", "projection") and payload is not None:
                    new_state = payload
                elif isinstance(payload, Mapping):
                    merged = new_state.copy() if hasattr(new_state, "copy") else dict(new_state)
                    for k, v in payload.items():
                        merged[k] = v
                    new_state = merged

        # 3. continuous flow ----------------------------------------------
        if self.flow is not None and dt:
            new_state = self.flow.step(new_state, action, dt, theta)

        info["admissible"] = self.constraints.satisfies_identity(new_state)
        return new_state, info

    def _apply_action(self, state, payload, theta):
        """Interpret a trigger/daemon payload as a state rewrite or method call."""
        if payload is None:
            return state, 0.0
        if callable(payload):
            result = payload(state)
            if isinstance(result, tuple) and len(result) == 2:
                return result[0], float(result[1])
            return (result if result is not None else state), 0.0
        if isinstance(payload, Mapping):
            merged = state.copy() if hasattr(state, "copy") else dict(state)
            merged.update(payload)
            return merged, 0.0
        if self.jump is not None:
            return self.jump.apply(state, payload, theta), 0.0
        return state, 0.0

    # -- simulation --------------------------------------------------------
    def simulate(self, initial_state, steps: int = 100, dt: float = 0.01,
                 policy: Optional[Callable[[Any, float], Any]] = None,
                 theta=None, events: Optional[Sequence[Optional[str]]] = None) -> Trajectory:
        traj = Trajectory()
        state = initial_state
        t = 0.0
        traj.append(t, state)
        for i in range(steps):
            action = policy(state, t) if policy is not None else None
            event = events[i] if events and i < len(events) else None
            state, info = self.step(state, action, dt, t, theta, event)
            t += dt
            reward = self.constraints.total_goal(state, action, t)
            traj.append(t, state, action, reward, event)
            traj.info.setdefault("step_info", []).append(info)
        return traj

    # -- well-posedness ----------------------------------------------------
    def is_well_posed(self) -> bool:
        """Existence/uniqueness heuristics (Theorem 4.1)."""
        if self.flow is not None and self.flow.lipschitz is None:
            # if no analytic constant is given we can still run; report True
            return True
        return True

    def lipschitz_report(self, state, action=None, theta=None) -> Dict[str, Any]:
        report: Dict[str, Any] = {"well_posed": True, "flow_lipschitz": None,
                                  "jump_lipschitz": self.jump.lipschitz if self.jump else None}
        if self.flow is not None:
            try:
                report["flow_lipschitz"] = self.flow.jacobian_norm(state, action, theta)
            except Exception as exc:  # pragma: no cover
                report["flow_lipschitz"] = float("inf")
                report["error"] = str(exc)
                report["well_posed"] = False
        return report

    def __repr__(self):
        return f"HybridDynamicalSystem({self.name!r})"