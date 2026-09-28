# ============================================================
# FILE: gismol/dynamical/hybrid.py
# ============================================================
"""Explicit hybrid-automaton view of a COH object (Definition 4.1)."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from ..core.coh_object import COHObject, State
from ..core.dynamics import ContinuousFlow, DiscreteTransition, Trajectory

__all__ = ["Mode", "Guard", "Transition", "HybridAutomaton", "coh_to_hybrid"]


class Mode:
    """A continuous mode with its own flow and invariant."""

    def __init__(self, name: str, flow: Optional[ContinuousFlow] = None,
                 invariant: Optional[Callable[[Any], bool]] = None):
        self.name = name
        self.flow = flow
        self.invariant = invariant or (lambda s: True)

    def __repr__(self):
        return f"Mode({self.name!r})"


class Guard:
    """Switching surface: an event name plus a state predicate."""

    def __init__(self, event: str, predicate: Callable[[Any], bool], name: str = ""):
        self.event = event
        self.predicate = predicate
        self.name = name or event

    def satisfied(self, state, event: Optional[str] = None) -> bool:
        if self.event not in ("*", None) and event != self.event:
            return False
        return bool(self.predicate(state))

    def __repr__(self):
        return f"Guard({self.name!r}, on={self.event!r})"


class Transition:
    """A discrete transition ``source --guard/reset--> target``."""

    def __init__(self, source: str, target: str, guard: Guard,
                 reset: Optional[Callable[[Any], Any]] = None, name: str = ""):
        self.source = source
        self.target = target
        self.guard = guard
        self.reset = reset or (lambda s: s)
        self.name = name or f"{source}->{target}"

    def __repr__(self):
        return f"Transition({self.name!r})"


class HybridAutomaton:
    """A hybrid automaton assembled from modes and guarded transitions."""

    def __init__(self, modes: Sequence[Mode], transitions: Sequence[Transition],
                 initial: str, daemons=None, name: str = "HA"):
        self.modes: Dict[str, Mode] = {m.name: m for m in modes}
        self.transitions: List[Transition] = list(transitions)
        self.initial = initial
        self.current = initial
        self.daemons = daemons
        self.name = name

    def enabled(self, state, event: Optional[str] = None) -> List[Transition]:
        return [t for t in self.transitions
                if t.source == self.current and t.guard.satisfied(state, event)]

    def step(self, state, dt: float = 0.01, event: Optional[str] = None):
        info: Dict[str, Any] = {"mode": self.current, "jumped": False}
        for t in self.enabled(state, event):
            state = t.reset(state)
            self.current = t.target
            info["jumped"] = True
            info["transition"] = t.name
            info["mode"] = self.current
            break
        mode = self.modes[self.current]
        if mode.flow is not None and dt:
            state = mode.flow.step(state, None, dt)
        if self.daemons is not None:
            act = self.daemons.observe(state, 0.0)
            if act is not None and isinstance(act.payload, Mapping):
                merged = state.copy() if hasattr(state, "copy") else dict(state)
                merged.update(act.payload)
                state = merged
                info["daemon"] = act.name
        info["invariant_ok"] = mode.invariant(state)
        return state, info

    def run(self, initial_state, steps: int = 100, dt: float = 0.01,
            events: Optional[Sequence[Optional[str]]] = None) -> Trajectory:
        traj = Trajectory()
        state = initial_state
        traj.append(0.0, state)
        for i in range(steps):
            event = events[i] if events and i < len(events) else None
            state, info = self.step(state, dt, event)
            traj.append((i + 1) * dt, state, None, 0.0, event)
            traj.info.setdefault("modes", []).append(info["mode"])
        return traj

    def __repr__(self):
        return f"HybridAutomaton({self.name!r}, modes={list(self.modes)}, current={self.current!r})"


# --------------------------------------------------------------------------
def coh_to_hybrid(coh: COHObject, name: Optional[str] = None) -> HybridAutomaton:
    """Build a :class:`HybridAutomaton` from a COH object.

    * each method becomes a discrete mode-local reset,
    * each trigger constraint becomes a guard + transition,
    * the continuous flow (if any) is attached to the default mode.
    """
    default_mode = Mode(
        "default",
        flow=coh.flow,
        invariant=lambda s: coh.satisfies_constraints(s if isinstance(s, State) else State(s)),
    )
    modes = [default_mode]
    transitions: List[Transition] = []
    for trg in coh.T:
        guard = Guard(trg.event, trg.condition, name=trg.name)

        def reset(state, _t=trg):
            payload = _t.fire(state)
            if isinstance(payload, Mapping):
                merged = state.copy() if hasattr(state, "copy") else dict(state)
                merged.update(payload)
                return merged
            if callable(payload):
                out = payload(state)
                return out[0] if isinstance(out, tuple) else (out if out is not None else state)
            return state

        transitions.append(Transition("default", "default", guard, reset, name=trg.name))
    return HybridAutomaton(modes, transitions, "default", daemons=coh.D,
                           name=name or f"{coh.name}.automaton")
