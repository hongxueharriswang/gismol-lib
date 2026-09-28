# ============================================================
# FILE: gismol/reasoners/general.py
# ============================================================
"""General-purpose reasoners over COH objects."""

from __future__ import annotations

import itertools
import random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..core import expression as E
from ..core.coh_object import COHObject, State
from ..core.exceptions import ReasoningError
from .base import KnowledgeBase, Query, ReasoningResult, Reasoner, Rule

__all__ = [
    "ForwardChainer", "ConstraintReasoner", "GoalSolver",
    "AbductiveReasoner", "GeneralReasoner",
]


# --------------------------------------------------------------------------
class ForwardChainer(Reasoner):
    """Forward chaining over facts derived from the COH state."""

    name = "forward_chainer"

    def __init__(self, rules: Optional[Iterable[Rule]] = None, max_iterations: int = 64):
        self.rules = list(rules or [])
        self.max_iterations = max_iterations

    def add_rule(self, rule: Rule) -> Rule:
        self.rules.append(rule)
        return rule

    def infer(self, coh: COHObject, query: Query) -> ReasoningResult:
        kb = KnowledgeBase(_flatten_state(coh.state), self.rules + _rules_from_triggers(coh))
        kb.add_fact("object", coh.name)
        kb.add_fact("n_children", len(coh.C))
        kb.add_fact("n_goals", len(coh.G))
        trace = [f"seeded {len(kb.facts)} facts", f"loaded {len(kb.rules)} rules"]
        derived = kb.infer(self.max_iterations)
        trace.append(f"derived {len(derived)} facts")
        key = query.target
        if key in derived:
            return ReasoningResult(derived[key], 0.9, trace, derived, self.name)
        matches = {k: v for k, v in derived.items() if key in k}
        answer = matches if matches else None
        confidence = 0.75 if matches else 0.2
        return ReasoningResult(answer, confidence, trace, derived, self.name)


# --------------------------------------------------------------------------
class ConstraintReasoner(Reasoner):
    """Evaluates invariants, finds minimal violations and proposes repairs."""

    name = "constraint_reasoner"

    def infer(self, coh: COHObject, query: Query) -> ReasoningResult:
        state = coh.state if query.parameters.get("state") is None else query.parameters["state"]
        broken = [c for c in coh.I if not c.satisfied(state)]
        trace = [f"checked {len(coh.I)} invariants"]
        if not broken:
            return ReasoningResult(True, 1.0, trace + ["all invariants hold"],
                                   {"violations": []}, self.name)

        repairs = [self._repair(coh, c, state) for c in broken]
        trace.append("violations: " + ", ".join(c.name for c in broken))
        for r in repairs:
            if r:
                trace.append("repair: " + ", ".join(f"{k}={v}" for k, v in r.items()))
        answer = {"violations": [c.name for c in broken],
                  "repairs": repairs}
        return ReasoningResult(answer, 0.8, trace, answer, self.name)

    def _repair(self, coh: COHObject, constraint, state) -> Dict[str, Any]:
        """Cheap repair: search the attribute grid for a satisfying nudge."""
        if constraint.expression is None:
            return {}
        variables = constraint.expression.variables()
        repair: Dict[str, Any] = {}
        for var in variables:
            try:
                current = float(state.get_path(var, 0.0))
            except (TypeError, ValueError):
                continue
            for delta in (1.0, -1.0, 0.5, -0.5, 0.1, -0.1):
                probe = state.copy()
                probe.set_path(var, current + delta)
                if all(c.satisfied(probe) for c in coh.I):
                    repair[var] = current + delta
                    break
            if repair:
                break
        return repair

    def check(self, coh: COHObject, state: Optional[State] = None) -> bool:
        return coh.constraints.satisfies_identity(state or coh.state)


# --------------------------------------------------------------------------
class GoalSolver(Reasoner):
    """Greedy / beam search over methods to maximise the goal constraints."""

    name = "goal_solver"

    def __init__(self, depth: int = 3, beam: int = 4):
        self.depth = depth
        self.beam = beam

    def infer(self, coh: COHObject, query: Query) -> ReasoningResult:
        if not coh.M:
            return ReasoningResult(None, 0.1, ["no methods available"], {}, self.name)

        initial = coh.state.copy()
        beam: List[Tuple[float, State, List[str]]] = [
            (coh.constraints.total_goal(initial), initial, [])
        ]
        trace: List[str] = [f"depth={self.depth}, beam={self.beam}",
                            f"methods={list(coh.M)}"]

        for level in range(self.depth):
            candidates: List[Tuple[float, State, List[str]]] = []
            for _, state, path in beam:
                for mname, method in coh.M.items():
                    try:
                        new_state, _ = method(state.copy())
                    except Exception:
                        continue
                    if not isinstance(new_state, State):
                        new_state = State(new_state)
                    if not coh.constraints.satisfies_identity(new_state):
                        continue
                    score = coh.constraints.total_goal(new_state)
                    candidates.append((score, new_state, path + [mname]))
            if not candidates:
                break
            candidates.sort(key=lambda t: -t[0])
            beam = candidates[:self.beam]
            trace.append(f"level {level + 1}: best={beam[0][0]:.4f} "
                         f"via {beam[0][2]}")

        best_score, best_state, best_path = beam[0]
        return ReasoningResult(
            {"plan": best_path, "value": best_score, "state": dict(best_state)},
            0.85 if best_path else 0.3, trace,
            {"score": best_score, "plan": best_path}, self.name)


# --------------------------------------------------------------------------
class AbductiveReasoner(Reasoner):
    """Finds the minimal set of state changes that makes the invariants hold."""

    name = "abductive_reasoner"

    def infer(self, coh: COHObject, query: Query) -> ReasoningResult:
        state = query.parameters.get("state", coh.state)
        broken = [c for c in coh.I if not c.satisfied(state)]
        if not broken:
            return ReasoningResult([], 1.0, ["nothing to explain"], {}, self.name)

        explanation: Dict[str, Any] = {}
        trace = []
        for constraint in broken:
            if constraint.expression is None:
                continue
            for candidate in _candidate_values(constraint):
                probe = state.copy()
                probe.set_path(candidate[0], candidate[1])
                if all(c.satisfied(probe) for c in coh.I):
                    explanation[candidate[0]] = candidate[1]
                    trace.append(f"assume {candidate[0]}={candidate[1]!r}")
                    break
            if explanation:
                break
        confidence = 0.7 if explanation else 0.15
        return ReasoningResult(explanation, confidence, trace, explanation, self.name)


def _candidate_values(constraint) -> List[Tuple[str, Any]]:
    out: List[Tuple[str, Any]] = []
    expr = constraint.expression
    if expr is None:
        return out
    for var in expr.variables():
        for value in (0.0, 1.0, -1.0, 10.0, 100.0):
            out.append((var, value))
    return out


# --------------------------------------------------------------------------
class GeneralReasoner(Reasoner):
    """Dispatches to the most appropriate specialised reasoner."""

    name = "general_reasoner"

    def __init__(self):
        self.forward = ForwardChainer()
        self.constraints = ConstraintReasoner()
        self.goals = GoalSolver()
        self.abductive = AbductiveReasoner()

    def infer(self, coh: COHObject, query: Query) -> ReasoningResult:
        kind = query.kind if query.kind != "auto" else self._classify(query.target)
        trace = [f"dispatched to {kind}"]
        if kind == "constraint":
            result = self.constraints.infer(coh, query)
        elif kind == "goal":
            result = self.goals.infer(coh, query)
        elif kind == "abduction":
            result = self.abductive.infer(coh, query)
        else:
            result = self.forward.infer(coh, query)
        result.trace = trace + result.trace
        result.reasoner = f"{self.name}->{result.reasoner}"
        return result

    @staticmethod
    def _classify(target: str) -> str:
        low = (target or "").lower()
        if any(k in low for k in ("violat", "satisf", "invariant", "constraint")):
            return "constraint"
        if any(k in low for k in ("max", "min", "optim", "plan", "best", "goal")):
            return "goal"
        if any(k in low for k in ("why", "explain", "because", "reason")):
            return "abduction"
        return "forward"


# --------------------------------------------------------------------------
def _flatten_state(state) -> Dict[str, Any]:
    flat = getattr(state, "flatten", None)
    if callable(flat):
        return dict(flat())
    if isinstance(state, Mapping):
        return {str(k): v for k, v in state.items()}
    return {}


def _rules_from_triggers(coh: COHObject) -> List[Rule]:
    rules: List[Rule] = []
    for trg in coh.T:
        rules.append(Rule(
            name=f"trigger:{trg.name}",
            condition=trg.condition,
            conclusion={f"triggered_{trg.name}": True, "next_action": trg.action},
            confidence=0.9,
            description=trg.description,
        ))
    return rules