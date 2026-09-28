# ============================================================
# FILE: gismol/reasoners/base.py
# ============================================================
"""Reasoner interfaces, rules and knowledge bases."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from ..core import expression as E
from ..core.coh_object import COHObject
from ..core.exceptions import ReasoningError

__all__ = ["Query", "ReasoningResult", "Rule", "KnowledgeBase", "Reasoner"]


class Query:
    """A reasoning query."""

    def __init__(self, target: str, kind: str = "auto",
                 parameters: Optional[Mapping[str, Any]] = None):
        self.target = target
        self.kind = kind
        self.parameters = dict(parameters or {})

    def __repr__(self):
        return f"Query({self.target!r}, kind={self.kind!r})"


class ReasoningResult:
    """Outcome of a reasoning episode."""

    def __init__(self, answer: Any, confidence: float = 1.0,
                 trace: Optional[Sequence[str]] = None,
                 derived: Optional[Mapping[str, Any]] = None,
                 reasoner: str = ""):
        self.answer = answer
        self.confidence = float(confidence)
        self.trace: List[str] = list(trace or [])
        self.derived: Dict[str, Any] = dict(derived or {})
        self.reasoner = reasoner

    def __bool__(self):
        return bool(self.answer)

    def to_dict(self):
        return {"answer": self.answer, "confidence": self.confidence,
                "trace": self.trace, "derived": self.derived,
                "reasoner": self.reasoner}

    def __repr__(self):
        return (f"ReasoningResult({self.answer!r}, conf={self.confidence:.2f}, "
                f"by={self.reasoner!r})")


class Rule:
    """A production rule: condition (over facts/state) implies conclusion."""

    def __init__(self, name: str, condition, conclusion: Any,
                 confidence: float = 1.0, description: str = ""):
        self.name = name
        self.condition = condition if isinstance(condition, E.Expr) else None
        self._condition = (lambda env, _e=condition: bool(_e.evaluate(env))
                           if isinstance(_e, E.Expr)
                           else bool(_e(env)))
        self.conclusion = conclusion
        self.confidence = float(confidence)
        self.description = description
        self.firings = 0

    def applies(self, facts: Mapping[str, Any]) -> bool:
        try:
            return self._condition(facts)
        except KeyError:
            return False
        except Exception:
            return False

    def fire(self) -> Any:
        self.firings += 1
        return self.conclusion

    def __repr__(self):
        return f"Rule({self.name!r})"


class KnowledgeBase:
    """Facts plus rules with forward chaining."""

    def __init__(self, facts: Optional[Mapping[str, Any]] = None,
                 rules: Optional[Iterable[Rule]] = None):
        self.facts: Dict[str, Any] = dict(facts or {})
        self.rules: List[Rule] = list(rules or [])

    def add_fact(self, key: str, value: Any):
        self.facts[key] = value
        return value

    def add_rule(self, rule: Rule) -> Rule:
        self.rules.append(rule)
        return rule

    def infer(self, max_iterations: int = 100) -> Dict[str, Any]:
        """Naive forward chaining to fixpoint."""
        for _ in range(max_iterations):
            changed = False
            for rule in self.rules:
                if not rule.applies(self.facts):
                    continue
                conclusion = rule.fire()
                if isinstance(conclusion, Mapping):
                    for k, v in conclusion.items():
                        if self.facts.get(k) != v:
                            self.facts[k] = v
                            changed = True
                elif isinstance(conclusion, str) and conclusion not in self.facts:
                    self.facts[conclusion] = True
                    changed = True
            if not changed:
                break
        return dict(self.facts)

    def matches(self, pattern: str) -> List[str]:
        return [k for k in self.facts if pattern in k]

    def __repr__(self):
        return f"KnowledgeBase(facts={len(self.facts)}, rules={len(self.rules)})"


class Reasoner:
    """Base class for all reasoners."""

    name = "reasoner"

    def infer(self, coh: COHObject, query: Query) -> ReasoningResult:  # pragma: no cover
        raise NotImplementedError

    def explain(self, result: ReasoningResult) -> str:
        header = f"[{result.reasoner or self.name}] confidence={result.confidence:.2f}"
        if not result.trace:
            return f"{header}\nanswer: {result.answer}"
        return header + "\n" + "\n".join(f"  - {step}" for step in result.trace)

    def __call__(self, coh: COHObject, query: Query) -> ReasoningResult:
        return self.infer(coh, query)

    def __repr__(self):
        return f"{type(self).__name__}(name={self.name!r})"
    