# ============================================================
# FILE: gismol/nlp/constraint_parser.py
# ============================================================
"""Translate natural language into COH constraint objects.

Handles three families of statements:

* **invariants** — "x must be less than 10", "the temperature shall not exceed 90"
* **triggers**  — "if pressure exceeds 100 then vent", "when x > 5 do cool"
* **goals**     — "maximize throughput", "minimise cost"

Explicit expressions such as ``"constraint: x > 0 and x < 10"`` are parsed with
:func:`gismol.core.expression.parse_expression`.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..core import expression as E
from ..core.constraints import GoalConstraint, IdentityConstraint, TriggerConstraint
from ..core.exceptions import ParseError
from .normalizers import normalize

__all__ = ["ParsedConstraints", "ConstraintParser", "parse_constraints"]

_NUM = r"-?\d+(?:\.\d+)?"
_IDENT = r"[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*"

_COMPARATORS = [
    (r"\b(?:no\s+more\s+than|at\s+most|not\s+(?:more|greater|higher)\s+than|"
     r"less\s+than\s+or\s+equal\s+to|maximum\s+of|<=?)\b", "<="),
    (r"\b(?:no\s+less\s+than|at\s+least|not\s+(?:less|smaller|lower)\s+than|"
     r"greater\s+than\s+or\s+equal\s+to|minimum\s+of|>=?)\b", ">="),
    (r"\b(?:greater|larger|higher|more)\s+than\b", ">"),
    (r"\b(?:less|smaller|lower|fewer)\s+than\b", "<"),
    (r"\b(?:equals?|is\s+equal\s+to|exactly)\b", "=="),
    (r"\b(?:is|are|shall\s+be|must\s+be)\b", "=="),
]

_NEGATIONS = [r"\bnot\b", r"\bnever\b", r"\bno\b", r"\bforbidden\b", r"\bprohibited\b"]

_GOAL_MAX = r"\b(?:maximi[sz]e|maximi[sz]ing|maximi[sz]ation|increase|maximize)\b"
_GOAL_MIN = r"\b(?:minimi[sz]e|minimi[sz]ing|minimi[sz]ation|reduce|minimize|lower)\b"

_TRIGGER_SPLIT = re.compile(r"\b(?:then|,|do|perform|execute)\b", re.IGNORECASE)


class ParsedConstraints:
    """Result of parsing a text into COH constraints."""

    def __init__(self, identity=None, triggers=None, goals=None,
                 expressions=None, unparsed=None):
        self.identity: List[IdentityConstraint] = list(identity or [])
        self.triggers: List[TriggerConstraint] = list(triggers or [])
        self.goals: List[GoalConstraint] = list(goals or [])
        self.expressions: List[E.Expr] = list(expressions or [])
        self.unparsed: List[str] = list(unparsed or [])

    def __len__(self):
        return len(self.identity) + len(self.triggers) + len(self.goals)

    def __repr__(self):
        return (f"ParsedConstraints(I={len(self.identity)}, T={len(self.triggers)}, "
                f"G={len(self.goals)}, unparsed={len(self.unparsed)})")

    def to_dict(self):
        return {
            "identity": [c.to_dict() for c in self.identity],
            "triggers": [c.to_dict() for c in self.triggers],
            "goals": [c.to_dict() for c in self.goals],
            "unparsed": self.unparsed,
        }


class ConstraintParser:
    """Rule-based natural-language -> constraint compiler."""

    def __init__(self, namespace: Optional[Mapping[str, float]] = None,
                 unknown_policy: str = "keep"):
        self.namespace: Dict[str, float] = dict(namespace or {})
        self.unknown_policy = unknown_policy  # "keep" | "drop" | "error"

    # ------------------------------------------------------------------
    def parse(self, text: str, source: str = "") -> ParsedConstraints:
        result = ParsedConstraints()
        for sentence in _sentences(text):
            self._parse_sentence(sentence, result, source)
        return result

    # ------------------------------------------------------------------
    def _parse_sentence(self, sentence: str, result: ParsedConstraints, source: str):
        raw = sentence.strip()
        if not raw:
            return

        # explicit expression form: "constraint: <expr>"
        explicit = re.match(r"^(?:constraint|invariant|rule)\s*[:=]\s*(.+)$", raw, re.I)
        if explicit:
            expr_text = explicit.group(1).strip()
            try:
                expr = E.parse_expression(expr_text)
            except ParseError:
                result.unparsed.append(raw)
                return
            result.expressions.append(expr)
            result.identity.append(IdentityConstraint(
                name=f"expr_{len(result.identity)}", predicate=expr,
                description=raw))
            return

        # goals
        if re.search(_GOAL_MAX, raw, re.I) or re.search(_GOAL_MIN, raw, re.I):
            direction = "max" if re.search(_GOAL_MAX, raw, re.I) else "min"
            variable = self._extract_subject(raw)
            if variable:
                weight = self._extract_weight(raw)
                result.goals.append(GoalConstraint(
                    name=f"goal_{len(result.goals)}",
                    objective=variable,
                    description=raw, direction=direction, weight=weight))
                return

        # triggers
        trigger = self._parse_trigger(raw)
        if trigger is not None:
            result.triggers.append(trigger)
            return

        # invariants
        invariant = self._parse_invariant(raw)
        if invariant is not None:
            result.identity.append(invariant)
            return

        result.unparsed.append(raw)

    # ------------------------------------------------------------------
    def _parse_trigger(self, raw: str) -> Optional[TriggerConstraint]:
        m = re.match(r"^(?:if|when|whenever|upon|once)\s+(.+?)\s*[,:]?\s*(?:then\s+)?"
                     r"([a-zA-Z_][\w\s]*)$", raw, re.I)
        if not m:
            m2 = re.match(r"^(?:if|when|whenever|upon|once)\s+(.+?)\s+"
                          r"(?:then\s+|do\s+|perform\s+|execute\s+)(.+)$", raw, re.I)
            if not m2:
                return None
            condition_text, action_text = m2.group(1), m2.group(2)
        else:
            condition_text, action_text = m.group(1), m.group(2)

        condition = self._condition_from_text(condition_text)
        if condition is None:
            return None

        action = re.sub(r"\s+", " ", action_text.strip().rstrip(".")) or "respond"
        event = self._event_from_text(condition_text)
        return TriggerConstraint(
            name=f"trigger_{abs(hash(raw)) % 10 ** 8}",
            event=event, condition=condition, action=action,
            description=raw, priority=1 if "critical" in raw.lower() else 0)

    @staticmethod
    def _event_from_text(text: str) -> str:
        low = text.lower()
        for token in ("exceeds", "drops", "reaches", "fails", "detects", "enters",
                      "leaves", "receives", "loses"):
            if token in low:
                return token
        return "*"

    def _condition_from_text(self, text: str) -> Optional[Any]:
        text = text.strip().rstrip(".").rstrip(",")
        try:
            return E.parse_expression(text)
        except ParseError:
            pass

        ident = self._extract_identifier(text)
        comparator, op = self._extract_comparator(text)
        number = self._extract_number(text)
        if ident and op and number is not None:
            return E.Cmp(op, E.Var(ident), E.Const(number))

        negated = any(re.search(p, text, re.I) for p in _NEGATIONS)
        if ident:
            if negated:
                return E.Not(E.Var(ident))
            return E.Var(ident)

        if number is not None:
            return E.Cmp(">", E.Const(number), E.Const(0.0))

        return None

    # ------------------------------------------------------------------
    def _parse_invariant(self, raw: str) -> Optional[IdentityConstraint]:
        ident = self._extract_identifier(raw)
        comparator, op = self._extract_comparator(raw)
        number = self._extract_number(raw)
        negated = any(re.search(p, raw, re.I) for p in _NEGATIONS)

        if ident and op and number is not None:
            expr = E.Cmp(op, E.Var(ident), E.Const(number))
            if negated and op in ("<=", ">=", "=="):
                expr = E.Not(expr)
            name = f"inv_{ident}_{len(ident)}"
            return IdentityConstraint(name, expr, description=raw)

        # "x between a and b"
        between = re.search(rf"({_IDENT})\s+(?:is\s+)?between\s+({_NUM})\s+and\s+({_NUM})",
                            raw, re.I)
        if between:
            name, low, high = between.group(1), float(between.group(2)), float(between.group(3))
            expr = E.And(E.Cmp(">=", E.Var(name), E.Const(low)),
                         E.Cmp("<=", E.Var(name), E.Const(high)))
            return IdentityConstraint(f"inv_{name}_range", expr, description=raw)

        if ident and negated:
            return IdentityConstraint(f"inv_not_{ident}", E.Not(E.Var(ident)),
                                      description=raw)
        return None

    # ------------------------------------------------------------------
    def _extract_identifier(self, text: str) -> Optional[str]:
        candidates = re.findall(_IDENT, text, re.I)
        stop = {"must", "shall", "should", "the", "a", "an", "is", "are", "be",
                "if", "then", "when", "and", "or", "not", "no", "at", "least",
                "most", "more", "less", "than", "between", "exceeds", "equals"}
        for candidate in candidates:
            if candidate.lower() not in stop and not candidate.lower().startswith("max"):
                return candidate
        return None

    @staticmethod
    def _extract_comparator(text: str) -> Tuple[str, Optional[str]]:
        low = text.lower()
        for pattern, op in _COMPARATORS:
            m = re.search(pattern, low)
            if m:
                return m.group(0), op
        return "", None

    @staticmethod
    def _extract_number(text: str) -> Optional[float]:
        m = re.search(_NUM, text)
        return float(m.group(0)) if m else None

    @staticmethod
    def _extract_subject(text: str) -> Optional[str]:
        m = re.search(rf"(?:maximi[sz]e|minimi[sz]e|optimi[sz]e|reduce|increase|lower)\s+"
                      rf"(?:the\s+)?({_IDENT})", text, re.I)
        if m:
            return m.group(1)
        m = re.search(rf"({_IDENT})\s+(?:should\s+be\s+)?(?:maximi[sz]ed|minimi[sz]ed)",
                      text, re.I)
        if m:
            return m.group(1)
        return None

    @staticmethod
    def _extract_weight(text: str) -> float:
        m = re.search(r"weight\s*[:=]?\s*(-?\d+(?:\.\d+)?)", text, re.I)
        return float(m.group(1)) if m else 1.0


# --------------------------------------------------------------------------
def _sentences(text: str) -> List[str]:
    from .parsers import split_sentences
    parts = split_sentences(text)
    if len(parts) <= 1:
        parts = [p.strip() for p in re.split(r"[;\n]", text) if p.strip()]
    return parts or [text]


def parse_constraints(text: str, **kwargs) -> ParsedConstraints:
    return ConstraintParser(**kwargs).parse(text)
