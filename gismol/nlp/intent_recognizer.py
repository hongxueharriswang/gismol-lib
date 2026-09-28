# ============================================================
# FILE: gismol/nlp/intent_recognizer.py
# ============================================================
"""Rule-based intent recognition with an extensible intent registry."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = ["Intent", "IntentRecognizer", "DEFAULT_INTENTS"]


DEFAULT_INTENTS: Dict[str, Dict[str, Any]] = {
    "define": {
        "keywords": ["define", "create", "build", "make", "construct", "new", "add"],
        "weight": 1.0,
    },
    "describe": {
        "keywords": ["describe", "what is", "explain", "show", "tell me about", "summarise", "summarize"],
        "weight": 1.0,
    },
    "constrain": {
        "keywords": ["must", "shall", "should", "constraint", "require", "invariant",
                     "at most", "at least", "no more than", "no less than", "between",
                     "forbidden", "prohibited"],
        "weight": 1.3,
    },
    "optimize": {
        "keywords": ["maximize", "maximise", "minimize", "minimise", "optimize",
                     "optimise", "best", "optimal", "reduce", "increase"],
        "weight": 1.2,
    },
    "query": {
        "keywords": ["what", "which", "how", "why", "when", "where", "who", "?"],
        "weight": 0.8,
    },
    "simulate": {
        "keywords": ["simulate", "run", "step", "evolve", "trajectory", "predict"],
        "weight": 1.1,
    },
    "compare": {
        "keywords": ["compare", "difference", "versus", "vs", "similar", "like"],
        "weight": 1.0,
    },
    "validate": {
        "keywords": ["validate", "check", "verify", "sound", "consistent", "complete"],
        "weight": 1.1,
    },
    "chat": {"keywords": ["hello", "hi", "hey", "thanks", "thank you", "bye"], "weight": 0.6},
}

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


class Intent:
    """A recognised intent with a confidence and optional slots."""

    def __init__(self, name: str, confidence: float = 0.0,
                 slots: Optional[Mapping[str, Any]] = None, matched: Sequence[str] = ()):
        self.name = name
        self.confidence = float(confidence)
        self.slots: Dict[str, Any] = dict(slots or {})
        self.matched = list(matched)

    def __repr__(self):
        return f"Intent({self.name!r}, conf={self.confidence:.2f})"

    def to_dict(self):
        return {"name": self.name, "confidence": self.confidence,
                "slots": self.slots, "matched": self.matched}


class IntentRecognizer:
    """Scores each registered intent by keyword overlap and picks the best."""

    def __init__(self, intents: Optional[Mapping[str, Mapping[str, Any]]] = None,
                 threshold: float = 0.0):
        self.intents: Dict[str, Dict[str, Any]] = {
            k: {"keywords": list(v.get("keywords", [])),
                "weight": float(v.get("weight", 1.0))}
            for k, v in (intents or DEFAULT_INTENTS).items()
        }
        self.threshold = float(threshold)

    def add_intent(self, name: str, keywords: Sequence[str], weight: float = 1.0):
        self.intents[name] = {"keywords": list(keywords), "weight": float(weight)}

    def score(self, name: str, text: str) -> Tuple[float, List[str]]:
        entry = self.intents[name]
        low = text.lower()
        matched = [kw for kw in entry["keywords"] if kw in low]
        if not matched:
            return 0.0, []
        base = len(matched) / max(1, len(entry["keywords"]))
        length_bonus = min(1.0, sum(len(m) for m in matched) / max(8, len(text)))
        return entry["weight"] * (0.6 * base + 0.4 * length_bonus), matched

    def recognize(self, text: str) -> Intent:
        low = (text or "").lower()
        best_name, best_score, best_matched = "chat", 0.0, []
        for name in self.intents:
            score, matched = self.score(name, text)
            if score > best_score:
                best_name, best_score, best_matched = name, score, matched
        if best_score <= self.threshold:
            best_name, best_score, best_matched = "chat", 0.0, []
        slots = self._extract_slots(low)
        return Intent(best_name, best_score, slots, best_matched)

    @staticmethod
    def _extract_slots(text: str) -> Dict[str, Any]:
        slots: Dict[str, Any] = {}
        numbers = _NUMBER.findall(text)
        if numbers:
            slots["numbers"] = [float(n) for n in numbers]
        for comparator in ("at most", "at least", "no more than", "no less than",
                           "greater than", "less than", "between", "equals"):
            if comparator in text:
                slots.setdefault("comparators", []).append(comparator)
        return slots

    def __repr__(self):
        return f"IntentRecognizer(intents={sorted(self.intents)})"
    