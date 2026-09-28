# ============================================================
# FILE: gismol/nlp/entity_extractor.py
# ============================================================
"""Rule-based entity extraction."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = ["Entity", "EntityExtractor", "extract_entities", "DEFAULT_PATTERNS"]


DEFAULT_PATTERNS: Dict[str, str] = {
    "NUMBER": r"(?<!\w)-?\d+(?:\.\d+)?(?:e-?\d+)?(?!\w)",
    "PERCENT": r"-?\d+(?:\.\d+)?\s?%",
    "MONEY": r"[$€£]\s?-?\d+(?:[.,]\d+)?|\b\d+(?:\.\d+)?\s?(?:USD|EUR|GBP|dollars|euros)\b",
    "DURATION": r"\b\d+(?:\.\d+)?\s?(?:s|sec|secs|seconds|ms|min|mins|minutes|h|hrs|hours|days|years)\b",
    "DATE": r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    "EMAIL": r"[\w\.\-]+@[\w\.\-]+\.\w+",
    "URL": r"https?://[^\s]+",
    "IDENTIFIER": r"\b[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)+\b",
    "CONSTRAINT_WORD": r"\b(?:must|shall|should|required|mandatory|forbidden|prohibited|allowed|at most|at least|no more than|no less than|exactly|between)\b",
    "GOAL_WORD": r"\b(?:maximi[sz]e|maximi[sz]ing|minimi[sz]e|minimi[sz]ing|optimi[sz]e|optimi[sz]ing|reduce|increase)\b",
    "TRIGGER_WORD": r"\b(?:if|when|whenever|upon|once|in case)\b",
    "MODAL": r"\b(?:must|shall|should|may|might|can|could|will|would)\b",
}


class Entity:
    """A span of text with a label and an optional typed value."""

    def __init__(self, text: str, label: str, start: int = -1, end: int = -1,
                 value: Any = None, confidence: float = 1.0):
        self.text = text
        self.label = label
        self.start = start
        self.end = end
        self.value = value if value is not None else text
        self.confidence = float(confidence)

    def to_dict(self):
        return {"text": self.text, "label": self.label, "start": self.start,
                "end": self.end, "value": self.value, "confidence": self.confidence}

    def __repr__(self):
        return f"Entity({self.text!r}, {self.label})"


class EntityExtractor:
    """Regex-driven entity extractor with an extensible pattern registry."""

    def __init__(self, patterns: Optional[Dict[str, str]] = None,
                 include_defaults: bool = True):
        self.patterns: Dict[str, re.Pattern] = {}
        if include_defaults:
            for label, pattern in DEFAULT_PATTERNS.items():
                self.patterns[label] = re.compile(pattern, re.IGNORECASE)
        for label, pattern in (patterns or {}).items():
            self.patterns[label] = re.compile(pattern, re.IGNORECASE)

    def add_pattern(self, label: str, pattern: str):
        self.patterns[label] = re.compile(pattern, re.IGNORECASE)

    def extract(self, text: str, labels: Optional[Sequence[str]] = None) -> List[Entity]:
        found: List[Entity] = []
        selected = labels or list(self.patterns)
        for label in selected:
            regex = self.patterns.get(label)
            if regex is None:
                continue
            for match in regex.finditer(text or ""):
                value = self._coerce(label, match.group(0))
                found.append(Entity(match.group(0), label, match.start(),
                                    match.end(), value))
        found.sort(key=lambda e: (e.start, -len(e.text)))
        return self._dedupe(found)

    @staticmethod
    def _coerce(label: str, raw: str) -> Any:
        text = raw.strip()
        if label in ("NUMBER", "PERCENT", "DURATION"):
            number = re.search(r"-?\d+(?:\.\d+)?", text)
            return float(number.group(0)) if number else text
        if label == "MONEY":
            number = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
            return float(number.group(0)) if number else text
        return text

    @staticmethod
    def _dedupe(entities: List[Entity]) -> List[Entity]:
        out: List[Entity] = []
        occupied: List[Tuple[int, int]] = []
        for e in entities:
            if any(not (e.end <= s or e.start >= t) for s, t in occupied):
                continue
            occupied.append((e.start, e.end))
            out.append(e)
        return out

    def by_label(self, entities: Iterable[Entity]) -> Dict[str, List[Entity]]:
        grouped: Dict[str, List[Entity]] = {}
        for e in entities:
            grouped.setdefault(e.label, []).append(e)
        return grouped

    def __repr__(self):
        return f"EntityExtractor(labels={sorted(self.patterns)})"


def extract_entities(text: str, labels: Optional[Sequence[str]] = None) -> List[Entity]:
    return EntityExtractor().extract(text, labels)
