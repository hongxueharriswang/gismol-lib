# ============================================================
# FILE: gismol/nlp/relations_miner.py
# ============================================================
"""Pattern-based relation mining over text and extracted entities."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .entity_extractor import Entity
from .parsers import Document, parse

__all__ = ["Relation", "RelationMiner", "mine_relations", "DEFAULT_RELATION_PATTERNS"]


DEFAULT_RELATION_PATTERNS: List[Tuple[str, str, float]] = [
    (r"(?P<s>[\w\.]+)\s+is\s+(?:a|an|the)\s+(?P<o>[\w\.]+)", "is_a", 0.9),
    (r"(?P<s>[\w\.]+)\s+(?:has|have|contains|includes)\s+(?P<o>[\w\.]+)", "has", 0.85),
    (r"(?P<s>[\w\.]+)\s+depends\s+on\s+(?P<o>[\w\.]+)", "depends_on", 0.9),
    (r"(?P<s>[\w\.]+)\s+(?:causes|leads\s+to|results\s+in)\s+(?P<o>[\w\.]+)", "causes", 0.85),
    (r"(?P<s>[\w\.]+)\s+requires\s+(?P<o>[\w\.]+)", "requires", 0.9),
    (r"(?P<s>[\w\.]+)\s+controls\s+(?P<o>[\w\.]+)", "controls", 0.85),
    (r"(?P<s>[\w\.]+)\s+(?:must|should|shall)\s+be\s+(?P<o>[\w\.<>=]+)", "must_be", 0.9),
    (r"(?P<s>[\w\.]+)\s+(?:greater|larger|higher)\s+than\s+(?P<o>[\w\.\-]+)", "greater_than", 0.9),
    (r"(?P<s>[\w\.]+)\s+(?:less|smaller|lower)\s+than\s+(?P<o>[\w\.\-]+)", "less_than", 0.9),
    (r"(?P<s>[\w\.]+)\s+(?:equals|is\s+equal\s+to)\s+(?P<o>[\w\.\-]+)", "equals", 0.9),
    (r"(?P<s>[\w\.]+)\s+produces\s+(?P<o>[\w\.]+)", "produces", 0.8),
    (r"(?P<s>[\w\.]+)\s+influences\s+(?P<o>[\w\.]+)", "influences", 0.75),
]

_PREDICATE_LABELS = {
    "greater_than": ">", "less_than": "<", "equals": "==", "must_be": "==",
}


class Relation:
    """A triple ``(subject, predicate, object)`` with evidence."""

    def __init__(self, subject: str, predicate: str, obj: str,
                 confidence: float = 1.0, evidence: str = "", sentence: int = 0):
        self.subject = subject
        self.predicate = predicate
        self.obj = obj
        self.confidence = float(confidence)
        self.evidence = evidence
        self.sentence = sentence

    @property
    def triple(self) -> Tuple[str, str, str]:
        return (self.subject, self.predicate, self.obj)

    def to_dict(self):
        return {"subject": self.subject, "predicate": self.predicate,
                "object": self.obj, "confidence": self.confidence,
                "evidence": self.evidence, "sentence": self.sentence}

    def __repr__(self):
        return f"Relation({self.subject!r} --{self.predicate}--> {self.obj!r})"


class RelationMiner:
    """Extracts relations using lexical patterns plus co-occurrence."""

    def __init__(self, patterns: Optional[Sequence[Tuple[str, str, float]]] = None,
                 cooccurrence_window: int = 6, cooccurrence_confidence: float = 0.35):
        self.patterns = [
            (re.compile(p, re.IGNORECASE), label, conf)
            for p, label, conf in (patterns or DEFAULT_RELATION_PATTERNS)
        ]
        self.window = int(cooccurrence_window)
        self.cooc_conf = float(cooccurrence_confidence)

    # -- public API --------------------------------------------------------
    def mine(self, text: str, entities: Optional[Sequence[Entity]] = None,
             document: Optional[Document] = None) -> List[Relation]:
        doc = document or parse(text)
        relations: List[Relation] = []
        for i, sentence in enumerate(doc.sentences or [text]):
            relations.extend(self._pattern_relations(sentence, i))
        relations.extend(self._cooccurrence_relations(doc, entities or []))
        return self._dedupe(relations)

    def _pattern_relations(self, sentence: str, index: int) -> List[Relation]:
        out: List[Relation] = []
        for regex, label, conf in self.patterns:
            for m in regex.finditer(sentence):
                subject = m.group("s").strip()
                obj = m.group("o").strip()
                if subject.lower() == obj.lower():
                    continue
                out.append(Relation(subject, label, obj, conf, sentence, index))
        return out

    def _cooccurrence_relations(self, doc: Document, entities: Sequence[Entity]) -> List[Relation]:
        if len(entities) < 2:
            return []
        out: List[Relation] = []
        for i, a in enumerate(entities):
            for b in entities[i + 1:]:
                if abs(a.start - b.start) <= self.window * 8:
                    out.append(Relation(a.text, "co_occurs_with", b.text,
                                        self.cooc_conf, "", 0))
        return out

    @staticmethod
    def _dedupe(relations: List[Relation]) -> List[Relation]:
        seen = set()
        out = []
        for r in relations:
            if r.triple in seen:
                continue
            seen.add(r.triple)
            out.append(r)
        return out

    # -- conversion --------------------------------------------------------
    def to_expression(self, relation: Relation) -> Optional[str]:
        """Turn a comparison relation into a constraint expression string."""
        op = _PREDICATE_LABELS.get(relation.predicate)
        if op is None:
            return None
        try:
            float(relation.obj)
            rhs = relation.obj
        except ValueError:
            rhs = repr(relation.obj)
        return f"{relation.subject} {op} {rhs}"

    def __repr__(self):
        return f"RelationMiner(patterns={len(self.patterns)})"


def mine_relations(text: str, entities: Optional[Sequence[Entity]] = None) -> List[Relation]:
    return RelationMiner().mine(text, entities)
