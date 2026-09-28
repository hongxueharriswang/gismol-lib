# ============================================================
# FILE: gismol/nlp/text_to_coh.py
# ============================================================
"""Compile natural-language descriptions into :class:`COHObject` instances."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from ..core.coh_object import Attribute, COHObject, Method
from ..core.constraints import GoalConstraint, IdentityConstraint, TriggerConstraint
from ..core.daemons import BoundaryDaemon, InvariantDaemon
from ..neural.components import MLPComponent
from .constraint_parser import ConstraintParser, ParsedConstraints
from .embedders import BagOfWordsEmbedder, TfidfEmbedder
from .entity_extractor import Entity, EntityExtractor
from .normalizers import normalize
from .parsers import Document, parse
from .relations_miner import Relation, RelationMiner

__all__ = ["TextToCOH", "text_to_coh"]

_NUMERIC_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


class TextToCOH:
    """Pipeline: text -> Document -> entities/relations -> constraints -> COH."""

    def __init__(self, extractor: Optional[EntityExtractor] = None,
                 miner: Optional[RelationMiner] = None,
                 parser: Optional[ConstraintParser] = None,
                 attach_neural: bool = True,
                 attach_daemons: bool = True):
        self.extractor = extractor or EntityExtractor()
        self.miner = miner or RelationMiner()
        self.parser = parser or ConstraintParser()
        self.attach_neural = attach_neural
        self.attach_daemons = attach_daemons

    # ------------------------------------------------------------------
    def build(self, text: str, name: str = "system",
              description: str = "") -> COHObject:
        doc = parse(text)
        entities = self.extractor.extract(text)
        relations = self.miner.mine(text, entities, doc)
        parsed = self.parser.parse(text)

        coh = COHObject(name=name, description=description or text[:200])

        # ---- attributes from numeric entities and identifiers -----------
        for ident in self._identifiers(doc, entities):
            coh.add_attribute(Attribute(ident, 0.0, description="extracted from text"))
        for ent in entities:
            if ent.label == "NUMBER" and isinstance(ent.value, float):
                key = self._nearest_identifier(doc, ent) or f"quantity_{int(ent.value)}"
                if key not in coh.A:
                    coh.add_attribute(Attribute(key, float(ent.value),
                                                description=f"from {ent.text!r}"))

        # ---- children from is-a / has relations -------------------------
        for rel in relations:
            if rel.predicate in ("is_a", "has", "contains", "includes"):
                child_name = self._clean(rel.obj)
                if child_name and child_name not in coh.C:
                    coh.add_child(COHObject(name=child_name,
                                            description=f"from relation {rel.predicate}"))

        # ---- methods from action-like relations -------------------------
        for rel in relations:
            if rel.predicate in ("controls", "produces", "influences"):
                mname = self._clean(rel.predicate + "_" + rel.obj)
                if mname not in coh.M:
                    coh.add_method(Method(mname, lambda s, a=None: (s, 0.0),
                                          description=rel.evidence))

        # ---- constraints -------------------------------------------------
        for c in parsed.identity:
            coh.add_identity_constraint(c)
        for c in parsed.triggers:
            coh.add_trigger_constraint(c)
        for c in parsed.goals:
            coh.add_goal_constraint(c)

        # ---- neural components ------------------------------------------
        if self.attach_neural:
            n_features = max(1, len(coh.A))
            coh.add_neural(MLPComponent("predictor", [n_features, max(4, n_features * 2), 1],
                                        seed=abs(hash(name)) % 10 ** 6),
                           name="predictor")

        # ---- daemons -----------------------------------------------------
        if self.attach_daemons:
            if coh.I:
                coh.add_daemon(InvariantDaemon(constraints=list(coh.I)))
            bounds = {a.name: (a.bounds or (-1e6, 1e6)) for a in coh.A.values()}
            if bounds:
                coh.add_daemon(BoundaryDaemon(bounds))

        # ---- embedder ----------------------------------------------------
        coh.set_embedder(BagOfWordsEmbedder())

        coh.metadata["source_text"] = text
        coh.metadata["entities"] = [e.to_dict() for e in entities]
        coh.metadata["relations"] = [r.to_dict() for r in relations]
        return coh

    # ------------------------------------------------------------------
    @staticmethod
    def _identifiers(doc: Document, entities: Sequence[Entity]) -> List[str]:
        stop = {"must", "shall", "should", "the", "a", "an", "is", "are", "be",
                "if", "then", "when", "and", "or", "not", "no", "at", "least",
                "most", "more", "less", "than", "between", "of", "in", "on"}
        seen: List[str] = []
        for token in doc.tokens:
            if token.pos in ("NN", "NNP") and token.lemma not in stop:
                cleaned = TextToCOH._clean(token.text)
                if cleaned and cleaned not in seen:
                    seen.append(cleaned)
        for ent in entities:
            if ent.label == "IDENTIFIER":
                cleaned = TextToCOH._clean(ent.text)
                if cleaned not in seen:
                    seen.append(cleaned)
        return seen[:24]

    @staticmethod
    def _nearest_identifier(doc: Document, entity: Entity) -> Optional[str]:
        best, best_distance = None, 10 ** 9
        for token in doc.tokens:
            if token.pos in ("NN", "NNP"):
                distance = abs(token.index * 6 - entity.start)
                if distance < best_distance:
                    best, best_distance = TextToCOH._clean(token.text), distance
        return best

    @staticmethod
    def _clean(name: str) -> str:
        return re.sub(r"[^0-9a-zA-Z_\.]", "", name.replace(" ", "_")).lower()

    def __repr__(self):
        return "TextToCOH(...)"


def text_to_coh(text: str, name: str = "system", **kwargs) -> COHObject:
    return TextToCOH(**kwargs).build(text, name=name)
