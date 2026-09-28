# ============================================================
# FILE: gismol/nlp/chat_module.py
# ============================================================
"""Single-turn chat pipeline over the COH toolkit."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..core.coh_object import COHObject
from ..core.exceptions import COHError
from .constraint_parser import ConstraintParser, ParsedConstraints
from .entity_extractor import Entity, EntityExtractor
from .intent_recognizer import Intent, IntentRecognizer
from .normalizers import normalize
from .parsers import Document, parse
from .relations_miner import Relation, RelationMiner
from .response_generator import ResponseGenerator
from .text_to_coh import TextToCOH
from .validation import COHValidator

__all__ = ["ChatResponse", "ChatModule"]


class ChatResponse:
    """Structured result of a single conversational turn."""

    def __init__(self, text: str, intent: Intent,
                 entities: Sequence[Entity] = (), relations: Sequence[Relation] = (),
                 constraints: Optional[ParsedConstraints] = None,
                 coh: Optional[COHObject] = None, document: Optional[Document] = None,
                 issues: Sequence[Any] = (), raw_input: str = ""):
        self.text = text
        self.intent = intent
        self.entities = list(entities)
        self.relations = list(relations)
        self.constraints = constraints
        self.coh = coh
        self.document = document
        self.issues = list(issues)
        self.raw_input = raw_input

    def to_dict(self):
        return {
            "text": self.text,
            "intent": self.intent.to_dict(),
            "entities": [e.to_dict() for e in self.entities],
            "relations": [r.to_dict() for r in self.relations],
            "constraints": self.constraints.to_dict() if self.constraints else None,
            "issues": [i.to_dict() for i in self.issues],
        }

    def __repr__(self):
        return f"ChatResponse(intent={self.intent.name!r}, text={self.text[:48]!r})"


class ChatModule:
    """Full NLP pipeline: normalise -> parse -> extract -> reason -> respond."""

    def __init__(self, coh: Optional[COHObject] = None,
                 normalizer=None, intent_recognizer: Optional[IntentRecognizer] = None,
                 entity_extractor: Optional[EntityExtractor] = None,
                 relation_miner: Optional[RelationMiner] = None,
                 constraint_parser: Optional[ConstraintParser] = None,
                 response_generator: Optional[ResponseGenerator] = None,
                 builder: Optional[TextToCOH] = None,
                 validator: Optional[COHValidator] = None):
        from .normalizers import TextNormalizer
        self.normalizer = normalizer or TextNormalizer()
        self.intents = intent_recognizer or IntentRecognizer()
        self.entities = entity_extractor or EntityExtractor()
        self.relations = relation_miner or RelationMiner()
        self.constraints = constraint_parser or ConstraintParser()
        self.responses = response_generator or ResponseGenerator()
        self.builder = builder or TextToCOH(self.entities, self.relations, self.constraints)
        self.validator = validator or COHValidator()
        self.coh: Optional[COHObject] = coh

    # ------------------------------------------------------------------
    def process(self, text: str) -> ChatResponse:
        normalized = self.normalizer(text)
        doc = parse(text)
        intent = self.intents.recognize(normalized)
        entities = self.entities.extract(text)
        relations = self.relations.mine(text, entities, doc)
        constraints = self.constraints.parse(text)

        handler = getattr(self, f"_handle_{intent.name}", None)
        try:
            if handler is not None:
                reply = handler(text, normalized, intent, entities, relations,
                                constraints, doc)
            else:
                reply = self.responses.generate(intent.name)
        except COHError as exc:
            reply = f"I hit a COH error: {exc}"

        return ChatResponse(reply, intent, entities, relations, constraints,
                            self.coh, doc, raw_input=text)

    # ------------------------------------------------------------------
    def _handle_define(self, raw, norm, intent, entities, relations, constraints, doc):
        name = self._guess_name(norm) or "system"
        self.coh = self.builder.build(raw, name=name)
        report = self.coh.component_report()
        return self.responses.generate("define", {
            "name": self.coh.name,
            "n_attrs": report["A"],
            "n_children": report["C"],
        })

    def _handle_constrain(self, raw, norm, intent, entities, relations, constraints, doc):
        if self.coh is None:
            self.coh = self.builder.build(raw, name=self._guess_name(norm) or "system")
            return ("No object existed, so I created one from your constraints.\n" +
                    self.responses.explain_constraints(self.coh))
        for c in constraints.identity:
            self.coh.add_identity_constraint(c)
        for c in constraints.triggers:
            self.coh.add_trigger_constraint(c)
        for c in constraints.goals:
            self.coh.add_goal_constraint(c)
        return self.responses.explain_constraints(self.coh)

    def _handle_optimize(self, raw, norm, intent, entities, relations, constraints, doc):
        if self.coh is None:
            self.coh = self.builder.build(raw, name=self._guess_name(norm) or "system")
        for c in constraints.goals:
            self.coh.add_goal_constraint(c)
        n = len(self.coh.G)
        return self.responses.generate("optimize", {"n_goals": n})

    def _handle_describe(self, raw, norm, intent, entities, relations, constraints, doc):
        if self.coh is None:
            return "There is no COH object in this session yet. Try 'define ...'."
        return self.responses.describe_coh(self.coh, detailed=True)

    def _handle_validate(self, raw, norm, intent, entities, relations, constraints, doc):
        if self.coh is None:
            return "Nothing to validate yet."
        report = self.coh.soundness_report()
        issues = self.validator.validate(self.coh)
        lines = [self.responses.render_soundness(report)]
        lines.extend(str(i) for i in issues)
        return "\n".join(lines)

    def _handle_simulate(self, raw, norm, intent, entities, relations, constraints, doc):
        if self.coh is None:
            return "Define an object before simulating it."
        steps = 10
        if intent.slots.get("numbers"):
            steps = max(1, int(intent.slots["numbers"][0]))
        traj = self.coh.simulate(steps=steps)
        return self.responses.generate("simulate", {
            "steps": steps,
            "goal": traj.cumulative_reward(),
        })

    def _handle_query(self, raw, norm, intent, entities, relations, constraints, doc):
        if self.coh is None:
            return "I do not have a COH object to answer that from."
        details = ", ".join(f"{k}={v!r}" for k, v in list(self.coh.state.items())[:6])
        return self.responses.generate("query", {"details": details})

    def _handle_compare(self, raw, norm, intent, entities, relations, constraints, doc):
        if self.coh is None:
            return "Define at least one object first."
        other = self.builder.build(raw, name="other")
        return self.responses.generate("compare", {
            "similarity": self.coh.similarity(other)})

    def _handle_chat(self, raw, norm, intent, entities, relations, constraints, doc):
        return self.responses.generate("chat")

    # ------------------------------------------------------------------
    @staticmethod
    def _guess_name(normalized: str) -> Optional[str]:
        import re
        m = re.search(r"(?:called|named)\s+([a-z_][a-z0-9_]*)", normalized)
        if m:
            return m.group(1)
        m = re.search(r"define\s+(?:a|an|the)?\s*([a-z_][a-z0-9_]*)", normalized)
        return m.group(1) if m else None

    def reset(self):
        self.coh = None

    def __repr__(self):
        return f"ChatModule(coh={self.coh.name if self.coh else None!r})"
    