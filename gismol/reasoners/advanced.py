# ============================================================
# FILE: gismol/reasoners/advanced.py
# ============================================================
"""Higher-order, analogical, meta and causal reasoners."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from ..categorical.category import COHCategory, COHMorphism
from ..categorical.functors import ComponentFunctor
from ..core.coh_object import COHObject
from .base import Query, ReasoningResult, Reasoner, Rule
from .domain import DomainReasoner
from .general import ConstraintReasoner, ForwardChainer, GoalSolver

__all__ = ["CategoricalReasoner", "AnalogicalReasoner", "MetaReasoner", "CausalReasoner"]


# --------------------------------------------------------------------------
class CategoricalReasoner(Reasoner):
    """Reasoning that exploits the cartesian-closed structure of **COH**."""

    name = "categorical_reasoner"

    def __init__(self):
        self.category = COHCategory()
        self.functor = ComponentFunctor()

    def infer(self, coh: COHObject, query: Query) -> ReasoningResult:
        self.category.register(coh)
        operation = query.parameters.get("operation", "structure")
        trace = [f"registered {coh.name!r} in COH"]
        derived: Dict[str, Any] = {}

        if operation == "product":
            other = query.parameters.get("other")
            if not isinstance(other, COHObject):
                return ReasoningResult(None, 0.1, trace + ["no 'other' object"], {}, self.name)
            product = self.category.product(coh, other)
            derived["product"] = product.name
            answer = {"product": product.name, "components": list(product.C)}
            trace.append("constructed product")
        elif operation == "exponential":
            other = query.parameters.get("other")
            if not isinstance(other, COHObject):
                return ReasoningResult(None, 0.1, trace + ["no 'other' object"], {}, self.name)
            exp = self.category.exponential(coh, other)
            derived["exponential"] = exp.name
            answer = {"exponential": exp.name, "arity": len(exp.A)}
            trace.append("constructed exponential (function object)")
        else:
            graph = self.functor.map_object(coh)
            derived["dag"] = graph.to_dict()
            answer = {
                "nodes": graph.nodes,
                "edges": graph.edges,
                "acyclic": not graph.has_cycle(),
                "cartesian_closed": self.category.is_cartesian_closed(),
            }
            trace.append(f"DAG has {len(graph.nodes)} nodes")

        return ReasoningResult(answer, 0.9, trace, derived, self.name)


# --------------------------------------------------------------------------
class AnalogicalReasoner(Reasoner):
    """Structure-mapping analogy between two COH objects."""

    name = "analogical_reasoner"

    def __init__(self, similarity_threshold: float = 0.2):
        self.threshold = float(similarity_threshold)

    def infer(self, coh: COHObject, query: Query) -> ReasoningResult:
        other = query.parameters.get("other")
        if not isinstance(other, COHObject):
            return ReasoningResult(None, 0.1, ["no comparison object supplied"], {}, self.name)

        embedding_similarity = coh.similarity(other)
        mapping = self._structure_mapping(coh, other)
        trace = [f"embedding similarity = {embedding_similarity:.3f}",
                 f"mapped {len(mapping)} components"]
        answer = {
            "embedding_similarity": embedding_similarity,
            "structure_mapping": mapping,
            "analogous": embedding_similarity >= self.threshold and bool(mapping),
        }
        confidence = min(1.0, abs(embedding_similarity) + 0.2 * bool(mapping))
        return ReasoningResult(answer, confidence, trace, answer, self.name)

    @staticmethod
    def _structure_mapping(a: COHObject, b: COHObject) -> Dict[str, str]:
        """Greedy matching of components by (kind, arity) signature."""
        sig_a = {name: _signature(child) for name, child in a.C.items()}
        sig_b = {name: _signature(child) for name, child in b.C.items()}
        mapping: Dict[str, str] = {}
        used: Set[str] = set()
        for aname, asig in sig_a.items():
            best, best_score = None, 0.0
            for bname, bsig in sig_b.items():
                if bname in used:
                    continue
                score = _signature_similarity(asig, bsig)
                if score > best_score:
                    best, best_score = bname, score
            if best is not None and best_score > 0.0:
                mapping[aname] = best
                used.add(best)
        return mapping


def _signature(coh: COHObject) -> Tuple[int, int, int, int]:
    return (len(coh.A), len(coh.M), len(coh.I), len(coh.G))


def _signature_similarity(a: Tuple[int, ...], b: Tuple[int, ...]) -> float:
    if not a or not b:
        return 0.0
    total = sum(1 for x, y in zip(a, b) if x == y)
    closeness = sum(1.0 / (1.0 + abs(x - y)) for x, y in zip(a, b)) / len(a)
    return 0.6 * (total / len(a)) + 0.4 * closeness


# --------------------------------------------------------------------------
class MetaReasoner(Reasoner):
    """Selects and combines reasoners based on the query and object profile."""

    name = "meta_reasoner"

    def __init__(self, reasoners: Optional[Mapping[str, Reasoner]] = None):
        self.reasoners: Dict[str, Reasoner] = dict(reasoners or {
            "general": ForwardChainer(),
            "constraint": ConstraintReasoner(),
            "goal": GoalSolver(),
            "domain": DomainReasoner(),
            "analogy": AnalogicalReasoner(),
            "categorical": CategoricalReasoner(),
        })

    def select(self, coh: COHObject, query: Query) -> List[str]:
        low = (query.target or "").lower()
        chosen: List[str] = []
        if any(k in low for k in ("violat", "constraint", "invariant", "satisf")):
            chosen.append("constraint")
        if any(k in low for k in ("max", "min", "optimi", "plan", "best")):
            chosen.append("goal")
        if any(k in low for k in ("similar", "analog", "like", "compare")):
            chosen.append("analogy")
        if any(k in low for k in ("structure", "product", "functor", "dag", "categor")):
            chosen.append("categorical")
        if any(k in low for k in ("domain", "physics", "biology", "financ", "city")):
            chosen.append("domain")
        if not chosen:
            chosen = ["general"]
            if coh.G:
                chosen.append("goal")
            if coh.I:
                chosen.append("constraint")
        return chosen

    def infer(self, coh: COHObject, query: Query) -> ReasoningResult:
        selected = self.select(coh, query)
        trace = [f"selected {selected}"]
        results: Dict[str, Any] = {}
        confidences: List[float] = []
        for key in selected:
            reasoner = self.reasoners.get(key)
            if reasoner is None:
                continue
            try:
                result = reasoner.infer(coh, query)
            except Exception as exc:  # pragma: no cover - defensive
                trace.append(f"{key} failed: {exc}")
                continue
            results[key] = result.answer
            confidences.append(result.confidence)
            trace.append(f"{key}: confidence={result.confidence:.2f}")

        if not results:
            return ReasoningResult(None, 0.0, trace, {}, self.name)
        best = max(selected, key=lambda k: self.reasoners[k].name if k in results else "")
        confidence = sum(confidences) / len(confidences)
        return ReasoningResult(results, confidence, trace, results, self.name)


# --------------------------------------------------------------------------
class CausalReasoner(Reasoner):
    """Causal reasoning over an intervention graph extracted from the COH."""

    name = "causal_reasoner"

    def __init__(self):
        self.graph: Dict[str, List[str]] = {}

    def build_graph(self, coh: COHObject) -> Dict[str, List[str]]:
        graph: Dict[str, List[str]] = {name: [] for name in coh.A}
        for trg in coh.T:
            variables = trg.condition_expr.variables() if trg.condition_expr else set()
            target = trg.action if isinstance(trg.action, str) else None
            for var in variables:
                if target:
                    graph.setdefault(var, []).append(target)
        for rel in coh.metadata.get("relations", []):
            if rel.get("predicate") in ("causes", "influences", "leads_to", "produces"):
                graph.setdefault(rel["subject"], []).append(rel["object"])
        self.graph = graph
        return graph

    def infer(self, coh: COHObject, query: Query) -> ReasoningResult:
        graph = self.build_graph(coh)
        action = query.parameters.get("do", query.target)
        trace = [f"causal graph has {len(graph)} nodes"]

        if query.parameters.get("intervene"):
            effect = self._descendants(action)
            trace.append(f"do({action}) affects {sorted(effect)}")
            answer = {"intervention": action, "effects": sorted(effect)}
            return ReasoningResult(answer, 0.85, trace, answer, self.name)

        causes = [src for src, dsts in graph.items() if action in dsts]
        effects = graph.get(action, [])
        answer = {"causes_of": causes, "effects_of": effects,
                  "graph": graph}
        confidence = 0.8 if causes or effects else 0.3
        return ReasoningResult(answer, confidence, trace, answer, self.name)

    def _descendants(self, node: str) -> Set[str]:
        seen: Set[str] = set()
        stack = list(self.graph.get(node, []))
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(self.graph.get(current, []))
        return seen
    