# ============================================================
# FILE: gismol/reasoners/domain.py
# ============================================================
"""Domain-specialised reasoning using declarative domain profiles."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from ..core import expression as E
from ..core.coh_object import COHObject
from .base import KnowledgeBase, Query, ReasoningResult, Reasoner, Rule

__all__ = [
    "DomainProfile", "DomainReasoner", "register_domain", "get_domain",
    "DOMAIN_PROFILES",
]


class DomainProfile:
    """A declarative description of a problem domain.

    Attributes
    ----------
    name:
        Domain identifier.
    vocabulary:
        Domain-specific terms used for keyword matching.
    rules:
        List of :class:`Rule` applied during forward chaining.
    invariants:
        Human-readable invariants the domain expects.
    metrics:
        Names of the objective attributes.
    entity_types:
        Labels the entity extractor should pay attention to.
    """

    def __init__(self, name: str, vocabulary: Sequence[str] = (),
                 rules: Optional[Iterable[Rule]] = None,
                 invariants: Sequence[str] = (),
                 metrics: Sequence[str] = (),
                 entity_types: Sequence[str] = (),
                 description: str = ""):
        self.name = name
        self.vocabulary = list(vocabulary)
        self.rules = list(rules or [])
        self.invariants = list(invariants)
        self.metrics = list(metrics)
        self.entity_types = list(entity_types)
        self.description = description

    def score(self, text: str) -> float:
        if not self.vocabulary:
            return 0.0
        low = (text or "").lower()
        hits = sum(1 for term in self.vocabulary if term in low)
        return hits / max(1, len(self.vocabulary))

    def knowledge_base(self, coh: Optional[COHObject] = None) -> KnowledgeBase:
        kb = KnowledgeBase(rules=list(self.rules))
        kb.add_fact("domain", self.name)
        if coh is not None:
            for key, value in getattr(coh.state, "flatten", lambda: {})().items():
                kb.add_fact(key, value)
        return kb

    def __repr__(self):
        return f"DomainProfile({self.name!r}, rules={len(self.rules)})"


# --------------------------------------------------------------------------
DOMAIN_PROFILES: Dict[str, DomainProfile] = {}


def register_domain(profile: DomainProfile) -> DomainProfile:
    DOMAIN_PROFILES[profile.name] = profile
    return profile


def get_domain(name: str) -> DomainProfile:
    try:
        return DOMAIN_PROFILES[name]
    except KeyError as exc:
        raise KeyError(f"unknown domain {name!r}; known: {sorted(DOMAIN_PROFILES)}") from exc


# --------------------------------------------------------------------------
class DomainReasoner(Reasoner):
    """Applies a :class:`DomainProfile` to a COH object."""

    name = "domain_reasoner"

    def __init__(self, profile: Optional[DomainProfile] = None, auto_detect: bool = True):
        self.profile = profile
        self.auto_detect = auto_detect

    def select_profile(self, text: str) -> Optional[DomainProfile]:
        if not DOMAIN_PROFILES:
            return None
        best = max(DOMAIN_PROFILES.values(), key=lambda p: p.score(text))
        return best if best.score(text) > 0.0 else None

    def infer(self, coh: COHObject, query: Query) -> ReasoningResult:
        profile = self.profile or self.select_profile(
            query.target + " " + (coh.description or "") + " " + coh.name)
        if profile is None:
            return ReasoningResult(None, 0.1, ["no matching domain profile"], {}, self.name)

        kb = profile.knowledge_base(coh)
        trace = [f"domain={profile.name}", f"vocabulary hits applied"]
        facts = kb.infer(50)
        trace.append(f"derived {len(facts)} facts")

        metric_values = {m: facts.get(m) for m in profile.metrics if m in facts}
        invariant_status = {inv: self._check_invariant(coh, inv)
                            for inv in profile.invariants}

        answer = {
            "domain": profile.name,
            "metrics": metric_values,
            "invariants": invariant_status,
            "facts": {k: v for k, v in facts.items() if k not in ("domain",)},
        }
        confidence = 0.8 if metric_values or invariant_status else 0.4
        return ReasoningResult(answer, confidence, trace, answer, self.name)

    @staticmethod
    def _check_invariant(coh: COHObject, text: str) -> Optional[bool]:
        for constraint in coh.I:
            if constraint.description and text.lower() in constraint.description.lower():
                return constraint.satisfied(coh.state)
        return None

    def __repr__(self):
        return f"DomainReasoner(profile={self.profile.name if self.profile else None!r})"


# --------------------------------------------------------------------------
# Built-in domain profiles (illustrative, drawn from Section 7 of the paper)
# --------------------------------------------------------------------------
register_domain(DomainProfile(
    name="physics",
    vocabulary=["force", "mass", "acceleration", "energy", "momentum", "velocity",
                "position", "gravity", "temperature", "pressure"],
    rules=[
        Rule("newton_second", lambda f: "mass" in f and "acceleration" in f,
             {"force": None}, 0.95, "F = m a"),
        Rule("kinetic_energy", lambda f: "mass" in f and "velocity" in f,
             {"kinetic_energy": None}, 0.9, "E = 1/2 m v^2"),
        Rule("ideal_gas", lambda f: "pressure" in f and "temperature" in f,
             {"volume": None}, 0.8, "PV = nRT"),
    ],
    invariants=["energy conservation", "mass balance"],
    metrics=["force", "energy"],
    entity_types=["NUMBER", "IDENTIFIER"],
    description="Classical mechanics and thermodynamics",
))

register_domain(DomainProfile(
    name="biology",
    vocabulary=["gene", "protein", "cell", "metabolite", "expression", "enzyme",
                "pathway", "tissue", "homeostasis", "concentration"],
    rules=[
        Rule("gene_expression", lambda f: "gene" in f,
             {"protein": None}, 0.8, "gene -> protein"),
        Rule("mass_balance", lambda f: "metabolite" in f,
             {"flux": None}, 0.85, "metabolite balance"),
        Rule("homeostasis", lambda f: "setpoint" in f and "concentration" in f,
             {"homeostatic": True}, 0.9, "set-point regulation"),
    ],
    invariants=["mass balance", "energy conservation", "homeostasis"],
    metrics=["expression", "concentration"],
    entity_types=["NUMBER", "DURATION"],
    description="Systems biology and physiology",
))

register_domain(DomainProfile(
    name="finance",
    vocabulary=["price", "portfolio", "risk", "liquidity", "order", "market",
                "arbitrage", "margin", "capital", "volatility"],
    rules=[
        Rule("no_arbitrage", lambda f: "price" in f and "price_forward" in f,
             {"arbitrage": False}, 0.9, "no-arbitrage condition"),
        Rule("margin_requirement", lambda f: "portfolio" in f and "margin" in f,
             {"margin_ok": True}, 0.85, "capital adequacy"),
        Rule("circuit_breaker", lambda f: f.get("volatility", 0) > 0.2,
             {"trading_halt": True}, 0.9, "circuit breaker trigger"),
    ],
    invariants=["no-arbitrage", "market clearing", "capital adequacy"],
    metrics=["price", "risk"],
    entity_types=["MONEY", "PERCENT", "NUMBER"],
    description="Financial markets and systemic risk",
))

register_domain(DomainProfile(
    name="smart_city",
    vocabulary=["traffic", "energy", "grid", "water", "demand", "congestion",
                "emissions", "transport", "utility", "resilience"],
    rules=[
        Rule("load_balance", lambda f: "demand" in f and "supply" in f,
             {"balanced": True}, 0.85, "generation matches demand"),
        Rule("congestion", lambda f: f.get("traffic", 0) > 0.8,
             {"reroute": True}, 0.9, "congestion response"),
        Rule("flow_conservation", lambda f: "flow_in" in f and "flow_out" in f,
             {"conserved": True}, 0.85, "conservation law"),
    ],
    invariants=["flow conservation", "safety margins", "service level"],
    metrics=["demand", "emissions"],
    entity_types=["NUMBER", "PERCENT"],
    description="Urban infrastructure management",
))

register_domain(DomainProfile(
    name="aerospace",
    vocabulary=["orbit", "thrust", "fuel", "delta-v", "attitude", "telemetry",
                "trajectory", "radiation", "payload", "window"],
    rules=[
        Rule("delta_v", lambda f: "thrust" in f and "mass" in f,
             {"delta_v": None}, 0.9, "Tsiolkovsky rocket equation"),
        Rule("communication_window", lambda f: "position" in f,
             {"window_open": None}, 0.75, "ground-station visibility"),
        Rule("thermal_limit", lambda f: f.get("temperature", 0) > 350,
             {"thermal_alarm": True}, 0.9, "thermal constraint"),
    ],
    invariants=["orbital mechanics", "mass balance", "thermal limits"],
    metrics=["fuel", "science_return"],
    entity_types=["NUMBER", "DURATION", "DATE"],
    description="Space mission planning and control",
))
