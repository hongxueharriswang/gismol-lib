# ============================================================
# FILE: gismol/nlp/response_generator.py
# ============================================================
"""Template-based natural-language generation for COH objects."""

from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from ..core.coh_object import COHObject

__all__ = ["ResponseGenerator"]


_TEMPLATES = {
    "greeting": ["Hello. I am a COH-based assistant.", "Hi there — ready to model."],
    "define": ["Created COH object {name!r} with {n_attrs} attributes and "
               "{n_children} components."],
    "describe": ["{name}: {n_attrs} attributes, {n_methods} methods, "
                 "{n_children} components, {n_ident} invariants, "
                 "{n_triggers} triggers, {n_goals} goals, {n_daemons} daemons."],
    "constrain": ["Added {n_ident} invariants and {n_triggers} triggers."],
    "optimize": ["Registered {n_goals} objective(s)."],
    "query": ["Here is what I know: {details}"],
    "simulate": ["Simulated {steps} steps; final goal value {goal:.4f}."],
    "compare": ["Similarity between the two objects is {similarity:.3f}."],
    "validate": ["Soundness: {sound}. Acyclic: {acyclic}."],
    "chat": ["I can define, constrain, optimise, simulate and validate COH objects."],
    "unknown": ["I did not understand that; try 'define a system with x and y'."],
}


class ResponseGenerator:
    """Produces natural-language responses from structured COH results."""

    def __init__(self, templates: Optional[Mapping[str, Sequence[str]]] = None,
                 seed: int = 0, verbose: bool = False):
        self.templates: Dict[str, List[str]] = {
            k: list(v) for k, v in (templates or _TEMPLATES).items()
        }
        self.rng = random.Random(seed)
        self.verbose = verbose

    def generate(self, intent: str, context: Optional[Mapping[str, Any]] = None) -> str:
        templates = self.templates.get(intent) or self.templates.get("unknown", ["{data}"])
        template = self.rng.choice(templates)
        context = dict(context or {})
        try:
            return template.format(**context)
        except (KeyError, IndexError, ValueError):
            return template

    # -- specialised renderers ---------------------------------------------
    def describe_coh(self, coh: COHObject, detailed: bool = False) -> str:
        report = coh.component_report()
        base = (f"COH object '{coh.name}': "
                f"{report['A']} attributes, {report['M']} methods, "
                f"{report['C']} components, {report['I']} invariants, "
                f"{report['T']} triggers, {report['G']} goals, "
                f"{report['D']} daemons, embedder={report['E']}.")
        if not detailed:
            return base
        lines = [base]
        if coh.I:
            lines.append("Invariants: " + ", ".join(c.name for c in coh.I))
        if coh.T:
            lines.append("Triggers: " + ", ".join(c.name for c in coh.T))
        if coh.G:
            lines.append("Goals: " + ", ".join(f"{c.name}({c.direction})" for c in coh.G))
        if coh.D:
            lines.append("Daemons: " + ", ".join(d.name for d in coh.D))
        if coh.C:
            lines.append("Components: " + ", ".join(coh.C))
        return "\n".join(lines)

    def explain_constraints(self, coh: COHObject) -> str:
        lines = []
        for c in coh.I:
            lines.append(f"Invariant {c.name}: {c.description or 'must always hold'}")
        for c in coh.T:
            lines.append(f"Trigger {c.name}: on event {c.event!r} -> {c.action}")
        for c in coh.G:
            lines.append(f"Objective {c.name}: {c.direction}imise "
                         f"(weight {c.weight})")
        return "\n".join(lines) or "No constraints registered."

    def describe_violations(self, coh: COHObject, state=None) -> str:
        state = state if state is not None else coh.state
        broken = [c.name for c in coh.I if not c.satisfied(state)]
        if not broken:
            return "All invariants are satisfied."
        return "Violated invariants: " + ", ".join(broken)

    def render_soundness(self, report: Mapping[str, Any]) -> str:
        return (f"Sound: {report.get('sound')}; "
                f"hierarchical: {report.get('hierarchical_sound')}; "
                f"violations: {report.get('violations') or 'none'}")

    def __repr__(self):
        return f"ResponseGenerator(templates={len(self.templates)})"
    