# ============================================================
# FILE: gismol/nlp/validation.py
# ============================================================
"""Validation of COH objects (hand-built or produced by the NLP pipeline)."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from ..core.coh_object import COHObject
from ..core.exceptions import ValidationError

__all__ = ["ValidationIssue", "COHValidator", "validate_coh"]


class ValidationIssue:
    """A single problem found in a COH object."""

    def __init__(self, severity: str, message: str, path: str = "",
                 code: str = ""):
        self.severity = severity  # "error" | "warning" | "info"
        self.message = message
        self.path = path
        self.code = code

    def __repr__(self):
        return f"[{self.severity.upper()}] {self.path or '<root>'}: {self.message}"

    def to_dict(self):
        return {"severity": self.severity, "message": self.message,
                "path": self.path, "code": self.code}


class COHValidator:
    """Structural and semantic validation of COH objects.

    Checks:
      * the component hierarchy is acyclic;
      * all nine components are present (minimality);
      * identity constraints are satisfiable on the initial state;
      * goal constraints reference existing attributes;
      * daemons monitor existing attributes;
      * the embedder is set.
    """

    def __init__(self, require_all_components: bool = False,
                 strict_goals: bool = True):
        self.require_all_components = require_all_components
        self.strict_goals = strict_goals

    # ------------------------------------------------------------------
    def validate(self, coh: COHObject) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        self._validate_object(coh, "", issues)
        return issues

    def _validate_object(self, coh: COHObject, path: str, issues: List[ValidationIssue]):
        here = f"{path}/{coh.name}" if path else coh.name

        if not coh.is_acyclic():
            issues.append(ValidationIssue("error", "component hierarchy contains a cycle",
                                          here, "cyclic"))

        report = coh.component_report()
        missing = [k for k in ("C", "A", "M", "N", "E", "I", "T", "G", "D")
                   if not report[k] or report[k] == 0]
        if missing:
            severity = "error" if self.require_all_components else "warning"
            issues.append(ValidationIssue(severity,
                                          f"missing components: {missing}", here, "minimality"))

        initial = coh.initial_state()
        broken = [c.name for c in coh.I if not c.satisfied(initial)]
        if broken:
            issues.append(ValidationIssue("error",
                                          f"initial state violates invariants: {broken}",
                                          here, "initial_infeasible"))

        attribute_names = set(coh.A)
        for goal in coh.G:
            variables = goal.expression.variables() if goal.expression is not None else set()
            unknown = {v for v in variables if v.split(".")[0] not in attribute_names}
            if unknown and self.strict_goals:
                issues.append(ValidationIssue("warning",
                                              f"goal {goal.name!r} references unknown "
                                              f"attributes {sorted(unknown)}", here, "goal_ref"))

        for daemon in coh.D:
            attr = getattr(daemon, "attribute", None)
            if attr and attr.split(".")[0] not in attribute_names and attr not in coh.C:
                issues.append(ValidationIssue("warning",
                                              f"daemon {daemon.name!r} monitors unknown "
                                              f"attribute {attr!r}", here, "daemon_ref"))

        for name, child in coh.C.items():
            self._validate_object(child, here, issues)

    # ------------------------------------------------------------------
    def is_valid(self, coh: COHObject) -> bool:
        return not any(i.severity == "error" for i in self.validate(coh))

    def raise_if_invalid(self, coh: COHObject):
        issues = [i for i in self.validate(coh) if i.severity == "error"]
        if issues:
            raise ValidationError("; ".join(str(i) for i in issues))

    def summary(self, coh: COHObject) -> str:
        issues = self.validate(coh)
        if not issues:
            return f"{coh.name}: valid (no issues)"
        return "\n".join(str(i) for i in issues)

    def __repr__(self):
        return f"COHValidator(require_all_components={self.require_all_components})"


def validate_coh(coh: COHObject, **kwargs) -> bool:
    return COHValidator(**kwargs).is_valid(coh)
