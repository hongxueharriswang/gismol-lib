# ============================================================
# FILE: gismol/core/exceptions.py
# ============================================================
"""Exception hierarchy for the gismol COH toolkit."""

from __future__ import annotations

__all__ = [
    "COHError",
    "ConstraintViolationError",
    "UnsatisfiableConstraintError",
    "CyclicHierarchyError",
    "MethodExecutionError",
    "DaemonError",
    "EmbeddingError",
    "ConvergenceError",
    "MorphismError",
    "ValidationError",
    "RepositoryError",
    "ParseError",
    "ReasoningError",
]


class COHError(Exception):
    """Base class for every error raised by gismol."""


class ConstraintViolationError(COHError):
    """Raised when a state violates an identity constraint."""

    def __init__(self, constraint=None, state=None, message=None):
        self.constraint = constraint
        self.state = state
        name = getattr(constraint, "name", constraint)
        super().__init__(message or f"constraint {name!r} violated")


class UnsatisfiableConstraintError(COHError):
    """Raised when a set of identity constraints has an empty feasible region."""

    def __init__(self, constraints=None, message=None):
        self.constraints = list(constraints or [])
        names = [getattr(c, "name", str(c)) for c in self.constraints]
        super().__init__(message or f"unsatisfiable constraint set: {names}")


class CyclicHierarchyError(COHError):
    """Raised when the component DAG of a COH object contains a cycle."""

    def __init__(self, path=None, message=None):
        self.path = list(path or [])
        super().__init__(
            message or f"cyclic component hierarchy detected: {' -> '.join(map(str, self.path))}"
        )


class MethodExecutionError(COHError):
    """Raised when a method fails, e.g. violates a post-condition."""


class DaemonError(COHError):
    """Raised by constraint daemons on unrecoverable violations."""


class EmbeddingError(COHError):
    """Raised when an object cannot be embedded into the target Hilbert space."""


class ConvergenceError(COHError):
    """Raised when an iterative procedure fails to converge."""


class MorphismError(COHError):
    """Raised when a map between COH objects is not structure preserving."""


class ValidationError(COHError):
    """Raised by :mod:`gismol.nlp.validation` on malformed COH objects."""


class RepositoryError(COHError):
    """Raised by :class:`gismol.core.repository.COHRepository`."""


class ParseError(COHError):
    """Raised when a textual expression or constraint cannot be parsed."""


class ReasoningError(COHError):
    """Raised by the reasoners subpackage."""
    