# ============================================================
# FILE: gismol/categorical/limits.py
# ============================================================
"""Limits and colimits; constraint satisfaction as a limit (Theorem 3.3)."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from ..core.constraints import IdentityConstraint

__all__ = ["Diagram", "Cone", "Limit", "constraint_limit", "pullback", "product_set"]


class Diagram:
    """A small diagram: a set of labelled nodes with morphisms between them."""

    def __init__(self, name: str = "D"):
        self.name = name
        self.nodes: Dict[str, Any] = {}
        self.edges: List[Tuple[str, str, Any]] = []

    def add_node(self, key: str, value: Any):
        self.nodes[key] = value
        return value

    def add_edge(self, src: str, dst: str, fn: Callable):
        self.edges.append((src, dst, fn))

    def __repr__(self):
        return f"Diagram({self.name!r}, nodes={list(self.nodes)})"


class Cone:
    """A cone over a diagram: an apex plus one leg per diagram node."""

    def __init__(self, apex: Any, legs: Mapping[str, Callable]):
        self.apex = apex
        self.legs = dict(legs)

    def commutes(self, diagram: Diagram) -> bool:
        for src, dst, fn in diagram.edges:
            for x in self.apex if isinstance(self.apex, (list, tuple, set)) else [self.apex]:
                try:
                    if self.legs[dst](x) != fn(self.legs[src](x)):
                        return False
                except Exception:
                    return False
        return True

    def __repr__(self):
        return f"Cone(apex={self.apex!r}, legs={list(self.legs)})"


class Limit:
    """The universal cone over a diagram."""

    def __init__(self, cone: Cone, universal: bool = True, name: str = "lim"):
        self.cone = cone
        self.universal = universal
        self.name = name

    @property
    def apex(self):
        return self.cone.apex

    def __repr__(self):
        return f"Limit({self.name!r}, universal={self.universal}, |apex|={len(self.apex) if hasattr(self.apex, '__len__') else '?'})"


# --------------------------------------------------------------------------
def constraint_limit(constraints: Sequence[IdentityConstraint],
                     domain: Iterable[Any]) -> Limit:
    """The admissible region as the limit of the constraint preimages
    (Theorem 3.3)::

        S_adm = lim { phi_i^{-1}(true) }

    ``domain`` is a finite sample of candidate states (the ambient object of
    the diagram).  The apex of the resulting limit is the intersection of the
    preimages.
    """
    states = list(domain)
    diagram = Diagram("constraints")
    legs: Dict[str, Callable] = {}
    for i, constraint in enumerate(constraints):
        key = f"phi_{i}"
        diagram.add_node(key, set())
        legs[key] = lambda s, c=constraint: c.satisfied(s)
    admissible = [s for s in states if all(c.satisfied(s) for c in constraints)]
    cone = Cone(apex=admissible, legs=legs)
    return Limit(cone, universal=True, name="S_adm")


def product_set(sets: Sequence[Iterable[Any]]) -> Set[Tuple]:
    """Cartesian product of a family of sets (the categorical product in Set)."""
    if not sets:
        return {()}
    result: Set[Tuple] = {()}
    for s in sets:
        result = {t + (x,) for t in result for x in s}
    return result


def pullback(f: Callable, g: Callable, a_domain: Iterable[Any],
             b_domain: Iterable[Any]) -> Set[Tuple[Any, Any]]:
    """Categorical pullback ``{(a, b) | f(a) == g(b)}``."""
    result: Set[Tuple[Any, Any]] = set()
    for a in a_domain:
        fa = f(a)
        for b in b_domain:
            if fa == g(b):
                result.add((a, b))
    return result
