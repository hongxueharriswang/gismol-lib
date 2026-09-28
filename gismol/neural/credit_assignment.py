# ============================================================
# FILE: gismol/neural/credit_assignment.py
# ============================================================
"""Hierarchical credit assignment and value decomposition (Section 5.3)."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..core.coh_object import COHObject

__all__ = ["HierarchicalValueDecomposition", "CreditAssigner"]


class HierarchicalValueDecomposition:
    """Implements ``V(O) = sum_{O' in C(O)} w_{O'} V(O') + V_coord(O)``
    (Definition 5.2)."""

    def __init__(self, coh: COHObject, weights: Optional[Mapping[str, float]] = None):
        self.coh = coh
        self._weights: Dict[str, float] = dict(weights or {})
        if not self._weights:
            self._weights = self._uniform_weights()
        self.coordination: float = 0.0

    # -- weights -----------------------------------------------------------
    def _uniform_weights(self) -> Dict[str, float]:
        children = list(self.coh.children())
        if not children:
            return {}
        w = 1.0 / len(children)
        return {name: w for name in children}

    @property
    def weights(self) -> Dict[str, float]:
        return dict(self._weights)

    def set_weights(self, weights: Mapping[str, float]):
        self._weights = {k: float(v) for k, v in weights.items()}
        return self

    # -- decomposition -----------------------------------------------------
    def decompose(self, values: Mapping[str, float],
                  coordination: Optional[float] = None) -> float:
        """Compute the value of the parent from its children's values."""
        total = 0.0
        for name, w in self._weights.items():
            total += w * float(values.get(name, 0.0))
        total += self.coordination if coordination is None else float(coordination)
        return float(total)

    def decomposition_table(self, values: Mapping[str, float]) -> Dict[str, float]:
        table = {name: w * float(values.get(name, 0.0))
                 for name, w in self._weights.items()}
        table["coordination"] = self.coordination
        table["total"] = sum(table.values())
        return table

    def fit(self, samples: Sequence[Tuple[Mapping[str, float], float]],
            lr: float = 0.05, epochs: int = 200) -> Dict[str, float]:
        """Least-mean-squares fit of the weights and the coordination term
        (Theorem 5.3: an optimal decomposition exists)."""
        names = list(self._weights)
        weights = [self._weights[n] for n in names]
        coord = self.coordination
        for _ in range(epochs):
            g_w = [0.0] * len(names)
            g_c = 0.0
            for values, target in samples:
                pred = sum(w * float(values.get(n, 0.0)) for n, w in zip(names, weights)) + coord
                err = pred - float(target)
                for i, n in enumerate(names):
                    g_w[i] += 2.0 * err * float(values.get(n, 0.0))
                g_c += 2.0 * err
            m = max(1, len(samples))
            weights = [w - lr * g / m for w, g in zip(weights, g_w)]
            coord -= lr * g_c / m
        self._weights = dict(zip(names, weights))
        self.coordination = coord
        return {"weights": dict(self._weights), "coordination": coord}

    def bellman_residual(self, samples: Sequence[Tuple[Mapping[str, float], float]]) -> float:
        total = 0.0
        for values, target in samples:
            total += (self.decompose(values) - float(target)) ** 2
        return float(total / max(1, len(samples)))

    def __repr__(self):
        return (f"HierarchicalValueDecomposition({self.coh.name!r}, "
                f"children={list(self._weights)})")


# --------------------------------------------------------------------------
class CreditAssigner:
    """Distributes a scalar return across a COH hierarchy.

    Children receive credit proportional to their value contribution; the
    remainder is assigned to the coordination term (Theorem 5.3).
    """

    def __init__(self, coh: COHObject):
        self.coh = coh

    def contribution_weights(self) -> Dict[str, float]:
        children = list(self.coh.children())
        if not children:
            return {}
        raw = {}
        for name, child in self.coh.children().items():
            raw[name] = 1.0 + 0.1 * len(child.M) + 0.1 * len(child.I)
        total = sum(raw.values()) or 1.0
        return {k: v / total for k, v in raw.items()}

    def assign(self, total_value: float,
               child_values: Optional[Mapping[str, float]] = None) -> Dict[str, Any]:
        weights = self.contribution_weights()
        if not weights:
            return {"self": float(total_value), "children": {}, "coordination": 0.0}

        if child_values is None:
            child_values = {name: float(total_value) for name in weights}

        assigned = {name: weights[name] * float(child_values.get(name, 0.0))
                    for name in weights}
        coordination = float(total_value) - sum(assigned.values())
        return {
            "self": float(total_value) - sum(assigned.values()),
            "children": assigned,
            "coordination": coordination,
            "weights": weights,
        }

    def deep_assign(self, total_value: float) -> Dict[str, Any]:
        """Recursively distribute value through the entire hierarchy."""
        result: Dict[str, Any] = {"value": float(total_value), "children": {}}
        weights = self.contribution_weights()
        for name, child in self.coh.children().items():
            share = weights.get(name, 0.0)
            result["children"][name] = CreditAssigner(child).deep_assign(
                share * float(total_value))
        return result

    def __repr__(self):
        return f"CreditAssigner({self.coh.name!r})"
    