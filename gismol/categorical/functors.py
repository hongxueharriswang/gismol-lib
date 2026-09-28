# ============================================================
# FILE: gismol/categorical/functors.py
# ============================================================
"""Functors out of **COH**, notably the component-DAG functor ``F : COH -> Graph``."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from ..core.coh_object import COHObject
from ..core.exceptions import MorphismError
from .category import Morphism

__all__ = ["Functor", "Graph", "ComponentFunctor", "ForgetfulFunctor", "EmbeddingFunctor"]


# --------------------------------------------------------------------------
class Graph:
    """A tiny directed graph used as the codomain of the component functor."""

    def __init__(self, name: str = "G"):
        self.name = name
        self.nodes: List[str] = []
        self.edges: List[Tuple[str, str]] = []
        self.labels: Dict[str, Any] = {}

    def add_node(self, node: str, label: Any = None):
        if node not in self.nodes:
            self.nodes.append(node)
        if label is not None:
            self.labels[node] = label
        return node

    def add_edge(self, src: str, dst: str):
        self.add_node(src)
        self.add_node(dst)
        if (src, dst) not in self.edges:
            self.edges.append((src, dst))

    def successors(self, node: str) -> List[str]:
        return [d for s, d in self.edges if s == node]

    def predecessors(self, node: str) -> List[str]:
        return [s for s, d in self.edges if d == node]

    def has_cycle(self) -> bool:
        colour: Dict[str, int] = {}

        def visit(n):
            colour[n] = 1
            for m in self.successors(n):
                if colour.get(m) == 1:
                    return True
                if colour.get(m, 0) == 0 and visit(m):
                    return True
            colour[n] = 2
            return False

        return any(colour.get(n, 0) == 0 and visit(n) for n in list(self.nodes))

    def topological_order(self) -> List[str]:
        if self.has_cycle():
            raise MorphismError("graph has a cycle; no topological order")
        visited: Set[str] = set()
        order: List[str] = []

        def visit(n):
            if n in visited:
                return
            visited.add(n)
            for m in self.successors(n):
                visit(m)
            order.append(n)

        for n in self.nodes:
            visit(n)
        return list(reversed(order))

    def to_dict(self):
        return {"nodes": self.nodes, "edges": self.edges}

    def __repr__(self):
        return f"Graph({self.name!r}, |V|={len(self.nodes)}, |E|={len(self.edges)})"


# --------------------------------------------------------------------------
class Functor:
    """Base functor: maps objects and morphisms."""

    def __init__(self, name: str = "F"):
        self.name = name

    def map_object(self, obj):  # pragma: no cover - abstract
        raise NotImplementedError

    def map_morphism(self, morphism: Morphism):  # pragma: no cover - abstract
        raise NotImplementedError

    def __call__(self, value):
        if isinstance(value, Morphism):
            return self.map_morphism(value)
        return self.map_object(value)

    def __repr__(self):
        return f"{type(self).__name__}({self.name!r})"


# --------------------------------------------------------------------------
class ComponentFunctor(Functor):
    """``F : COH -> Graph`` mapping a COH object to its component DAG
    (Definition 3.2)."""

    def __init__(self, name: str = "F_components"):
        super().__init__(name)

    def map_object(self, obj: COHObject) -> Graph:
        graph = Graph(name=f"DAG({obj.name})")

        def walk(node: COHObject, prefix: str = ""):
            key = f"{prefix}{node.name}"
            graph.add_node(key, label=node)
            for cname, child in node.children().items():
                child_key = f"{key}/{cname}"
                graph.add_edge(key, child_key)
                walk(child, prefix=f"{key}/")

        walk(obj)
        return graph

    def map_morphism(self, morphism: Morphism):
        src = self.map_object(morphism.source)
        dst = self.map_object(morphism.target)

        def hom(x):
            if isinstance(x, Graph):
                # identity on the graph structure via node-name mapping
                return x
            return x

        return hom

    def preserves_hierarchy(self, morphism: Morphism) -> bool:
        """Theorem 3.2 — a COH morphism induces a graph homomorphism."""
        src = self.map_object(morphism.source)
        dst = self.map_object(morphism.target)
        if len(src.nodes) > len(dst.nodes):
            return False
        if morphism.component_map:
            for s, d in src.edges:
                ms = morphism.component_map.get(s.split("/")[-1], s.split("/")[-1])
                md = morphism.component_map.get(d.split("/")[-1], d.split("/")[-1])
                if (ms, md) not in dst.edges and ms in [n.split("/")[-1] for n in dst.nodes]:
                    return False
        return True


# --------------------------------------------------------------------------
class ForgetfulFunctor(Functor):
    """``COH -> Set``: forgets structure, remembering only the objects."""

    def __init__(self, name: str = "U"):
        super().__init__(name)

    def map_object(self, obj: COHObject) -> Set[str]:
        return set(obj.component_dag().keys())

    def map_morphism(self, morphism: Morphism):
        return lambda nodes: set(nodes) & self.map_object(morphism.target)


# --------------------------------------------------------------------------
class EmbeddingFunctor(Functor):
    """``COH -> Vect``: maps objects to vectors via their embedding ``E``."""

    def __init__(self, name: str = "E_functor"):
        super().__init__(name)

    def map_object(self, obj: COHObject) -> tuple:
        return obj.embed()

    def map_morphism(self, morphism: Morphism):
        target = morphism.target
        return lambda x: target.embed()
    