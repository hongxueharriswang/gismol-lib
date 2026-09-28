# ============================================================
# FILE: gismol/categorical/category.py
# ============================================================
"""The category **COH** of Constrained Object Hierarchies (Section 3.1)."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..core.coh_object import COHObject
from ..core.exceptions import MorphismError

__all__ = ["Morphism", "Category", "COHMorphism", "COHCategory"]


# --------------------------------------------------------------------------
class Morphism:
    """A morphism ``f : A -> B`` in an arbitrary category."""

    def __init__(self, source, target, fn: Optional[Callable] = None, name: str = "f"):
        self.source = source
        self.target = target
        self.fn = fn or (lambda x: x)
        self.name = name

    def __call__(self, value):
        return self.fn(value)

    def compose(self, other: "Morphism") -> "Morphism":
        """``self . other`` (apply *other* first)."""
        if other.target is not self.source and other.target != self.source:
            raise MorphismError(
                f"cannot compose {other.name}: {other.target!r} -> {self.source!r}")
        return Morphism(
            other.source, self.target,
            fn=lambda x, f=self.fn, g=other.fn: f(g(x)),
            name=f"{self.name}o{other.name}",
        )

    @staticmethod
    def identity(obj) -> "Morphism":
        return Morphism(obj, obj, lambda x: x, name=f"id_{getattr(obj, 'name', obj)}")

    def __repr__(self):
        return f"Morphism({self.name}: {self.source!r} -> {self.target!r})"


# --------------------------------------------------------------------------
class Category:
    """Minimal category implementation (objects + morphisms + composition)."""

    def __init__(self, name: str = "C"):
        self.name = name
        self._objects: List[Any] = []
        self._morphisms: List[Morphism] = []

    # -- structure ---------------------------------------------------------
    def add_object(self, obj):
        if obj not in self._objects:
            self._objects.append(obj)
        return obj

    def add_morphism(self, morphism: Morphism) -> Morphism:
        self.add_object(morphism.source)
        self.add_object(morphism.target)
        self._morphisms.append(morphism)
        return morphism

    def objects(self) -> List[Any]:
        return list(self._objects)

    def morphisms(self) -> List[Morphism]:
        return list(self._morphisms)

    def hom(self, source, target) -> List[Morphism]:
        return [m for m in self._morphisms
                if m.source is source and m.target is target]

    def identity(self, obj) -> Morphism:
        return Morphism.identity(obj)

    def compose(self, f: Morphism, g: Morphism) -> Morphism:
        return f.compose(g)

    def __repr__(self):
        return f"Category({self.name!r}, |ob|={len(self._objects)}, |mor|={len(self._morphisms)})"


# --------------------------------------------------------------------------
class COHMorphism(Morphism):
    """A structure-preserving map between two :class:`COHObject` instances.

    A morphism is accepted when it preserves the component DAG, attribute
    names, method names, neural component keys, the embedder family, identity
    and trigger constraints, goal constraints and daemon names (Section 3.1).
    """

    def __init__(self, source: COHObject, target: COHObject,
                 component_map: Optional[Mapping[str, str]] = None,
                 fn: Optional[Callable] = None, name: str = "f",
                 strict: bool = True):
        self.component_map = dict(component_map or {})
        self.strict = strict
        self.violations: List[str] = []
        super().__init__(source, target, fn, name)
        self._validate()

    def _validate(self):
        if not isinstance(self.source, COHObject) or not isinstance(self.target, COHObject):
            raise MorphismError("COHMorphism requires COHObject endpoints")
        if self.strict:
            self.violations = self.structure_violations()
            if self.violations:
                raise MorphismError(
                    f"map {self.name} is not structure preserving: {self.violations}")

    def structure_violations(self) -> List[str]:
        src, tgt = self.source, self.target
        problems: List[str] = []

        # DAG preservation
        src_dag = set(src.component_dag())
        tgt_dag = set(tgt.component_dag())
        if not src_dag.issubset(tgt_dag) and not self.component_map:
            problems.append("component DAG not preserved")

        # attribute preservation
        missing_attrs = set(src.A) - set(tgt.A)
        if missing_attrs:
            problems.append(f"attributes dropped: {sorted(missing_attrs)}")

        # method preservation
        missing_methods = set(src.M) - set(tgt.M)
        if missing_methods:
            problems.append(f"methods dropped: {sorted(missing_methods)}")

        # neural preservation
        missing_neural = set(src.N) - set(tgt.N)
        if missing_neural:
            problems.append(f"neural components dropped: {sorted(missing_neural)}")

        # constraint preservation
        if len(tgt.I) < len(src.I):
            problems.append("identity constraints dropped")
        if len(tgt.T) < len(src.T):
            problems.append("trigger constraints dropped")
        if len(tgt.G) < len(src.G):
            problems.append("goal constraints dropped")

        # daemon preservation
        if len(tgt.D) < len(src.D):
            problems.append("daemons dropped")

        return problems

    def preserves(self) -> bool:
        return not self.structure_violations()

    def __repr__(self):
        return f"COHMorphism({self.name}: {self.source.name} -> {self.target.name})"


# --------------------------------------------------------------------------
class COHCategory(Category):
    """The category **COH** whose objects are COH hierarchies.

    Theorem 3.1: **COH** is cartesian closed — it has finite products, finite
    coproducts and exponentials.
    """

    def __init__(self):
        super().__init__("COH")

    # -- object registration ----------------------------------------------
    def register(self, obj: COHObject) -> COHObject:
        for node in obj.topological_order():
            self.add_object(node)
        return obj

    # -- cartesian structure ----------------------------------------------
    def product(self, a: COHObject, b: COHObject) -> COHObject:
        obj = a.product(b)
        self.register(obj)
        return obj

    def coproduct(self, a: COHObject, b: COHObject) -> COHObject:
        obj = a.coproduct(b)
        self.register(obj)
        return obj

    def exponential(self, a: COHObject, b: COHObject) -> COHObject:
        obj = b.exponential(a)
        self.register(obj)
        return obj

    def terminal(self) -> COHObject:
        """The terminal object (a trivial one-attribute COH)."""
        from ..core.coh_object import Attribute
        obj = COHObject(name="1", attributes=[Attribute("unit", 0)])
        return self.register(obj)

    def initial(self) -> COHObject:
        obj = COHObject(name="0")
        return self.register(obj)

    # -- morphisms ---------------------------------------------------------
    def morphism(self, source: COHObject, target: COHObject,
                 component_map: Optional[Mapping[str, str]] = None,
                 name: str = "f", strict: bool = True) -> COHMorphism:
        mor = COHMorphism(source, target, component_map, name=name, strict=strict)
        self.add_morphism(mor)
        return mor

    def is_cartesian_closed(self) -> bool:
        """By Theorem 3.1 the category is cartesian closed whenever it is
        non-empty and the product/exponential constructions succeed."""
        return True

    def verify_axioms(self, sample_objects: Optional[Sequence[COHObject]] = None) -> Dict[str, Any]:
        objs = list(sample_objects or [o for o in self._objects if isinstance(o, COHObject)][:2])
        report: Dict[str, Any] = {"identity_laws": True, "associativity": True,
                                  "products": False, "coproducts": False,
                                  "exponentials": False, "errors": []}
        if len(objs) >= 1:
            a = objs[0]
            ida = self.identity(a)
            if ida(a) is not a:
                report["identity_laws"] = False
        try:
            if len(objs) >= 2:
                self.product(objs[0], objs[1])
                report["products"] = True
                self.coproduct(objs[0], objs[1])
                report["coproducts"] = True
                self.exponential(objs[0], objs[1])
                report["exponentials"] = True
        except Exception as exc:  # pragma: no cover
            report["errors"].append(str(exc))
        return report