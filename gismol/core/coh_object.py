# ============================================================
# FILE: gismol/core/coh_object.py
# ============================================================
"""The :class:`COHObject` base class — the 9-tuple of Definition 2.1.

Every intelligent system in gismol is (a subclass of) a ``COHObject``::

    O = (C, A, M, N, E, I, T, G, D)

The class also implements the constructions used by the categorical layer
(products, coproducts, exponentials) and the hybrid-dynamical-system view.
"""

from __future__ import annotations

import copy as _copy
import uuid
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from . import expression as E
from .constraints import (
    ConstraintSet,
    GoalConstraint,
    IdentityConstraint,
    TriggerConstraint,
    as_predicate,
)
from .daemons import Daemon, DaemonNetwork, InvariantDaemon
from .dynamics import ContinuousFlow, DiscreteTransition, HybridDynamicalSystem
from .embedding import CompositeEmbedder, Embedder, FeatureEmbedder, HashEmbedder
from .exceptions import (
    ConstraintViolationError,
    CyclicHierarchyError,
    MethodExecutionError,
    MorphismError,
    UnsatisfiableConstraintError,
)

__all__ = ["COHObject", "Attribute", "Method", "State"]


# ==========================================================================
# State
# ==========================================================================
class State(dict):
    """Hierarchical state assignment.

    Own attributes live as top-level keys; each child's state is nested under
    the child's name.  Dotted paths (``"arm.joint.angle"``) traverse the
    hierarchy.
    """

    def copy(self) -> "State":
        return State({k: (v.copy() if isinstance(v, State) else v)
                      for k, v in self.items()})

    __copy__ = copy

    def get_path(self, path: str, default: Any = None) -> Any:
        current: Any = self
        for part in path.split("."):
            if isinstance(current, Mapping) and part in current:
                current = current[part]
            else:
                return default
        return current

    def set_path(self, path: str, value: Any) -> "State":
        parts = path.split(".")
        current: Any = self
        for part in parts[:-1]:
            nxt = current.get(part)
            if not isinstance(nxt, State):
                nxt = State()
                current[part] = nxt
            current = nxt
        current[parts[-1]] = value
        return self

    def flatten(self, prefix: str = "") -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, value in self.items():
            full = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, State):
                out.update(value.flatten(full))
            elif isinstance(value, Mapping):
                out.update(State(value).flatten(full))
            else:
                out[full] = value
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {k: (v.to_dict() if isinstance(v, State) else v) for k, v in self.items()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "State":
        return cls({k: (cls.from_dict(v) if isinstance(v, Mapping) else v)
                    for k, v in data.items()})

    def __repr__(self):
        return f"State({dict.__repr__(self)})"


# ==========================================================================
# Attribute / Method
# ==========================================================================
class Attribute:
    """A measurable state variable ``a_i : O -> V_i`` (Definition 2.3)."""

    def __init__(self, name: str, default: Any = None, space: str = "real",
                 description: str = "", bounds: Optional[Tuple[float, float]] = None,
                 unit: str = ""):
        self.name = name
        self.default = default
        self.space = space
        self.description = description
        self.bounds = tuple(bounds) if bounds else None
        self.unit = unit

    def __repr__(self):
        return f"Attribute({self.name!r}, default={self.default!r})"

    def to_dict(self):
        return {"name": self.name, "default": self.default, "space": self.space,
                "description": self.description, "bounds": self.bounds, "unit": self.unit}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Attribute":
        return cls(**{k: v for k, v in data.items() if k in
                      {"name", "default", "space", "description", "bounds", "unit"}})


class Method:
    """An executable transformation ``m : S x A_act -> S x R`` (Definition 2.4)."""

    def __init__(self, name: str, fn: Callable, description: str = "",
                 preconditions: Optional[Sequence[Any]] = None,
                 postconditions: Optional[Sequence[Any]] = None,
                 tags: Sequence[str] = ()):
        self.name = name
        self.fn = fn
        self.description = description
        self.preconditions = [as_predicate(p) for p in (preconditions or [])]
        self.postconditions = [as_predicate(p) for p in (postconditions or [])]
        self.tags = list(tags)
        self.calls = 0

    def __call__(self, state, action=None, **kwargs) -> Tuple[Any, float]:
        for pre in self.preconditions:
            if not pre(state):
                raise MethodExecutionError(f"precondition failed for method {self.name!r}")
        self.calls += 1
        result = self.fn(state, action, **kwargs) if _accepts_action(self.fn) else self.fn(state)
        if isinstance(result, tuple) and len(result) == 2:
            new_state, reward = result
        else:
            new_state, reward = result, 0.0
        if new_state is None:
            new_state = state
        for post in self.postconditions:
            if not post(new_state):
                raise MethodExecutionError(f"postcondition failed for method {self.name!r}")
        return new_state, float(reward)

    def __repr__(self):
        return f"Method({self.name!r})"

    def to_dict(self):
        return {"name": self.name, "description": self.description,
                "tags": list(self.tags)}


def _accepts_action(fn) -> bool:
    import inspect
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return len(params) >= 2


# ==========================================================================
# COHObject
# ==========================================================================
class COHObject:
    """Base class for every Constrained Object Hierarchy.

    Parameters
    ----------
    name:
        Human readable identifier.
    children:
        Iterable of :class:`COHObject` (component ``C``).
    attributes:
        Iterable of :class:`Attribute` (component ``A``).
    methods:
        Iterable of :class:`Method` (component ``M``).
    neural:
        Mapping ``name -> NeuralComponent`` (component ``N``).
    embedder:
        :class:`Embedder` (component ``E``).  Defaults to a
        :class:`HashEmbedder`.
    identity / triggers / goals:
        Constraint lists (components ``I``, ``T``, ``G``).
    daemons:
        A :class:`Daemon` or iterable of daemons (component ``D``).
    """

    #: version tag stored in serialised payloads
    SCHEMA_VERSION = 1

    def __init__(self, name: str,
                 children: Optional[Iterable["COHObject"]] = None,
                 attributes: Optional[Iterable[Attribute]] = None,
                 methods: Optional[Iterable[Method]] = None,
                 neural: Optional[Mapping[str, Any]] = None,
                 embedder: Optional[Embedder] = None,
                 identity: Optional[Iterable[IdentityConstraint]] = None,
                 triggers: Optional[Iterable[TriggerConstraint]] = None,
                 goals: Optional[Iterable[GoalConstraint]] = None,
                 daemons: Optional[Any] = None,
                 description: str = "",
                 metadata: Optional[Mapping[str, Any]] = None,
                 flow: Optional[ContinuousFlow] = None,
                 jump: Optional[DiscreteTransition] = None):

        self.uid: str = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.metadata: Dict[str, Any] = dict(metadata or {})

        # ---- C : hierarchical compositional structure -------------------
        self._children: Dict[str, COHObject] = {}
        for child in (children or []):
            self.add_child(child)

        # ---- A : attributes --------------------------------------------
        self._attributes: Dict[str, Attribute] = {}
        for attr in (attributes or []):
            self.add_attribute(attr)

        # ---- M : methods ------------------------------------------------
        self._methods: Dict[str, Method] = {}
        for m in (methods or []):
            self.add_method(m)

        # ---- N : neural components --------------------------------------
        self._neural: Dict[str, Any] = dict(neural or {})

        # ---- E : embedding ----------------------------------------------
        self._embedder: Embedder = embedder or HashEmbedder()

        # ---- I, T, G -----------------------------------------------------
        self.constraints = ConstraintSet(identity=identity, triggers=triggers, goals=goals)

        # ---- D : constraint daemons -------------------------------------
        self._daemons: DaemonNetwork = DaemonNetwork(name=f"{name}.daemons")
        if daemons is not None:
            self.set_daemons(daemons)

        # ---- induced hybrid dynamics (Definition 4.1) --------------------
        self.flow: Optional[ContinuousFlow] = flow
        self.jump: Optional[DiscreteTransition] = jump

        # internal state
        self._state: State = self.initial_state()
        self._t: float = 0.0
        self.history: List[Tuple[float, State]] = []

    # ------------------------------------------------------------------
    # 9-tuple accessors
    # ------------------------------------------------------------------
    @property
    def C(self) -> Dict[str, "COHObject"]:
        """Hierarchical compositional structure (sub-objects by name)."""
        return self._children

    @property
    def A(self) -> Dict[str, Attribute]:
        """Attributes (measurable state variables)."""
        return self._attributes

    @property
    def M(self) -> Dict[str, Method]:
        """Methods (executable transformations)."""
        return self._methods

    @property
    def N(self) -> Dict[str, Any]:
        """Neural components (adaptive functions)."""
        return self._neural

    @property
    def E(self) -> Embedder:
        """Embedding function ``E : O -> H``."""
        return self._embedder

    @property
    def I(self) -> List[IdentityConstraint]:
        """Identity constraints (invariants)."""
        return self.constraints.identity

    @property
    def T(self) -> List[TriggerConstraint]:
        """Trigger constraints (event-condition-action rules)."""
        return self.constraints.triggers

    @property
    def G(self) -> List[GoalConstraint]:
        """Goal constraints (objectives)."""
        return self.constraints.goals

    @property
    def D(self) -> DaemonNetwork:
        """Constraint daemons (continuous monitors)."""
        return self._daemons

    # ------------------------------------------------------------------
    # C : hierarchy
    # ------------------------------------------------------------------
    def add_child(self, child: "COHObject", key: Optional[str] = None) -> "COHObject":
        if not isinstance(child, COHObject):
            raise TypeError("children must be COHObject instances")
        if child is self:
            raise CyclicHierarchyError([self.name, self.name])
        if self._would_cycle(child):
            raise CyclicHierarchyError([self.name, child.name])
        self._children[key or child.name] = child
        if isinstance(self._state, State):
            self._state.setdefault(child.name, child.initial_state())
        return child

    def _would_cycle(self, candidate: "COHObject") -> bool:
        stack = [candidate]
        seen = set()
        while stack:
            node = stack.pop()
            if node is self:
                return True
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.extend(node._children.values())
        return False

    def remove_child(self, key: str):
        self._children.pop(key, None)
        self._state.pop(key, None)

    def children(self) -> Dict[str, "COHObject"]:
        return self._children

    def child(self, key: str) -> "COHObject":
        return self._children[key]

    def descendants(self) -> List["COHObject"]:
        out: List[COHObject] = []
        for child in self._children.values():
            out.append(child)
            out.extend(child.descendants())
        return out

    def is_leaf(self) -> bool:
        return not self._children

    def component_dag(self) -> Dict[str, List[str]]:
        """Return the component DAG as ``{object_name: [child names]}``."""
        dag: Dict[str, List[str]] = {}

        def walk(obj):
            dag[obj.name] = [c.name for c in obj._children.values()]
            for c in obj._children.values():
                walk(c)

        walk(self)
        return dag

    def topological_order(self) -> List["COHObject"]:
        """Bottom-up evaluation order; raises on cycles."""
        order: List[COHObject] = []
        visiting: Set[int] = set()
        done: Set[int] = set()

        def visit(obj, path):
            oid = id(obj)
            if oid in done:
                return
            if oid in visiting:
                raise CyclicHierarchyError(path + [obj.name])
            visiting.add(oid)
            for child in obj._children.values():
                visit(child, path + [obj.name])
            visiting.discard(oid)
            done.add(oid)
            order.append(obj)

        visit(self, [])
        return order

    def is_acyclic(self) -> bool:
        try:
            self.topological_order()
            return True
        except CyclicHierarchyError:
            return False

    # ------------------------------------------------------------------
    # A : attributes
    # ------------------------------------------------------------------
    def add_attribute(self, attribute: Attribute | str, default: Any = None, **kwargs) -> Attribute:
        if isinstance(attribute, str):
            attribute = Attribute(attribute, default=default, **kwargs)
        self._attributes[attribute.name] = attribute
        if isinstance(self._state, State) and attribute.name not in self._state:
            self._state[attribute.name] = attribute.default
        return attribute

    def attribute(self, name: str) -> Attribute:
        return self._attributes[name]

    def get(self, path: str, default: Any = None) -> Any:
        return self._state.get_path(path, default)

    def set(self, path: str, value: Any) -> "COHObject":
        self._state.set_path(path, value)
        return self

    # ------------------------------------------------------------------
    # M : methods
    # ------------------------------------------------------------------
    def add_method(self, method: Method | Callable, name: Optional[str] = None, **kwargs) -> Method:
        if not isinstance(method, Method):
            method = Method(name or getattr(method, "__name__", "method"), method, **kwargs)
        self._methods[method.name] = method
        return method

    def method(self, name: str) -> Method:
        return self._methods[name]

    def invoke(self, name: str, state: Optional[State] = None, action=None, **kwargs):
        method = self._methods[name]
        target = self._state if state is None else state
        new_state, reward = method(target, action, **kwargs)
        if state is None:
            self._state = new_state if isinstance(new_state, State) else State(new_state)
        return new_state, reward

    # ------------------------------------------------------------------
    # N : neural components
    # ------------------------------------------------------------------
    def add_neural(self, component, name: Optional[str] = None):
        key = name or getattr(component, "name", f"neural_{len(self._neural)}")
        self._neural[key] = component
        return component

    def neural(self, key: str):
        return self._neural[key]

    def parameters(self) -> List[float]:
        params: List[float] = []
        for comp in self._neural.values():
            theta = getattr(comp, "theta", None)
            if theta is not None:
                params.extend(list(theta))
        for child in self._children.values():
            params.extend(child.parameters())
        return params

    # ------------------------------------------------------------------
    # E : embedding
    # ------------------------------------------------------------------
    def set_embedder(self, embedder: Embedder) -> Embedder:
        self._embedder = embedder
        return embedder

    def embed(self) -> tuple:
        """Embed this object (and, recursively, its components)."""
        return self._embedder(self)

    def similarity(self, other: "COHObject") -> float:
        return self._embedder.space.similarity(self.embed(), other.embed())

    def canonical_form(self) -> str:
        """Deterministic string used by :class:`HashEmbedder`."""
        attrs = ",".join(
            f"{n}={self._state.get(n, a.default)!r}"
            for n, a in sorted(self._attributes.items())
        )
        kids = ",".join(f"{k}:{v.canonical_form()}"
                        for k, v in sorted(self._children.items()))
        return f"{self.name}[{attrs}]{{{kids}}}"

    # ------------------------------------------------------------------
    # I, T, G : constraints
    # ------------------------------------------------------------------
    def add_identity_constraint(self, constraint) -> IdentityConstraint:
        if isinstance(constraint, str):
            constraint = IdentityConstraint(constraint, constraint)
        self.constraints.add_identity(constraint)
        return constraint

    def add_trigger_constraint(self, constraint) -> TriggerConstraint:
        self.constraints.add_trigger(constraint)
        return constraint

    def add_goal_constraint(self, constraint) -> GoalConstraint:
        self.constraints.add_goal(constraint)
        return constraint

    def satisfies_constraints(self, state: Optional[State] = None) -> bool:
        return self.constraints.satisfies_identity(self._state if state is None else state)

    def check(self, state: Optional[State] = None):
        self.constraints.check_identity(self._state if state is None else state)

    def goal_value(self, state: Optional[State] = None, action=None, t: Optional[float] = None) -> float:
        s = self._state if state is None else state
        return self.constraints.total_goal(s, action, self._t if t is None else t)

    # ------------------------------------------------------------------
    # D : daemons
    # ------------------------------------------------------------------
    def set_daemons(self, daemons) -> DaemonNetwork:
        if isinstance(daemons, DaemonNetwork):
            self._daemons = daemons
        elif isinstance(daemons, Daemon):
            self._daemons = DaemonNetwork([daemons], name=f"{self.name}.daemons")
        else:
            self._daemons = DaemonNetwork(daemons, name=f"{self.name}.daemons")
        return self._daemons

    def add_daemon(self, daemon: Daemon) -> Daemon:
        self._daemons.add(daemon)
        return daemon

    def monitor(self, state: Optional[State] = None):
        """Run all daemons once; returns the highest-severity action."""
        return self._daemons.observe(self._state if state is None else state, self._t)

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------
    @property
    def state(self) -> State:
        return self._state

    @state.setter
    def state(self, value):
        self._state = value if isinstance(value, State) else State(value)

    def initial_state(self) -> State:
        st = State()
        for name, attr in self._attributes.items():
            st[name] = _copy.deepcopy(attr.default)
        for key, child in self._children.items():
            st[key] = child.initial_state()
        return st

    def reset(self):
        self._state = self.initial_state()
        self._t = 0.0
        self.history.clear()
        self._daemons.reset()
        for child in self._children.values():
            child.reset()

    def snapshot(self) -> State:
        snap = self._state.copy()
        for key, child in self._children.items():
            snap[key] = child.snapshot()
        return snap

    # ------------------------------------------------------------------
    # dynamics
    # ------------------------------------------------------------------
    def to_hybrid_system(self) -> HybridDynamicalSystem:
        return HybridDynamicalSystem(
            flow=self.flow, jump=self.jump,
            constraints=self.constraints, daemons=self._daemons,
            name=f"{self.name}.hybrid",
        )

    def step(self, action=None, dt: float = 0.01, event: Optional[str] = None):
        """Advance the induced hybrid system by one step."""
        system = self.to_hybrid_system()
        new_state, info = system.step(self._state, action, dt, self._t, None, event)
        self._state = new_state if isinstance(new_state, State) else State(new_state)
        self._t += dt
        self.history.append((self._t, self._state.copy()))
        return self._state, info

    def simulate(self, steps: int = 100, dt: float = 0.01, policy=None):
        system = self.to_hybrid_system()
        traj = system.simulate(self._state, steps, dt, policy, None)
        if traj.states:
            self._state = traj.states[-1]
            self._t = traj.times[-1]
        return traj

    # ------------------------------------------------------------------
    # soundness / completeness diagnostics
    # ------------------------------------------------------------------
    def soundness_report(self, sample_state: Optional[State] = None) -> Dict[str, Any]:
        """Check the two soundness conditions of Section 6.1."""
        sample = sample_state or self.initial_state()
        acyclic = self.is_acyclic()
        violations = [c.name for c in self.constraints.identity if not c.satisfied(sample)]
        trigger_conflicts = []
        for trg in self.constraints.triggers:
            if trg.preserves is None:
                continue
            if trg.matches(trg.event, sample):
                outcome = trg.fire(sample)
                if isinstance(outcome, Mapping):
                    merged = dict(sample)
                    merged.update(outcome)
                    if not self.constraints.satisfies_identity(merged):
                        trigger_conflicts.append(trg.name)
        return {
            "hierarchical_sound": acyclic,
            "constraints_satisfied_at_sample": not violations,
            "violations": violations,
            "triggers_breaking_invariants": trigger_conflicts,
            "sound": acyclic and not violations and not trigger_conflicts,
        }

    def component_report(self) -> Dict[str, Any]:
        """Which of the nine components are populated (minimality check)."""
        return {
            "C": len(self._children),
            "A": len(self._attributes),
            "M": len(self._methods),
            "N": len(self._neural),
            "E": type(self._embedder).__name__,
            "I": len(self.constraints.identity),
            "T": len(self.constraints.triggers),
            "G": len(self.constraints.goals),
            "D": len(self._daemons),
        }

    def is_minimal(self) -> bool:
        """True when all nine components are present (Theorem 6.6)."""
        return all([
            bool(self._children), bool(self._attributes), bool(self._methods),
            bool(self._neural), self._embedder is not None,
            bool(self.constraints.identity), bool(self.constraints.triggers),
            bool(self.constraints.goals), len(self._daemons) > 0,
        ])

    # ------------------------------------------------------------------
    # categorical constructions (Section 3)
    # ------------------------------------------------------------------
    def product(self, other: "COHObject") -> "COHObject":
        """Parallel composition ``O1 x O2`` (Theorem 3.1)."""
        merged = COHObject(
            name=f"({self.name} x {other.name})",
            description="categorical product",
            attributes=list(self._attributes.values()) + list(other._attributes.values()),
            methods=list(self._methods.values()) + list(other._methods.values()),
            neural={**self._neural, **other._neural},
            embedder=CompositeEmbedder(self._embedder),
            identity=list(self.constraints.identity) + list(other.constraints.identity),
            triggers=list(self.constraints.triggers) + list(other.constraints.triggers),
            goals=list(self.constraints.goals) + list(other.constraints.goals),
        )
        merged.add_child(self, key=f"{self.name}")
        merged.add_child(other, key=f"{other.name}")
        merged.set_daemons(list(self._daemons.daemons) + list(other._daemons.daemons))
        return merged

    def coproduct(self, other: "COHObject") -> "COHObject":
        """Disjoint / alternative composition ``O1 + O2`` (Theorem 3.1)."""
        return COHObject(
            name=f"({self.name} + {other.name})",
            description="categorical coproduct",
            children=[self, other],
            identity=[IdentityConstraint(
                "coproduct_exclusive",
                lambda s: True,
                description="either branch may be active",
            )],
            embedder=CompositeEmbedder(self._embedder),
            daemons=DaemonNetwork(name="coproduct.daemons"),
        )

    def exponential(self, other: "COHObject") -> "COHObject":
        """Function object ``O2^O1`` (higher-order COH system)."""
        def apply_fn(state, action=None):
            payload = state.get("argument") if isinstance(state, Mapping) else None
            result = other.snapshot() if payload is None else payload
            out = State(state)
            out["result"] = result
            return out, 0.0

        return COHObject(
            name=f"({other.name}^{self.name})",
            description="categorical exponential / function object",
            attributes=[Attribute("argument", None), Attribute("result", None)],
            methods=[Method("apply", apply_fn, description="evaluation map")],
            children=[self, other],
            embedder=self._embedder,
        )

    def clone(self, deep: bool = True) -> "COHObject":
        return _copy.deepcopy(self) if deep else _copy.copy(self)

    # ------------------------------------------------------------------
    # serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.SCHEMA_VERSION,
            "uid": self.uid,
            "name": self.name,
            "description": self.description,
            "metadata": dict(self.metadata),
            "attributes": [a.to_dict() for a in self._attributes.values()],
            "methods": [m.to_dict() for m in self._methods.values()],
            "neural": {k: type(v).__name__ for k, v in self._neural.items()},
            "embedder": type(self._embedder).__name__,
            "constraints": self.constraints.to_dict(),
            "daemons": [d.name for d in self._daemons],
            "children": [c.to_dict() for c in self._children.values()],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "COHObject":
        obj = cls(
            name=data.get("name", "object"),
            description=data.get("description", ""),
            metadata=data.get("metadata", {}),
            attributes=[Attribute.from_dict(a) for a in data.get("attributes", [])],
        )
        obj.uid = data.get("uid", obj.uid)
        for child_data in data.get("children", []):
            obj.add_child(cls.from_dict(child_data))
        for ident in data.get("constraints", {}).get("identity", []):
            if ident.get("expression"):
                obj.add_identity_constraint(IdentityConstraint(
                    ident["name"], _expr_from_dict(ident["expression"]),
                    ident.get("description", "")))
        return obj

    # ------------------------------------------------------------------
    def __repr__(self):
        return (f"{type(self).__name__}(name={self.name!r}, "
                f"children={len(self._children)}, "
                f"attrs={len(self._attributes)}, "
                f"I/T/G={len(self.I)}/{len(self.T)}/{len(self.G)})")

    def describe(self, indent: int = 0) -> str:
        pad = "  " * indent
        lines = [f"{pad}{self.name}  {self.component_report()}"]
        for child in self._children.values():
            lines.append(child.describe(indent + 1))
        return "\n".join(lines)


# --------------------------------------------------------------------------
def _expr_from_dict(data: Mapping[str, Any]) -> E.Expr:
    op = data.get("op")
    if op == "const":
        return E.Const(data.get("value"))
    if op == "var":
        return E.Var(data.get("name"))
    if op == "not":
        return E.Not(_expr_from_dict(data["arg"]))
    if op == "and":
        return E.And(*[_expr_from_dict(a) for a in data["args"]])
    if op == "or":
        return E.Or(*[_expr_from_dict(a) for a in data["args"]])
    if op == "cmp":
        args = data["args"]
        return E.Cmp(data["operator"], _expr_from_dict(args[0]), _expr_from_dict(args[1]))
    if op == "arith":
        args = data["args"]
        right = _expr_from_dict(args[1]) if len(args) > 1 else None
        return E.Arith(data["operator"], _expr_from_dict(args[0]), right)
    if op == "call":
        return E.Call(data["name"], [_expr_from_dict(a) for a in data.get("args", [])])
    return E.Const(None)