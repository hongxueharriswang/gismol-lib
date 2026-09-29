# GISMOL Technical Specification

**Version 0.1.0 · Draft Standard**

**General Intelligent Systems Modelling Language — Toolkit Specification**

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Conformance and Normative Language](#2-conformance-and-normative-language)
3. [System Architecture](#3-system-architecture)
4. [Type Specifications](#4-type-specifications)
5. [Interface Contracts](#5-interface-contracts)
6. [Module Specifications](#6-module-specifications)
7. [Algorithms](#7-algorithms)
8. [Error Handling](#8-error-handling)
9. [Serialisation Format](#9-serialisation-format)
10. [Performance Characteristics](#10-performance-characteristics)
11. [Security Considerations](#11-security-considerations)
12. [Conformance Classes](#12-conformance-classes)
13. [Testing Requirements](#13-testing-requirements)
14. [Versioning and Compatibility](#14-versioning-and-compatibility)
15. [References](#15-references)
16. [Appendix A · Grammar of the Constraint Language](#appendix-a--grammar-of-the-constraint-language)
17. [Appendix B · JSON Schema](#appendix-b--json-schema)
18. [Appendix C · UML-Style Class Diagram](#appendix-c--uml-style-class-diagram)

---

## 1. Introduction

### 1.1 Scope

This document specifies **GISMOL** (General Intelligent Systems Modelling Language), a Python toolkit implementing **Constrained Object Hierarchies (COH)**. The specification covers:

- The abstract data model of a COH object (the 9-tuple)
- The concrete Python API exposed by the `gismol` package
- The semantics of every public operation
- The serialisation format
- Performance and security properties
- Conformance classes for implementations

This specification does **not** cover:

- The underlying mathematical theory of COH (see [COH-2025])
- Application-specific modelling patterns (see [GISMOL-Guide])
- Deployment topologies beyond single-process execution

### 1.2 Purpose

The purpose of this specification is to:

1. Define the **normative behaviour** of a GISMOL implementation
2. Enable **independent implementations** in other languages or runtimes
3. Provide a **stable contract** for downstream applications
4. Support **formal verification** of tooling and reasoners

### 1.3 Audience

This document is intended for:

- **Implementers** of the GISMOL toolkit or compatible toolkits
- **Application developers** building intelligent systems on GISMOL
- **Researchers** validating the COH theory against its implementation
- **Auditors** performing safety or security assessments

### 1.4 Conventions

#### 1.4.1 Normative Language

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHALL**, **SHALL NOT**, **SHOULD**, **SHOULD NOT**, **RECOMMENDED**, **MAY**, and **OPTIONAL** in this document are to be interpreted as described in [RFC 2119].

#### 1.4.2 Mathematical Notation

| Symbol | Meaning |
|---|---|
| `𝒪` | A COH object |
| `S` | The state space |
| `A_act` | The action space |
| `T` | The time domain (usually ℝ≥0) |
| `H` | A Hilbert space |
| `Θ` | A parameter space |
| `ℝ` | The real numbers |
| `𝔹` | The booleans `{true, false}` |
| `→` | Function type constructor |
| `×` | Cartesian product |
| `≅` | Isomorphism |

#### 1.4.3 Code Conventions

Code fragments in this document are Python 3.9+ unless otherwise stated. Type annotations use PEP 484 syntax. Variable names in code fragments follow the toolkit's own conventions (snake_case for functions and variables, PascalCase for classes).

#### 1.4.4 Formal Model References

Where this specification references theorems from [COH-2025], they are cited as "Theorem N.M". Where it references definitions, they are cited as "Definition N.M".

---

## 2. Conformance and Normative Language

### 2.1 Conformance Classes

An implementation claiming conformance to this specification MUST declare one of three conformance classes:

| Class | Requirement |
|---|---|
| **Class 1 (Minimal)** | Implements the core 9-tuple, constraint evaluation, daemons, and simulation |
| **Class 2 (Standard)** | Class 1, plus categorical composition, neural components, and NLP front-end |
| **Class 3 (Full)** | Class 2, plus reasoners, stability analysis, and free-energy correspondence |

See §12 for detailed requirements.

### 2.2 Normative vs Informative

Clauses marked with the word "MUST" or "SHALL" are **normative**. All other clauses are **informative** unless they specify a mathematical property or a data format.

### 2.3 Extension Points

An implementation MAY:

- Add new daemons, reasoners, or domain profiles
- Extend the constraint language with additional functions
- Add optional runtime dependencies for acceleration
- Provide additional serialisation formats

An implementation MUST NOT:

- Change the semantics of the 9-tuple
- Alter the meaning of the standard `add_*` methods
- Break the structural invariants described in §4
- Introduce side effects into the safe expression evaluator

---

## 3. System Architecture

### 3.1 Layered Architecture

A GISMOL implementation MUST be organised into the following layers, each building on the one below:

```
┌─────────────────────────────────────────────────────────────┐
│  L6  Applications       (user code)                          │
├─────────────────────────────────────────────────────────────┤
│  L5  Reasoning          reasoners, nlp, validation           │
├─────────────────────────────────────────────────────────────┤
│  L4  Learning           neural, stability, free_energy       │
├─────────────────────────────────────────────────────────────┤
│  L3  Composition        categorical, dynamical               │
├─────────────────────────────────────────────────────────────┤
│  L2  Model              core (COHObject, constraints, d…)    │
├─────────────────────────────────────────────────────────────┤
│  L1  Expression         core.expression                      │
├─────────────────────────────────────────────────────────────┤
│  L0  Standard library   math, ast, json, typing, …           │
└─────────────────────────────────────────────────────────────┘
```

**Requirement 3.1.1** — Layers MUST NOT have circular dependencies. A layer MAY depend on any lower layer; it MUST NOT depend on any higher layer.

### 3.2 Package Structure

A conforming implementation MUST provide the following top-level modules:

| Module | Layer | Purpose |
|---|---|---|
| `gismol.core` | L2 | The 9-tuple and its components |
| `gismol.core.expression` | L1 | Safe expression language |
| `gismol.categorical` | L3 | Category-theoretic layer |
| `gismol.dynamical` | L3 | Hybrid dynamics and free energy |
| `gismol.stability` | L4 | Lyapunov analysis |
| `gismol.neural` | L4 | Neural components and optimisers |
| `gismol.nlp` | L5 | Natural-language front-end |
| `gismol.reasoners` | L5 | Reasoning engines |

**Requirement 3.2.1** — Every module listed above MUST be importable without side effects.

**Requirement 3.2.2** — Importing `gismol` MUST NOT import any third-party package.

### 3.3 Dependency Rules

**Requirement 3.3.1** — A Class 1 implementation MUST have zero mandatory third-party runtime dependencies.

**Requirement 3.3.2** — A Class 2 or Class 3 implementation MAY optionally depend on `numpy`, `networkx`, `matplotlib`, `sentence-transformers`, or `jsonschema`. Such dependencies MUST be guarded by runtime checks so that the toolkit imports cleanly if they are absent.

**Requirement 3.3.3** — No module in layers L0–L4 MAY import from `gismol.nlp` or `gismol.reasoners`.

---

## 4. Type Specifications

### 4.1 The 9-Tuple

**Definition 4.1.1 (COH Object).** A COH object is a 9-tuple:

$$
\mathcal{O} = (C,\, A,\, M,\, N,\, E,\, I,\, T,\, G,\, D)
$$

where:

| Symbol | Type | Description |
|---|---|---|
| `C` | `dict[str, COHObject]` | Hierarchical compositional structure |
| `A` | `dict[str, Attribute]` | Attributes (state variables) |
| `M` | `dict[str, Method]` | Methods (transformations) |
| `N` | `dict[str, NeuralComponent]` | Neural components |
| `E` | `Embedder` | Embedding function |
| `I` | `list[IdentityConstraint]` | Identity constraints |
| `T` | `list[TriggerConstraint]` | Trigger constraints |
| `G` | `list[GoalConstraint]` | Goal constraints |
| `D` | `DaemonNetwork` | Constraint daemons |

**Requirement 4.1.1** — The tuple MUST be exposed via read-only properties named `C`, `A`, `M`, `N`, `E`, `I`, `T`, `G`, `D` on the `COHObject` class.

**Requirement 4.1.2** — Mutation of the components MUST go through the `add_*`, `remove_*`, `set_*` methods. Direct mutation of the returned collections MUST be discouraged (return copies where feasible).

### 4.2 COHObject

**Class 4.2.1 (COHObject).** The base class for every intelligent system.

```
class COHObject:
    # Identity
    uid: str
    name: str
    description: str
    metadata: dict[str, Any]

    # The 9-tuple (read-only properties)
    C: dict[str, COHObject]
    A: dict[str, Attribute]
    M: dict[str, Method]
    N: dict[str, NeuralComponent]
    E: Embedder
    I: list[IdentityConstraint]
    T: list[TriggerConstraint]
    G: list[GoalConstraint]
    D: DaemonNetwork

    # Induced hybrid dynamics
    flow: Optional[ContinuousFlow]
    jump: Optional[DiscreteTransition]

    # Current state
    state: State
```

**Requirement 4.2.1 (Uniqueness of uid).** Every `COHObject` MUST have a `uid` that is unique within its runtime. Implementations SHOULD use UUID4.

**Requirement 4.2.2 (Name uniqueness among siblings).** Two children of the same parent MUST NOT have the same name. Adding a child whose name collides with an existing child MUST either replace the existing child or raise an error; implementations MUST document which.

**Requirement 4.2.3 (Acyclicity invariant).** The directed graph formed by parent–child relationships MUST be acyclic. Attempting to introduce a cycle MUST raise `CyclicHierarchyError` and MUST leave the hierarchy unchanged.

**Requirement 4.2.4 (Topological order).** The method `topological_order()` MUST return nodes in an order where every child appears before its parent. If the graph contains a cycle, the method MUST raise `CyclicHierarchyError`.

### 4.3 State

**Definition 4.3.1 (State).** A `State` is a hierarchical mapping from string keys to values. It is a subclass of `dict` with the following additional behaviours:

- Sub-states are nested under the child's name
- Attribute values live at the leaves
- Dotted paths access nested values

**Requirement 4.3.1** — `State` MUST support the following methods:

| Method | Signature | Behaviour |
|---|---|---|
| `get_path` | `(path: str, default=None) -> Any` | Resolve dotted path |
| `set_path` | `(path: str, value: Any) -> State` | Set dotted path; return self |
| `flatten` | `(prefix: str = "") -> dict[str, Any]` | Return flat view |
| `copy` | `() -> State` | Shallow-copy that deep-copies nested states |
| `to_dict` | `() -> dict` | Serializable representation |
| `from_dict` | `(data: dict) -> State` | Classmethod inverse |

**Requirement 4.3.2** — `State.copy()` MUST deep-copy nested `State` objects to prevent aliasing bugs between parent and child state.

**Requirement 4.3.3** — Path resolution MUST NOT distinguish missing keys from keys whose value is `None`. If the implementation needs to distinguish, it MUST use a sentinel.

### 4.4 Attribute

**Class 4.4.1 (Attribute).** A measurable state variable.

```
class Attribute:
    name: str
    default: Any
    space: Literal["real", "bool", "categorical", "vector"]
    description: str
    bounds: Optional[tuple[float, float]]
    unit: str
```

**Requirement 4.4.1** — An `Attribute` MUST be immutable after construction. Implementations MAY use `dataclass(frozen=True)` or manual `__setattr__` guards.

**Requirement 4.4.2** — `bounds`, if provided, MUST satisfy `bounds[0] <= bounds[1]`.

### 4.5 Method

**Class 4.5.1 (Method).** An executable state transformation.

```
class Method:
    name: str
    fn: Callable[..., Any]
    description: str
    preconditions: list[Callable[[State], bool]]
    postconditions: list[Callable[[State], bool]]
    tags: list[str]
```

**Requirement 4.5.1 (Signature).** The wrapped function `fn` MUST accept one of:

- `fn(state) -> new_state | (new_state, reward)`
- `fn(state, action) -> new_state | (new_state, reward)`

Implementations MUST inspect the function signature to determine which form is accepted.

**Requirement 4.5.2 (Precondition semantics).** If any precondition evaluates to `False`, the method MUST raise `MethodExecutionError` and MUST NOT execute `fn`.

**Requirement 4.5.3 (Postcondition semantics).** If any postcondition evaluates to `False` after execution, the method MUST raise `MethodExecutionError`. The state change MUST be recorded regardless.

**Requirement 4.5.4 (Return value normalisation).** If `fn` returns a bare state, the reward MUST default to `0.0`. If it returns `None`, the state MUST be treated as unchanged.

### 4.6 Constraints

#### 4.6.1 IdentityConstraint

**Class 4.6.1.1 (IdentityConstraint).** An invariant predicate.

```
class IdentityConstraint:
    name: str
    predicate: Callable[[State], bool]
    description: str
    weight: float
    margin: Optional[Callable[[State], float]]
```

**Requirement 4.6.1.1** — `predicate` MUST be a pure function: it MUST NOT modify the state, raise exceptions on valid inputs, or have side effects.

**Requirement 4.6.1.2** — `satisfied(state)` MUST return `True` iff `predicate(state)` is truthy.

**Requirement 4.6.1.3** — `violation(state)` MUST return `0.0` when satisfied and a non-negative magnitude otherwise. If `margin` is provided, `violation` SHOULD return `abs(margin(state)) * weight`; otherwise it SHOULD return `weight`.

#### 4.6.2 TriggerConstraint

**Class 4.6.2.1 (TriggerConstraint).** An event-condition-action rule.

```
class TriggerConstraint:
    name: str
    event: str                # or "*" for any event
    condition: Callable[[State], bool]
    action: str | Callable | dict
    description: str
    weight: float
    preserves: Optional[Callable[[State], bool]]
    priority: int
```

**Requirement 4.6.2.1** — `matches(event, state)` MUST return `True` iff:

- `event` matches `self.event` (identical string or `self.event == "*"`), AND
- `condition(state)` is truthy.

**Requirement 4.6.2.2** — `fire(state)` MUST return the action payload:

- If `action` is a `str`, return it as-is
- If `action` is a `Callable`, return `action(state)`
- If `action` is a `dict`, return it as-is

**Requirement 4.6.2.3 (Preserves).** If `preserves` is provided and the resulting state (after applying the action) violates it, the trigger SHOULD be flagged by `soundness_report`. Implementations MAY choose to reject such triggers at construction time.

#### 4.6.3 GoalConstraint

**Class 4.6.3.1 (GoalConstraint).** An optimisation objective.

```
class GoalConstraint:
    name: str
    objective: Callable[..., float]
    direction: Literal["max", "min"]
    weight: float
    target: Optional[float]
```

**Requirement 4.6.3.1 (Utility normalisation).** The method `utility(state, action, t)` MUST return a value such that larger is always better:

- If `direction == "max"`: `utility = weight * value`
- If `direction == "min"`: `utility = -weight * value`
- If `target` is set: `utility = -weight * abs(value - target)`

**Requirement 4.6.3.2 (Non-failing).** Goals MUST NOT cause simulation to fail. Violations of a goal are informational; only identity constraints are hard.

#### 4.6.4 ConstraintSet

**Class 4.6.4.1 (ConstraintSet).** Container for `I`, `T`, `G`.

**Requirement 4.6.4.1** — `total_goal(state, action, t)` MUST return the aggregate goal value:

$$
G_\text{total} = \sum_j w_j \, g_j(s, a, t) - \lambda R(\theta)
$$

where `R(θ)` is the optional regulariser.

### 4.7 Daemons

**Class 4.7.1 (Daemon).** A continuous monitoring process.

```
class Daemon:
    name: str
    description: str
    enabled: bool
    history: list[CorrectiveAction]

    def observe(self, state: State, t: float = 0.0) -> Optional[CorrectiveAction]:
        ...
```

**Requirement 4.7.1.1** — `observe` MUST be idempotent for a given state: repeated calls with the same arguments MUST return actions with the same name and severity.

**Requirement 4.7.1.2** — `observe` MUST NOT mutate the state.

**Requirement 4.7.1.3 (History).** Actions returned by `observe` MUST be appended to `history` via `self._emit(action)`.

**Class 4.7.2 (CorrectiveAction).** The payload returned by a daemon.

```
class CorrectiveAction:
    name: str
    payload: Any
    daemon: str
    severity: float
    message: str
```

**Requirement 4.7.2.1** — `severity` MUST be a non-negative finite float. `NaN` and `inf` MUST be rejected.

**Class 4.7.3 (DaemonNetwork).** A composite daemon.

**Requirement 4.7.3.1** — When multiple sub-daemons fire, `observe` MUST return the one with the highest `severity`. Ties MUST be broken by declaration order.

### 4.8 Embedder

**Class 4.8.1 (Embedder).** A map from COH objects to vectors.

```
class Embedder(ABC):
    dim: Optional[int]
    space: HilbertSpace

    @abstractmethod
    def __call__(self, obj: COHObject) -> tuple[float, ...]:
        ...
```

**Requirement 4.8.1.1** — `__call__` MUST be a pure function of `obj`. Two calls with the same object MUST return the same vector (for non-learning embedders).

**Requirement 4.8.1.2** — The returned vector MUST be a tuple of floats, not a list.

**Requirement 4.8.1.3** — `similarity(a, b)` MUST return a value in `[-1.0, 1.0]`.

**Class 4.8.2 (HilbertSpace).** A finite-dimensional real vector space with inner product.

**Requirement 4.8.2.1** — `similarity(u, v)` MUST be the cosine similarity, clamped to `[-1, 1]`. If either vector has zero norm, the result MUST be `0.0`.

### 4.9 Neural Components

**Class 4.9.1 (NeuralComponent).** A parameterised adaptive function.

```
class NeuralComponent(ABC):
    name: str
    input_dim: Optional[int]
    output_dim: Optional[int]

    @abstractmethod
    def __call__(self, x: Sequence[float]) -> Sequence[float]:
        ...

    @property
    def theta(self) -> list[float]:
        ...

    def set_theta(self, theta: Sequence[float]) -> NeuralComponent:
        ...
```

**Requirement 4.9.1.1** — `theta` MUST return the parameters as a flat list of floats. `set_theta` MUST accept a list of the same length and MUST restore the parameters exactly (bit-identical for repeated round trips).

**Requirement 4.9.1.2** — `n_parameters()` MUST return `len(theta)` in O(1) time.

**Requirement 4.9.1.3 (Determinism).** Given the same inputs and same parameters, `__call__` MUST return the same output.

### 4.10 Trajectory

**Class 4.10.1 (Trajectory).** The result of a simulation.

```
class Trajectory:
    times: list[float]
    states: list[State]
    actions: list[Any]
    rewards: list[float]
    events: list[Optional[str]]
    info: dict[str, Any]
```

**Requirement 4.10.1.1** — `len(traj)` MUST equal `len(traj.states)`.

**Requirement 4.10.1.2** — Iterating over a trajectory MUST yield `(t, state)` pairs.

**Requirement 4.10.1.3** — `cumulative_reward()` MUST return the sum of rewards; `discounted_return(gamma)` MUST return `sum(gamma^i * r_i)`.

**Requirement 4.10.1.4** — `to_dict()` MUST produce JSON-serialisable output.

---

## 5. Interface Contracts

### 5.1 COHObject Public Interface

The following methods MUST be exposed. Implementations MAY add more.

#### 5.1.1 Construction

| Method | Signature | Semantics |
|---|---|---|
| `__init__` | `(name, children=(), attributes=(), methods=(), neural={}, embedder=None, identity=(), triggers=(), goals=(), daemons=None, description="", metadata={}, flow=None, jump=None)` | Construct object |
| `from_dict` | `classmethod(data: dict) -> COHObject` | Deserialise |
| `clone` | `(deep: bool = True) -> COHObject` | Copy |

#### 5.1.2 Hierarchy

| Method | Signature | Semantics |
|---|---|---|
| `add_child` | `(child: COHObject, key: str = None) -> COHObject` | Attach child |
| `remove_child` | `(key: str) -> None` | Detach child |
| `child` | `(key: str) -> COHObject` | Look up |
| `children` | `() -> dict[str, COHObject]` | Direct children |
| `descendants` | `() -> list[COHObject]` | All nested |
| `is_leaf` | `() -> bool` | No children |
| `topological_order` | `() -> list[COHObject]` | Bottom-up |
| `component_dag` | `() -> dict[str, list[str]]` | DAG |

#### 5.1.3 State

| Method | Signature | Semantics |
|---|---|---|
| `get` | `(path: str, default=None) -> Any` | Resolve |
| `set` | `(path: str, value: Any) -> COHObject` | Assign |
| `initial_state` | `() -> State` | Fresh state |
| `snapshot` | `() -> State` | Deep copy of current |
| `reset` | `() -> None` | Return to initial |

#### 5.1.4 Constraint Management

| Method | Signature | Semantics |
|---|---|---|
| `add_identity_constraint` | `(c) -> IdentityConstraint` | Add invariant |
| `add_trigger_constraint` | `(c) -> TriggerConstraint` | Add trigger |
| `add_goal_constraint` | `(c) -> GoalConstraint` | Add goal |
| `add_daemon` | `(d: Daemon) -> Daemon` | Attach daemon |
| `satisfies_constraints` | `(state=None) -> bool` | Check invariants |
| `check` | `(state=None) -> None` | Raise on violation |

#### 5.1.5 Learning

| Method | Signature | Semantics |
|---|---|---|
| `add_neural` | `(component, name=None) -> NeuralComponent` | Attach |
| `neural` | `(key: str) -> NeuralComponent` | Look up |
| `parameters` | `() -> list[float]` | All params in hierarchy |
| `set_embedder` | `(e: Embedder) -> Embedder` | Assign embedder |
| `embed` | `() -> tuple[float, ...]` | Embed object |
| `similarity` | `(other: COHObject) -> float` | Cosine similarity |

#### 5.1.6 Simulation

| Method | Signature | Semantics |
|---|---|---|
| `step` | `(action=None, dt=0.01, event=None) -> (State, dict)` | One step |
| `simulate` | `(steps=100, dt=0.01, policy=None) -> Trajectory` | Full run |
| `to_hybrid_system` | `() -> HybridDynamicalSystem` | Build system |

#### 5.1.7 Diagnostics

| Method | Signature | Semantics |
|---|---|---|
| `component_report` | `() -> dict[str, int \| str]` | Count each component |
| `soundness_report` | `(sample=None) -> dict` | Soundness check |
| `is_minimal` | `() -> bool` | All nine components populated |
| `is_acyclic` | `() -> bool` | Cycle check |
| `canonical_form` | `() -> str` | Deterministic string |

#### 5.1.8 Composition

| Method | Signature | Semantics |
|---|---|---|
| `product` | `(other: COHObject) -> COHObject` | Categorical product |
| `coproduct` | `(other: COHObject) -> COHObject` | Categorical coproduct |
| `exponential` | `(other: COHObject) -> COHObject` | Function object |

### 5.2 Daemon Interface

**Requirement 5.2.1** — Every daemon MUST implement `observe`. Implementations MUST provide `monitor(trajectory)` and `reset()`.

**Requirement 5.2.2** — A `DaemonNetwork` MUST be iterable. Iterating MUST yield sub-daemons in declaration order.

**Requirement 5.2.3** — Calling `add` on a `DaemonNetwork` with a name that already exists MUST replace the existing daemon.

### 5.3 Reasoner Interface

**Requirement 5.3.1** — Every reasoner MUST implement:

```
def infer(self, coh: COHObject, query: Query) -> ReasoningResult
```

**Requirement 5.3.2** — `ReasoningResult` MUST carry `answer`, `confidence` in `[0, 1]`, `trace` (list of strings), `derived` (dict), and `reasoner` (string).

**Requirement 5.3.3** — Every reasoner MUST be deterministic given the same inputs.

### 5.4 Optimiser Interface

**Requirement 5.4.1** — Every optimiser MUST implement:

```
def step(self, params: Sequence[float], grads: Sequence[float]) -> list[float]
```

**Requirement 5.4.2** — `step` MUST NOT modify `params` or `grads` in place.

**Requirement 5.4.3** — `step` MUST return a new list of the same length as `params`.

---

## 6. Module Specifications

### 6.1 `gismol.core`

#### 6.1.1 `coh_object`

Contains `COHObject`, `Attribute`, `Method`, `State`. The class hierarchy is specified in §4.2.

**Requirement 6.1.1.1** — `COHObject.__init__` MUST NOT execute any user-supplied callable. Construction is declarative.

**Requirement 6.1.1.2** — `COHObject.state` MUST be recomputed from children on demand. If cached, the cache MUST be invalidated when a child is added or removed.

#### 6.1.2 `constraints`

Contains `Constraint`, `IdentityConstraint`, `TriggerConstraint`, `GoalConstraint`, `ConstraintSet`.

**Requirement 6.1.2.1** — All constraint classes MUST expose `satisfied(state) -> bool` and `violation(state) -> float`.

**Requirement 6.1.2.2** — `ConstraintSet.check_identity(state)` MUST raise `ConstraintViolationError` on the first violated constraint.

#### 6.1.3 `daemons`

Contains `Daemon`, `CorrectiveAction`, `InvariantDaemon`, `BoundaryDaemon`, `RateDaemon`, `SafetyDaemon`, `DaemonNetwork`.

**Requirement 6.1.3.1** — `BoundaryDaemon.project(state)` MUST return a state where every bound attribute lies within `[lo, hi]`. It MUST NOT modify other attributes.

**Requirement 6.1.3.2** — `RateDaemon` MUST track the previous state and previous time internally. It MUST reset when `reset()` is called.

#### 6.1.4 `daemon_impl`

Contains `HomeostasisDaemon`, `ProjectionDaemon`, `BarrierDaemon`, `PrecisionWeightedDaemon`, `EventDaemon`, `RateLimitDaemon`.

**Requirement 6.1.4.1** — `BarrierDaemon` MUST compute `h(state)` and, if `derivative` is provided, `dh/dt + alpha(h)`. It MUST emit an action whenever either value violates the corresponding threshold.

#### 6.1.5 `dynamics`

Contains `ContinuousFlow`, `DiscreteTransition`, `HybridDynamicalSystem`, `Trajectory`.

**Requirement 6.1.5.1** — `ContinuousFlow.step` MUST use a fourth-order Runge–Kutta integration scheme.

**Requirement 6.1.5.2** — `HybridDynamicalSystem.step` MUST execute in the following order:

1. Evaluate triggers and apply the highest-priority action.
2. Run daemons and apply the highest-severity correction.
3. Integrate the continuous flow over `dt`.

**Requirement 6.1.5.3** — The `info` dict returned from `step` MUST contain the keys `"events"`, `"corrections"`, and `"admissible"`.

#### 6.1.6 `embedding`

Contains `Embedder`, `HilbertSpace`, `FeatureEmbedder`, `HashEmbedder`, `CompositeEmbedder`, `ConstantEmbedder`, and vector helpers.

**Requirement 6.1.6.1** — `HashEmbedder` MUST be deterministic. Two identical objects (per `canonical_form`) MUST embed to the same vector.

**Requirement 6.1.6.2** — `FeatureEmbedder` MUST normalise features by min-max when `ranges` are provided, mapping to `[0, 1]`.

#### 6.1.7 `expression`

Contains `Expr`, `Const`, `Var`, `Cmp`, `Arith`, `Call`, `And`, `Or`, `Not`, `Implies`, `parse_expression`.

**Requirement 6.1.7.1** — The parser MUST accept only a whitelisted subset of Python's AST. It MUST reject:

- `ast.Attribute`
- `ast.Subscript`
- `ast.Lambda`
- `ast.ListComp`, `ast.SetComp`, `ast.DictComp`, `ast.GeneratorExp`
- `ast.Import`, `ast.ImportFrom`
- `ast.Assign`, `ast.AugAssign`, `ast.AnnAssign`
- `ast.Await`, `ast.Yield`, `ast.YieldFrom`
- Any node not on the whitelist

**Requirement 6.1.7.2** — `parse_expression` MUST NOT use `eval` or `exec`. It MUST interpret the AST directly.

**Requirement 6.1.7.3** — The parser MUST support the functions listed in Table 6.1.7.3. Additional functions MAY be registered.

| Function | Arity | Behaviour |
|---|---|---|
| `abs` | 1 | Absolute value |
| `min` | n≥1 | Minimum |
| `max` | n≥1 | Maximum |
| `sum` | 1 | Sum of iterable |
| `len` | 1 | Length |
| `round` | 1 or 2 | Round |
| `int` | 1 | Integer conversion |
| `float` | 1 | Float conversion |
| `bool` | 1 | Boolean conversion |
| `sqrt` | 1 | Square root |
| `exp` | 1 | Exponential |
| `log` | 1 | Natural logarithm |
| `sin`, `cos`, `tanh` | 1 | Trigonometric / hyperbolic |
| `true`, `false` | 0 | Boolean constants |

#### 6.1.8 `repository`

Contains `COHRepository`.

**Requirement 6.1.8.1** — `add` MUST reject duplicate identifiers unless the existing object is identical to the new one.

**Requirement 6.1.8.2** — `save` MUST write a JSON file that `load` can round-trip.

**Requirement 6.1.8.3** — `load` MUST NOT execute arbitrary code. In particular, it MUST NOT restore methods or neural weights from the file.

#### 6.1.9 `exceptions`

The exception hierarchy is specified in §8.

### 6.2 `gismol.categorical`

Contains `Category`, `Morphism`, `COHCategory`, `COHMorphism`, `Functor`, `ComponentFunctor`, `ForgetfulFunctor`, `EmbeddingFunctor`, `Graph`, `Diagram`, `Cone`, `Limit`, `constraint_limit`, `pullback`, `product_set`.

**Requirement 6.2.1** — `COHCategory.product(a, b)` MUST return a `COHObject` whose children include both `a` and `b`, whose identity constraints are the conjunction of those of `a` and `b`, and whose goals and triggers are the concatenation.

**Requirement 6.2.2** — `COHMorphism(source, target)` MUST raise `MorphismError` if any structural element of `source` is missing from `target` (attributes, methods, neural components, constraints, daemons).

### 6.3 `gismol.dynamical`

Contains `HybridAutomaton`, `Mode`, `Guard`, `Transition`, `LyapunovAnalysis`, `StabilityResult`, `FreeEnergy`, `GenerativeModel`, `VariationalPosterior`.

**Requirement 6.3.1** — `LyapunovAnalysis.analyse(V, states)` MUST compute the Lie derivative of `V` along the flow for each state and classify stability based on its sign.

**Requirement 6.3.2** — `FreeEnergy.variational(observation, q)` MUST compute `E_q[-log p(o, s)] - H[q]` exactly (within floating-point tolerance).

### 6.4 `gismol.stability`

Contains `ConstraintStabilityAnalyzer`, `LyapunovFunction`, `LyapunovCertificate`, `lyapunov_from_constraints`, `goal_lyapunov`.

**Requirement 6.4.1** — `lyapunov_from_constraints(coh)` MUST return a function `V(s) = sum_i violation_i(s)`.

**Requirement 6.4.2** — The certificate returned by `analyse` MUST contain at minimum `stable` (bool), `asymptotic` (bool), `rate` (float), `samples` (int).

### 6.5 `gismol.neural`

Contains `MLP`, `Dense`, `Activation`, `MLPComponent`, `PolicyComponent`, `ValueComponent`, `PredictorComponent`, `PerceptionComponent`, `NeuralComponentSet`, `SGD`, `Adam`, `RMSProp`, `AdaGrad`, `Momentum`, `LagrangianOptimizer`, `ConstrainedMDP`, `PrimalDualOptimizer`, `LagrangianAgent`, `HierarchicalValueDecomposition`, `CreditAssigner`, `NeuralEmbedder`, `AutoencoderEmbedder`, `SiameseEmbedder`.

**Requirement 6.5.1** — `MLP.forward` MUST NOT mutate the input.

**Requirement 6.5.2** — `MLP.backward` MUST accumulate gradients into the layers' `grad_*` fields. It MUST NOT update parameters.

**Requirement 6.5.3** — `MLP.fit` MUST apply the optimiser once per batch. It MUST NOT update parameters before gradients are averaged over the batch.

**Requirement 6.5.4** — `LagrangianAgent.train` MUST perform both primal and dual updates on each iteration.

### 6.6 `gismol.nlp`

Contains `TextNormalizer`, `normalize`, `parse`, `tokenize`, `split_sentences`, `EntityExtractor`, `RelationMiner`, `IntentRecognizer`, `ConstraintParser`, `parse_constraints`, `BagOfWordsEmbedder`, `TfidfEmbedder`, `HashingEmbedder`, `CharNGramEmbedder`, `SimilarityIndex`, `ResponseGenerator`, `TextToCOH`, `text_to_coh`, `ChatModule`, `ChatManager`, `COHValidator`, `validate_coh`.

**Requirement 6.6.1** — `normalize(text)` MUST be idempotent: `normalize(normalize(t)) == normalize(t)`.

**Requirement 6.6.2** — `text_to_coh(text, name)` MUST return a `COHObject` with at least one attribute, one identity constraint, one trigger, one goal, one daemon (when the input supports them).

**Requirement 6.6.3** — `COHValidator.validate(coh)` MUST return a list of `ValidationIssue` objects with `severity`, `message`, `path`, `code`.

**Requirement 6.6.4** — `ChatModule.process(text)` MUST return a `ChatResponse` with `.text`, `.intent`, `.entities`, `.relations`, `.constraints`, `.coh`.

### 6.7 `gismol.reasoners`

Contains `Reasoner`, `Rule`, `KnowledgeBase`, `Query`, `ReasoningResult`, `ForwardChainer`, `ConstraintReasoner`, `GoalSolver`, `AbductiveReasoner`, `GeneralReasoner`, `DomainProfile`, `DomainReasoner`, `register_domain`, `get_domain`, `CategoricalReasoner`, `AnalogicalReasoner`, `MetaReasoner`, `CausalReasoner`.

**Requirement 6.7.1** — `KnowledgeBase.infer` MUST run rules to fixpoint or until `max_iterations` is reached.

**Requirement 6.7.2** — `ConstraintReasoner.infer` MUST return the list of violated invariants and, when feasible, a repair that satisfies them.

**Requirement 6.7.3** — `GoalSolver.infer` MUST only consider states that satisfy identity constraints.

**Requirement 6.7.4** — `MetaReasoner` MUST select at least one reasoner and MUST return aggregated results.

---

## 7. Algorithms

### 7.1 Simulation Loop

**Algorithm 7.1 (Simulation Step).**

```
INPUT: coh, state, action, dt, event
OUTPUT: new_state, info

info ← {events: [], corrections: [], admissible: True}

# Phase 1: triggers
triggers ← coh.T.applicable_triggers(event, state)  # sorted by priority desc
FOR each trigger IN triggers:
    payload ← trigger.fire(state)
    info.events.append(trigger.name)
    state ← apply_action(state, payload)

# Phase 2: daemons
daemon_action ← coh.D.observe(state, t)
IF daemon_action IS NOT None:
    info.corrections.append(daemon_action.name)
    state ← apply_correction(state, daemon_action)

# Phase 3: continuous flow
IF coh.flow IS NOT None AND dt > 0:
    state ← rk4_step(coh.flow, state, action, dt)

info.admissible ← coh.constraints.satisfies_identity(state)
RETURN state, info
```

**Complexity.** O(|T| + |D| + c) per step, where `c` is the cost of one flow evaluation.

### 7.2 Runge–Kutta 4 Integration

**Algorithm 7.2 (RK4 Step).**

$$
\begin{aligned}
k_1 &= f(s, a, \theta) \\
k_2 &= f(s + \tfrac{\Delta t}{2} k_1, a, \theta) \\
k_3 &= f(s + \tfrac{\Delta t}{2} k_2, a, \theta) \\
k_4 &= f(s + \Delta t \cdot k_3, a, \theta) \\
s' &= s + \tfrac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)
\end{aligned}
$$

**Complexity.** O(v) per step, `v` = number of state variables.

### 7.3 Cycle Detection

**Algorithm 7.3 (Cycle Detection via DFS Colouring).**

```
INPUT: root
OUTPUT: True if acyclic

colours ← {}
STACK ← [(root, 0)]

FUNCTION visit(node):
    colour[node] ← GREY
    FOR child IN node.children:
        IF colour[child] == GREY: RETURN False
        IF colour[child] IS UNDEFINED: 
            IF NOT visit(child): RETURN False
    colour[node] ← BLACK
    RETURN True

RETURN visit(root)
```

**Complexity.** O(n) time, O(n) space.

### 7.4 Categorical Product Construction

**Algorithm 7.4.**

```
INPUT: a, b: COHObject
OUTPUT: p: COHObject

p.name ← f"({a.name} x {b.name})"
p.attributes ← a.attributes ++ b.attributes
p.methods    ← a.methods    ++ b.methods
p.neural     ← a.neural     ++ b.neural
p.identity   ← a.identity   ++ b.identity
p.triggers   ← a.triggers   ++ b.triggers
p.goals      ← a.goals      ++ b.goals
p.daemons    ← a.daemons    ++ b.daemons
p.add_child(a, key=a.name)
p.add_child(b, key=b.name)
RETURN p
```

**Complexity.** O(|a| + |b|) for shallow composition.

### 7.5 Lyapunov Lie Derivative

**Algorithm 7.5.**

```
INPUT: flow, V, state, eps=1e-6
OUTPUT: dV/dt along flow

d ← flow.derivative(state)
state' ← state
FOR each (var, delta) IN zip(flow.variables, d):
    state'[var] ← state'[var] + eps * delta
RETURN (V(state') - V(state)) / eps
```

**Complexity.** O(v) per state.

### 7.6 Primal-Dual Update

**Algorithm 7.6.**

```
INPUT: policy, trajectories, lambdas, lr_theta, lr_lambda
OUTPUT: updated policy, lambdas

# Primal: gradient ascent on expected return minus weighted cost
grad ← 0
FOR traj IN trajectories:
    weight ← traj.return - Σ lambdas[i] * traj.cost[i]
    grad += weight * ∇log π(a|s)
theta ← theta + lr_theta * grad / |trajectories|

# Dual: ascent on constraint violation
FOR i IN range(n_constraints):
    violation_i ← mean(traj.cost[i] for traj) - cost_limit[i]
    lambdas[i] ← max(0, lambdas[i] + lr_lambda * violation_i)

RETURN theta, lambdas
```

**Complexity.** O(E · H · |θ|) per iteration, `E` = trajectories, `H` = horizon.

### 7.7 Hierarchical Value Decomposition Fit

**Algorithm 7.7 (Least-Mean-Squares).**

```
INPUT: samples [(child_values, parent_value)], lr, epochs
OUTPUT: weights, coordination

w ← uniform weights; c ← 0
FOR epoch IN 1..epochs:
    FOR (values, target) IN samples:
        pred ← Σ w_i * values_i + c
        err ← pred - target
        FOR i: g_w[i] += 2 * err * values_i
        g_c += 2 * err
    w ← w - lr * g_w / |samples|
    c ← c - lr * g_c / |samples|
RETURN w, c
```

**Complexity.** O(epochs · samples · children).

---

## 8. Error Handling

### 8.1 Exception Hierarchy

```
COHError (base)
├── ConstraintViolationError
├── UnsatisfiableConstraintError
├── CyclicHierarchyError
├── MethodExecutionError
├── DaemonError
├── EmbeddingError
├── ConvergenceError
├── MorphismError
├── ValidationError
├── RepositoryError
├── ParseError
└── ReasoningError
```

**Requirement 8.1.1** — Every exception raised by a GISMOL function MUST inherit from `COHError`, either directly or transitively.

**Requirement 8.1.2** — Every `COHError` subclass MUST accept an optional `message` keyword argument that overrides the default message.

### 8.2 Error Semantics

| Exception | Raised when | Recovery |
|---|---|---|
| `ConstraintViolationError` | An identity constraint is not satisfied at a required check | Fix the state or relax the constraint |
| `UnsatisfiableConstraintError` | A set of identity constraints has no solution in the given domain | Reconcile contradictory constraints |
| `CyclicHierarchyError` | Adding a child would create a cycle | Restructure the hierarchy |
| `MethodExecutionError` | A precondition or postcondition fails | Fix the state or the contract |
| `DaemonError` | A daemon raises an unrecoverable error | Inspect the daemon's implementation |
| `EmbeddingError` | Embedding fails (e.g., dimension mismatch) | Check embedder configuration |
| `ConvergenceError` | An iterative procedure fails to converge | Increase iterations or relax tolerance |
| `MorphismError` | A map between COH objects is not structure-preserving | Review the mapping |
| `ValidationError` | `COHValidator.raise_if_invalid` finds errors | Fix the reported issues |
| `RepositoryError` | Registering a duplicate, missing key, or IO failure | Check identifiers |
| `ParseError` | `parse_expression` rejects a string | Fix the syntax |
| `ReasoningError` | A reasoner cannot complete | Inspect the query and inputs |

### 8.3 Atomicity

**Requirement 8.3.1** — If `add_child` raises `CyclicHierarchyError`, the parent MUST remain unchanged.

**Requirement 8.3.2** — If `Method.__call__` raises `MethodExecutionError`, the state change from the method MUST NOT be persisted. The postcondition failure is informational and MUST be logged.

### 8.4 Error Reporting

**Requirement 8.4.1** — Exceptions SHOULD carry sufficient context to identify the failing object, constraint, or path. In particular, `ConstraintViolationError` MUST expose `.constraint` and `.state`.

---

## 9. Serialisation Format

### 9.1 JSON Schema

The canonical serialisation format is JSON. The schema is defined in Appendix B.

**Requirement 9.1.1** — `COHObject.to_dict()` MUST return a dict conforming to the schema.

**Requirement 9.1.2** — `COHObject.from_dict(data)` MUST reconstruct the object such that:

- Names, descriptions, and metadata are preserved.
- Attribute defaults are preserved.
- Identity constraints expressed as strings are preserved and re-parsed.
- Children are recursively reconstructed.

**Requirement 9.1.3** — Callables (methods, daemons, neural weights) MUST NOT be serialised. Their names and descriptions MUST be preserved. Implementations SHOULD emit a warning when attempting to serialise an object with substantial callable state.

### 9.2 Schema Versioning

**Requirement 9.2.1** — Every serialised payload MUST include a `schema` field with an integer value. The current version is `1`.

**Requirement 9.2.2** — `from_dict` MUST reject payloads whose `schema` is greater than the implementation's supported version.

**Requirement 9.2.3** — When the schema version increases, `from_dict` MUST support loading payloads from the previous version for at least one major release cycle.

### 9.3 Round-Trip Guarantee

**Requirement 9.3.1** — For any `COHObject` `o` that contains only string-based identity constraints (no lambdas), the following MUST hold:

$$
\text{structural\_equiv}(o,\, \text{COHObject.from\_dict}(o.\text{to\_dict}())) = \text{True}
$$

where `structural_equiv` compares component counts, names, and constraint texts.

---

## 10. Performance Characteristics

### 10.1 Complexity Classes

| Operation | Complexity | Notes |
|---|---|---|
| `COHObject.__init__` | O(c + a + m + n + i + t + g + d) | Linear in components |
| `add_child` | O(n) | Cycle check via DFS |
| `topological_order` | O(n) | DFS-based |
| `add_attribute` | O(1) | Amortised |
| `add_identity_constraint` | O(1) | Amortised |
| `satisfies_constraints` | O(|I|) | Per check |
| `step` | O(|T| + |D| + v) | Per step |
| `simulate(N steps)` | O(N · (|T| + |D| + v)) | |
| `embed` | O(|A| + Σ children) | Feature-based |
| `product(a, b)` | O(|a| + |b|) | Shallow |
| `COHMorphism.validate` | O(|a| + |b|) | Name set comparison |

### 10.2 Memory Model

**Requirement 10.2.1** — A `COHObject` MUST store its attributes, methods, neural components, and constraints as dictionaries or lists whose total memory footprint is linear in the number of components.

**Requirement 10.2.2** — A `Trajectory` of length `N` MUST NOT retain full copies of intermediate states beyond what is necessary for iteration. Implementations MAY provide a streaming API.

### 10.3 Scaling Guidelines

The following guidelines describe expected performance on a single modern CPU core (Python 3.11):

| System size | Steps/second | Notes |
|---|---|---|
| 10 objects, 3 constraints | ~10,000 | Small controller |
| 100 objects, 20 constraints | ~2,000 | Sub-system |
| 1,000 objects, 100 constraints | ~100 | Medium plant |
| 10,000 objects, 1,000 constraints | ~5 | Large plant |

### 10.4 Numpy Acceleration

**Requirement 10.4.1** — If `numpy` is importable, `MLP.fit` MUST use vectorised operations. Implementations MUST fall back to pure Python if numpy is absent.

**Requirement 10.4.2** — The presence or absence of numpy MUST NOT change the numerical semantics of the toolkit.

---

## 11. Security Considerations

### 11.1 Safe Expression Evaluation

**Requirement 11.1.1** — The expression parser MUST reject any input that would invoke arbitrary code execution. The whitelist in §6.1.7.1 is normative.

**Requirement 11.1.2** — The parser MUST reject strings longer than an implementation-defined limit (RECOMMENDED: 10,000 characters).

**Requirement 11.1.3** — The parser MUST reject expressions with more than an implementation-defined number of nodes (RECOMMENDED: 10,000).

### 11.2 Deserialisation

**Requirement 11.2.1** — `from_dict` MUST NOT execute any callable encoded in the payload.

**Requirement 11.2.2** — `from_dict` MUST reject payloads that reference unknown classes or modules.

**Requirement 11.2.3** — `COHRepository.load` MUST NOT execute code from the file. In particular, it MUST NOT use `pickle`.

### 11.3 Resource Limits

**Requirement 11.3.1** — Implementations SHOULD provide a mechanism to limit:

- Maximum hierarchy depth
- Maximum number of children per object
- Maximum number of constraints
- Maximum `simulate` iterations
- Maximum trajectory length

**Requirement 11.3.2** — If a resource limit is exceeded, the implementation MUST raise a `COHError` and MUST NOT partially apply the offending operation.

### 11.4 Trust Boundary

**Requirement 11.4.1** — Callables (`Method.fn`, `Daemon.observe`, `TriggerConstraint.action`, `GoalConstraint.objective`) MUST be treated as trusted code. Implementations MUST document that loading user-supplied callables is a security-sensitive operation.

**Requirement 11.4.2** — The NLP front-end MUST NOT be exposed to untrusted input without a timeout or length limit, because regular-expression matching can exhibit super-linear behaviour.

---

## 12. Conformance Classes

### 12.1 Class 1 (Minimal)

A Class 1 implementation MUST provide:

- `COHObject` with the full 9-tuple
- `Attribute`, `Method`, `State`
- `IdentityConstraint`, `TriggerConstraint`, `GoalConstraint`, `ConstraintSet`
- `InvariantDaemon`, `BoundaryDaemon`, `DaemonNetwork`, `CorrectiveAction`
- `ContinuousFlow`, `DiscreteTransition`, `HybridDynamicalSystem`, `Trajectory`
- `Embedder`, `HilbertSpace`, `FeatureEmbedder`, `HashEmbedder`
- `Expr`, `Const`, `Var`, `Cmp`, `Arith`, `Call`, `And`, `Or`, `Not`, `parse_expression`
- The exception hierarchy of §8

A Class 1 implementation MAY omit categorical, neural, NLP, and reasoner modules.

### 12.2 Class 2 (Standard)

A Class 2 implementation MUST provide everything in Class 1, plus:

- `Category`, `Morphism`, `COHCategory`, `COHMorphism`
- `ComponentFunctor`, `ForgetfulFunctor`, `EmbeddingFunctor`, `Graph`
- `Diagram`, `Cone`, `Limit`, `constraint_limit`, `pullback`
- `MLP`, `Dense`, `Activation`
- `MLPComponent`, `PolicyComponent`, `ValueComponent`, `PredictorComponent`
- `SGD`, `Adam`, `RMSProp`, `AdaGrad`, `Momentum`
- `TextNormalizer`, `parse`, `EntityExtractor`, `RelationMiner`
- `IntentRecognizer`, `ConstraintParser`, `TextToCOH`, `ChatModule`
- `COHValidator`

### 12.3 Class 3 (Full)

A Class 3 implementation MUST provide everything in Class 2, plus:

- `LyapunovAnalysis`, `StabilityResult`, `verify_decrease_condition`
- `ConstraintStabilityAnalyzer`, `LyapunovCertificate`
- `FreeEnergy`, `GenerativeModel`, `VariationalPosterior`, `kl_divergence`
- `ConstrainedMDP`, `PrimalDualOptimizer`, `LagrangianAgent`
- `HierarchicalValueDecomposition`, `CreditAssigner`
- `ForwardChainer`, `ConstraintReasoner`, `GoalSolver`, `AbductiveReasoner`
- `GeneralReasoner`, `MetaReasoner`, `DomainReasoner`, `CausalReasoner`
- `AnalogicalReasoner`, `CategoricalReasoner`
- `DomainProfile`, `register_domain`, `get_domain`
- `ChatManager`, `SimilarityIndex`

---

## 13. Testing Requirements

### 13.1 Mandatory Tests

Every conforming implementation MUST include tests for:

- **Cycle detection:** Adding a child that creates a cycle MUST raise `CyclicHierarchyError`.
- **Topological ordering:** Every child MUST precede its parent.
- **Constraint evaluation:** A satisfied constraint MUST return `True`; a violated one MUST return `False`.
- **Trigger matching:** A trigger MUST fire only when its event matches and its condition holds.
- **Daemon arbitration:** A network with two firing daemons MUST return the one with higher severity.
- **State path resolution:** Dotted paths MUST resolve correctly across multiple levels.
- **Expression parsing:** The whitelist MUST reject attribute access, subscripting, lambdas, comprehensions, imports, and assignment.
- **Serialisation round-trip:** `from_dict(to_dict(o))` MUST be structurally equivalent to `o`.
- **Simulation determinism:** The same initial state, flow, and policy MUST produce the same trajectory.
- **Conformance class coverage:** Every public class in the claimed class MUST have at least one test.

### 13.2 Coverage

**Requirement 13.2.1** — A Class 3 implementation SHOULD maintain ≥ 80% line coverage across the `gismol` package.

**Requirement 13.2.2** — Every public class MUST have at least one test that constructs it and exercises at least one method.

### 13.3 Conformance Test Suite

A conformance test suite is under development. An implementation SHOULD run against it before claiming conformance.

---

## 14. Versioning and Compatibility

### 14.1 Semantic Versioning

GISMOL follows [SemVer 2.0.0](https://semver.org/). Given a version `MAJOR.MINOR.PATCH`:

- `MAJOR` is incremented on incompatible API changes
- `MINOR` is incremented on backward-compatible additions
- `PATCH` is incremented on backward-compatible bug fixes

### 14.2 Python Version Support

**Requirement 14.2.1** — The toolkit MUST support Python 3.9 or later.

**Requirement 14.2.2** — The toolkit SHOULD support the four most recent minor releases of Python at any given time.

### 14.3 Deprecation Policy

**Requirement 14.3.1** — Deprecation MUST be announced at least one MINOR version before removal.

**Requirement 14.3.2** — Deprecated APIs MUST emit a `DeprecationWarning` on first use.

**Requirement 14.3.3** — The CHANGELOG MUST list all deprecations and the version in which the API will be removed.

### 14.4 Schema Compatibility

**Requirement 14.4.1** — Schema changes MUST be backward-compatible for at least one MAJOR version cycle.

**Requirement 14.4.2** — The `schema` field in serialised payloads MUST be bumped whenever the payload format changes incompatibly.

---

## 15. References

### 15.1 Normative

- [RFC 2119] Bradner, S., "Key words for use in RFCs to Indicate Requirement Levels", BCP 14, RFC 2119, March 1997.
- [SemVer] Preston-Werner, T., "Semantic Versioning 2.0.0", 2013.
- [JSON Schema] Wright, A., et al., "JSON Schema: A Media Type for Describing JSON Documents", 2020.
- [PEP 484] van Rossum, G., et al., "Type Hints", 2014.
- [PEP 8] van Rossum, G., et al., "Style Guide for Python Code", 2001.

### 15.2 Informative

- [COH-2025] "The Mathematical Foundations of Constrained Object Hierarchies: A Unified Framework for World Modelling and Artificial General Intelligence", 2025.
- [GISMOL-Guide] "The GISMOL Developer Guide", 2025.
- [Mac Lane 1978] Mac Lane, S., *Categories for the Working Mathematician*, Springer, 1978.
- [Khalil 2002] Khalil, H. K., *Nonlinear Systems*, 3rd ed., Prentice Hall, 2002.
- [Sutton & Barto 2018] Sutton, R. S., Barto, A. G., *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018.
- [Friston 2010] Friston, K., "The free-energy principle: a unified brain theory?", *Nat. Rev. Neurosci.*, vol. 11, no. 2, pp. 127–138, 2010.
- [Cover & Thomas 2006] Cover, T. M., Thomas, J. A., *Elements of Information Theory*, 2nd ed., Wiley, 2006.

---

## Appendix A · Grammar of the Constraint Language

### A.1 Lexical Grammar

```
IDENTIFIER   ::= [A-Za-z_][A-Za-z0-9_]*
PATH         ::= IDENTIFIER ("." IDENTIFIER)*
NUMBER       ::= "-"? [0-9]+ ("." [0-9]+)? ("e" [-+]? [0-9]+)?
STRING       ::= "'" [^']* "'" | '"' [^"]* '"'
BOOLEAN      ::= "True" | "False"
NULL         ::= "None"
```

### A.2 Syntactic Grammar

```
expression   ::= or_expr
or_expr      ::= and_expr ("or" and_expr)*
and_expr     ::= not_expr ("and" not_expr)*
not_expr     ::= "not" not_expr | comparison
comparison   ::= arith (comp_op arith)*
comp_op      ::= "==" | "!=" | "<" | "<=" | ">" | ">=" | "in" | "not in" | "is" | "is not"
arith        ::= term (("+" | "-") term)*
term         ::= factor (("*" | "/" | "%") factor)*
factor       ::= power ("**" factor)?
power        ::= unary
unary        ::= "-" unary | primary
primary      ::= NUMBER
               | STRING
               | BOOLEAN
               | NULL
               | PATH
               | "(" expression ")"
               | call
call         ::= IDENTIFIER "(" [expression ("," expression)*] ")"
```

### A.3 Rejected Productions

The following Python constructs MUST be rejected by the parser:

- Attribute access: `expr.name`
- Subscripting: `expr[index]`
- Slicing: `expr[a:b]`
- Lambda: `lambda x: ...`
- Comprehensions: `[x for x in ...]`
- Generator expressions: `(x for x in ...)`
- Assignment: `x = 1`
- Augmented assignment: `x += 1`
- `await`, `yield`, `yield from`
- `import`, `from ... import`
- `del`
- `assert`
- `global`, `nonlocal`
- Class and function definitions
- `try` / `except` / `finally`
- `with`
- `raise`

---

## Appendix B · JSON Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://gismol.example/schemas/coh-object-v1.json",
  "title": "COHObject",
  "type": "object",
  "required": ["schema", "uid", "name", "attributes", "constraints", "children"],
  "properties": {
    "schema": { "type": "integer", "const": 1 },
    "uid": { "type": "string", "format": "uuid" },
    "name": { "type": "string", "minLength": 1 },
    "description": { "type": "string" },
    "metadata": { "type": "object" },
    "attributes": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["name"],
        "properties": {
          "name": { "type": "string" },
          "default": {},
          "space": { "enum": ["real", "bool", "categorical", "vector"] },
          "description": { "type": "string" },
          "bounds": {
            "type": ["array", "null"],
            "items": { "type": "number" },
            "minItems": 2,
            "maxItems": 2
          },
          "unit": { "type": "string" }
        }
      }
    },
    "methods": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["name"],
        "properties": {
          "name": { "type": "string" },
          "description": { "type": "string" },
          "tags": { "type": "array", "items": { "type": "string" } }
        }
      }
    },
    "neural": {
      "type": "object",
      "additionalProperties": { "type": "string" }
    },
    "embedder": { "type": "string" },
    "constraints": {
      "type": "object",
      "properties": {
        "identity": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["name", "kind"],
            "properties": {
              "kind": { "const": "identity" },
              "name": { "type": "string" },
              "description": { "type": "string" },
              "weight": { "type": "number" },
              "expression": { "type": ["object", "null"] }
            }
          }
        },
        "triggers": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["name", "event"],
            "properties": {
              "kind": { "const": "trigger" },
              "name": { "type": "string" },
              "event": { "type": "string" },
              "description": { "type": "string" },
              "action": { "type": "string" },
              "priority": { "type": "integer" },
              "condition": { "type": ["object", "null"] }
            }
          }
        },
        "goals": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["name", "direction"],
            "properties": {
              "kind": { "const": "goal" },
              "name": { "type": "string" },
              "description": { "type": "string" },
              "direction": { "enum": ["max", "min"] },
              "weight": { "type": "number" },
              "target": { "type": ["number", "null"] },
              "expression": { "type": ["object", "null"] }
            }
          }
        },
        "lam": { "type": "number" }
      }
    },
    "daemons": {
      "type": "array",
      "items": { "type": "string" }
    },
    "children": {
      "type": "array",
      "items": { "$ref": "#" }
    }
  }
}
```

---

## Appendix C · UML-Style Class Diagram

```
                    ┌──────────────────┐
                    │   COHObject      │
                    ├──────────────────┤
                    │ uid: str         │
                    │ name: str        │
                    │ description: str │
                    │ metadata: dict   │
                    ├──────────────────┤
                    │ C: dict[COHObject]│
                    │ A: dict[Attribute]│
                    │ M: dict[Method]  │
                    │ N: dict[Neural]  │
                    │ E: Embedder      │
                    │ I: list[Identity]│
                    │ T: list[Trigger] │
                    │ G: list[Goal]    │
                    │ D: DaemonNetwork │
                    ├──────────────────┤
                    │ flow: Flow?      │
                    │ jump: Jump?      │
                    │ state: State     │
                    ├──────────────────┤
                    │ add_child()      │
                    │ add_attribute()  │
                    │ add_method()     │
                    │ add_identity()   │
                    │ add_trigger()    │
                    │ add_goal()       │
                    │ add_daemon()     │
                    │ add_neural()     │
                    │ step()           │
                    │ simulate()       │
                    │ embed()          │
                    │ product()        │
                    │ coproduct()      │
                    │ exponential()    │
                    │ soundness_report()│
                    └──────────────────┘
                            │
         ┌──────────────────┼──────────────────┬──────────────────┐
         │                  │                  │                  │
         ▼                  ▼                  ▼                  ▼
   ┌───────────┐      ┌───────────┐      ┌───────────┐      ┌───────────┐
   │ Attribute │      │  Method   │      │ Embedder  │      │  Neural   │
   ├───────────┤      ├───────────┤      ├───────────┤      ├───────────┤
   │ name      │      │ name      │      │ dim       │      │ name      │
   │ default   │      │ fn        │      │ space     │      │ theta     │
   │ space     │      │ pre       │      ├───────────┤      ├───────────┤
   │ bounds    │      │ post      │      │ __call__  │      │ __call__  │
   │ unit      │      ├───────────┤      │ embed     │      │ set_theta │
   └───────────┘      │ __call__  │      │ similarity│      │ n_params  │
                      └───────────┘      └───────────┘      └───────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │  ConstraintSet  │
                   ├─────────────────┤
                   │ identity: [I]   │
                   │ triggers: [T]   │
                   │ goals: [G]      │
                   ├─────────────────┤
                   │ check_identity()│
                   │ total_goal()    │
                   │ to_dict()       │
                   └─────────────────┘
                       │   │   │
         ┌─────────────┘   │   └─────────────┐
         ▼                 ▼                 ▼
   ┌───────────┐    ┌───────────┐    ┌───────────┐
   │ Identity  │    │  Trigger  │    │   Goal    │
   ├───────────┤    ├───────────┤    ├───────────┤
   │ predicate │    │ event     │    │ objective │
   │ margin    │    │ condition │    │ direction │
   │ weight    │    │ action    │    │ target    │
   │           │    │ priority  │    │ weight    │
   ├───────────┤    ├───────────┤    ├───────────┤
   │ satisfied │    │ matches   │    │ value     │
   │ violation │    │ fire      │    │ utility   │
   └───────────┘    └───────────┘    └───────────┘

                   ┌──────────────────┐
                   │  DaemonNetwork   │
                   ├──────────────────┤
                   │ daemons: [Daemon]│
                   ├──────────────────┤
                   │ observe()        │
                   │ add()            │
                   │ by_name()        │
                   └──────────────────┘
                            │
                            ▼
                   ┌──────────────────┐
                   │     Daemon       │
                   ├──────────────────┤
                   │ name: str        │
                   │ enabled: bool    │
                   │ history: [Action]│
                   ├──────────────────┤
                   │ observe()        │
                   │ monitor()        │
                   │ reset()          │
                   └──────────────────┘
                            △
                            │ (inherits)
                            │
         ┌──────────┬───────┴───────┬────────────┬─────────────┐
         │          │               │            │             │
   ┌──────────┐ ┌────────┐ ┌────────────┐ ┌──────────┐ ┌──────────────┐
   │Invariant │ │Boundary│ │Homeostasis │ │  Safety  │ │   Barrier    │
   │ Daemon   │ │ Daemon │ │  Daemon    │ │  Daemon  │ │   Daemon     │
   └──────────┘ └────────┘ └────────────┘ └──────────┘ └──────────────┘

                   ┌──────────────────┐
                   │ HybridDynamical  │
                   │     System       │
                   ├──────────────────┤
                   │ flow: Flow?      │
                   │ jump: Jump?      │
                   │ constraints: CS  │
                   │ daemons: Network │
                   ├──────────────────┤
                   │ step()           │
                   │ simulate()       │
                   └──────────────────┘
                            │
                            ▼
                   ┌──────────────────┐
                   │   Trajectory     │
                   ├──────────────────┤
                   │ times: [float]   │
                   │ states: [State]  │
                   │ actions: [Any]   │
                   │ rewards: [float] │
                   │ events: [str?]   │
                   ├──────────────────┤
                   │ cumulative_rew() │
                   │ discounted_ret() │
                   │ to_dict()        │
                   └──────────────────┘
```

---

<p align="center">
  <b>End of GISMOL Technical Specification v0.1.0</b><br>
  <i>This specification is a living document. Propose changes via the GitHub issue tracker.</i>
</p>