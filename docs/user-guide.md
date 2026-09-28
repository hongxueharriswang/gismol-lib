# The GISMOL Developer Guide

### A Comprehensive Guide to Building Intelligent Systems with Constrained Object Hierarchies and GISMOL library that implements the COH theory for general intelligence and intelligent systems.

**Version 0.1.0 · For developers of intelligent systems**

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Installation and Project Setup](#2-installation-and-project-setup)
3. [Mental Model — How to Think in COH](#3-mental-model--how-to-think-in-coh)
4. [Your First Intelligent System](#4-your-first-intelligent-system)
5. [Building Hierarchies (Component C)](#5-building-hierarchies-component-c)
6. [Attributes and State (Component A)](#6-attributes-and-state-component-a)
7. [Methods and Actions (Component M)](#7-methods-and-actions-component-m)
8. [Constraint Language in Depth (Components I, T, G)](#8-constraint-language-in-depth-components-i-t-g)
9. [Daemons in Depth (Component D)](#9-daemons-in-depth-component-d)
10. [Neural Components (Component N)](#10-neural-components-component-n)
11. [Embeddings (Component E)](#11-embeddings-component-e)
12. [Hybrid Dynamics — Simulation and Control](#12-hybrid-dynamics--simulation-and-control)
13. [Categorical Composition — Building Complex from Simple](#13-categorical-composition--building-complex-from-simple)
14. [Stability Analysis](#14-stability-analysis)
15. [Free-Energy Correspondence](#15-free-energy-correspondence)
16. [The Natural-Language Front-End](#16-the-natural-language-front-end)
17. [Reasoners](#17-reasoners)
18. [Validation and Soundness](#18-validation-and-soundness)
19. [Persistence and Repositories](#19-persistence-and-repositories)
20. [Design Patterns and Recipes](#20-design-patterns-and-recipes)
21. [Anti-Patterns to Avoid](#21-anti-patterns-to-avoid)
22. [Performance and Scaling](#22-performance-and-scaling)
23. [Debugging](#23-debugging)
24. [Case Study — Building a Full System End to End](#24-case-study--building-a-full-system-end-to-end)
25. [API Cheatsheet](#25-api-cheatsheet)
26. [Glossary](#26-glossary)

---

## 1. Introduction

### 1.1 What GISMOL Is

GISMOL — **General Intelligent Systems Modelling Language** — is a Python-embedded domain-specific language (DSL) for modelling intelligent systems. It implements **Constrained Object Hierarchies (COH)**, a mathematically grounded framework that unifies symbolic reasoning, neural learning, dynamical systems, and constraint-driven regulation into a single architecture.

When you write GISMOL code, you are not writing an application. You are **declaring the mathematical structure of an intelligent system**. The toolkit then gives you, for free:

- Hybrid simulation (continuous flow + discrete events)
- Constraint enforcement with real-time monitors
- Stability certificates
- Compositional algebra (products, coproducts, exponentials)
- Learning loops (constrained RL, hierarchical credit assignment)
- Reasoning engines
- Natural-language input and output
- Validation and soundness guarantees

### 1.2 Who This Guide Is For

This guide assumes you:

- Are comfortable with Python (functions, classes, lambdas, decorators)
- Have a system you want to make intelligent — a robot, a scheduler, a control loop, a market simulator, a tutor, a game agent
- Understand basic concepts like state, action, objective, and constraint

You do **not** need prior exposure to category theory, Lyapunov stability, or free-energy minimisation. Every advanced concept is introduced from first principles as we go.

### 1.3 How to Read This Guide

- **§2–§4** get you running in fifteen minutes.
- **§5–§11** are the core reference: one chapter per component of the 9-tuple.
- **§12–§17** cover the advanced capabilities (dynamics, category theory, learning, NLP, reasoning).
- **§18–§23** are operational: validation, persistence, patterns, performance, debugging.
- **§24** builds a complete system from scratch.
- **§25–§26** are quick reference.

Code snippets are runnable and self-contained unless marked otherwise. `⚠️` marks common pitfalls.

---

## 2. Installation and Project Setup

### 2.1 Requirements

GISMOL requires **Python 3.9 or newer** and has **zero mandatory runtime dependencies**. Everything you need is in the standard library.

Optional extras unlock acceleration and visualisation:

```bash
pip install -e ".[fast]"    # numpy acceleration
pip install -e ".[plot]"    # networkx + matplotlib
pip install -e ".[nlp]"     # sentence-transformers (better embeddings)
```

### 2.2 Installing

From PyPI (when published):

```bash
pip install gismol
```

From source:

```bash
git clone https://github.com/hongxueharriswang/gismol-lib.git
cd gismol
pip install -e ".[dev]"
```

Verify:

```python
import gismol
print(gismol.__version__)   # 0.1.0
```

### 2.3 Recommended Project Layout

For any project using GISMOL, use this layout:

```
my_intelligent_system/
├── systems/                 # One module per intelligent system
│   ├── __init__.py
│   ├── robot.py             # build_robot() -> COHObject
│   ├── scheduler.py
│   └── market.py
├── constraints/             # Reusable constraint factories
│   └── safety.py
├── daemons/                 # Custom daemon implementations
│   └── thermal.py
├── neural/                  # Neural architectures
│   └── policies.py
├── configs/                 # JSON/YAML configuration
│   └── defaults.json
├── tests/
│   └── test_systems.py
├── requirements.txt
└── README.md
```

**Convention.** Every intelligent system should be constructed by a single `build_<name>() -> COHObject` function. This makes systems testable, serialisable, and composable.

### 2.4 Recommended Import Style

Import the top-level package for everyday use:

```python
from gismol import COHObject, Attribute, Method
from gismol.core.constraints import IdentityConstraint, TriggerConstraint, GoalConstraint
```

Import submodules explicitly when you need depth:

```python
from gismol.neural.components import MLPComponent, PolicyComponent
from gismol.reasoners import GeneralReasoner, Query
```

Never use `from gismol import *` — the framework is deliberately explicit.

---

## 3. Mental Model — How to Think in COH

Before writing code, internalise the four guiding principles of COH.

### 3.1 Intelligence Is Constrained Optimisation

An intelligent system is not a program that computes outputs from inputs. It is an **optimiser** that maximises objectives while satisfying invariants:

$$
\text{intelligence} \;=\; \max_{\pi} \; \mathbb{E}\left[\sum_t \gamma^t r_t\right] \;-\; \lambda \cdot \mathrm{KL}(p_\pi \,\|\, p_\text{constraints})
$$

Translated into plain English: *pursue goals, but stay within bounds.* Everything in GISMOL serves this equation.

### 3.2 Structure Precedes Behaviour

Do not start by writing functions. Start by asking:

1. **What are the parts?** — the hierarchy (C)
2. **What do the parts measure?** — attributes (A)
3. **What can they do?** — methods (M)
4. **What must always be true?** — identity constraints (I)
5. **What reacts to events?** — triggers (T)
6. **What is being optimised?** — goals (G)

Only then add neural components (N), embeddings (E), and daemons (D). Structure first, learning second.

### 3.3 Everything Is a COHObject

A `COHObject` can represent:

- A physical device (a robot arm, a battery, a turbine)
- A logical subsystem (a planner, a scheduler, a controller)
- An abstract concept (a market, a policy, a learner)
- A whole society (a fleet, a power grid, a trading desk)

Whether the system is 3 objects deep or 30, whether it runs on a microcontroller or a cluster, the interface is identical: `COHObject`. This uniformity is what makes composition and reasoning possible.

### 3.4 Constraints Are First-Class

Most frameworks treat constraints as afterthoughts — assertions scattered through code. In GISMOL, constraints are declarative objects that participate in simulation, learning, reasoning, and validation. A constraint is not something you *check*; it is something the system *maintains*.

---

## 4. Your First Intelligent System

Let's build a minimal but complete intelligent system: a **smart light** that maintains target brightness while never exceeding power limits.

### 4.1 The Complete Example

```python
from gismol import COHObject, Attribute, Method
from gismol.core.constraints import (
    IdentityConstraint, TriggerConstraint, GoalConstraint,
)
from gismol.core.daemons import InvariantDaemon, BoundaryDaemon

def build_smart_light() -> COHObject:
    light = COHObject("smart_light", description="Adaptive LED controller")

    # A — attributes
    light.add_attribute(Attribute("brightness", 0.5, bounds=(0.0, 1.0)))
    light.add_attribute(Attribute("power_watts", 5.0, bounds=(0.0, 20.0)))
    light.add_attribute(Attribute("target", 0.8))
    light.add_attribute(Attribute("ambient_lux", 300.0, bounds=(0.0, 1000.0)))

    # M — methods
    def increase(state, action=None):
        new = state.copy()
        new["brightness"] = min(1.0, new["brightness"] + 0.1)
        new["power_watts"] = 20.0 * new["brightness"]
        return new, 0.1

    def decrease(state, action=None):
        new = state.copy()
        new["brightness"] = max(0.0, new["brightness"] - 0.1)
        new["power_watts"] = 20.0 * new["brightness"]
        return new, -0.05

    light.add_method(Method("increase", increase))
    light.add_method(Method("decrease", decrease))

    # I — identity constraints
    light.add_identity_constraint(IdentityConstraint(
        "power_limit", lambda s: s["power_watts"] <= 15.0,
        description="never exceed the circuit breaker limit",
    ))
    light.add_identity_constraint(IdentityConstraint(
        "brightness_bounds", lambda s: 0.0 <= s["brightness"] <= 1.0,
    ))

    # T — trigger constraints
    light.add_trigger_constraint(TriggerConstraint(
        "night_boost", "ambient_dark",
        condition=lambda s: s["ambient_lux"] < 100.0 and s["brightness"] < 0.9,
        action="increase", priority=1,
    ))
    light.add_trigger_constraint(TriggerConstraint(
        "power_emergency", "overpower",
        condition=lambda s: s["power_watts"] > 12.0,
        action="decrease", priority=5,
    ))

    # G — goal constraints
    light.add_goal_constraint(GoalConstraint(
        "brightness_match",
        lambda s: -abs(s["brightness"] - s["target"]),
        direction="max", weight=2.0,
    ))
    light.add_goal_constraint(GoalConstraint(
        "energy_saving",
        lambda s: -s["power_watts"],
        direction="max", weight=1.0,
    ))

    # D — daemons
    light.add_daemon(InvariantDaemon(constraints=list(light.I)))
    light.add_daemon(BoundaryDaemon({
        "brightness": (0.0, 1.0),
        "power_watts": (0.0, 20.0),
    }))

    return light


# Build, inspect, simulate
light = build_smart_light()
print(light.component_report())
# {'C': 0, 'A': 4, 'M': 2, 'N': 0, 'E': 'HashEmbedder',
#  'I': 2, 'T': 2, 'G': 2, 'D': 2}

traj = light.simulate(steps=20, dt=0.1)
print(f"Final brightness: {traj.final['brightness']:.3f}")
print(f"Cumulative reward: {traj.cumulative_reward():.3f}")
```

### 4.2 What Just Happened

When you called `light.simulate(steps=20, dt=0.1)`, the toolkit ran this loop twenty times:

1. **Trigger phase** — check every trigger whose condition matches; execute the highest-priority action.
2. **Daemon phase** — run every daemon; apply corrections (e.g., project out-of-bounds state).
3. **Continuous phase** — integrate the flow `ds/dt = f(s, a, θ)` (here empty, so state is unchanged).
4. **Record** — append the resulting state and reward to a trajectory.

The trajectory captures everything: times, states, actions, rewards, events. It is a first-class object you can analyse, embed, or replay.

### 4.3 What's Missing (and Why)

This example has no neural components, no embedding, no hierarchy, and no dynamics. That's fine — GISMOL is **pay-for-what-you-need**. The following chapters introduce each missing piece.

---

## 5. Building Hierarchies (Component C)

### 5.1 The Composition Operator

Every `COHObject` can contain named child `COHObject`s. Adding a child is `add_child`:

```python
arm = COHObject("arm")
joint1 = arm.add_child(COHObject("joint1"))
joint2 = arm.add_child(COHObject("joint2"))
gripper = arm.add_child(COHObject("gripper"))
```

The resulting structure is a **directed acyclic graph** (DAG). Each parent has an ordered dict of children; each child has exactly one parent.

### 5.2 Depth Guidance

Aim for **3–5 levels**. Common patterns:

| Depth | Role | Example |
|:-:|---|---|
| 0 | System | A whole warehouse |
| 1 | Subsystem | Picking zone, packing zone |
| 2 | Unit | A single robot |
| 3 | Component | Navigation, gripper, battery |
| 4 | Micro-component | Kalman filter, force controller |

If you find yourself going deeper, you probably want attributes rather than objects. If you find yourself going shallower, you may be missing modularity.

### 5.3 Traversal

```python
fleet = COHObject("fleet")
zone = fleet.add_child(COHObject("zone"))
robot = zone.add_child(COHObject("robot"))

# Direct access
fleet.C                # {'zone': <COHObject>}
fleet.children()       # same as above
zone.child("robot")    # <COHObject robot>

# Full traversal
fleet.descendants()    # [zone, robot]
fleet.topological_order()  # [robot, zone, fleet] — bottom-up

# DAG
fleet.component_dag()
# {'fleet': ['zone'], 'zone': ['robot'], 'robot': []}
```

### 5.4 Cycle Safety

GISMOL detects cycles at `add_child` time:

```python
a = COHObject("a")
b = COHObject("b")
a.add_child(b)
b.add_child(a)   # raises CyclicHierarchyError
```

This preserves **hierarchical soundness** (Theorem 6.2 in the theory): well-founded semantics require acyclicity.

### 5.5 Naming

Use descriptive names, lower-case with underscores. Names appear in serialised output, log messages, and reasoning traces, so they matter:

```python
# Good
zone = COHObject("northern_picking_zone")
robot = COHObject("robot_R2")

# Avoid
z = COHObject("z1")
r = COHObject("r")
```

When using `add_child(obj, key=...)`, you can override the storage key while keeping the object's internal name:

```python
arm.add_child(joint1, key="j1")
arm.add_child(joint2, key="j2")
```

### 5.6 Removing Children

```python
arm.remove_child("j1")
```

This removes the child and its nested state. Note: child objects themselves are not destroyed — keep a reference if you need them.

---

## 6. Attributes and State (Component A)

### 6.1 Declaring Attributes

An `Attribute` describes a measurable state variable — its name, default value, allowed bounds, and optional metadata:

```python
from gismol import Attribute

Attribute("temperature", 20.0, description="°C",
          bounds=(-40.0, 150.0), space="real", unit="°C")
```

| Parameter | Meaning |
|---|---|
| `name` | Identifier, used as key in state dict |
| `default` | Initial value (any Python object) |
| `space` | `"real"`, `"bool"`, `"categorical"`, `"vector"` |
| `bounds` | `(low, high)` used by `BoundaryDaemon` |
| `description` | Human-readable note |
| `unit` | Physical unit, for documentation |

### 6.2 Attaching to a `COHObject`

```python
robot = COHObject("robot")
robot.add_attribute(Attribute("x", 0.0, bounds=(-100, 100)))
robot.add_attribute(Attribute("y", 0.0, bounds=(-100, 100)))
robot.add_attribute(Attribute("heading", 0.0))
robot.add_attribute(Attribute("payload_kg", 0.0, bounds=(0, 25)))
```

The object's state is initialised from the defaults:

```python
robot.state
# State({'x': 0.0, 'y': 0.0, 'heading': 0.0, 'payload_kg': 0.0})
```

### 6.3 Hierarchical State

Children's states nest automatically under their parent:

```python
arm = COHObject("arm")
arm.add_attribute(Attribute("reach", 1.0))
joint = arm.add_child(COHObject("joint"))
joint.add_attribute(Attribute("angle", 0.0))

arm.state
# State({'reach': 1.0, 'joint': State({'angle': 0.0})})
```

Access nested values with **dotted paths**:

```python
arm.get("joint.angle")           # 0.0
arm.set("joint.angle", 0.5)
arm.get("joint.angle")           # 0.5
```

Internally, `State` is a dict subclass with `get_path` and `set_path`.

### 6.4 Attributes Are Not Magic

Attributes are conventional: they are documented, serialisable, and known to daemons and validators, but the toolkit does not enforce that every state key corresponds to a declared attribute. You *can* store ad-hoc values in the state dict; you *should* declare everything you want validated.

### 6.5 Choosing Granularity

Rule of thumb: **one attribute per quantity you would plot on a time series**. If a value never changes, it's a constant — put it in `metadata`. If a value is only meaningful during a method call, it's a local variable — do not add it as an attribute.

---

## 7. Methods and Actions (Component M)

### 7.1 Signatures

A `Method` wraps a function that transforms state and (optionally) returns a reward:

```python
Method("name", fn, description="", preconditions=[], postconditions=[])
```

The wrapped function can have any of these signatures:

```python
fn(state)                          -> new_state | (new_state, reward)
fn(state, action)                  -> new_state | (new_state, reward)
```

`new_state` can be a `State`, a plain dict, or `None` (meaning "state unchanged"). `reward` defaults to `0.0`.

### 7.2 Building a Method

```python
def move_forward(state, action=None):
    new = state.copy()
    new["x"] += 0.5 * math.cos(new["heading"])
    new["y"] += 0.5 * math.sin(new["heading"])
    return new, 0.1      # small positive reward for progress

robot.add_method(Method("move_forward", move_forward,
                        description="advance 0.5 m along heading"))
```

### 7.3 Preconditions and Postconditions

Contracts make methods self-checking:

```python
robot.add_method(Method(
    "grasp",
    lambda s, a=None: ({**s, "holding": True}, 1.0),
    preconditions=[
        lambda s: s["payload_kg"] <= 20.0,   # weight limit
        lambda s: not s["holding"],          # not already holding
    ],
    postconditions=[
        lambda s: s["holding"] is True,
    ],
))
```

Preconditions raise `MethodExecutionError` if violated; postconditions likewise.

### 7.4 Invoking Methods

Direct invocation:

```python
new_state, reward = robot.invoke("move_forward")
```

Or from a trigger:

```python
robot.add_trigger_constraint(TriggerConstraint(
    "auto_advance", "tick",
    condition=lambda s: s["x"] < 10.0,
    action="move_forward",
))
```

When `action` is a string matching a method name, the trigger executes the method. When it is a callable, the callable is applied directly. When it is a dict, the dict is merged into state.

### 7.5 Naming and Discovery

Method names should be verbs: `move`, `open`, `close`, `charge`, `discharge`. Reasoners use these names in their traces.

```python
robot.M              # {'move_forward': <Method>, ...}
robot.method("move_forward")   # <Method>
```

### 7.6 Determinism and Purity

Prefer **pure methods** that do not mutate the input state. The toolkit never relies on mutation and copying is cheap for small states. If you need stochasticity, use `random.Random(seed)` inside the method and record the seed in `metadata`.

---

## 8. Constraint Language in Depth (Components I, T, G)

Constraints are the **heart of COH**. This chapter covers them exhaustively.

### 8.1 Writing Constraints — Three Ways

Every constraint accepts a predicate in one of three forms:

```python
# (a) Lambda
IdentityConstraint("positive", lambda s: s["x"] > 0)

# (b) Expression tree
from gismol.core import expression as E
IdentityConstraint("positive", E.Var("x") > 0)

# (c) String (parsed safely)
IdentityConstraint("positive", "x > 0")
```

Form (c) is convenient and safe: GISMOL parses a **whitelisted subset of Python's AST** — no `eval`, no attribute access, no imports.

### 8.2 Identity Constraints (I)

**Identity constraints are invariants.** They must hold in every admissible state. Violating one means the system has left its feasible region.

```python
IdentityConstraint(
    name="temp_bounds",
    predicate=lambda s: 0.0 <= s["temperature"] <= 100.0,
    description="Operating temperature must stay within safe envelope",
    weight=2.0,        # severity weight, used by daemons
    margin=None,       # optional signed-distance function
)
```

**When to use identity constraints:**

- Physical limits (pressure, temperature, current)
- Logical invariants (mutual exclusion, conservation laws)
- Safety requirements (collision avoidance, dose limits)
- Legal/ethical boundaries (consent, privacy, budget caps)

**When *not* to use them:** for soft preferences, use goals. For event responses, use triggers.

**The `margin` parameter** lets you express signed distance to the boundary — used by advanced daemons for projection:

```python
IdentityConstraint(
    "temp_bounds",
    lambda s: s["temp"] <= 100.0,
    margin=lambda s: 100.0 - s["temp"],   # positive when safe
)
```

### 8.3 Trigger Constraints (T)

**Triggers are event-condition-action rules.** They fire when an event name matches and a condition holds:

```python
TriggerConstraint(
    name="emergency_shutdown",
    event="overheat",                       # or "*" for any event
    condition=lambda s: s["temp"] > 90.0,
    action="shutdown",                      # a method name or callable
    priority=10,                            # higher fires first
    preserves=lambda s: s["temp"] < 100.0,  # optional post-check
)
```

Multiple triggers may match the same event. They fire in descending priority order.

**Action types:**

```python
# String: invokes a method by that name
TriggerConstraint("t1", "evt", cond, action="increase")

# Callable: called with the state; return new state or (state, reward)
TriggerConstraint("t2", "evt", cond,
                  action=lambda s: ({**s, "flag": True}, 0.0))

# Dict: merged into the state
TriggerConstraint("t3", "evt", cond, action={"alarm": True})
```

**The `event` field** matches the `event` argument passed to `COHObject.step()`. Use `"*"` to listen to every event.

### 8.4 Goal Constraints (G)

**Goals are objectives.** Every goal has a direction (`"max"` or `"min"`) and a weight. Goals never "fail" — they contribute to an aggregate score:

$$
G_\text{total}(s, a, t) = \sum_j w_j \, g_j(s, a, t) - \lambda R(\theta)
$$

```python
GoalConstraint(
    name="throughput",
    objective=lambda s: s["items_per_hour"],
    direction="max",
    weight=2.0,
)

GoalConstraint(
    name="energy_cost",
    objective=lambda s: s["watts"],
    direction="min",
    weight=1.0,
)

# Objective with target (uses |g - target|)
GoalConstraint(
    name="hold_temp",
    objective=lambda s: s["temp"],
    direction="max",
    target=22.0,
    weight=1.5,
)
```

The objective can also accept `(state, action, t)` — three-argument callables get the full context:

```python
GoalConstraint(
    "smooth_control",
    lambda s, a, t: -abs(a) if a is not None else 0.0,
    direction="max",
)
```

### 8.5 The `ConstraintSet` Container

Internally, I/T/G live in a `ConstraintSet`:

```python
from gismol.core.constraints import ConstraintSet

cs = ConstraintSet()
cs.add_identity(IdentityConstraint("i1", "x > 0"))
cs.add_trigger(TriggerConstraint("t1", "evt", "x > 5", "increase"))
cs.add_goal(GoalConstraint("g1", "x", direction="max"))

cs.check_identity({"x": 1})           # passes
cs.check_identity({"x": -1})          # raises ConstraintViolationError
cs.satisfies_identity({"x": 1})       # True
cs.violation_score({"x": -5})         # positive magnitude
cs.total_goal({"x": 3})               # -3.0 (maximise x means utility = +x)
```

You usually interact with these through `COHObject`:

```python
coh.add_identity_constraint(...)
coh.I                # list of IdentityConstraint
coh.constraints      # the underlying ConstraintSet
```

### 8.6 Expression Language Reference

The safe expression language supports:

| Category | Syntax | Example |
|---|---|---|
| Variables | identifiers, dotted paths | `x`, `arm.joint.angle` |
| Constants | numbers, strings, booleans, `None` | `1.5`, `"on"`, `True` |
| Arithmetic | `+ - * / % **`, unary `-` | `(x + y) / 2` |
| Comparison | `== != < <= > >= in not in is is not` | `x >= 0 and x < 10` |
| Boolean | `and or not` | `(x > 0) & (y < 5)` |
| Functions | whitelisted calls | `abs(x)`, `min(a, b)`, `sqrt(x)`, `exp(x)`, `log(x)` |
| Containers | tuples, lists | `x in (1, 2, 3)` |

**Not supported:** attribute access (`.foo`), subscripting (`x[0]`), lambdas, comprehensions, imports, assignment, `eval`, `exec`.

Register your own functions:

```python
from gismol.core import expression as E

@E.register_function("clamp")
def _clamp(x, lo, hi):
    return max(lo, min(hi, x))

# Now usable in parsed expressions
expr = E.parse_expression("clamp(x, 0, 100) > 50")
expr.evaluate({"x": 75})   # True
```

### 8.7 Constraint Evaluation Cost

Constraints are evaluated on every simulation step and every daemon tick. Keep them cheap:

- Prefer arithmetic on scalars
- Avoid string operations in hot paths
- Cache complex sub-expressions in attributes

For expensive constraints (e.g., collision checks against large meshes), decompose them: use a cheap *proxy* constraint for every step and a full check inside a method that runs less often.

---

## 9. Daemons in Depth (Component D)

### 9.1 What Daemons Do

A **daemon** continuously monitors state and emits **corrective actions**. Unlike triggers, daemons are not tied to events — they run every step, ensuring invariants hold in real time.

Daemons are the mechanism by which GISMOL systems remain **safe under distribution shift** (Section 9.2 of the theory). Even if the learned policy fails, daemons enforce invariants.

### 9.2 The Daemon Interface

```python
class Daemon:
    def __init__(self, name, description="", enabled=True): ...
    def observe(self, state, t=0.0) -> Optional[CorrectiveAction]: ...
    def monitor(self, trajectory) -> List[CorrectiveAction]: ...
```

`observe` returns `None` (no action) or a `CorrectiveAction(name, payload, severity, message)`.

### 9.3 Built-In Daemons

| Daemon | Purpose | Key parameters |
|---|---|---|
| `InvariantDaemon` | Reports identity violations | `constraints`, `on_violation` |
| `BoundaryDaemon` | Projects state into a box | `bounds={"x": (lo, hi), ...}` |
| `RateDaemon` | Limits rate of change | `attribute`, `max_rate` |
| `SafetyDaemon` | Hazard-margin monitoring | `hazard`, `threshold` |
| `HomeostasisDaemon` | Set-point regulation | `attribute`, `setpoint`, `gain` |
| `ProjectionDaemon` | Custom projection onto feasible region | `projection` |
| `BarrierDaemon` | Control-barrier forward invariance | `barrier`, `derivative`, `alpha` |
| `PrecisionWeightedDaemon` | Precision-weighted prediction error | `predictor`, `target`, `precision` |
| `EventDaemon` | Edge-triggered action | `predicate`, `action` |
| `DaemonNetwork` | Composite; highest severity wins | `daemons=[...]` |

### 9.4 Attaching Daemons

```python
from gismol.core.daemons import InvariantDaemon, BoundaryDaemon

robot.add_daemon(InvariantDaemon(constraints=list(robot.I)))
robot.add_daemon(BoundaryDaemon({
    "x": (-50.0, 50.0),
    "y": (-50.0, 50.0),
    "payload_kg": (0.0, 25.0),
}))
```

Daemons are automatically aggregated into a `DaemonNetwork` on the parent:

```python
robot.D                    # DaemonNetwork
robot.D.by_name("invariant")   # the InvariantDaemon
robot.monitor()            # runs all daemons; returns highest-severity action
```

### 9.5 Homeostatic Regulation

The `HomeostasisDaemon` implements a simple proportional controller for a set-point:

```python
from gismol.core.daemon_impl import HomeostasisDaemon

battery.add_daemon(HomeostasisDaemon(
    attribute="temperature",
    setpoint=25.0,
    gain=0.5,          # correction proportional to error
    tolerance=0.5,     # dead-band
))
```

Each tick, it emits a `regulate` action carrying `{attribute: -gain * (value - setpoint)}` as payload. A parent can integrate this into its state update.

### 9.6 Control-Barrier Supervision

`BarrierDaemon` implements the forward-invariance check from modern control theory. Given a barrier function `h(state) >= 0` and a decay class `α`, it verifies:

$$
\dot h(s) + \alpha(h(s)) \ge 0
$$

```python
from gismol.core.daemon_impl import BarrierDaemon

robot.add_daemon(BarrierDaemon(
    barrier=lambda s: 1.0 - (s["x"] ** 2 + s["y"] ** 2) / 100.0,   # inside a circle
    derivative=lambda s: -2 * (s["x"] * s["x_dot"] + s["y"] * s["y_dot"]) / 100.0,
    alpha=lambda h: 0.5 * h,
    intervention="emergency_stop",
))
```

### 9.7 Writing a Custom Daemon

Subclass `Daemon` and implement `observe`:

```python
from gismol.core.daemons import Daemon, CorrectiveAction

class ThermalThrottleDaemon(Daemon):
    """Reduce power when temperature approaches a limit."""

    def __init__(self, attribute="temperature", limit=80.0, margin=10.0):
        super().__init__("thermal_throttle")
        self.attribute = attribute
        self.limit = limit
        self.margin = margin

    def observe(self, state, t=0.0):
        temp = state.get(self.attribute, 0.0)
        if temp < self.limit - self.margin:
            return None
        severity = (temp - (self.limit - self.margin)) / self.margin
        throttle = max(0.0, 1.0 - severity)
        return self._emit(CorrectiveAction(
            name="throttle",
            payload={"power_limit": throttle},
            daemon=self.name,
            severity=severity,
            message=f"temp={temp:.1f} → throttle={throttle:.2f}",
        ))
```

Attach it:

```python
reactor.add_daemon(ThermalThrottleDaemon("core_temp", limit=900.0))
```

### 9.8 Severity and Priority

When multiple daemons fire, the **network** returns the one with the highest `severity`. If you need deterministic ordering, give your daemon a positive severity in `[0, 1]` and combine with priority encoded in the name. For critical safety daemons, use severity values well above 1.0 to guarantee dominance.

---

## 10. Neural Components (Component N)

### 10.1 Why Neural Components?

Identity constraints and triggers capture *known* rules. Neural components capture *learned* mappings — perception, prediction, control policies. They live inside `COHObject`s so that learning is scoped and composable.

### 10.2 Neural Component Types

| Class | Role | Typical size |
|---|---|---|
| `MLPComponent` | Generic `f_θ : X → Y` | 2–3 layers |
| `PerceptionComponent` | Encode observations to latent | input → 4–32 dims |
| `PredictorComponent` | Predict next state/attribute | input → 1–4 outputs |
| `PolicyComponent` | Softmax over discrete actions | input → n_actions |
| `ValueComponent` | Approximate `V(s)` | input → 1 |

```python
from gismol.neural.components import (
    MLPComponent, PolicyComponent, ValueComponent, PredictorComponent,
)

robot.add_neural(PolicyComponent("pick_policy", input_dim=6, n_actions=4, seed=42),
                 name="policy")
robot.add_neural(ValueComponent("V", input_dim=6, seed=43), name="value")
robot.add_neural(PredictorComponent("next_pos", input_dim=6, output_dim=2, seed=44),
                 name="predictor")
```

### 10.3 Using a Neural Component

Components are callable:

```python
policy = robot.N["policy"]
probs = policy([0.1, 0.5, -0.2, 0.0, 0.3, 0.8])   # softmax over actions
action = policy.sample([...])                     # sampled action index
```

### 10.4 Training

Every MLP-based component exposes `.fit(X, Y, ...)`:

```python
import random

X = [[random.random() for _ in range(6)] for _ in range(500)]
Y = [[sum(x[:3]) - sum(x[3:])] for x in X]   # some target function

predictor = robot.N["predictor"]
history = predictor.fit(
    X, Y,
    epochs=200,
    lr=0.05,
    batch_size=32,
    optimizer=Adam(0.01),
    verbose=True,
)
print(f"final loss: {history[-1]:.6f}")
```

### 10.5 Optimisers

```python
from gismol.neural.optimizers import SGD, Adam, RMSProp, AdaGrad, Momentum

opt = Adam(lr=0.01, beta1=0.9, beta2=0.999)
```

For constrained problems, `LagrangianOptimizer` performs dual ascent on multipliers.

### 10.6 Parameter Access

Every neural component exposes its parameters as a flat list:

```python
comp = robot.N["policy"]
flat = comp.theta                       # list[float]
comp.set_theta([0.0] * len(flat))       # reset
comp.n_parameters()                     # count
```

The whole object aggregates parameters recursively:

```python
robot.parameters()                      # all params in the hierarchy
```

### 10.7 Custom Neural Components

Subclass `NeuralComponent` for anything non-MLP:

```python
from gismol.neural.components import NeuralComponent

class ThresholdUnit(NeuralComponent):
    def __init__(self, name="threshold", threshold=0.5):
        super().__init__(name, input_dim=1, output_dim=1)
        self.threshold = threshold

    def __call__(self, x):
        return [1.0 if x[0] > self.threshold else 0.0]

    @property
    def theta(self):
        return [self.threshold]

    def set_theta(self, theta):
        self.threshold = theta[0]
        return self
```

### 10.8 Best Practices

- **Seed everything.** Neural components accept a `seed` argument. Use it in tests.
- **Keep networks small.** Neural components live inside hierarchies; complexity multiplies.
- **Train offline, deploy frozen.** For deterministic systems, freeze `theta` after training and record it in `metadata`.
- **Never let N override I.** Neural outputs must respect identity constraints; if they don't, use daemons or constrained RL.

---

## 11. Embeddings (Component E)

### 11.1 What Embeddings Are For

An embedding is a function `E : O → H` into a Hilbert space. It gives every `COHObject` a **vector representation** that supports similarity, clustering, and downstream learning.

```python
vec = coh.embed()                        # tuple[float, ...]
sim = coh1.similarity(coh2)              # in [-1, 1]
dist = coh1.E.space.distance(coh1.embed(), coh2.embed())
```

### 11.2 Built-In Embedders

| Embedder | Purpose |
|---|---|
| `HashEmbedder` | Deterministic structural hashing (default) |
| `FeatureEmbedder` | Selected attributes with optional normalisation |
| `CompositeEmbedder` | Combines own + children embeddings |
| `NeuralEmbedder` | Learned MLP-based projection |
| `AutoencoderEmbedder` | Encoder/decoder pair |
| `SiameseEmbedder` | Contrastive pairs |
| `ConstantEmbedder` | Trivial zero vector |

### 11.3 Choosing an Embedder

- **For structural similarity** (does this object look like that one?) → `HashEmbedder` or `CompositeEmbedder`.
- **For semantic similarity** (do these attributes mean similar things?) → `FeatureEmbedder` with normalisation.
- **For learned task-specific similarity** → `NeuralEmbedder` trained on your data.

### 11.4 FeatureEmbedder with Ranges

The most common choice:

```python
from gismol.core.embedding import FeatureEmbedder

robot.set_embedder(FeatureEmbedder(
    features=["x", "y", "heading", "payload_kg"],
    ranges={
        "x": (-100, 100),
        "y": (-100, 100),
        "heading": (-3.14, 3.14),
        "payload_kg": (0, 25),
    },
))
```

Missing attributes fall back to `default` (0.0 by default).

### 11.5 Composite Embeddings

For hierarchies, use `CompositeEmbedder` to concatenate parent features with child embeddings:

```python
from gismol.core.embedding import CompositeEmbedder, FeatureEmbedder

fleet.set_embedder(CompositeEmbedder(
    self_embedder=FeatureEmbedder(["total_throughput", "fleet_health"]),
    child_weight=0.5,
    mode="concat",   # or "sum"
))
```

### 11.6 Similarity in Practice

```python
index = SimilarityIndex()
for coh in systems:
    index.add(coh.name, coh.embed(), payload=coh)

similar = index.query(target.embed(), k=5)
for name, score, payload in similar:
    print(f"{name}: {score:.3f}")
```

---

## 12. Hybrid Dynamics — Simulation and Control

### 12.1 The Hybrid Model

A COH system's dynamics are **hybrid**: continuous evolution interleaved with discrete jumps. This is the natural semantics for real systems — a drone flies continuously but lands discretely; a market clears in ticks but drifts between them.

GISMOL represents this as:

- **Continuous flow** — `ds/dt = f(s, a, θ)` (an ODE)
- **Discrete jumps** — `s_{t+1} = g(s_t, a_t, θ)` (method execution)
- **Switching conditions** — triggers decide when jumps occur

### 12.2 Attaching a Flow

```python
from gismol.core.dynamics import ContinuousFlow

reactor.flow = ContinuousFlow(
    f=lambda s, a, th: {
        "temperature": -0.01 * (s["temperature"] - 20.0) + 0.5 * s["power"],
        "pressure": 0.05 * s["temperature"],
    },
    variables=["temperature", "pressure"],
)
```

The flow is integrated with classical **Runge-Kutta 4** for stability and accuracy.

### 12.3 Attaching a Jump

```python
from gismol.core.dynamics import DiscreteTransition

reactor.jump = DiscreteTransition(
    g=lambda s, a, th: {**s, "temperature": 0.9 * s["temperature"]},
)
```

### 12.4 Stepping and Simulating

```python
# One step
state, info = reactor.step(action=None, dt=0.05, event="tick")
print(info)
# {'events': ['overheat'], 'corrections': ['project'], 'admissible': True}

# Full simulation
traj = reactor.simulate(steps=200, dt=0.01, policy=lambda s, t: "regulate")
```

### 12.5 Trajectory Object

The result of `simulate` is a `Trajectory`:

```python
traj.times           # [0.0, 0.01, 0.02, ...]
traj.states          # [State, State, ...]
traj.actions
traj.rewards
traj.events
traj.final           # last State
traj.cumulative_reward()
traj.discounted_return(gamma=0.99)
traj.to_dict()       # serialisable
```

### 12.6 Well-Posedness

Existence and uniqueness of trajectories require Lipschitz continuity:

```python
system = reactor.to_hybrid_system()
report = system.lipschitz_report(reactor.state)
# {'well_posed': True, 'flow_lipschitz': 0.62, 'jump_lipschitz': None}
```

If `flow_lipschitz` is extremely large, the flow is stiff — reduce `dt` or switch integrator.

### 12.7 Policies

A policy is any callable `(state, t) -> action`:

```python
def bang_bang(state, t):
    return "heat" if state["temperature"] < state["target"] else "cool"

traj = reactor.simulate(steps=100, policy=bang_bang)
```

For neural policies:

```python
policy = reactor.N["controller"]
def neural_policy(state, t):
    return policy(state.to_dict() if hasattr(state, "to_dict") else state)
```

### 12.8 Discrete Events

Pass event names to simulate trigger-based behaviour:

```python
traj = reactor.simulate(
    steps=100,
    events=["tick", None, "tick", "fault", None],  # per-step event
)
```

Any trigger whose `event` matches (or is `"*"`) will fire if its condition holds.

---

## 13. Categorical Composition — Building Complex from Simple

### 13.1 The Three Compositions

The **COH category is cartesian closed**, meaning it supports:

1. **Products** `O₁ × O₂` — parallel composition (both run)
2. **Coproducts** `O₁ + O₂` — alternative composition (either runs)
3. **Exponentials** `O₂^O₁` — function objects (higher-order systems)

```python
arm = build_arm()
camera = build_camera()

# Both must work together
robot_head = arm.product(camera)

# Either one or the other runs (switching)
either = arm.coproduct(camera)

# A system that takes an arm and produces a camera
transformer = camera.exponential(arm)
```

### 13.2 Semantics of the Product

The product:

- Merges attributes, methods, neural components of both operands
- Conjoins identity constraints (`I₁ ∧ I₂`)
- Concatenates triggers and goals
- Attaches both as named children

It is the categorical construction that makes parallel composition well-defined.

### 13.3 Semantics of the Coproduct

The coproduct:

- Attaches both operands as alternative children
- Adds an exclusive-choice invariant
- Provides a clean way to model mode switching (offline/online, day/night, standby/active)

### 13.4 Semantics of the Exponential

The exponential is a *function object*: it holds an "argument" and a "result" and exposes an `apply` method. You can use it to model:

- Meta-controllers (a controller that takes a controller)
- Learnt transformations (a system that produces systems)
- Higher-order policies

### 13.5 The COHCategory

For explicit categorical programming:

```python
from gismol.categorical import COHCategory

cat = COHCategory()
cat.register(arm)
cat.register(camera)

prod = cat.product(arm, camera)
coprod = cat.coproduct(arm, camera)
exp = cat.exponential(arm, camera)

cat.is_cartesian_closed()   # True
```

### 13.6 Morphisms

A `COHMorphism` is a **structure-preserving map** between COH objects. It respects:

- Component DAG
- Attribute names
- Method names
- Neural component names
- Constraint sets
- Daemon names

```python
from gismol.categorical import COHMorphism

mor = COHMorphism(source=small_robot, target=big_robot, name="upgrade")
mor.preserves()   # True if the map preserves all structure
mor.violations    # list of problems (empty if preserves() is True)
```

Use morphisms to:

- Verify that a refactoring preserves semantics
- Build abstractions and refinements
- Share structure between similar systems

### 13.7 Functors

```python
from gismol.categorical import ComponentFunctor

F = ComponentFunctor()
dag = F.map_object(robot)
dag.nodes       # ['robot', 'robot/nav', ...]
dag.edges       # [('robot', 'robot/nav'), ...]
```

The component functor formalises "the structure of a COH is itself a mathematical object" (Theorem 3.2).

---

## 14. Stability Analysis

### 14.1 Why Stability Matters

A system that satisfies its constraints at time `t` may violate them at time `t+Δ`. **Stability analysis** proves the system remains admissible indefinitely.

### 14.2 Lyapunov Functions

A Lyapunov function `V(s)` is a non-negative scalar that decreases along trajectories. If such a function exists inside the admissible region, the system is stable.

GISMOL builds them automatically:

```python
from gismol.stability import lyapunov_from_constraints, goal_lyapunov

V_constraints = lyapunov_from_constraints(reactor)   # sum of violations
V_goal = goal_lyapunov(reactor)                       # negative goal value
```

### 14.3 The Certificate

```python
from gismol.stability import ConstraintStabilityAnalyzer

analyzer = ConstraintStabilityAnalyzer(reactor)
cert = analyzer.analyze(states=sample_states)
print(cert)
# LyapunovCertificate(asymptotically stable, rate=0.0413, daemons=True)

print(cert.to_dict())
# {'stable': True, 'asymptotic': True, 'rate': 0.0413,
#  'admissible_fraction': 1.0, 'samples': 100,
#  'worst_derivative': -0.0413, 'daemons_enforcing': True, 'notes': []}
```

### 14.4 Sampling States

Stability certificates require sample states spanning the operating regime:

```python
import random
states = []
for _ in range(200):
    states.append({
        "temperature": random.uniform(20.0, 90.0),
        "pressure": random.uniform(0.5, 5.0),
    })

cert = analyzer.analyze(states=states)
```

Use the initial state plus perturbed versions of it for a quick sanity check.

### 14.5 Interpreting Failures

If `stable` is `False`:

1. Look at `worst_derivative` — it is positive.
2. Inspect which states contributed (sample them individually).
3. Common causes:
   - The goal sign is wrong (`direction="min"` on something you want to grow).
   - A flow pushes state out of the admissible region.
   - Constraints are mutually inconsistent.

Fixing usually means: adjust the flow, add a restoring daemon, or relax a constraint.

---

## 15. Free-Energy Correspondence

### 15.1 The Free-Energy Principle in One Paragraph

The **Free-Energy Principle (FEP)** says biological systems maintain order by minimising variational free energy — an information-theoretic quantity that upper-bounds surprise. GISMOL includes an explicit implementation to let you build systems that are *provably* interpretable in FEP terms.

### 15.2 Generative Models

```python
from gismol.dynamical.free_energy import GenerativeModel, VariationalPosterior, FreeEnergy

model = GenerativeModel(
    prior={"cold": 0.4, "warm": 0.6},
    likelihood={
        "cold": {"sensor_low": 0.9, "sensor_high": 0.1},
        "warm": {"sensor_low": 0.2, "sensor_high": 0.8},
    },
)

q = VariationalPosterior({"cold": 0.3, "warm": 0.7})
fe = FreeEnergy(model)
print(fe.variational("sensor_low", q))   # variational free energy
```

### 15.3 The COH ↔ FEP Correspondence

Every COH object can be mapped to a generative model (Theorem 8.1):

```python
from gismol.dynamical.free_energy import coh_free_energy_correspondence

mapping = coh_free_energy_correspondence(reactor)
# {'identity_constraints': [...], 'identity_role': 'structural priors p(s)',
#  'trigger_constraints': [...], 'trigger_role': 'conditional dependencies',
#  'goal_constraints': [...], 'goal_role': 'expected log-likelihood terms',
#  'daemons': [...], 'daemon_role': 'precision-weighted prediction-error minimisation'}
```

This lets you build hybrid systems that are simultaneously interpretable via:

- **Symbolic constraints** — logic and rules
- **Neural learning** — flexible function approximation
- **Information theory** — surprise minimisation

### 15.4 A Scalar Free-Energy Proxy

`free_energy_from_coh` gives a scalar you can log during simulation:

```python
from gismol.dynamical.free_energy import free_energy_from_coh

fe = free_energy_from_coh(reactor, reactor.state)
print(f"free energy: {fe:.4f}")
```

Track this alongside rewards to see the trade-off between task performance and constraint adherence.

---

## 16. The Natural-Language Front-End

### 16.1 What It Does

The NLP pipeline compiles English descriptions into runnable `COHObject`s:

```python
from gismol.nlp import text_to_coh

reactor = text_to_coh(
    "Define a thermal plant called reactor. "
    "The core temperature must be less than 900. "
    "The coolant flow must be at least 5. "
    "If core temperature exceeds 800 then increase coolant. "
    "Maximize energy output. Minimize coolant usage.",
    name="reactor",
)
```

This produces a `COHObject` with:

- Attributes for extracted identifiers and numeric values
- Children for "is-a" / "has" relations
- Methods for "controls" / "produces" relations
- Identity constraints from "must be less than" clauses
- Triggers from "if X then Y" clauses
- Goals from "maximize" / "minimize" clauses
- An `MLPComponent` predictor and default daemons

### 16.2 Pipeline Stages

| Stage | Module | What it does |
|---|---|---|
| Normalise | `normalizers.py` | Unicode + whitespace + case |
| Parse | `parsers.py` | Tokens, sentences, POS |
| Extract entities | `entity_extractor.py` | Numbers, dates, identifiers, constraint keywords |
| Mine relations | `relations_miner.py` | `is_a`, `has`, `requires`, `controls`, `causes`, … |
| Recognise intent | `intent_recognizer.py` | `define`, `constrain`, `optimize`, `simulate`, … |
| Compile constraints | `constraint_parser.py` | Text → I/T/G objects |
| Embed | `embedders.py` | Bag-of-words, TF-IDF, hashing, char-ngram |
| Validate | `validation.py` | Structural and semantic checks |
| Generate response | `response_generator.py` | Templates |

### 16.3 Explicit Constraint Syntax

For precise control, use the `constraint:` prefix:

```python
text = """
constraint: temperature > 0 and temperature < 900
constraint: pressure <= 10
if pressure > 8 then vent
maximize energy_output
minimize coolant_flow
"""
coh = text_to_coh(text, name="reactor")
```

### 16.4 The Chat Pipeline

For interactive systems:

```python
from gismol.nlp import ChatModule

chat = ChatModule()
print(chat.process("define a system called pump").text)
print(chat.process("the pressure must be less than 200").text)
print(chat.process("describe it").text)
print(chat.process("simulate 50 steps").text)
```

Every response is a `ChatResponse` with structured fields:

```python
r = chat.process("maximize throughput")
r.intent.name        # 'optimize'
r.entities           # list of Entity
r.relations          # list of Relation
r.constraints        # ParsedConstraints
r.coh                # the current COHObject
```

### 16.5 Multi-Session Management

```python
from gismol.nlp import ChatManager

manager = ChatManager()
session = manager.new_session("user_42")
session.say("define a robot called R2")
session.say("the payload must be under 20kg")

print(session.transcript())   # full dialogue
manager.close("user_42")
```

### 16.6 Customising the Pipeline

Every stage is replaceable:

```python
from gismol.nlp import (
    TextToCOH, EntityExtractor, RelationMiner, ConstraintParser,
)

extractor = EntityExtractor()
extractor.add_pattern("SENSOR_ID", r"sensor_\d+")

builder = TextToCOH(
    extractor=extractor,
    miner=RelationMiner(),
    parser=ConstraintParser(),
    attach_neural=False,
    attach_daemons=True,
)
```

### 16.7 Caveats

- The NL pipeline is **rule-based**, not ML-based. It handles clean declarative English well and complex prose poorly.
- Numbers are parsed greedily. Use explicit `constraint:` syntax for anything numeric and important.
- Relations are heuristic. Always inspect `metadata["relations"]` after building.

For critical systems, build programmatically and use NLP only for the outer, flexible parts.

---

## 17. Reasoners

### 17.1 The Reasoning Menu

| Reasoner | Question it answers |
|---|---|
| `ForwardChainer` | "What facts follow from the current state?" |
| `ConstraintReasoner` | "Which invariants are violated? How to repair?" |
| `GoalSolver` | "What method sequence maximises the goals?" |
| `AbductiveReasoner` | "What would have to be true for the constraints to hold?" |
| `CausalReasoner` | "What are the causes and effects of X?" |
| `AnalogicalReasoner` | "Which of these two systems is more like that one?" |
| `CategoricalReasoner` | "What is the categorical structure of this system?" |
| `DomainReasoner` | "How does this system behave under domain rules?" |
| `MetaReasoner` | "Which reasoner should I use?" |

### 17.2 Querying

```python
from gismol.reasoners import Query, GeneralReasoner

result = GeneralReasoner().infer(reactor, Query("maximize energy output"))
print(result.answer)
print(result.confidence)
print(result.trace)
```

`GeneralReasoner` classifies the query and dispatches to the right specialist.

### 17.3 Constraint Repair

```python
from gismol.reasoners import ConstraintReasoner

reactor.set("core_temperature", 1500.0)   # violate the invariant
result = ConstraintReasoner().infer(reactor, Query("violations"))
print(result.answer)
# {'violations': ['inv_core_temperature_10'],
#  'repairs': [{'core_temperature': 900.0}]}
```

The reasoner searches a small grid of candidate values and returns the smallest change that restores admissibility.

### 17.4 Planning

```python
from gismol.reasoners import GoalSolver

result = GoalSolver(depth=4, beam=6).infer(reactor, Query("best plan"))
print(result.answer["plan"])       # e.g., ['increase_coolant', 'vent', 'regulate']
print(result.answer["value"])      # total goal score
```

The planner does a bounded beam search through methods, discarding any state that violates identity constraints.

### 17.5 Causal Reasoning

```python
from gismol.reasoners import CausalReasoner

reasoner = CausalReasoner()
result = reasoner.infer(
    reactor,
    Query("coolant_flow", parameters={"intervene": True}),
)
print(result.answer["effects"])   # what changes when coolant_flow changes
```

### 17.6 Domain Reasoning

Built-in domain profiles cover physics, biology, finance, smart cities, and aerospace:

```python
from gismol.reasoners import DomainReasoner, get_domain

reasoner = DomainReasoner(profile=get_domain("finance"))
result = reasoner.infer(trading_desk, Query("analyse"))
print(result.answer["domain"])           # 'finance'
print(result.answer["metrics"])          # any financial metrics in the state
print(result.answer["invariants"])       # no-arbitrage, capital adequacy, ...
```

Register your own:

```python
from gismol.reasoners import DomainProfile, register_domain
from gismol.reasoners.base import Rule

register_domain(DomainProfile(
    name="agriculture",
    vocabulary=["soil", "yield", "irrigation", "nitrogen", "harvest"],
    rules=[
        Rule("yield_estimate", lambda f: "yield" in f, {"yield_kg": None}, 0.85),
        Rule("nitrogen_check", lambda f: f.get("nitrogen", 100) < 40,
             {"fertilizer_needed": True}, 0.9),
    ],
    invariants=["soil nitrogen ≥ 20", "irrigation ≤ quota"],
    metrics=["yield", "water_use"],
))
```

### 17.7 Composing Reasoners

Use `MetaReasoner` to automatically select and combine:

```python
from gismol.reasoners import MetaReasoner

meta = MetaReasoner()
result = meta.infer(reactor, Query("check violations and plan a fix"))
print(result.answer)   # dict of answers from multiple reasoners
```

---

## 18. Validation and Soundness

### 18.1 Two Kinds of Correctness

- **Soundness** — the object is internally consistent (no cycles, satisfiable constraints, well-typed components)
- **Validation** — the object matches your intent

GISMOL checks soundness automatically and gives you tools for validation.

### 18.2 The Soundness Report

```python
report = reactor.soundness_report()
print(report)
# {'hierarchical_sound': True,
#  'constraints_satisfied_at_sample': True,
#  'violations': [],
#  'triggers_breaking_invariants': [],
#  'sound': True}
```

This corresponds directly to Theorems 6.1 and 6.2: soundness requires an acyclic hierarchy and invariant-preserving triggers.

### 18.3 The Validator

```python
from gismol.nlp.validation import COHValidator

validator = COHValidator(require_all_components=False, strict_goals=True)
issues = validator.validate(reactor)

for issue in issues:
    print(issue)
    # [WARNING] reactor: goal 'comfort' references unknown attributes ['x']
```

Severities are `"error"`, `"warning"`, `"info"`. Errors block deployment; warnings flag suspicious patterns.

### 18.4 Full Component Check

```python
report = reactor.component_report()
# {'C': 3, 'A': 12, 'M': 5, 'N': 4, 'E': 'FeatureEmbedder',
#  'I': 6, 'T': 3, 'G': 2, 'D': 3}

reactor.is_minimal()   # True if all nine components are populated
```

For research papers, add `require_all_components=True`. For production, this is optional — many useful systems omit some components.

### 18.5 Raise on Invalid

```python
validator.raise_if_invalid(reactor)   # raises ValidationError on any error
```

Use this in CI to catch structural regressions.

---

## 19. Persistence and Repositories

### 19.1 Serialising a Single Object

```python
data = reactor.to_dict()
# {'schema': 1, 'uid': '...', 'name': 'reactor', 'description': '...',
#  'attributes': [...], 'methods': [...], 'neural': {...},
#  'embedder': 'FeatureEmbedder', 'constraints': {...}, 'daemons': [...],
#  'children': [...]}

restored = COHObject.from_dict(data)
```

Note: **callables and neural weights are not serialised** by default. If you need to persist them:

1. Save network weights separately using `comp.theta`.
2. Save method source code with `inspect.getsource`.
3. Save structural metadata (constraints, attributes) via `to_dict`.

### 19.2 The Repository

For collections of systems:

```python
from gismol.core.repository import COHRepository

repo = COHRepository("production")
oid = repo.add(reactor, tags=["thermal", "critical"])
repo.add(other_reactor, tags=["thermal"])

# Query
repo.find(tag="critical")             # list of COHObjects
repo.find(name="reactor")
repo.get(oid)                          # by id
repo.tags_of(oid)                      # {'thermal', 'critical'}

# Hierarchy
repo.index_hierarchy(fleet)            # register parent + all descendants

# Persistence
repo.save("production.json")
loaded = COHRepository.load("production.json")
```

### 19.3 Storing Trained Weights

```python
import json

weights = {
    name: comp.theta
    for name, comp in reactor.N.items()
    if comp.n_parameters() > 0
}
with open("weights.json", "w") as fh:
    json.dump(weights, fh)
```

Reload:

```python
with open("weights.json") as fh:
    weights = json.load(fh)

for name, theta in weights.items():
    reactor.N[name].set_theta(theta)
```

### 19.4 Best Practices

- **Save after every structural change.** The repository is cheap.
- **Tag systems by domain, criticality, and owner.**
- **Version your weights.** Store the git SHA alongside them.
- **Never trust a reloaded object without calling `soundness_report()`.**

---

## 20. Design Patterns and Recipes

### 20.1 The Standard Builder Pattern

Every system should have one builder function:

```python
def build_system() -> COHObject:
    sys = COHObject("system")
    _add_attributes(sys)
    _add_methods(sys)
    _add_constraints(sys)
    _add_daemons(sys)
    _add_neural(sys)
    return sys
```

Sub-functions make it easy to test pieces in isolation.

### 20.2 Sensor-Actuator Split

Model perception and actuation as separate child objects:

```python
system.add_child(build_sensors())     # A-heavy, no M
system.add_child(build_actuators())   # M-heavy, no A
system.add_child(build_controller())  # N-heavy, bridges them
```

The controller's neural components read from sensors and write to actuators via methods.

### 20.3 Safety Envelope

Encode safety as a triple of (identity constraint, safety daemon, barrier):

```python
def add_safety_envelope(coh, var, lo, hi):
    coh.add_identity_constraint(IdentityConstraint(
        f"{var}_bounds", lambda s: lo <= s[var] <= hi))
    coh.add_daemon(BoundaryDaemon({var: (lo, hi)}))
    coh.add_daemon(BarrierDaemon(
        barrier=lambda s: min(s[var] - lo, hi - s[var]),
        alpha=lambda h: 0.5 * h,
    ))
```

This gives you *both* passive (projection) and *active* (barrier) safety.

### 20.4 Two-Level Control

For systems with fast and slow loops:

```python
fast = COHObject("fast_loop")     # high-frequency regulator
slow = COHObject("slow_loop")     # low-frequency planner
system = slow.product(fast)
```

Give the fast loop a small `dt` and the slow loop triggers with long horizons.

### 20.5 Event-Driven Architecture

Use triggers with distinct events to model a state machine:

```python
for transition in [("idle", "start", "running"), ...]:
    src, evt, dst = transition
    coh.add_trigger_constraint(TriggerConstraint(
        f"{src}_to_{dst}", evt,
        condition=lambda s, src=src: s["mode"] == src,
        action={"mode": dst},
    ))
```

This gives you an explicit, inspectable state machine.

### 20.6 Learned Policy with Safety Net

Combine a neural policy with a barrier daemon:

```python
coh.add_neural(PolicyComponent("policy", input_dim=6, n_actions=4))
coh.add_daemon(BarrierDaemon(
    barrier=lambda s: 1.0 - (s["x"] ** 2 + s["y"] ** 2) / 100.0,
    intervention="safe_stop",
))
```

The policy runs when safe; the daemon intervenes when not.

### 20.7 Curriculum via Trigger Priority

Model learning curricula by escalating trigger priorities:

```python
coh.add_trigger_constraint(TriggerConstraint(
    "beginner_hint", "learner_stuck",
    condition=lambda s: s["attempts"] > 5 and s["mastery"] < 0.3,
    action="offer_hint", priority=1,
))
coh.add_trigger_constraint(TriggerConstraint(
    "advanced_challenge", "learner_bored",
    condition=lambda s: s["mastery"] > 0.95,
    action="escalate", priority=2,
))
```

### 20.8 Config-Driven System Construction

Read parameters from JSON:

```python
def build_from_config(cfg: dict) -> COHObject:
    coh = COHObject(cfg["name"])
    for a in cfg["attributes"]:
        coh.add_attribute(Attribute(**a))
    for i in cfg["invariants"]:
        coh.add_identity_constraint(IdentityConstraint(i["name"], i["expression"]))
    return coh
```

This lets non-programmers configure systems without touching code.

---

## 21. Anti-Patterns to Avoid

### 21.1 God Objects

**Bad:** one `COHObject` with 200 attributes and 50 methods.

**Good:** nested objects with focused responsibilities. If an object has more than ~15 attributes or ~10 methods, split it.

### 21.2 Constraints as Assertions

**Bad:** checking constraints inside methods with `assert`.

**Good:** declaring them as `IdentityConstraint`s so daemons, reasoners, and validators can act on them.

### 21.3 Mutable State

**Bad:** methods that mutate `state` in place.

**Good:** methods that return a new state. Mutation makes trajectories unpredictable and breaks daemon reasoning.

### 21.4 Magic Values

**Bad:** `if s["temp"] > 87.3:` scattered through code.

**Good:** declare an `Attribute("max_temp", 87.3)` and reference it.

### 21.5 Overly Deep Hierarchies

**Bad:** 8-level nesting for a simple controller.

**Good:** 3–5 levels. Depth should reflect conceptual decomposition, not implementation convenience.

### 21.6 Neural Components Without Constraints

**Bad:** a policy that can output any action.

**Good:** a policy whose outputs are gated by identity constraints and daemon supervision. Neural learning is powerful but not trustworthy.

### 21.7 Skipping Validation

**Bad:** deploying without calling `soundness_report()`.

**Good:** run `soundness_report` and `COHValidator` in CI on every commit.

### 21.8 Over-Using Triggers

**Bad:** every state change is a trigger.

**Good:** triggers for *events*, goals for *preferences*, methods for *actions*. If you have more than ~15 triggers, look for abstractions.

### 21.9 Confusing Goals and Constraints

**Bad:** using `GoalConstraint` for safety.

**Good:** safety is an `IdentityConstraint` with a daemon; performance is a `GoalConstraint`.

### 21.10 Ignoring Embeddings

**Bad:** never setting an embedder.

**Good:** set a `FeatureEmbedder` or `CompositeEmbedder` — it enables similarity, clustering, and cross-object reasoning.

---

## 22. Performance and Scaling

### 22.1 Where Time Goes

For a typical system, per-step costs are:

| Phase | Typical share |
|---|---|
| Constraints | 30–50% |
| Daemons | 20–40% |
| Neural forward | 10–30% |
| Flow integration | 5–15% |
| Bookkeeping | 5% |

Measure before optimising. `time.perf_counter` around `step()` is enough.

### 22.2 Constraint Caching

If a constraint depends only on a few attributes, cache its evaluation per tick:

```python
cached = {}
def get_constraint_value(state, key):
    if key not in cached or cached[key + "_state"] is not state:
        cached[key] = constraint.satisfied(state)
        cached[key + "_state"] = state
    return cached[key]
```

### 22.3 Daemon Throttling

Not every daemon needs to run every tick. Gate them:

```python
class ThrottledDaemon(Daemon):
    def __init__(self, inner, period=10):
        super().__init__(f"throttled_{inner.name}")
        self.inner = inner
        self.period = period
        self.tick = 0

    def observe(self, state, t=0.0):
        self.tick += 1
        if self.tick % self.period != 0:
            return None
        return self.inner.observe(state, t)
```

### 22.4 Hierarchical Decomposition

Split a big object into children. Per-component simulation is O(n) not O(n²) if you use the product construction wisely. The theory guarantees polynomial scaling with bounded interaction degree (Theorem 9.1).

### 22.5 Neural Component Sizing

Keep MLPs small. A controller with 3 inputs and 4 actions needs at most one hidden layer of 32 units. Deep networks make sense only when the perception problem is genuinely hard.

### 22.6 Parallel Simulation

For independent sub-systems, run them in parallel processes:

```python
from concurrent.futures import ProcessPoolExecutor

def run_one(seed):
    system = build_system()
    return system.simulate(steps=100).cumulative_reward()

with ProcessPoolExecutor() as ex:
    results = list(ex.map(run_one, range(100)))
```

### 22.7 Numpy Acceleration

If numpy is installed, `MLP.fit` automatically uses vectorised gradients:

```bash
pip install -e ".[fast]"
```

10–50× speedup on large networks with no code changes.

### 22.8 Avoiding Redundant Embeddings

`embed()` is expensive for large hierarchies. Compute it once and cache:

```python
@functools.lru_cache(maxsize=1024)
def embed_cached(uid):
    return coh.embed()
```

Invalidate the cache when state changes materially.

---

## 23. Debugging

### 23.1 The Describe Method

`coh.describe()` prints a hierarchical overview:

```python
print(reactor.describe())
# reactor  {'C': 3, 'A': 12, 'M': 5, 'N': 4, 'E': 'FeatureEmbedder', ...}
#   sensors  {'C': 0, 'A': 6, 'M': 1, ...}
#   actuators  {'C': 0, 'A': 3, 'M': 4, ...}
```

### 23.2 The Soundness Report

```python
print(reactor.soundness_report())
# {'hierarchical_sound': True, 'constraints_satisfied_at_sample': True, ...}
```

### 23.3 The Component Report

```python
print(reactor.component_report())
# Check that all nine are populated appropriately
```

### 23.4 Simulating with Traces

```python
traj = reactor.simulate(steps=10, dt=0.1)
for t, state in traj:
    print(f"t={t:.2f}  temp={state['temperature']:.2f}  admissible={reactor.satisfies_constraints(state)}")
```

### 23.5 Per-Step Info

`step()` returns a dict with events, corrections, and admissibility:

```python
state, info = reactor.step(dt=0.1)
print(info)
# {'events': ['overheat'], 'corrections': ['project'], 'admissible': True}
```

Log this to see which triggers and daemons fire.

### 23.6 Daemon History

Every daemon keeps its history:

```python
for d in reactor.D:
    print(f"{d.name}: {len(d.history)} actions")
    for a in d.history[-3:]:
        print(f"  {a.name}  severity={a.severity:.3f}  {a.message}")
```

### 23.7 Constraint Diagnostics

```python
for c in reactor.I:
    if not c.satisfied(reactor.state):
        print(f"VIOLATED: {c.name}  margin={c.violation(reactor.state):.3f}")
```

### 23.8 Reasoner Traces

Every reasoning result carries a trace:

```python
result = GeneralReasoner().infer(reactor, Query("maximize energy"))
for step in result.trace:
    print(" ", step)
```

### 23.9 The `pdb` Route

GISMOL code is ordinary Python. Set a breakpoint inside a method:

```python
import pdb

def inspect_method(state, action=None):
    pdb.set_trace()
    return state, 0.0
```

### 23.10 Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| `CyclicHierarchyError` | `a.add_child(b)` and `b.add_child(a)` | Restructure hierarchy |
| `ConstraintViolationError` | `check_identity` on inadmissible state | Fix the state or the constraint |
| `MethodExecutionError` | Precondition failed | Check preconditions |
| `UnsatisfiableConstraintError` | Contradictory constraints | Reconcile or relax |
| `ParseError` | Invalid expression string | Check whitelist |
| `MorphismError` | Structure dropped in map | Review component_map |
| `ValidationError` | Validator found errors | Fix issues before deploying |

---

## 24. Case Study — Building a Full System End to End

Let's build a **drone inspection system** with four hierarchy levels, all nine components, and full validation.

### 24.1 Requirements

- A drone flies a patrol route over a solar farm
- It must not fly in high wind or low battery
- It must detect thermal anomalies on panels
- It must return to base when anomalous or out of bounds

### 24.2 Step 1 — Hierarchy

```
drone_system                      (level 0)
├── flight_controller             (level 1)
│   ├── attitude_control          (level 2)
│   │   ├── roll_ctrl             (level 3)
│   │   ├── pitch_ctrl            (level 3)
│   │   └── yaw_ctrl              (level 3)
│   └── navigation                (level 2)
│       └── gps                   (level 3)
├── power                         (level 1)
│   └── battery                   (level 2)
│       └── cell_monitor          (level 3)
└── payload                       (level 1)
    ├── thermal_camera            (level 2)
    │   └── image_processor       (level 3)
    └── anomaly_reporter          (level 2)
```

### 24.3 Step 2 — Implementation

```python
import math
from gismol import COHObject, Attribute, Method
from gismol.core.constraints import (
    IdentityConstraint, TriggerConstraint, GoalConstraint,
)
from gismol.core.daemons import BoundaryDaemon, InvariantDaemon, SafetyDaemon
from gismol.core.daemon_impl import BarrierDaemon
from gismol.core.embedding import FeatureEmbedder
from gismol.neural.components import MLPComponent, PolicyComponent
from gismol.nlp.validation import COHValidator


def build_drone_system() -> COHObject:
    # ---- level 3 : micro-components ----
    def axis_ctrl(axis):
        c = COHObject(f"{axis}_ctrl")
        c.add_attribute(Attribute(f"angle_{axis}", 0.0, bounds=(-1.5, 1.5)))
        c.add_attribute(Attribute(f"rate_{axis}", 0.0, bounds=(-3.0, 3.0)))
        c.add_method(Method(
            "correct",
            lambda s, a=None: ({**s, f"rate_{axis}": s[f"rate_{axis}"] * 0.9}, 0.0)))
        c.add_identity_constraint(IdentityConstraint(
            f"{axis}_angle_ok", lambda s, ax=axis: abs(s[f"angle_{ax}"]) < 1.4))
        attach_standard_daemons(c)
        return c

    gps = COHObject("gps")
    gps.add_attribute(Attribute("lat", 0.0))
    gps.add_attribute(Attribute("lon", 0.0))
    gps.add_attribute(Attribute("accuracy_m", 1.0, bounds=(0.0, 10.0)))
    gps.add_method(Method("fix", lambda s, a=None: ({**s, "accuracy_m": max(0.5, s["accuracy_m"] * 0.95)}, 0.0)))
    gps.add_identity_constraint(IdentityConstraint("gps_ok", lambda s: s["accuracy_m"] < 5.0))
    attach_standard_daemons(gps)

    cell_monitor = COHObject("cell_monitor")
    cell_monitor.add_attribute(Attribute("min_cell_v", 4.0, bounds=(3.0, 4.2)))
    cell_monitor.add_attribute(Attribute("max_cell_temp", 30.0, bounds=(-10.0, 70.0)))
    cell_monitor.add_method(Method("scan", lambda s, a=None: (s, 0.0)))
    cell_monitor.add_identity_constraint(IdentityConstraint("cell_temp_ok", lambda s: s["max_cell_temp"] < 60.0))
    attach_standard_daemons(cell_monitor)

    image_processor = COHObject("image_processor")
    image_processor.add_attribute(Attribute("frame_rate_hz", 30.0, bounds=(0.0, 60.0)))
    image_processor.add_attribute(Attribute("last_anomaly_score", 0.0, bounds=(0.0, 1.0)))
    image_processor.add_method(Method("process", lambda s, a=None: (s, 0.0)))
    image_processor.add_neural(MLPComponent("anomaly_cnn", [16, 32, 1], seed=1), name="anomaly_detector")
    attach_standard_daemons(image_processor)

    # ---- level 2 : subsystems ----
    attitude = COHObject("attitude_control")
    attitude.add_attribute(Attribute("stability", 1.0, bounds=(0.0, 1.0)))
    attitude.add_child(axis_ctrl("roll"))
    attitude.add_child(axis_ctrl("pitch"))
    attitude.add_child(axis_ctrl("yaw"))
    attitude.add_method(Method("stabilise", lambda s, a=None: ({**s, "stability": min(1.0, s["stability"] + 0.01)}, 0.0)))
    attitude.add_goal_constraint(GoalConstraint("max_stability", lambda s: s["stability"], direction="max"))
    attach_standard_daemons(attitude)

    navigation = COHObject("navigation")
    navigation.add_attribute(Attribute("waypoint_index", 0.0, bounds=(0.0, 100.0)))
    navigation.add_attribute(Attribute("distance_to_wp_m", 0.0, bounds=(0.0, 1000.0)))
    navigation.add_child(gps)
    navigation.add_method(Method("next_waypoint", lambda s, a=None: ({**s, "waypoint_index": s["waypoint_index"] + 1}, 0.5)))
    navigation.add_goal_constraint(GoalConstraint("advance", lambda s: s["waypoint_index"], direction="max"))
    attach_standard_daemons(navigation)

    battery = COHObject("battery")
    battery.add_attribute(Attribute("charge_pct", 100.0, bounds=(0.0, 100.0)))
    battery.add_attribute(Attribute("voltage", 22.0, bounds=(18.0, 25.2)))
    battery.add_child(cell_monitor)
    battery.add_method(Method("drain", lambda s, a=None: ({**s, "charge_pct": max(0.0, s["charge_pct"] - 0.05)}, 0.0)))
    battery.add_identity_constraint(IdentityConstraint("voltage_ok", lambda s: s["voltage"] >= 19.0))
    battery.add_trigger_constraint(TriggerConstraint(
        "low_battery_rtb", "charge_low",
        condition=lambda s: s["charge_pct"] < 20.0,
        action="return_to_base", priority=8))
    battery.add_goal_constraint(GoalConstraint("endurance", lambda s: s["charge_pct"], direction="max"))
    attach_standard_daemons(battery)

    thermal_camera = COHObject("thermal_camera")
    thermal_camera.add_attribute(Attribute("panel_temp_c", 25.0, bounds=(-20.0, 150.0)))
    thermal_camera.add_child(image_processor)
    thermal_camera.add_method(Method("capture", lambda s, a=None: (s, 0.0)))
    thermal_camera.add_trigger_constraint(TriggerConstraint(
        "panel_hot", "temp_high",
        condition=lambda s: s["panel_temp_c"] > 80.0,
        action="report_anomaly", priority=5))
    attach_standard_daemons(thermal_camera)

    anomaly_reporter = COHObject("anomaly_reporter")
    anomaly_reporter.add_attribute(Attribute("anomalies_found", 0.0, bounds=(0.0, 1000.0)))
    anomaly_reporter.add_method(Method("report", lambda s, a=None: ({**s, "anomalies_found": s["anomalies_found"] + 1}, 1.0)))
    anomaly_reporter.add_goal_constraint(GoalConstraint("find_anomalies", lambda s: s["anomalies_found"], direction="max"))
    attach_standard_daemons(anomaly_reporter)

    # ---- level 1 : major subsystems ----
    fc = COHObject("flight_controller")
    fc.add_attribute(Attribute("mode", "patrol"))
    fc.add_attribute(Attribute("airspeed_mps", 12.0, bounds=(0.0, 25.0)))
    fc.add_child(attitude)
    fc.add_child(navigation)
    fc.add_method(Method("fly", lambda s, a=None: ({**s, "airspeed_mps": max(8.0, s["airspeed_mps"] - 0.01)}, 0.0)))
    fc.add_identity_constraint(IdentityConstraint("airspeed_ok", lambda s: 8.0 <= s["airspeed_mps"] <= 22.0))
    fc.add_trigger_constraint(TriggerConstraint(
        "wind_slowdown", "wind_high",
        condition=lambda s: s["airspeed_mps"] > 20.0,
        action={"airspeed_mps": 15.0}, priority=6))
    fc.add_goal_constraint(GoalConstraint("patrol_speed", lambda s: s["airspeed_mps"], direction="max"))
    attach_standard_daemons(fc)

    power = COHObject("power")
    power.add_attribute(Attribute("total_current_a", 10.0, bounds=(0.0, 50.0)))
    power.add_child(battery)
    power.add_method(Method("regulate", lambda s, a=None: ({**s, "total_current_a": min(40.0, s["total_current_a"])}, 0.0)))
    power.add_identity_constraint(IdentityConstraint("current_ok", lambda s: s["total_current_a"] < 45.0))
    attach_standard_daemons(power)

    payload = COHObject("payload")
    payload.add_attribute(Attribute("mass_kg", 1.2, bounds=(0.0, 5.0)))
    payload.add_child(thermal_camera)
    payload.add_child(anomaly_reporter)
    payload.add_method(Method("scan_step", lambda s, a=None: (s, 0.0)))
    payload.add_goal_constraint(GoalConstraint("min_weight", lambda s: s["mass_kg"], direction="min"))
    attach_standard_daemons(payload)

    # ---- level 0 : system ----
    drone = COHObject("drone_system", description="Autonomous solar farm inspection UAV")
    drone.add_attribute(Attribute("mission_time_s", 0.0, bounds=(0.0, 3600.0)))
    drone.add_attribute(Attribute("coverage_pct", 0.0, bounds=(0.0, 100.0)))
    drone.add_attribute(Attribute("wind_speed_mps", 5.0, bounds=(0.0, 30.0)))
    drone.add_attribute(Attribute("mission_status", "active"))

    drone.add_child(fc, key="flight_controller")
    drone.add_child(power, key="power")
    drone.add_child(payload, key="payload")

    drone.add_method(Method(
        "mission_step",
        lambda s, a=None: ({**s,
                            "mission_time_s": s["mission_time_s"] + 1.0,
                            "coverage_pct": min(100.0, s["coverage_pct"] + 0.1)},
                           0.5)))

    drone.add_identity_constraint(IdentityConstraint(
        "mission_time_limit", lambda s: s["mission_time_s"] <= 3000.0))
    drone.add_identity_constraint(IdentityConstraint(
        "coverage_bounded", lambda s: 0.0 <= s["coverage_pct"] <= 100.0))

    drone.add_trigger_constraint(TriggerConstraint(
        "high_wind_rtb", "wind_alert",
        condition=lambda s: s["wind_speed_mps"] > 15.0,
        action={"mission_status": "returning"}, priority=9))
    drone.add_trigger_constraint(TriggerConstraint(
        "mission_complete", "coverage_done",
        condition=lambda s: s["coverage_pct"] >= 95.0,
        action={"mission_status": "complete"}, priority=3))

    drone.add_goal_constraint(GoalConstraint("max_coverage", lambda s: s["coverage_pct"], direction="max", weight=3.0))
    drone.add_goal_constraint(GoalConstraint("min_time", lambda s: -s["mission_time_s"], direction="max"))

    drone.add_daemon(InvariantDaemon(constraints=list(drone.I)))
    drone.add_daemon(BoundaryDaemon({
        "wind_speed_mps": (0.0, 25.0),
        "coverage_pct": (0.0, 100.0),
    }))
    drone.add_daemon(BarrierDaemon(
        barrier=lambda s: 25.0 - s["wind_speed_mps"],
        intervention={"mission_status": "returning"},
    ))

    drone.add_neural(PolicyComponent("mission_policy", input_dim=5, n_actions=4, seed=42),
                     name="mission_policy")
    drone.add_neural(MLPComponent("wind_predictor", [4, 16, 1], seed=43),
                     name="wind_predictor")

    drone.set_embedder(FeatureEmbedder(
        ["mission_time_s", "coverage_pct", "wind_speed_mps"],
        ranges={"mission_time_s": (0, 3600), "coverage_pct": (0, 100),
                "wind_speed_mps": (0, 30)},
    ))

    return drone


def attach_standard_daemons(coh: COHObject) -> COHObject:
    if coh.I:
        coh.add_daemon(InvariantDaemon(constraints=list(coh.I)))
    bounds = {a.name: (a.bounds or (-1e9, 1e9)) for a in coh.A.values()}
    if bounds:
        coh.add_daemon(BoundaryDaemon(bounds))
    return coh


# Build and inspect
drone = build_drone_system()

print(f"Total descendants: {len(drone.descendants())}")
print(f"Component report : {drone.component_report()}")
print(f"Sound            : {drone.soundness_report()['sound']}")
print(f"Acyclic          : {drone.is_acyclic()}")
print(f"Embedding dim    : {len(drone.embed())}")

# Validate
validator = COHValidator()
issues = validator.validate(drone)
errors = [i for i in issues if i.severity == "error"]
print(f"Validator errors : {len(errors)}")
assert not errors, errors

# Simulate
traj = drone.simulate(steps=200, dt=0.1)
print(f"Final coverage   : {traj.final['coverage_pct']:.1f}%")
print(f"Final status     : {traj.final['mission_status']}")
print(f"Total reward     : {traj.cumulative_reward():.2f}")

# Reason
from gismol.reasoners import GeneralReasoner, Query

result = GeneralReasoner().infer(drone, Query("maximize coverage"))
print(f"Reasoner answer  : {result.answer}")
print(f"Confidence       : {result.confidence:.2f}")
```

### 24.4 What We Achieved

- **Four levels of hierarchy** — system → controller/power/payload → attitude/nav/battery/camera → individual control axes / GPS / cells / image processor
- **All nine components** populated with domain-appropriate content
- **Hybrid dynamics** — continuous state evolution + trigger-driven mode switching
- **Layered safety** — invariants + boundary daemons + barrier daemon
- **Learned policy** — a neural component ready to be trained
- **Full validation** — no errors, soundness confirmed
- **Reasoning** — the query engine can plan, check, and explain

This is what a production GISMOL system looks like.

---

## 25. API Cheatsheet

### 25.1 The 9-tuple at a glance

```python
coh = COHObject("name", description="...")
coh.C           # children (dict)
coh.A           # attributes (dict)
coh.M           # methods (dict)
coh.N           # neural components (dict)
coh.E           # embedder
coh.I           # identity constraints (list)
coh.T           # trigger constraints (list)
coh.G           # goal constraints (list)
coh.D           # daemon network
```

### 25.2 Construction

```python
COHObject(name, children=[], attributes=[], methods=[], neural={},
          embedder=None, identity=[], triggers=[], goals=[], daemons=None,
          description="", metadata={}, flow=None, jump=None)

coh.add_child(COHObject("child"))
coh.add_attribute(Attribute("x", 0.0, bounds=(0, 10)))
coh.add_method(Method("do", lambda s, a=None: (s, 1.0)))
coh.add_identity_constraint(IdentityConstraint("i1", "x > 0"))
coh.add_trigger_constraint(TriggerConstraint("t1", "evt", "x > 5", "do"))
coh.add_goal_constraint(GoalConstraint("g1", "x", direction="max"))
coh.add_daemon(BoundaryDaemon({"x": (0, 10)}))
coh.add_neural(MLPComponent("nn", [3, 4, 1]))
coh.set_embedder(FeatureEmbedder(["x"]))
```

### 25.3 State

```python
coh.state                    # State (dict-like)
coh.get("x")                 # top-level
coh.get("child.x")           # dotted
coh.set("child.x", 5.0)
coh.initial_state()          # fresh state from defaults
coh.reset()                  # reinitialise
coh.snapshot()               # deep copy of state
```

### 25.4 Methods

```python
coh.invoke("method_name")    # -> (new_state, reward)
coh.method("method_name")    # -> Method object
```

### 25.5 Simulation

```python
coh.step(action=None, dt=0.01, event=None)    # -> (state, info)
coh.simulate(steps=100, dt=0.01, policy=None) # -> Trajectory

traj.final
traj.cumulative_reward()
traj.discounted_return(gamma=0.99)
traj.to_dict()
```

### 25.6 Constraints

```python
coh.constraints.check_identity(state)
coh.constraints.satisfies_identity(state)
coh.constraints.violation_score(state)
coh.constraints.total_goal(state, action, t)
```

### 25.7 Daemons

```python
coh.monitor(state=None)      # run all daemons, return highest-severity action
coh.D                        # the DaemonNetwork
coh.D.by_name("...")         # look up a daemon
for d in coh.D: ...          # iterate
```

### 25.8 Embeddings

```python
coh.embed()                  # -> tuple[float, ...]
coh.similarity(other)        # -> float in [-1, 1]
coh.canonical_form()         # deterministic string
```

### 25.9 Reasoning

```python
from gismol.reasoners import GeneralReasoner, Query

GeneralReasoner().infer(coh, Query("target"))
ConstraintReasoner().infer(coh, Query("check"))
GoalSolver(depth=3, beam=4).infer(coh, Query("plan"))
```

### 25.10 NLP

```python
from gismol.nlp import text_to_coh, ChatModule

coh = text_to_coh("...text...", name="sys")
chat = ChatModule()
chat.process("...")
```

### 25.11 Validation

```python
from gismol.nlp.validation import COHValidator

COHValidator().validate(coh)         # -> list[ValidationIssue]
COHValidator().is_valid(coh)         # -> bool
coh.soundness_report()                # -> dict
coh.component_report()                # -> dict
coh.is_minimal()                      # -> bool
```

### 25.12 Persistence

```python
data = coh.to_dict()
coh2 = COHObject.from_dict(data)

repo = COHRepository()
repo.add(coh, tags=["x"])
repo.save("repo.json")
COHRepository.load("repo.json")
```

---

## 26. Glossary

| Term | Definition |
|---|---|
| **Admissible region** | The set of states where all identity constraints hold |
| **Attribute (A)** | A measurable state variable of a COHObject |
| **Barrier function** | A scalar function whose non-negativity certifies safety |
| **COH** | Constrained Object Hierarchy — the underlying framework |
| **COHObject** | The base class implementing the 9-tuple |
| **Constraint** | A predicate that restricts admissible behaviour |
| **Coproduct** | Categorical alternative composition `O₁ + O₂` |
| **Corrective action** | An action emitted by a daemon |
| **Daemon (D)** | A continuous monitor enforcing invariants |
| **Embedding (E)** | A map `O → H` into a Hilbert space |
| **Exponential** | Categorical function object `O₂^O₁` |
| **Free energy** | An information-theoretic bound on surprise |
| **Goal constraint (G)** | An optimisation objective |
| **GISMOL** | General Intelligent Systems Modelling Language |
| **Hybrid system** | A system combining continuous and discrete dynamics |
| **Identity constraint (I)** | An invariant predicate |
| **Lyapunov function** | A scalar function decreasing along trajectories |
| **Method (M)** | An executable state transformation |
| **Morphism** | A structure-preserving map between COHObjects |
| **Neural component (N)** | A parameterised adaptive function |
| **Product** | Categorical parallel composition `O₁ × O₂` |
| **Soundness** | Internal consistency of a COHObject |
| **State** | A hierarchical assignment of values to attributes |
| **Trigger constraint (T)** | An event-condition-action rule |
| **9-tuple** | `(C, A, M, N, E, I, T, G, D)` |

---

## Closing Note

You now have everything you need to build intelligent systems with GISMOL. The framework gives you structure, safety, learning, reasoning, and language — but the design of *your* system is up to you.

Start small. Build a two-level hierarchy with a handful of constraints. Add daemons. Simulate. Then grow. The toolkit scales with your ambition.

The COH thesis is simple: **intelligence is constrained hierarchical optimisation.** Every system you build with GISMOL is an instance of that thesis. Build well.

---

<p align="center">
  <b>GISMOL — General Intelligent Systems Modelling Language</b><br>
  <i>Structure first. Learning second. Safety always.</i>
</p>
