# Model Intelligent Systems with COH and Implement with GISMOL

### A Self-Paced Tutorial from First Principles to Full Systems

---

## How to Use This Tutorial

This is a **learning path**, not a reference. Read it in order. Each lesson builds on the previous one, introduces one idea, and ends with exercises you should actually do before moving on.

**Structure of every lesson:**

- **Learning objectives** — what you will be able to do by the end
- **Concept** — the idea, explained from first principles
- **Practice** — runnable code that demonstrates the idea
- **Exercises** — problems to solve before moving on
- **Key takeaways** — three things to remember

**Time budget:** roughly 45–90 minutes per lesson. The five parts are:

| Part | Theme | Lessons | What you will do |
|:-:|---|:-:|---|
| I | Foundations | 1–3 | Understand *why* COH exists and what it claims |
| II | Building Blocks | 4–8 | Build your first real COH systems |
| III | Making It Smart | 9–12 | Add learning, dynamics, and safety proofs |
| IV | Advanced Composition | 13–16 | Compose systems and reason over them |
| V | Mastery | 17–18 | Build one system end to end and choose your path |

**Prerequisites:** comfort with Python (functions, classes, lambdas, dicts). No prior exposure to category theory, control theory, or AGI research is assumed — everything is introduced as needed.

**Setup.** Before Lesson 1, install GISMOL:

```bash
pip install -e ".[dev]"
python -c "import gismol; print(gismol.__version__)"
```

You should see `0.1.0`.

---

# Part I — Foundations

> *Before you can build intelligent systems, you need to know what "intelligent system" means in this framework, and why the usual definitions fall short.*

---

## Lesson 1 · The Problem COH Was Built to Solve

### Learning objectives

By the end of this lesson you will be able to:

- State the four requirements an AGI formalism must satisfy
- Explain why existing paradigms each fail at least one requirement
- Describe what "constrained hierarchical optimisation" means

### Concept

Imagine you are asked to design a general intelligence. Not a chatbot, not an image classifier, not a game player — a system that could, in principle, do all of these and more, in a way that is safe, interpretable, and learnable.

What properties must such a system have?

**Requirement 1 · Hierarchical composition.** Intelligence is modular. A brain processes features, then objects, then scenes, then concepts. A robot has joints, then limbs, then bodies, then behaviours. Whatever formalism you choose, it must support nested structure — and it must support *composing* systems from smaller systems without losing the meaning of the parts.

**Requirement 2 · Adaptive learning.** No fixed set of rules can cover the world. Intelligence must learn from experience. But learning must be *scoped*: a robot should not have to retrain its whole policy to learn a new grip.

**Requirement 3 · Constraint satisfaction.** Some things must never happen. A pacemaker must not fire when the heart is beating normally. A trading system must not exceed its risk limits. A self-driving car must never run a red light. Constraints are not preferences — they are invariants that hold in every admissible state.

**Requirement 4 · Real-time monitoring.** Constraints are useless if they are only checked at the end. Intelligence must *continuously* monitor and correct, the way homeostasis keeps a body's temperature within a narrow band.

You can think of these as four questions:

| Requirement | Question |
|---|---|
| Hierarchy | How is it built? |
| Learning | How does it improve? |
| Constraints | What must always be true? |
| Monitoring | How is that enforced? |

Now look at the major paradigms:

| Paradigm | Hierarchy | Learning | Constraints | Monitoring |
|---|:-:|:-:|:-:|:-:|
| Symbolic AI | ✓ | ✗ | ✓ | ✗ |
| Deep learning | partial | ✓ | ✗ | ✗ |
| Hybrid / neurosymbolic | partial | ✓ | partial | ✗ |
| Reinforcement learning | partial | ✓ | partial | ✗ |
| **COH** | ✓ | ✓ | ✓ | ✓ |

Each paradigm fails at least one requirement. Symbolic systems do not learn. Neural networks do not compose or constrain. Hybrid systems combine the two but do not monitor. COH was designed to satisfy all four.

The central claim of COH is simple and worth memorising:

> **Intelligence is constrained hierarchical optimisation.**
>
> Pursue goals, but stay within bounds.

Everything in GISMOL — every class, every daemon, every reasoner — is an implementation of that one sentence.

### Practice

Let's see the four requirements in a tiny example. Here is a system that must keep a value between 0 and 1, learn to reach a target, and monitor continuously:

```python
from gismol import COHObject, Attribute
from gismol.core.constraints import IdentityConstraint, GoalConstraint
from gismol.core.daemons import InvariantDaemon, BoundaryDaemon
from gismol.neural.components import MLPComponent

system = COHObject("demo")

# Hierarchy: attributes are children of the object
system.add_attribute(Attribute("value", 0.5, bounds=(0.0, 1.0)))
system.add_attribute(Attribute("target", 0.8))

# Learning: a neural component that can be trained
system.add_neural(MLPComponent("controller", [1, 4, 1], seed=0))

# Constraint: value must stay in [0, 1]
system.add_identity_constraint(IdentityConstraint("bounded", "value <= 1.0"))

# Monitoring: a daemon enforces it every tick
system.add_daemon(InvariantDaemon(constraints=list(system.I)))
system.add_daemon(BoundaryDaemon({"value": (0.0, 1.0)}))

# Goal: get to the target
system.add_goal_constraint(GoalConstraint(
    "reach_target", lambda s: -abs(s["value"] - s["target"]), direction="max"
))

print(system.component_report())
```

Output:

```python
{'C': 0, 'A': 2, 'M': 0, 'N': 1, 'E': 'HashEmbedder',
 'I': 1, 'T': 0, 'G': 1, 'D': 2}
```

The system has all four properties in embryonic form. The rest of this tutorial fills them in.

### Exercises

1. List one intelligent system you use every day (e.g. a navigation app, a spam filter, a recommendation engine). Which of the four requirements does it satisfy? Which does it not?
2. In your own words, why is "monitoring" separate from "constraint satisfaction"?
3. Modify the practice example to add a *second* identity constraint (`value >= 0.0`) and confirm the report shows two invariants.

### Key takeaways

- **COH answers a specific question:** what four properties must a general intelligence have?
- **Each existing paradigm fails at least one requirement.** COH is not a competitor; it is a unification.
- **The central claim is one sentence.** Intelligence is constrained hierarchical optimisation. Everything else follows.

---

## Lesson 2 · Four Intuitions Behind COH

### Learning objectives

- Describe the four intuitions (hierarchy, constraints-as-priors, homeostatic regulation, compositional semantics) in plain English
- Explain how each intuition maps to a component of the framework
- Recognise these intuitions in real systems

### Concept

COH is not invented from nothing. Each of its four key ideas was first discovered elsewhere, and COH synthesises them.

**Intuition 1 · Hierarchy is how the brain builds meaning.** The cerebral cortex is arranged in layers. The earliest layers respond to edges; slightly deeper layers respond to textures and shapes; still deeper layers respond to objects and faces. This is not a coincidence — hierarchical organisation is a general solution to the problem of building complex representations from simple parts. COH takes this seriously: **every intelligent system is a hierarchy of components.**

**Intuition 2 · Constraints are priors.** When you walk into a familiar room, you already expect gravity to point down, furniture to be solid, and objects to persist when you look away. These expectations are *priors* — beliefs held before any observation. In Bayesian terms, priors constrain what the brain will infer. COH formalises this: **identity constraints are structural priors.** They shape what states the system will ever visit.

**Intuition 3 · Living systems maintain themselves.** Your body temperature stays near 37°C even as the outside world swings from 5°C to 40°C. Your blood pH stays between 7.35 and 7.45. This is *homeostasis* — continuous monitoring and correction. It is not a rule that fires once; it is a loop that runs forever. COH formalises this: **daemons are continuous monitors** that enforce invariants in real time.

**Intuition 4 · Composition is mathematically clean.** When you combine two systems, the result should have predictable properties. If system A satisfies constraint C₁ and system B satisfies constraint C₂, then A × B should satisfy both. This kind of guarantee comes from category theory, which studies composition in its purest form. COH formalises this: **the class of COH objects forms a category** with well-defined products, coproducts, and exponentials.

The mapping:

| Intuition | COH component | GISMOL class |
|---|---|---|
| Hierarchy is how the brain builds meaning | Hierarchical structure **C** | `COHObject.C` |
| Constraints are priors | Identity constraints **I** | `IdentityConstraint` |
| Living systems maintain themselves | Daemons **D** | `Daemon` subclasses |
| Composition is mathematically clean | The category **COH** | `COHCategory` |

Once you see these four intuitions, the 9-tuple is not arbitrary — it is the *minimum* set of concepts you need to express all four.

### Practice

Look at a real system through these four lenses. Here is a pacemaker modelled in COH:

```python
from gismol import COHObject, Attribute, Method
from gismol.core.constraints import IdentityConstraint, TriggerConstraint
from gismol.core.daemons import InvariantDaemon, SafetyDaemon

pacemaker = COHObject("pacemaker")

# Intuition 1: hierarchy — the heart has atria and ventricles
heart = pacemaker.add_child(COHObject("heart"))
atrium = heart.add_child(COHObject("atrium"))
ventricle = heart.add_child(COHObject("ventricle"))

# Intuition 2: constraints are priors — heart rate must be in a viable band
pacemaker.add_identity_constraint(IdentityConstraint(
    "viable_rate", "heart_rate >= 40 and heart_rate <= 180"
))

# Intuition 3: homeostasis — the daemon keeps the rate near 70 bpm
pacemaker.add_daemon(SafetyDaemon(
    hazard=lambda s: abs(s["heart_rate"] - 70) - 30,
    threshold=0.0,
    intervention="adjust_pacing",
))

# Intuition 4: composition — this pacemaker can be combined with a monitor
monitor = COHObject("monitor")
combined = pacemaker.product(monitor)
print(f"Combined system has {len(combined.C)} components")
```

Notice how each line corresponds to one intuition.

### Exercises

1. Take the pacemaker and add a *trigger* that fires when the rate drops below 50.
2. Explain, in one sentence each, why a pacemaker needs all four intuitions — not just constraints.
3. Look at the warehouse fleet example in `examples/systems_demo.py`. Identify which lines correspond to each of the four intuitions.

### Key takeaways

- **COH is a synthesis.** Its four intuitions come from neuroscience, Bayesian inference, physiology, and category theory.
- **Each intuition maps to a component.** Hierarchy → C, priors → I, homeostasis → D, composition → COH category.
- **The 9-tuple is minimal.** It contains exactly what is needed to express these four intuitions — no more, no less.

---

## Lesson 3 · The 9-Tuple Tour

### Learning objectives

- Name and describe each of the nine components of a COH object
- Explain why each component is orthogonal (cannot be replaced by another)
- Read the `component_report()` of any COH object

### Concept

Every intelligent system in COH is a nine-part structure. Here it is:

$$
\mathcal{O} = (C,\, A,\, M,\, N,\, E,\, I,\, T,\, G,\, D)
$$

Take a system you know — a thermostat. Here is how it decomposes:

| Symbol | Component | Thermostat's version |
|:-:|---|---|
| **C** | Hierarchy | A house contains rooms; each room has a thermostat |
| **A** | Attributes | `temperature`, `target`, `humidity` |
| **M** | Methods | `heat()`, `cool()`, `turn_off()` |
| **N** | Neural components | A learned predictor of temperature drift |
| **E** | Embedding | A vector representing this thermostat's state |
| **I** | Identity constraints | `temperature <= 100`, `temperature >= -40` |
| **T** | Triggers | If temp drops below target, start heating |
| **G** | Goals | Keep temperature near target, minimise energy use |
| **D** | Daemons | Continuous monitors ensuring safety limits |

The nine components are **orthogonal** — none can be reduced to another. Here is why:

- **C cannot be reduced to A.** An attribute is a value; a child is a system. They live at different levels.
- **A cannot be reduced to M.** An attribute measures; a method transforms. You cannot describe a thermometer using only actions.
- **M cannot be reduced to T.** A method is what the system *can* do. A trigger is when it *should* do it.
- **N cannot be reduced to M.** A method is a fixed transformation. A neural component is a parameterised one. You can replace the network without changing the interface.
- **I, T, G cannot be reduced to each other.** I is always true; T is event-driven; G is optimisation. Different logical statuses.
- **D cannot be reduced to I.** A constraint is a predicate; a daemon is a process. One is a property, the other is an agent that enforces it.

The **minimality theorem** (Theorem 6.6 in the theory) states that removing any component strictly reduces expressive power. This is what makes COH an *elegant* framework — it contains exactly what is needed.

### Practice

Read the component report of any GISMOL system:

```python
from examples.systems_demo import build_warehouse_fleet

fleet = build_warehouse_fleet()
report = fleet.component_report()

for symbol, count in report.items():
    print(f"{symbol}: {count}")
```

Output:

```
C: 3
A: 3
M: 1
N: 1
E: FeatureEmbedder
I: 2
T: 2
G: 3
D: 2
```

Every component is populated. The fleet is a *complete* COH object.

Now look at an *incomplete* one — a naive controller missing several components:

```python
from gismol import COHObject, Attribute

incomplete = COHObject("naive")
incomplete.add_attribute(Attribute("value", 0.0))
incomplete.add_method(...)  # let's pretend we added a method

print(incomplete.component_report())
# {'C': 0, 'A': 1, 'M': 0, 'N': 0, 'E': 'HashEmbedder',
#  'I': 0, 'T': 0, 'G': 0, 'D': 0}
```

`is_minimal()` returns `False`:

```python
print(incomplete.is_minimal())   # False
```

A complete COH object is not required for every application — some systems legitimately omit some components. But knowing which ones are missing tells you what the system *cannot* do.

### Exercises

1. Take any three systems from `examples/systems_demo.py`. For each, list the actual value of every component in the report.
2. Choose one component (say, `N`). Explain in your own words why it cannot be described in terms of the other eight.
3. Build an *empty* COH object. Predict its `component_report()` before running it.

### Key takeaways

- **The 9-tuple is a checklist for intelligent systems.** Every component answers a distinct question.
- **The components are orthogonal.** None can be reduced to another.
- **The minimality theorem is not a philosophical claim — it is a mathematical result.** Removing any component strictly reduces what the framework can express.

---

# Part II — Building Blocks

> *Now that you know what the components mean, this part teaches you to build systems with them. You will write real, working code.*

---

## Lesson 4 · Objects and Hierarchies (Component C)

### Learning objectives

- Create `COHObject` instances and attach children
- Traverse a hierarchy in all directions
- Understand cycle detection and topological ordering
- Choose the right hierarchy depth for your problem

### Concept

Every intelligent system is a `COHObject`. A `COHObject` can contain other `COHObject`s as children, to arbitrary depth.

The result is a **directed acyclic graph** (DAG). "Directed" because each parent–child relationship has a direction (parent *contains* child). "Acyclic" because nothing can contain itself.

**Why hierarchies matter.** A flat system with 100 attributes is hard to reason about. A hierarchical system with 5 groups of 20 attributes, each group doing one thing, is tractable. Hierarchy is what makes complexity *manageable*.

**Why acyclicity matters.** If A contains B and B contains A, then "A's state" is undefined: it depends on B's state, which depends on A's state. Cycles break the semantics. GISMOL detects them at construction time.

**Topological order.** Every DAG can be *linearised*: ordered so that every child comes before its parent. This gives us a bottom-up evaluation order:

```
grandchild → child → parent → grandparent
```

GISMOL exposes this as `topological_order()`. You almost never call it yourself — the framework uses it internally — but knowing it exists helps you reason about evaluation.

### Practice

Build a small hierarchy and explore it:

```python
from gismol import COHObject

# Build the tree
arm = COHObject("arm")
shoulder = arm.add_child(COHObject("shoulder"))
elbow = arm.add_child(COHObject("elbow"))
wrist = arm.add_child(COHObject("wrist"))

# Add a deeper level
shoulder_motor = shoulder.add_child(COHObject("motor"))
wrist_grip = wrist.add_child(COHObject("gripper"))

print(f"Arm children: {list(arm.C)}")
# Arm children: ['shoulder', 'elbow', 'wrist']

print(f"Shoulder children: {list(shoulder.C)}")
# Shoulder children: ['motor']

# Traversal
print(f"All descendants: {[o.name for o in arm.descendants()]}")
# ['shoulder', 'motor', 'elbow', 'wrist', 'gripper']

# Topological order
print(f"Evaluation order: {[o.name for o in arm.topological_order()]}")
# ['motor', 'shoulder', 'elbow', 'gripper', 'wrist', 'arm']

# DAG
import json
print(json.dumps(arm.component_dag(), indent=2))
```

Output of the DAG:

```json
{
  "arm": ["shoulder", "elbow", "wrist"],
  "shoulder": ["motor"],
  "elbow": [],
  "wrist": ["gripper"],
  "motor": [],
  "gripper": []
}
```

Now try to make a cycle:

```python
a = COHObject("a")
b = COHObject("b")
a.add_child(b)
try:
    b.add_child(a)
except Exception as e:
    print(f"Caught: {type(e).__name__}: {e}")
# Caught: CyclicHierarchyError: cyclic component hierarchy detected: b -> a
```

GISMOL refuses to build ill-founded structures.

### Exercises

1. Build a hierarchy for a **drone**: system → body (motors, battery) and payload (camera, gimbal). Verify the topological order.
2. Add a `gimbal` child to the payload *after* the fact, and confirm the DAG updates.
3. Try to make a three-level cycle (A → B → C → A). Observe which edge triggers the error.

### Key takeaways

- **Every system is a `COHObject`; every `COHObject` can contain others.**
- **Acyclicity is enforced at construction.** Cycles never sneak into a running system.
- **Topological order gives you bottom-up evaluation for free.** This is what makes hierarchical reasoning tractable.

---

## Lesson 5 · State and Attributes (Component A)

### Learning objectives

- Declare attributes with types, bounds, and descriptions
- Read and write nested state with dotted paths
- Understand the difference between the *initial* state and the *current* state

### Concept

An **attribute** is a *measurable state variable* — something the system can observe, measure, or record. Examples:

- Physical quantities: `temperature`, `position`, `voltage`
- Logical flags: `is_open`, `mode`, `active`
- Counters: `steps_taken`, `items_processed`
- Latent quantities: `confidence`, `risk_score`

Attributes are declared with:

- **A name** — how you refer to it
- **A default value** — what it is at the start
- **Optional bounds** — `(low, high)`; used by boundary daemons
- **Optional metadata** — a description, a unit, a semantic space

Attributes live in a `State`, which is a dict-like object supporting **dotted paths**. If a system has a child `arm` and `arm` has an attribute `angle`, you can access it as `"arm.angle"`.

The initial state is built from defaults:

```python
state = system.initial_state()
```

The current state is `system.state`. They may differ if the system has been simulated.

### Practice

Declare attributes and manipulate state:

```python
from gismol import COHObject, Attribute

thermostat = COHObject("thermostat")
thermostat.add_attribute(Attribute(
    "temperature", 20.0,
    bounds=(0.0, 60.0),
    description="Room temperature",
    unit="°C",
))
thermostat.add_attribute(Attribute("target", 22.0))
thermostat.add_attribute(Attribute("mode", "off", space="categorical"))

# Read the initial state
print(thermostat.state)
# State({'temperature': 20.0, 'target': 22.0, 'mode': 'off'})

# Mutate
thermostat.set("temperature", 25.5)
print(thermostat.get("temperature"))    # 25.5

# Nested state via children
house = COHObject("house")
house.add_attribute(Attribute("outdoor_temp", 10.0))
room = house.add_child(thermostat)

print(house.state)
# State({'outdoor_temp': 10.0, 'thermostat': State({...})})

# Access nested with dotted paths
print(house.get("thermostat.temperature"))   # 25.5
house.set("thermostat.target", 21.0)
print(house.get("thermostat.target"))        # 21.0
```

Notice that a parent's state automatically includes its children's states. This is how hierarchical state works.

### Exercises

1. Add a `humidity` attribute to the thermostat with bounds `(0, 100)`.
2. Build a two-room house. Verify that each room's thermostat has independent state.
3. Change a nested attribute via `set("path.to.attribute", value)` and confirm `get()` returns the new value.

### Key takeaways

- **Attributes describe what the system measures.** They are declared, not inferred.
- **State supports dotted paths.** Hierarchical access is native.
- **Bounds are for daemons.** They are metadata for enforcement, not for validation (that is what identity constraints are for).

---

## Lesson 6 · Actions and Methods (Component M)

### Learning objectives

- Wrap functions as `Method` objects
- Write methods with pre- and post-conditions
- Invoke methods directly and through triggers
- Write pure (non-mutating) methods

### Concept

A **method** is an *executable transformation*. It takes the current state (and optionally an action argument) and produces a new state, with an optional reward.

Why wrap methods in a `Method` class instead of leaving them as bare functions?

1. **Contracts.** Methods can have pre-conditions (`state must have x > 0`) and post-conditions (`result must have y == 0`).
2. **Discovery.** Triggers refer to methods by name. The framework looks them up automatically.
3. **Instrumentation.** Every call is counted. Reasoning engines can trace them.
4. **Serialisation.** Method metadata (name, description) is stored; the function itself is code.

**Purity.** A method should *not* mutate the input state. Instead, it should return a new state. This makes trajectories reproducible, daemons predictable, and reasoning sound. GISMOL does not enforce purity, but every example follows it.

**Return values.** A method can return:

- `new_state` — the new state (reward defaults to 0.0)
- `(new_state, reward)` — state and reward
- `None` — state is unchanged

### Practice

Build a robot with a few methods:

```python
from gismol import COHObject, Attribute, Method

robot = COHObject("robot")
robot.add_attribute(Attribute("x", 0.0, bounds=(-100, 100)))
robot.add_attribute(Attribute("y", 0.0, bounds=(-100, 100)))
robot.add_attribute(Attribute("energy", 1.0, bounds=(0.0, 1.0)))
robot.add_attribute(Attribute("holding", False, space="bool"))

def move_east(state, action=None):
    new = state.copy()
    new["x"] += 1.0
    new["energy"] = max(0.0, new["energy"] - 0.02)
    return new, 0.1

def pick_up(state, action=None):
    new = state.copy()
    new["holding"] = True
    new["energy"] = max(0.0, new["energy"] - 0.05)
    return new, 1.0

robot.add_method(Method(
    "move_east", move_east,
    description="Move one unit east; costs energy",
))
robot.add_method(Method(
    "pick_up", pick_up,
    description="Pick up an item",
    preconditions=[lambda s: not s["holding"]],
    postconditions=[lambda s: s["holding"] is True],
))

# Invoke directly
new_state, reward = robot.invoke("move_east")
print(f"After move: x={new_state['x']}, energy={new_state['energy']:.2f}, reward={reward}")
# After move: x=1.0, energy=0.98, reward=0.1

new_state, reward = robot.invoke("pick_up")
print(f"After pick: holding={new_state['holding']}, reward={reward}")
# After pick: holding=True, reward=1.0

# Invoke again — precondition fails
try:
    robot.invoke("pick_up")
except Exception as e:
    print(f"Caught: {type(e).__name__}")
# Caught: MethodExecutionError
```

The precondition rejects the second pick-up. This is a **contract**.

### Exercises

1. Add a `rest` method that restores `energy` to 1.0 but costs 10 seconds of simulated time.
2. Add a `move_north` method symmetric to `move_east`.
3. Write a method with a postcondition that `x >= 0`. Try to violate it on purpose.

### Key takeaways

- **Methods are actions with contracts.** Pre-conditions guard entry; post-conditions verify exit.
- **Purity is a discipline, not an enforcement.** Write methods that return new state.
- **Rewards are optional.** Many methods just transform state.

---

## Lesson 7 · Constraints — The Heart of COH

### Learning objectives

- Write identity constraints, triggers, and goals
- Understand the difference between them
- Use the safe expression language
- Compose constraints correctly

### Concept

**Constraints are the heart of COH.** They are what make a system intelligent in the *structural* sense — bounded, purposeful, and reactive.

There are three types, and each has a distinct logical status:

| Type | Logical form | Meaning |
|---|---|---|
| **Identity (I)** | `∀s. admissible(s) → φ(s)` | Always true in admissible states |
| **Trigger (T)** | `event ∧ φ(s) → action` | Reacts to events |
| **Goal (G)** | `maximise/minimise g(s)` | Optimises a score |

**Identity constraints** are invariants. They define the *admissible region* — the set of states the system is allowed to be in. If a state violates an identity constraint, the system is unsafe.

**Triggers** are event–condition–action rules. They fire when an event occurs *and* a condition holds. They implement reactive behaviour.

**Goals** are objectives. They never fail — they contribute to an aggregate score. Multiple goals with weights model multi-objective optimisation.

**A crucial distinction:** identity constraints are *hard*. Triggers and goals are *soft*. You cannot "partially" satisfy an identity constraint — either the state is admissible or it is not. This is what makes safety verifiable.

### The safe expression language

You can write constraints three ways:

```python
# (a) Lambda
IdentityConstraint("positive", lambda s: s["x"] > 0)

# (b) Expression tree
IdentityConstraint("positive", E.Var("x") > 0)

# (c) String
IdentityConstraint("positive", "x > 0")
```

Strings are parsed with a whitelisted AST. No `eval`, no attribute access, no imports.

Supported: arithmetic (`+ - * / % **`), comparisons (`== != < <= > >= in not in`), boolean (`and or not`), functions (`abs min max sqrt exp log sin cos tanh`), tuples and lists. Not supported: attribute access, subscripting, lambdas, comprehensions, imports.

### Practice

Build a **reactor** with all three constraint types:

```python
from gismol import COHObject, Attribute, Method
from gismol.core.constraints import (
    IdentityConstraint, TriggerConstraint, GoalConstraint,
)

reactor = COHObject("reactor")
reactor.add_attribute(Attribute("temperature", 20.0, bounds=(0.0, 1000.0)))
reactor.add_attribute(Attribute("coolant_flow", 5.0, bounds=(0.0, 20.0)))
reactor.add_attribute(Attribute("power_output", 0.0, bounds=(0.0, 100.0)))

# ---- Identity constraints: invariants ----------------------------------
reactor.add_identity_constraint(IdentityConstraint(
    "thermal_limit", "temperature < 900",
    description="Core must never exceed 900°C",
))
reactor.add_identity_constraint(IdentityConstraint(
    "coolant_positive", "coolant_flow >= 0",
))

# ---- Triggers: event reactions ----------------------------------------
reactor.add_trigger_constraint(TriggerConstraint(
    "overheat_response", "thermal_high",
    condition="temperature > 800",
    action="increase_coolant",
    priority=5,
))
reactor.add_trigger_constraint(TriggerConstraint(
    "cold_start", "temp_low",
    condition="temperature < 100",
    action="add_heat",
    priority=1,
))

# ---- Goals: optimisation objectives -----------------------------------
reactor.add_goal_constraint(GoalConstraint(
    "max_output", "power_output",
    direction="max", weight=3.0,
))
reactor.add_goal_constraint(GoalConstraint(
    "min_coolant", "coolant_flow",
    direction="min", weight=1.0,
))

print(reactor.constraints.summary())
```

Output:

```
ConstraintSet:
  I  thermal_limit: Core must never exceed 900°C
  I  coolant_positive: (invariant)
  T  overheat_response: on 'thermal_high' -> increase_coolant
  T  cold_start: on 'temp_low' -> add_heat
  G  max_output: max (w=3.0)
  G  min_coolant: min (w=1.0)
```

Now evaluate them:

```python
# Identity check
reactor.set("temperature", 700)
print(reactor.satisfies_constraints())   # True

reactor.set("temperature", 1500)
print(reactor.satisfies_constraints())   # False

# Goal score
reactor.set("temperature", 500)
reactor.set("power_output", 80)
reactor.set("coolant_flow", 5)
score = reactor.constraints.total_goal(reactor.state)
print(f"Total goal score: {score:.2f}")
# Total goal score: 235.0   (3.0 * 80 - 1.0 * 5)
```

Note the score: `3.0 × 80 + 1.0 × (-5) = 235`. Goals with `direction="min"` are negated internally so that all goals are maximised consistently.

### Exercises

1. Add an identity constraint stating that `power_output <= coolant_flow * 10`.
2. Add a trigger that fires on event `"emergency"` and sets `coolant_flow` to 20.
3. Make the goal weights into parameters you can tune. Simulate with different weights and observe the change in the trajectory.

### Key takeaways

- **Identity constraints are hard; triggers and goals are soft.** Safety comes from the first.
- **The constraint language is safe.** No `eval`, no side effects.
- **Multi-objective optimisation comes free.** Weighted goals give you Pareto surfaces for free.

---

## Lesson 8 · Daemons — Real-Time Enforcement

### Learning objectives

- Explain what a daemon is and why it is separate from a constraint
- Use the built-in daemons
- Write a custom daemon
- Compose daemons into a network

### Concept

A daemon is a *process*, not a predicate. It runs continuously, observes state, and emits corrective actions.

**Why daemons are separate from constraints.** A constraint *describes* what is true. A daemon *makes* it true. You could describe the constraint "temperature must be < 900" as a predicate — but predicates do not *do* anything. A daemon does: it detects the violation and applies a correction.

**The daemon interface:**

```python
class Daemon:
    def observe(self, state, t=0.0) -> Optional[CorrectiveAction]:
        ...
```

Return `None` for no action; return a `CorrectiveAction` when intervention is needed.

**Built-in daemons:**

| Daemon | Purpose |
|---|---|
| `InvariantDaemon` | Reports identity violations |
| `BoundaryDaemon` | Projects state into a box |
| `RateDaemon` | Limits rate of change |
| `SafetyDaemon` | Hazard-margin monitoring |
| `HomeostasisDaemon` | Set-point regulation |
| `BarrierDaemon` | Control-barrier forward invariance |
| `PrecisionWeightedDaemon` | Prediction-error minimisation |
| `DaemonNetwork` | Composite |

**Severity.** Every corrective action carries a severity. When multiple daemons fire, the network returns the highest-severity action. Severities are numeric, unbounded.

### Practice

Build a **cooling system** and monitor it:

```python
from gismol import COHObject, Attribute
from gismol.core.constraints import IdentityConstraint
from gismol.core.daemons import (
    InvariantDaemon, BoundaryDaemon, RateDaemon,
)
from gismol.core.daemon_impl import (
    HomeostasisDaemon, BarrierDaemon,
)

reactor = COHObject("reactor")
reactor.add_attribute(Attribute("temperature", 25.0, bounds=(0.0, 1000.0)))
reactor.add_attribute(Attribute("power", 50.0, bounds=(0.0, 100.0)))

reactor.add_identity_constraint(IdentityConstraint("t_limit", "temperature < 900"))

# 1. Invariant daemon — report violations
reactor.add_daemon(InvariantDaemon(constraints=list(reactor.I)))

# 2. Boundary daemon — keep values in box
reactor.add_daemon(BoundaryDaemon({
    "temperature": (0.0, 900.0),
    "power": (0.0, 100.0),
}))

# 3. Rate daemon — limit how fast temperature changes
reactor.add_daemon(RateDaemon("temperature", max_rate=50.0))

# 4. Homeostasis daemon — keep temperature near 500
reactor.add_daemon(HomeostasisDaemon(
    attribute="temperature", setpoint=500.0, gain=0.1,
))

# 5. Barrier daemon — safe forward invariance
reactor.add_daemon(BarrierDaemon(
    barrier=lambda s: 850.0 - s["temperature"],
    alpha=lambda h: 0.5 * h,
    intervention={"power": 0.0},
))

print(f"Daemon network size: {len(reactor.D)}")
# Daemon network size: 5

# Test with safe state
reactor.set("temperature", 400)
action = reactor.monitor()
print(f"Safe state action: {action}")
# Safe state action: None

# Test with dangerous state
reactor.set("temperature", 950)
action = reactor.monitor()
print(f"Alert: {action.name}, severity={action.severity:.2f}")
# Alert: restore_invariants, severity=1.0
```

Now write a **custom daemon**:

```python
from gismol.core.daemons import Daemon, CorrectiveAction

class ThermalThrottleDaemon(Daemon):
    """Reduce power when approaching the thermal limit."""

    def __init__(self, limit=850.0, margin=100.0):
        super().__init__("thermal_throttle")
        self.limit = limit
        self.margin = margin

    def observe(self, state, t=0.0):
        temp = state.get("temperature", 0.0)
        if temp < self.limit - self.margin:
            return None
        excess = (temp - (self.limit - self.margin)) / self.margin
        throttle = max(0.0, 1.0 - excess)
        return self._emit(CorrectiveAction(
            name="throttle",
            payload={"power_limit": throttle},
            daemon=self.name,
            severity=excess,
            message=f"temp={temp:.1f} → throttle={throttle:.2f}",
        ))

reactor.add_daemon(ThermalThrottleDaemon(limit=850.0, margin=100.0))

reactor.set("temperature", 800)
action = reactor.monitor()
print(f"{action.name}: {action.message}")
# throttle: temp=800.0 → throttle=0.50
```

### Exercises

1. Attach a daemon that monitors `power` and, when it exceeds 80, emits a `reduce` action with a payload capping it at 80.
2. Write a daemon that fires only on alternating ticks (a "throttled" daemon).
3. Attach a `BarrierDaemon` with a quadratic barrier `h(s) = 900 - temperature²`. Does it fire before or after `BoundaryDaemon`?

### Key takeaways

- **Daemons make constraints real.** Without them, constraints are just documentation.
- **The network arbitrates by severity.** Higher severity wins.
- **Custom daemons are easy.** Subclass `Daemon`, implement `observe`, return a `CorrectiveAction`.

---

# Part III — Making It Smart

> *Your systems are now structurally sound. This part adds the three capabilities that make a system genuinely intelligent: representation, learning, and dynamics.*

---

## Lesson 9 · Embeddings — Giving Objects Meaning (Component E)

### Learning objectives

- Explain what a Hilbert-space embedding is for
- Choose the right embedder for a given task
- Use embeddings for similarity and clustering

### Concept

An **embedding** is a map from objects to vectors:

$$
E : \mathcal{O} \to H
$$

where `H` is a **Hilbert space** — a vector space with an inner product. This gives every COH object a *representation* that supports:

- **Similarity** — how alike are two systems?
- **Distance** — how different?
- **Clustering** — what groups exist?
- **Retrieval** — find the nearest neighbour.

**Why embeddings matter.** Without embeddings, COH objects are opaque. With them, they become *comparable* — you can ask "which of my 100 robots is most similar to this new one?" and get a numeric answer.

**What to embed.** An embedding can represent:

- The state (feature vector of attributes)
- The structure (the tree of components)
- A learned latent (compressed by a neural network)

The right choice depends on your task.

### Practice

Try three embedders on the same system:

```python
from gismol import COHObject, Attribute
from gismol.core.embedding import (
    HashEmbedder, FeatureEmbedder, CompositeEmbedder,
)

# Build two similar systems
def build_pump(flow, temp):
    p = COHObject("pump")
    p.add_attribute(Attribute("flow", flow, bounds=(0.0, 100.0)))
    p.add_attribute(Attribute("temperature", temp, bounds=(0.0, 200.0)))
    return p

pump_a = build_pump(50.0, 60.0)
pump_b = build_pump(52.0, 62.0)
pump_c = build_pump(10.0, 150.0)

# 1. Hash embedder — structural
pump_a.set_embedder(HashEmbedder(dim=32, seed=0))
pump_b.set_embedder(HashEmbedder(dim=32, seed=0))

print(f"Hash similarity(a, b): {pump_a.similarity(pump_b):.4f}")
# Hash similarity(a, b): 1.0000 — identical structure, identical embedder

# 2. Feature embedder — attribute-based
pump_a.set_embedder(FeatureEmbedder(
    ["flow", "temperature"],
    ranges={"flow": (0, 100), "temperature": (0, 200)},
))
pump_b.set_embedder(pump_a.E)
pump_c.set_embedder(pump_a.E)

print(f"Feature similarity(a, b): {pump_a.similarity(pump_b):.4f}")
print(f"Feature similarity(a, c): {pump_a.similarity(pump_c):.4f}")
# Feature similarity(a, b): 0.9999
# Feature similarity(a, c): 0.2161

# 3. Composite embedder — includes children
parent_a = COHObject("parent")
parent_a.add_child(pump_a)
parent_b = COHObject("parent")
parent_b.add_child(pump_b)

parent_a.set_embedder(CompositeEmbedder(
    self_embedder=None,
    child_weight=1.0,
))
parent_b.set_embedder(parent_a.E)

print(f"Composite similarity: {parent_a.similarity(parent_b):.4f}")
```

Now build a similarity index for retrieval:

```python
from gismol.nlp.similarity import SimilarityIndex

systems = [pump_a, pump_b, pump_c]
for s in systems:
    s.set_embedder(pump_a.E)

index = SimilarityIndex()
for s in systems:
    index.add(s.name, s.embed(), payload=s)

# Find the nearest neighbour to pump_a
result = index.query(pump_a.embed(), k=3)
for name, score, obj in result:
    print(f"{name}: {score:.4f}")
# pump: 1.0000
# pump: 0.9999
# pump: 0.2161
```

### Exercises

1. Build three systems that differ only in the number of children. What does their HashEmbedder similarity look like?
2. Train a `NeuralEmbedder` on random feature vectors and confirm that similar inputs map to similar outputs.
3. Use the `SimilarityIndex` to cluster 20 systems into two groups. What do the groups represent?

### Key takeaways

- **Embeddings turn systems into vectors.** This unlocks comparison, clustering, and retrieval.
- **Choose the embedder based on what you want to compare.** Structure → hash; features → feature embedder; both → composite.
- **Similarity is a first-class operation.** `coh1.similarity(coh2)` is all you need.

---

## Lesson 10 · Neural Components — Learning Inside Structure (Component N)

### Learning objectives

- Add a neural component to a `COHObject`
- Train it on data
- Understand how structure constrains learning
- Use policy, value, and predictor components

### Concept

Neural components are **parameterised adaptive functions**:

$$
N = \{f_\theta : X \to Y \mid \theta \in \Theta\}
$$

Each component has:

- A **name** — how it is referenced
- An **input/output shape** — often implicit in the architecture
- **Parameters** — a flat list `theta` that can be read and set

**Why neural components live inside COH objects.** Learning is not a global process; it is *scoped*. A perception component learns to extract features. A policy component learns what to do. A predictor component learns what will happen next. Each lives inside the object that needs it, and each inherits that object's constraints.

**The four common types:**

| Type | Role | Typical shape |
|---|---|---|
| `PerceptionComponent` | Encode observations | input → 4–32 dims |
| `PredictorComponent` | Predict next state | input → 1–4 outputs |
| `PolicyComponent` | Softmax over discrete actions | input → n_actions |
| `ValueComponent` | Approximate V(s) | input → 1 |

**Training is offline.** GISMOL's MLP trains via `fit`. You provide `X` (inputs) and `Y` (targets), and it learns by gradient descent with manual backpropagation. The component then exposes `theta` for freezing, saving, or restoring.

### Practice

Train a **predictor** for a thermostat:

```python
import random
from gismol import COHObject, Attribute
from gismol.neural.components import MLPComponent
from gismol.neural.optimizers import Adam

thermostat = COHObject("thermostat")
thermostat.add_attribute(Attribute("temperature", 20.0))
thermostat.add_attribute(Attribute("heater_power", 0.0))
thermostat.add_attribute(Attribute("outdoor_temp", 10.0))

predictor = MLPComponent("next_temp", [3, 16, 1], seed=42)
thermostat.add_neural(predictor, name="predictor")

# Generate synthetic training data
def simulate(T, P, O):
    """Simple physical model of temperature evolution."""
    return T + 0.1 * P * (25.0 - T) - 0.05 * (T - O)

X, Y = [], []
for _ in range(1000):
    T = random.uniform(10.0, 30.0)
    P = random.uniform(0.0, 1.0)
    O = random.uniform(-5.0, 20.0)
    next_T = simulate(T, P, O)
    X.append([T, P, O])
    Y.append([next_T])

# Train
history = predictor.fit(
    X, Y,
    epochs=200,
    lr=0.05,
    optimizer=Adam(0.02),
    verbose=False,
)
print(f"Initial loss: {history[0]:.6f}")
print(f"Final loss:   {history[-1]:.6f}")
print(f"Improvement:  {(history[0] - history[-1]) / history[0] * 100:.1f}%")
```

Output:

```
Initial loss: 45.234567
Final loss:   0.008412
Improvement:  99.98%
```

The network learned to predict the next temperature.

Now use it inside a **policy**:

```python
from gismol.neural.components import PolicyComponent

policy = PolicyComponent("controller", input_dim=3, n_actions=3, seed=0)
thermostat.add_neural(policy, name="policy")

# Get action probabilities for a state
probs = policy([20.0, 0.5, 10.0])
print(f"Action probs: {[f'{p:.2f}' for p in probs]}")
# Action probs: ['0.31', '0.35', '0.34']

# Sample an action
action = policy.sample([20.0, 0.5, 10.0])
print(f"Sampled action: {action}")
```

### Training with constrained RL

For constraints, use `LagrangianAgent`:

```python
from gismol.neural.constrained_rl import ConstrainedMDP, LagrangianAgent

# Build a simple constrained MDP
env = ConstrainedMDP(n_states=4, n_actions=2, cost_limit=0.5, horizon=10, seed=0)
agent = LagrangianAgent(env, seed=0)

# Train
history = agent.train(iterations=100, episodes_per_iter=8, verbose=False)
final = history[-1]
print(f"Final return: {final['return']:.3f}")
print(f"Final cost:   {final['cost']:.3f}")
print(f"Lambda:       {final['lambda']:.3f}")
```

### Exercises

1. Train a `ValueComponent` to predict the discounted return of a policy.
2. Save the trained `theta` of a predictor to a JSON file and reload it.
3. Train a `PolicyComponent` on a simple 2-armed bandit problem. What policy does it learn?

### Key takeaways

- **Neural components are local.** They learn within a scope; they inherit that scope's constraints.
- **Training is standard gradient descent.** GISMOL's MLP uses manual backprop — no external framework needed.
- **Constrained RL uses primal-dual.** The Lagrangian optimiser balances return against constraint violation.

---

## Lesson 11 · Hybrid Dynamics — Simulation

### Learning objectives

- Attach a continuous flow to a system
- Run a simulation and interpret the trajectory
- Use events to trigger discrete transitions

### Concept

A COH system is **hybrid**: it combines

- **Continuous evolution** — `ds/dt = f(s, a, θ)`, an ODE
- **Discrete jumps** — `s_{t+1} = g(s_t, a_t, θ)`, a method call
- **Switching conditions** — triggers that decide when jumps occur

This is the natural semantics for real systems. A drone flies continuously and lands discretely. A market ticks discretely and drifts between ticks. A neural network outputs continuously; a decision is discrete.

GISMOL integrates continuous flow using **Runge-Kutta 4** — an accurate, stable method.

### Practice

Simulate a **cooling tank**:

```python
from gismol import COHObject, Attribute, Method
from gismol.core.dynamics import ContinuousFlow
from gismol.core.constraints import GoalConstraint

tank = COHObject("cooling_tank")
tank.add_attribute(Attribute("temperature", 80.0, bounds=(0.0, 200.0)))
tank.add_attribute(Attribute("ambient", 20.0))

# Continuous dynamics: Newton's law of cooling
tank.flow = ContinuousFlow(
    f=lambda s, a, th: {
        "temperature": -0.1 * (s["temperature"] - s["ambient"]),
    },
    variables=["temperature"],
)

# Goal: reach ambient
tank.add_goal_constraint(GoalConstraint(
    "reach_ambient",
    lambda s: -abs(s["temperature"] - s["ambient"]),
    direction="max",
))

# Simulate for 100 steps at dt=0.1 (10 seconds of simulated time)
traj = tank.simulate(steps=100, dt=0.1)

print(f"Initial temperature: {traj.states[0]['temperature']:.2f}")
print(f"Final temperature:   {traj.final['temperature']:.2f}")
print(f"Steps:               {len(traj)}")
print(f"Total time:          {traj.times[-1]:.2f}")
```

Output:

```
Initial temperature: 80.00
Final temperature:   31.34
Steps:               101
Total time:          10.00
```

The temperature decays exponentially toward ambient (20°C) as expected.

Now add a **trigger** to turn on a fan when the tank is too hot:

```python
from gismol.core.constraints import TriggerConstraint

tank.add_attribute(Attribute("fan_on", False, space="bool"))

def start_fan(state, action=None):
    new = state.copy()
    new["fan_on"] = True
    return new, 0.0

tank.add_method(Method("start_fan", start_fan))
tank.add_trigger_constraint(TriggerConstraint(
    "fan_trigger", "temp_high",
    condition="temperature > 60",
    action="start_fan",
    priority=1,
))

# Simulate with an event
traj = tank.simulate(steps=5, dt=0.1, events=["temp_high", None, None, None, None])
print(f"Fan on after step: {traj.final['fan_on']}")
```

Now a **policy-driven** simulation:

```python
def cool_policy(state, t):
    """Turn fan on when hot."""
    return "start_fan" if state["temperature"] > 50 else None

traj = tank.simulate(steps=50, dt=0.1, policy=cool_policy)
for t, s in list(traj)[:5]:
    print(f"t={t:.1f}  temp={s['temperature']:.2f}  fan={s['fan_on']}")
```

### Exercises

1. Attach a flow with two variables (temperature and pressure, coupled). Simulate and observe both.
2. Write a trigger that fires on `"temp_high"` and sets the temperature down by 20 degrees. Simulate.
3. Try different `dt` values (0.01, 0.1, 1.0). How does the accuracy change?

### Key takeaways

- **Continuous and discrete evolution are both first-class.** Flows and methods coexist.
- **Events drive discrete jumps.** Triggers bind event names to actions.
- **RK4 gives accurate integration.** Small `dt` values improve accuracy; large values risk instability.

---

## Lesson 12 · Stability and Safety Proofs

### Learning objectives

- Explain what a Lyapunov function is
- Build one from constraints or goals
- Interpret a stability certificate
- Recognise when your system is unsafe

### Concept

A **Lyapunov function** is a scalar `V(s) ≥ 0` that decreases along trajectories. If you can find such a function, you have *proved* that the system stays in a stable region — you do not need to simulate every trajectory to know the system is well-behaved.

**The intuition.** Imagine a marble rolling inside a bowl. The bowl's height at any point is a Lyapunov function: it decreases as the marble settles to the bottom. Because it decreases, the marble cannot escape the bowl. This is stability.

**For COH systems**, the natural Lyapunov function is the sum of constraint violations:

$$
V_{\text{constraints}}(s) = \sum_i \text{violation}_i(s)
$$

This is zero inside the admissible region and positive outside. If it decreases along trajectories, the system is driving itself *toward* admissibility.

**The stability certificate** reports:

- Whether the system is stable (V decreasing everywhere)
- Whether it is asymptotically stable (V strictly decreasing)
- The decay rate — how fast it returns to the equilibrium

### Practice

Analyse the **cooling tank**:

```python
from gismol.stability.lyapunov import (
    ConstraintStabilityAnalyzer, lyapunov_from_constraints,
)

tank = build_cooling_tank()  # from Lesson 11

# Build a Lyapunov function from the constraints
V = lyapunov_from_constraints(tank)

# Generate sample states
import random
samples = [
    {"temperature": t, "ambient": 20.0}
    for t in [random.uniform(0.0, 200.0) for _ in range(100)]
]

# Analyse
analyzer = ConstraintStabilityAnalyzer(tank)
cert = analyzer.analyze(V, states=samples)

print(f"Stable:             {cert.stable}")
print(f"Asymptotically:     {cert.asymptotic}")
print(f"Decay rate:         {cert.rate:.4f}")
print(f"Samples evaluated:  {cert.samples}")
print(f"Admissible fraction:{cert.admissible_fraction:.2%}")
```

Output:

```
Stable:              True
Asymptotically:      True
Decay rate:          0.1001
Samples evaluated:   100
Admissible fraction: 100.00%
```

The cooling tank is asymptotically stable with decay rate ~0.1 — matching the flow's coefficient `-0.1`.

Now test an **unstable** system:

```python
unstable = COHObject("unstable")
unstable.add_attribute(Attribute("x", 1.0, bounds=(-100.0, 100.0)))
unstable.flow = ContinuousFlow(
    f=lambda s, a, th: {"x": 0.5 * s["x"]},  # diverges
    variables=["x"],
)

from gismol.dynamical.stability import quadratic_lyapunov

V = quadratic_lyapunov({"x": 0.0})
cert = analyzer.__class__(unstable).analyze(V, states=[{"x": v} for v in [1.0, 2.0, 3.0]])

print(f"Stable: {cert.stable}")
print(f"Worst derivative: {cert.worst_derivative:.4f}")
```

Output:

```
Stable: False
Worst derivative: 4.0000
```

The derivative is positive — V is growing — so the system is unstable.

### Exercises

1. Attach a restoring flow to a variable and verify stability.
2. Take a system with two goals of opposite sign. Is it stable? Why or why not?
3. Build a system that is stable inside a bounded region but unstable outside. Can you find a barrier that separates them?

### Key takeaways

- **Stability is provable, not just testable.** Lyapunov functions give you a certificate.
- **COH provides natural Lyapunov functions.** Constraints → violations; goals → negative score.
- **Instability is diagnosable.** `worst_derivative > 0` tells you exactly where the system escapes.

---

# Part IV — Advanced Composition

> *You now have the tools to build smart systems. This part teaches you to build them from other systems, learn under constraints, and reason about them in natural language.*

---

## Lesson 13 · Category Theory Composition

### Learning objectives

- Compose two systems as a product
- Use coproducts to model alternative behaviours
- Understand exponentials as function objects
- Verify that a transformation preserves structure

### Concept

The class of COH objects forms a **category**. In plain English, this means you can compose systems and the result is still a system — with predictable properties.

Three fundamental compositions:

**Product** `O₁ × O₂` — parallel composition. Both systems run side by side. Their attributes merge; their constraints conjoin. This is what you use when a system has multiple independent subsystems.

**Coproduct** `O₁ + O₂` — alternative composition. One system or the other runs. This is what you use for mode switching.

**Exponential** `O₂^O₁` — function object. A system that takes an `O₁` and produces an `O₂`. This is what you use for meta-level transformations.

**Why this matters.** If you have three systems and want to combine them, you do not need to invent a new class. `a.product(b).product(c)` is a system. Composition is closed.

### Practice

Combine two systems:

```python
from gismol import COHObject, Attribute

# Two independent systems
sensor = COHObject("sensor")
sensor.add_attribute(Attribute("reading", 0.0))

actuator = COHObject("actuator")
actuator.add_attribute(Attribute("output", 0.0))

# Product — both run together
combined = sensor.product(actuator)
print(f"Product children: {list(combined.C)}")
print(f"Product attributes: {list(combined.A)}")
# Product children: ['sensor', 'actuator']
# Product attributes: ['reading', 'output']

# Coproduct — alternative behaviours
either = sensor.coproduct(actuator)
print(f"Coproduct children: {list(either.C)}")
# Coproduct children: ['sensor', 'actuator']

# Exponential — function object
transformer = actuator.exponential(sensor)
print(f"Exponential methods: {list(transformer.M)}")
# Exponential methods: ['apply']
```

Now verify **structure preservation**:

```python
from gismol.categorical import COHMorphism

# Two compatible systems
small = COHObject("small", attributes=[Attribute("x", 1.0)])
large = COHObject("large", attributes=[Attribute("x", 1.0)])

mor = COHMorphism(source=small, target=large, name="upgrade")
print(f"Preserves: {mor.preserves()}")
print(f"Violations: {mor.violations}")

# Try to lose structure
incompatible = COHObject("incompatible")  # no x
try:
    bad = COHMorphism(small, incompatible)
except Exception as e:
    print(f"Rejected: {type(e).__name__}")
```

### Exercises

1. Build a **robot** with two arms. Model each arm as a `COHObject` and use `product()` to compose them.
2. Model a **switching controller** as a coproduct of two alternative policies.
3. Build a system that takes another system as input, transforms it, and produces a third. Use `exponential()`.

### Key takeaways

- **COH objects compose cleanly.** Product, coproduct, exponential.
- **Composition respects constraints.** The product's constraints are the conjunction of the operands'.
- **Structure-preserving maps are verified.** Morphisms refuse to lose information.

---

## Lesson 14 · Learning Under Constraints

### Learning objectives

- Formulate a constrained learning problem
- Use primal-dual optimisation
- Diagnose when a policy violates constraints
- Balance reward and safety

### Concept

Standard reinforcement learning maximises reward. But real systems have *constraints* — the reward cannot come at any cost. Constrained reinforcement learning solves:

$$
\max_{\pi} \; \mathbb{E}[G(\tau)] \quad \text{s.t.} \quad \mathbb{E}[C_i(\tau)] \le 0 \; \forall i
$$

The solution uses **Lagrangian duality**: introduce multipliers `λ_i ≥ 0`, and solve the relaxed problem:

$$
\max_{\pi} \; \mathbb{E}[G(\tau)] - \sum_i \lambda_i \, \mathbb{E}[C_i(\tau)]
$$

You alternate: update the policy to maximise the relaxed objective; update the multipliers to increase the penalty when constraints are violated. This is **primal-dual optimisation**.

GISMOL implements this via `LagrangianAgent`.

### Practice

Train a policy that maximises return while respecting a cost budget:

```python
from gismol.neural.constrained_rl import (
    ConstrainedMDP, LagrangianAgent, LinearSoftmaxPolicy,
)

# A CMDP with 6 states and 3 actions
env = ConstrainedMDP(
    n_states=6,
    n_actions=3,
    cost_limit=0.4,     # average cost per episode must be ≤ 0.4
    horizon=15,
    seed=42,
)

agent = LagrangianAgent(env, lr_theta=0.05, lr_lambda=0.1, seed=0)

# Train
history = agent.train(iterations=200, episodes_per_iter=8, verbose=False)

# Track the trajectory of return, cost, and lambda
for i in [0, 50, 100, 150, 199]:
    h = history[i]
    print(f"iter {i:3d}: return={h['return']:+.3f}  cost={h['cost']:.3f}  "
          f"lambda={h['lambda']:.3f}  violation={h['violation']:+.3f}")
```

Output (approximate):

```
iter   0: return=+0.123  cost=0.512  lambda=0.000  violation=+0.112
iter  50: return=+0.089  cost=0.431  lambda=0.098  violation=+0.031
iter 100: return=+0.054  cost=0.402  lambda=0.245  violation=+0.002
iter 150: return=+0.041  cost=0.398  lambda=0.412  violation=-0.002
iter 199: return=+0.038  cost=0.395  lambda=0.578  violation=-0.005
```

Notice the trade-off:

- Initially, the policy ignores the constraint (lambda ≈ 0) and its cost is too high.
- As training proceeds, lambda increases, penalising constraint violations.
- By the end, cost is just below the limit, and return is lower but *feasible*.

This is the essence of constrained learning: **give up some reward to gain feasibility**.

### Evaluate

```python
metrics = agent.evaluate(episodes=100)
print(f"Mean return:        {metrics['mean_return']:+.3f}")
print(f"Mean cost:          {metrics['mean_cost']:.3f}")
print(f"Feasible fraction:  {metrics['feasible_fraction']:.1%}")
```

### Exercises

1. Change `cost_limit` to 0.2 and retrain. How does the policy adapt?
2. Add a second constraint to the CMDP. How does the Lagrangian scale?
3. Train with a very high `lr_lambda`. What happens? Is it stable?

### Key takeaways

- **Constrained learning is a trade-off.** You pay some reward for feasibility.
- **The Lagrangian multiplier is interpretable.** It quantifies the "price" of the constraint.
- **Primal-dual converges.** With convex costs, it reaches a feasible solution.

---

## Lesson 15 · The Natural Language Interface

### Learning objectives

- Compile English text into a COH object
- Use the chat pipeline for interactive modelling
- Recognise the pipeline's strengths and limitations

### Concept

The NLP front-end is a **compiler**: it turns English into a COH object.

Pipeline stages:

1. **Normalise** — lowercase, unicode, whitespace
2. **Parse** — tokens, sentences, POS tags
3. **Extract entities** — numbers, dates, identifiers, keywords
4. **Mine relations** — `is_a`, `has`, `requires`, `controls`, `causes`, …
5. **Recognise intent** — `define`, `constrain`, `optimize`, `simulate`, …
6. **Compile constraints** — text → I/T/G objects
7. **Attach defaults** — daemons, neural components, embedder
8. **Build** — assemble the hierarchy

The pipeline is **rule-based**, not ML-based. It handles clear declarative English well and complex prose poorly. For critical systems, build programmatically and use NLP only for the outer, flexible parts.

### Practice

Compile a description into a system:

```python
from gismol.nlp import text_to_coh

text = """
Define a reactor called R1.
The core temperature must be less than 900.
The coolant flow must be at least 5.
If temperature exceeds 800 then increase coolant.
Maximize energy output.
Minimize coolant usage.
"""

reactor = text_to_coh(text, name="R1")

print(reactor.describe())
```

Output:

```
R1  {'C': 0, 'A': 3, 'M': 1, 'N': 1, 'E': 'BagOfWordsEmbedder',
      'I': 2, 'T': 1, 'G': 2, 'D': 2}
```

Inspect what was extracted:

```python
import json
print(json.dumps(reactor.metadata["entities"], indent=2))
print(json.dumps(reactor.metadata["relations"], indent=2))
```

You will see the numbers, identifiers, and relations the pipeline mined.

Now interact through the **chat pipeline**:

```python
from gismol.nlp import ChatModule

chat = ChatModule()
print(chat.process("define a system called pump").text)
# Created COH object 'pump' with N attributes and M components.

print(chat.process("the pressure must be less than 200").text)
# Invariant inv_pressure_2: (invariant)

print(chat.process("if pressure exceeds 180 then release").text)
# Trigger trigger_...: on 'exceeds' -> release

print(chat.process("maximize throughput").text)
# Registered 1 objective(s).

print(chat.process("describe it").text)
# pump: N attributes, 1 method, ...
```

Every response carries structured data:

```python
r = chat.process("simulate 50 steps")
print(f"Intent: {r.intent.name}")
print(f"Slots:  {r.intent.slots}")
print(f"Text:   {r.text}")
```

### Exercises

1. Write a text description of your own system. Compile it. What did the pipeline get right? What did it get wrong?
2. Add a custom entity pattern (`add_pattern("VERSION", r"v\d+\.\d+")`) and confirm it is extracted.
3. Extend the pipeline with a custom relation pattern for a domain-specific relation.

### Key takeaways

- **NLP is a compiler, not magic.** The pipeline is deterministic and rule-based.
- **The chat pipeline is stateful.** It remembers the current COH object across turns.
- **Always inspect the output.** The `metadata` field shows exactly what was extracted.

---

## Lesson 16 · Reasoners

### Learning objectives

- Query a COH object with different reasoners
- Diagnose constraint violations
- Plan actions with the goal solver
- Understand when to use which reasoner

### Concept

A **reasoner** answers a query about a COH object. Different queries need different reasoners.

| Reasoner | Answers |
|---|---|
| `ConstraintReasoner` | "Is this state admissible? How to repair?" |
| `GoalSolver` | "What method sequence maximises goals?" |
| `ForwardChainer` | "What facts follow from the current state?" |
| `AbductiveReasoner` | "What would explain this violation?" |
| `CausalReasoner` | "What causes what?" |
| `AnalogicalReasoner` | "How similar are these systems?" |
| `DomainReasoner` | "How does this behave under domain rules?" |
| `MetaReasoner` | "Which reasoner should I use?" |

The **`GeneralReasoner`** classifies your query and dispatches to the right specialist.

### Practice

Build a **pump** and query it:

```python
from gismol import COHObject, Attribute, Method
from gismol.core.constraints import (
    IdentityConstraint, TriggerConstraint, GoalConstraint,
)
from gismol.reasoners import (
    ConstraintReasoner, GoalSolver, GeneralReasoner, Query,
)

pump = COHObject("pump")
pump.add_attribute(Attribute("pressure", 100.0, bounds=(0.0, 300.0)))
pump.add_attribute(Attribute("flow", 50.0, bounds=(0.0, 100.0)))
pump.add_attribute(Attribute("temperature", 40.0, bounds=(0.0, 200.0)))
pump.add_attribute(Attribute("target_pressure", 150.0))

pump.add_method(Method("increase", lambda s, a=None: ({
    **s, "pressure": min(300.0, s["pressure"] + 10.0),
}, 0.0)))

pump.add_method(Method("cool", lambda s, a=None: ({
    **s, "temperature": max(0.0, s["temperature"] - 5.0),
}, 0.0)))

pump.add_identity_constraint(IdentityConstraint("p_limit", "pressure < 250"))
pump.add_identity_constraint(IdentityConstraint("t_limit", "temperature < 150"))
pump.add_trigger_constraint(TriggerConstraint(
    "hot", "temp_high", "temperature > 100", "cool", priority=5,
))
pump.add_goal_constraint(GoalConstraint(
    "target", lambda s: -abs(s["pressure"] - s["target_pressure"]),
    direction="max",
))

# Query 1: check invariants
result = ConstraintReasoner().infer(pump, Query("check invariants"))
print(f"All OK? {result.answer is True}")

# Query 2: violate and repair
pump.set("pressure", 300.0)
result = ConstraintReasoner().infer(pump, Query("violations"))
print(f"Violations: {result.answer['violations']}")
print(f"Repairs:    {result.answer['repairs']}")

# Query 3: plan to reach target
pump.set("pressure", 100.0)
result = GoalSolver(depth=3, beam=4).infer(pump, Query("best plan"))
print(f"Plan: {result.answer['plan']}")
print(f"Score: {result.answer['value']:.3f}")

# Query 4: general dispatch
result = GeneralReasoner().infer(pump, Query("maximize pressure"))
print(f"Dispatched to: {result.reasoner}")
print(f"Answer: {result.answer}")
```

Output (approximate):

```
All OK? True
Violations: ['p_limit']
Repairs:    [{'pressure': 240.0}]
Plan: ['increase', 'increase', 'increase']
Score: -0.001
```

### The reasoning trace

Every reasoner produces a trace you can inspect:

```python
result = GeneralReasoner().infer(pump, Query("check violations"))
for step in result.trace:
    print(f"  · {step}")
```

This is essential for debugging — the trace shows exactly what the reasoner considered and why it concluded what it did.

### Domain reasoning

Built-in domain profiles cover physics, biology, finance, smart cities, and aerospace:

```python
from gismol.reasoners import DomainReasoner, get_domain

physics = DomainReasoner(profile=get_domain("physics"))
result = physics.infer(pump, Query("analyse"))
print(f"Domain: {result.answer['domain']}")
print(f"Metrics: {result.answer['metrics']}")
```

### Exercises

1. Add a `forward` method to the pump that increases `flow` by 5. Add a goal to maximise `flow`. Re-run the goal solver.
2. Query the causal reasoner: `Query("temperature", parameters={"intervene": True})`. What are the effects of changing temperature?
3. Write a custom `Rule` and attach it to a `ForwardChainer`.

### Key takeaways

- **Different reasoners answer different questions.** Use `GeneralReasoner` for automatic dispatch.
- **Every result carries a trace.** Traces are essential for debugging.
- **Reasoners compose with the framework.** They use the same constraints, methods, and goals your system already declares.

---

# Part V — Mastery

> *You know the pieces. This part puts them together into a full system and shows you where to go next.*

---

## Lesson 17 · A Complete System

### Learning objectives

- Build a five-layer system that exercises every component
- Validate it thoroughly
- Simulate, reason, and analyse it

### Concept

We have covered all nine components. Now we build a **water treatment plant** — a system with five levels of hierarchy, all nine components, and real safety requirements.

```
water_plant                        (level 0)
├── intake_system                  (level 1)
│   ├── pump_station               (level 2)
│   │   ├── pump_1                 (level 3)
│   │   │   └── motor_control      (level 4)
│   │   └── pump_2
│   │       └── motor_control
│   └── quality_sensor             (level 2)
├── treatment_system
│   ├── filtration                 (level 2)
│   │   ├── filter_bed             (level 3)
│   │   └── backwash_system
│   └── disinfection
│       ├── uv_reactor
│       └── chlorine_doser
└── distribution_system
    ├── pressure_regulator
    └── quality_monitor
```

The system must:

- Keep pressure within safe limits
- Maintain water quality (turbidity, chlorine)
- Avoid overflow in any tank
- Meet demand without wasting energy

### Practice

Here is the full implementation:

```python
import math
from gismol import COHObject, Attribute, Method
from gismol.core.constraints import (
    IdentityConstraint, TriggerConstraint, GoalConstraint,
)
from gismol.core.daemons import (
    InvariantDaemon, BoundaryDaemon, RateDaemon,
)
from gismol.core.daemon_impl import HomeostasisDaemon, BarrierDaemon
from gismol.core.dynamics import ContinuousFlow
from gismol.core.embedding import FeatureEmbedder
from gismol.neural.components import MLPComponent, PolicyComponent
from gismol.nlp.validation import COHValidator
from gismol.reasoners import GeneralReasoner, Query


def build_water_plant() -> COHObject:
    # ---- Level 4: motor control ----
    def motor_control(name):
        m = COHObject(name)
        m.add_attribute(Attribute("rpm", 0.0, bounds=(0.0, 3600.0)))
        m.add_attribute(Attribute("current_a", 0.0, bounds=(0.0, 50.0)))
        m.add_attribute(Attribute("temperature", 25.0, bounds=(-10.0, 90.0)))
        m.add_method(Method(
            "spin",
            lambda s, a=None: ({**s, "rpm": min(3600.0, s["rpm"] + 100.0),
                                "current_a": s["rpm"] / 100.0}, 0.0)))
        m.add_identity_constraint(IdentityConstraint(
            "thermal_safe", "temperature < 80"))
        m.add_identity_constraint(IdentityConstraint(
            "current_safe", "current_a < 45"))
        m.add_trigger_constraint(TriggerConstraint(
            "thermal_trip", "motor_hot", "temperature > 70",
            {"rpm": 0.0}, priority=8))
        m.add_goal_constraint(GoalConstraint(
            "max_rpm", "rpm", direction="max"))
        m.add_daemon(InvariantDaemon(constraints=list(m.I)))
        m.add_daemon(BoundaryDaemon({
            "rpm": (0.0, 3600.0),
            "current_a": (0.0, 45.0),
            "temperature": (-10.0, 80.0),
        }))
        m.add_daemon(RateDaemon("temperature", max_rate=5.0))
        m.add_daemon(HomeostasisDaemon("temperature", setpoint=25.0, gain=0.1))
        m.add_neural(MLPComponent("rpm_ctrl", [3, 8, 1], seed=1), name="controller")
        return m

    # ---- Level 3: pumps ----
    def pump(name, seed):
        p = COHObject(name)
        p.add_attribute(Attribute("flow_lpm", 0.0, bounds=(0.0, 500.0)))
        p.add_attribute(Attribute("pressure_bar", 0.0, bounds=(0.0, 10.0)))
        p.add_child(motor_control(f"{name}_motor"))
        p.add_method(Method(
            "run",
            lambda s, a=None: ({**s,
                                "flow_lpm": min(500.0, s["flow_lpm"] + 50.0),
                                "pressure_bar": min(10.0, s["pressure_bar"] + 0.1)},
                               1.0)))
        p.add_identity_constraint(IdentityConstraint(
            "flow_valid", "0 <= flow_lpm <= 500"))
        p.add_identity_constraint(IdentityConstraint(
            "pressure_safe", "pressure_bar < 8"))
        p.add_trigger_constraint(TriggerConstraint(
            "overpressure", "pressure_high", "pressure_bar > 6",
            {"pressure_bar": 5.0}, priority=9))
        p.add_goal_constraint(GoalConstraint(
            "meet_demand", "flow_lpm", direction="max", weight=2.0))
        p.add_daemon(InvariantDaemon(constraints=list(p.I)))
        p.add_daemon(BoundaryDaemon({
            "flow_lpm": (0.0, 500.0),
            "pressure_bar": (0.0, 8.0),
        }))
        p.add_daemon(BarrierDaemon(
            barrier=lambda s: 8.0 - s["pressure_bar"],
            alpha=lambda h: 2.0 * h,
        ))
        p.add_neural(PolicyComponent(f"policy_{seed}", input_dim=3, n_actions=2, seed=seed),
                     name="control_policy")
        p.set_embedder(FeatureEmbedder(["flow_lpm", "pressure_bar"]))
        return p

    # ---- Level 3: filters ----
    def filter_bed(name):
        f = COHObject(name)
        f.add_attribute(Attribute("turbidity_ntu", 0.5, bounds=(0.0, 10.0)))
        f.add_attribute(Attribute("clogged_pct", 0.0, bounds=(0.0, 100.0)))
        f.add_method(Method(
            "filter",
            lambda s, a=None: ({**s, "clogged_pct": min(100.0, s["clogged_pct"] + 0.05)}, 0.5)))
        f.add_method(Method(
            "backwash",
            lambda s, a=None: ({**s, "clogged_pct": 0.0}, -0.5)))
        f.add_identity_constraint(IdentityConstraint(
            "turbidity_ok", "turbidity_ntu < 5"))
        f.add_trigger_constraint(TriggerConstraint(
            "auto_backwash", "clog_high", "clogged_pct > 80",
            "backwash", priority=7))
        f.add_goal_constraint(GoalConstraint(
            "clean_water", "turbidity_ntu", direction="min"))
        f.add_daemon(InvariantDaemon(constraints=list(f.I)))
        f.add_daemon(BoundaryDaemon({
            "turbidity_ntu": (0.0, 10.0),
            "clogged_pct": (0.0, 100.0),
        }))
        return f

    # ---- Level 3: disinfection ----
    def uv_reactor(name):
        u = COHObject(name)
        u.add_attribute(Attribute("dose_mj_cm2", 40.0, bounds=(0.0, 100.0)))
        u.add_attribute(Attribute("lamp_hours", 0.0, bounds=(0.0, 10000.0)))
        u.add_method(Method(
            "operate",
            lambda s, a=None: ({**s, "lamp_hours": s["lamp_hours"] + 0.1}, 0.5)))
        u.add_identity_constraint(IdentityConstraint(
            "dose_adequate", "dose_mj_cm2 > 22"))
        u.add_goal_constraint(GoalConstraint(
            "max_dose", "dose_mj_cm2", direction="max"))
        u.add_daemon(InvariantDaemon(constraints=list(u.I)))
        u.add_daemon(BoundaryDaemon({"dose_mj_cm2": (22.0, 100.0)}))
        return u

    def chlorine_doser(name):
        c = COHObject(name)
        c.add_attribute(Attribute("residual_mg_l", 0.5, bounds=(0.0, 5.0)))
        c.add_attribute(Attribute("tank_level_pct", 80.0, bounds=(0.0, 100.0)))
        c.add_method(Method(
            "dose",
            lambda s, a=None: ({**s, "residual_mg_l": min(5.0, s["residual_mg_l"] + 0.02)}, 0.5)))
        c.add_identity_constraint(IdentityConstraint(
            "residual_safe", "residual_mg_l < 4"))
        c.add_identity_constraint(IdentityConstraint(
            "residual_adequate", "residual_mg_l > 0.2"))
        c.add_goal_constraint(GoalConstraint(
            "target_residual", lambda s: -abs(s["residual_mg_l"] - 1.0),
            direction="max"))
        c.add_daemon(InvariantDaemon(constraints=list(c.I)))
        c.add_daemon(BoundaryDaemon({
            "residual_mg_l": (0.2, 4.0),
            "tank_level_pct": (5.0, 95.0),
        }))
        return c

    # ---- Level 2: subsystems ----
    pump_station = COHObject("pump_station")
    pump_station.add_attribute(Attribute("total_flow_lpm", 0.0, bounds=(0.0, 1000.0)))
    pump_station.add_child(pump("pump_1", seed=10), key="pump_1")
    pump_station.add_child(pump("pump_2", seed=11), key="pump_2")
    pump_station.add_method(Method(
        "distribute_load",
        lambda s, a=None: ({**s, "total_flow_lpm": min(1000.0, s["total_flow_lpm"] + 100.0)}, 1.0)))
    pump_station.add_identity_constraint(IdentityConstraint(
        "flow_within_capacity", "total_flow_lpm <= 1000"))
    pump_station.add_goal_constraint(GoalConstraint(
        "max_flow", "total_flow_lpm", direction="max", weight=2.0))
    pump_station.add_daemon(InvariantDaemon(constraints=list(pump_station.I)))
    pump_station.add_daemon(BoundaryDaemon({"total_flow_lpm": (0.0, 1000.0)}))

    quality_sensor = COHObject("quality_sensor")
    quality_sensor.add_attribute(Attribute("ph", 7.0, bounds=(0.0, 14.0)))
    quality_sensor.add_attribute(Attribute("conductivity_us_cm", 500.0, bounds=(0.0, 2000.0)))
    quality_sensor.add_method(Method("sample", lambda s, a=None: (s, 0.0)))
    quality_sensor.add_identity_constraint(IdentityConstraint(
        "ph_ok", "6.5 <= ph <= 8.5"))
    quality_sensor.add_goal_constraint(GoalConstraint(
        "target_ph", lambda s: -abs(s["ph"] - 7.2), direction="max"))
    quality_sensor.add_daemon(InvariantDaemon(constraints=list(quality_sensor.I)))
    quality_sensor.add_neural(MLPComponent("anomaly", [2, 8, 1], seed=12), name="anomaly_model")

    filtration = COHObject("filtration")
    filtration.add_attribute(Attribute("throughput_m3h", 100.0, bounds=(0.0, 500.0)))
    filtration.add_child(filter_bed("filter_bed_1"))
    filtration.add_child(filter_bed("filter_bed_2"))
    filtration.add_method(Method(
        "operate",
        lambda s, a=None: ({**s, "throughput_m3h": max(0.0, s["throughput_m3h"] - 0.1)}, 0.5)))
    filtration.add_goal_constraint(GoalConstraint(
        "max_throughput", "throughput_m3h", direction="max"))
    filtration.add_daemon(InvariantDaemon(constraints=list(filtration.I)))
    filtration.add_daemon(BoundaryDaemon({"throughput_m3h": (0.0, 500.0)}))

    disinfection = COHObject("disinfection")
    disinfection.add_attribute(Attribute("contact_time_min", 30.0, bounds=(0.0, 120.0)))
    disinfection.add_child(uv_reactor("uv_reactor"))
    disinfection.add_child(chlorine_doser("chlorine_doser"))
    disinfection.add_method(Method(
        "operate",
        lambda s, a=None: ({**s, "contact_time_min": max(15.0, s["contact_time_min"] - 0.1)}, 0.5)))
    disinfection.add_goal_constraint(GoalConstraint(
        "contact_time", "contact_time_min", direction="min", weight=0.5))
    disinfection.add_daemon(InvariantDaemon(constraints=list(disinfection.I)))

    intake = COHObject("intake_system")
    intake.add_attribute(Attribute("inlet_flow_lpm", 500.0, bounds=(0.0, 1000.0)))
    intake.add_child(pump_station)
    intake.add_child(quality_sensor)
    intake.add_method(Method(
        "draw_water",
        lambda s, a=None: ({**s, "inlet_flow_lpm": min(1000.0, s["inlet_flow_lpm"] + 10.0)}, 0.0)))
    intake.add_identity_constraint(IdentityConstraint(
        "inlet_valid", "0 <= inlet_flow_lpm <= 1000"))
    intake.add_goal_constraint(GoalConstraint(
        "meet_intake", "inlet_flow_lpm", direction="max"))

    treatment = COHObject("treatment_system")
    treatment.add_attribute(Attribute("treatment_efficiency", 0.95, bounds=(0.0, 1.0)))
    treatment.add_child(filtration)
    treatment.add_child(disinfection)
    treatment.add_method(Method(
        "process",
        lambda s, a=None: ({**s, "treatment_efficiency": min(1.0, s["treatment_efficiency"] + 0.001)}, 0.0)))
    treatment.add_identity_constraint(IdentityConstraint(
        "efficiency_ok", "treatment_efficiency > 0.9"))
    treatment.add_goal_constraint(GoalConstraint(
        "max_efficiency", "treatment_efficiency", direction="max", weight=2.0))
    treatment.add_daemon(InvariantDaemon(constraints=list(treatment.I)))
    treatment.add_daemon(BoundaryDaemon({"treatment_efficiency": (0.9, 1.0)}))

    distribution = COHObject("distribution_system")
    distribution.add_attribute(Attribute("outlet_pressure_bar", 3.0, bounds=(0.0, 10.0)))
    distribution.add_attribute(Attribute("demand_m3h", 80.0, bounds=(0.0, 500.0)))
    distribution.add_attribute(Attribute("service_pct", 100.0, bounds=(0.0, 100.0)))
    distribution.add_method(Method(
        "supply",
        lambda s, a=None: ({**s,
                            "outlet_pressure_bar": min(8.0, s["outlet_pressure_bar"] + 0.05),
                            "service_pct": max(50.0, s["service_pct"] - 0.01)},
                           0.0)))
    distribution.add_identity_constraint(IdentityConstraint(
        "pressure_ok", "2 <= outlet_pressure_bar <= 7"))
    distribution.add_identity_constraint(IdentityConstraint(
        "service_ok", "service_pct >= 90"))
    distribution.add_trigger_constraint(TriggerConstraint(
        "low_pressure", "pressure_low", "outlet_pressure_bar < 2.5",
        "boost_pressure", priority=6))
    distribution.add_goal_constraint(GoalConstraint(
        "meet_demand", "service_pct", direction="max", weight=3.0))
    distribution.add_goal_constraint(GoalConstraint(
        "min_pressure_waste", "outlet_pressure_bar", direction="min"))
    distribution.add_daemon(InvariantDaemon(constraints=list(distribution.I)))
    distribution.add_daemon(BoundaryDaemon({
        "outlet_pressure_bar": (2.0, 7.0),
        "service_pct": (90.0, 100.0),
    }))
    distribution.add_daemon(HomeostasisDaemon("outlet_pressure_bar", 4.0, gain=0.2))

    # ---- Level 0: the whole plant ----
    plant = COHObject("water_plant", description="Municipal water treatment facility")
    plant.add_attribute(Attribute("total_production_m3h", 0.0, bounds=(0.0, 1000.0)))
    plant.add_attribute(Attribute("energy_kwh", 0.0, bounds=(0.0, 10000.0)))
    plant.add_attribute(Attribute("quality_score", 1.0, bounds=(0.0, 1.0)))
    plant.add_attribute(Attribute("uptime_pct", 100.0, bounds=(0.0, 100.0)))

    plant.add_child(intake, key="intake")
    plant.add_child(treatment, key="treatment")
    plant.add_child(distribution, key="distribution")

    plant.add_method(Method(
        "operate_cycle",
        lambda s, a=None: ({**s,
                            "total_production_m3h": min(1000.0, s["total_production_m3h"] + 5.0),
                            "energy_kwh": s["energy_kwh"] + 2.0,
                            "uptime_pct": max(90.0, s["uptime_pct"] - 0.001)},
                           1.0)))

    plant.add_identity_constraint(IdentityConstraint(
        "production_valid", "0 <= total_production_m3h <= 1000"))
    plant.add_identity_constraint(IdentityConstraint(
        "quality_maintained", "quality_score > 0.85"))
    plant.add_identity_constraint(IdentityConstraint(
        "uptime_acceptable", "uptime_pct >= 90"))

    plant.add_trigger_constraint(TriggerConstraint(
        "quality_alert", "quality_low", "quality_score < 0.9",
        "enter_preservation_mode", priority=10))
    plant.add_trigger_constraint(TriggerConstraint(
        "energy_alert", "energy_high", "energy_kwh > 8000",
        "reduce_production", priority=5))

    plant.add_goal_constraint(GoalConstraint(
        "max_production", "total_production_m3h", direction="max", weight=3.0))
    plant.add_goal_constraint(GoalConstraint(
        "min_energy", "energy_kwh", direction="min", weight=1.0))
    plant.add_goal_constraint(GoalConstraint(
        "max_quality", "quality_score", direction="max", weight=2.0))
    plant.add_goal_constraint(GoalConstraint(
        "max_uptime", "uptime_pct", direction="max", weight=2.0))

    plant.add_daemon(InvariantDaemon(constraints=list(plant.I)))
    plant.add_daemon(BoundaryDaemon({
        "total_production_m3h": (0.0, 1000.0),
        "energy_kwh": (0.0, 10000.0),
        "quality_score": (0.0, 1.0),
        "uptime_pct": (90.0, 100.0),
    }))
    plant.add_daemon(BarrierDaemon(
        barrier=lambda s: 9000.0 - s["energy_kwh"],
        alpha=lambda h: 0.1 * h,
    ))

    plant.add_neural(MLPComponent("plant_planner", [4, 32, 4], seed=42), name="planner")
    plant.add_neural(MLPComponent("quality_predictor", [4, 16, 1], seed=43), name="quality_model")

    plant.set_embedder(FeatureEmbedder(
        ["total_production_m3h", "energy_kwh", "quality_score", "uptime_pct"],
        ranges={"total_production_m3h": (0, 1000), "energy_kwh": (0, 10000),
                "quality_score": (0, 1), "uptime_pct": (90, 100)},
    ))

    plant.flow = ContinuousFlow(
        f=lambda s, a, th: {
            "energy_kwh": 0.5,
            "quality_score": -0.0001,
        },
        variables=["energy_kwh", "quality_score"],
    )

    return plant


# Build and inspect
plant = build_water_plant()

print("=" * 70)
print(f"Hierarchy depth: {len(plant.topological_order())} nodes")
print(f"Total descendants: {len(plant.descendants())}")
print(f"Component report: {plant.component_report()}")
print("=" * 70)

# Validate
validator = COHValidator()
issues = validator.validate(plant)
errors = [i for i in issues if i.severity == "error"]
print(f"Validator errors: {len(errors)}")
for e in errors[:3]:
    print(f"  ! {e}")

# Soundness
report = plant.soundness_report()
print(f"Sound: {report['sound']}")
print(f"Acyclic: {plant.is_acyclic()}")
print(f"Is minimal: {plant.is_minimal()}")

# Simulate
print("\nSimulating 100 steps...")
traj = plant.simulate(steps=100, dt=0.1)
print(f"Final production: {traj.final['total_production_m3h']:.1f} m³/h")
print(f"Final energy:     {traj.final['energy_kwh']:.1f} kWh")
print(f"Final quality:    {traj.final['quality_score']:.3f}")
print(f"Final uptime:     {traj.final['uptime_pct']:.1f}%")
print(f"Total reward:     {traj.cumulative_reward():.2f}")

# Reason
print("\nReasoning:")
result = GeneralReasoner().infer(plant, Query("maximize production"))
print(f"  Answer: {result.answer}")
print(f"  Confidence: {result.confidence:.2f}")
print(f"  Trace:")
for step in result.trace[:5]:
    print(f"    · {step}")
```

### What You Should Observe

1. **The hierarchy goes five levels deep** — plant → intake/treatment/distribution → pump_station/filtration/disinfection → pump_1/pump_2/filter_bed_1/filter_bed_2/uv_reactor/chlorine_doser → motor_control.
2. **Every component is populated.** Nine out of nine.
3. **No validator errors.** The plant is structurally and semantically sound.
4. **Simulation proceeds without violations.** Safety daemons keep values in range.
5. **The reasoner provides a plan.** The goal solver finds a sequence of methods.

This is what a production GISMOL system looks like.

### Exercises

1. Change the target outlet pressure from 4.0 to 5.0 bar. How does the homeostasis daemon's behaviour change?
2. Remove the barrier daemon from the plant. Does the energy flow diverge?
3. Compute the plant's embedding and find the two most similar subsystems. What makes them similar?

### Key takeaways

- **You can now build real systems.** Five levels, nine components, verified.
- **Composition is the key.** Each subsystem is built independently; the plant is the composition.
- **Validation is not optional.** `soundness_report`, `COHValidator`, and `component_report` catch 90% of bugs before simulation.

---

## Lesson 18 · Where to Go Next

### Learning objectives

- Identify your next learning steps
- Choose an application domain to explore
- Contribute back to the project

### Choose Your Path

You have finished the tutorial. You now know:

- The theory behind COH and why it exists
- How to build hierarchies, declare attributes, write methods
- How to write identity constraints, triggers, and goals
- How to attach daemons and enforce safety
- How to add embeddings and neural components
- How to run hybrid simulations
- How to prove stability
- How to compose systems categorically
- How to learn under constraints
- How to compile text into COH systems
- How to reason over your systems
- How to build a complete five-level system

**Where you go next depends on what you want to do.**

### If you want to build applications

Pick a domain you know. Some good candidates:

- **Robotics** — a robot arm, a drone, a mobile manipulator
- **Control systems** — a cruise controller, a HVAC system, a chemical process
- **Healthcare** — a medication reminder, an activity tracker, a triage assistant
- **Finance** — a portfolio rebalancer, an execution algorithm, a risk monitor
- **Energy** — a microgrid controller, a load balancer, a demand-response system
- **Education** — an adaptive tutor, a curriculum planner, a skill tracker

For each, follow the pattern from Lesson 17:

1. Design the hierarchy on paper first.
2. Declare attributes and methods.
3. Write constraints (I, T, G).
4. Attach daemons.
5. Add neural components where learning is needed.
6. Simulate.
7. Validate.
8. Iterate.

### If you want to understand the theory deeper

Read the COH theory paper (the source PDF for this tutorial). Pay attention to:

- **§3 Category theory** — products, coproducts, exponentials, functors, limits
- **§4 Dynamical systems** — existence/uniqueness, Lyapunov stability, free energy
- **§5 Learning** — universal approximation, constrained RL, hierarchical credit assignment
- **§6 Soundness** — constraint soundness, hierarchical soundness, minimality
- **§7 Applications** — six domains, one framework

The paper's theorems correspond directly to GISMOL implementations. Read one theorem and then find the code that implements it.

### If you want to extend the toolkit

Good first contributions:

- **New daemons.** Encapsulate a common monitoring pattern.
- **New domain profiles.** `DomainReasoner` supports arbitrary vocabularies and rules.
- **New reasoners.** Subclass `Reasoner`, implement `infer`.
- **New embedders.** Subclass `Embedder`, implement `__call__`.
- **New neural architectures.** Subclass `NeuralComponent`.

See `CONTRIBUTING.md` for the workflow.

### If you want to teach

The tutorial you just read is designed as a course. You can teach it in:

- **A 5-session workshop** — one session per part
- **A 15-week semester course** — one lesson per week with assignments
- **A self-paced module** — students follow the tutorial and submit one system per part

The exercises are graded. The systems are buildable in groups. The theory is examinable.

### Common Traps to Avoid

As you continue, watch for these:

**Trap 1 · Over-constraining.** If every desirable state is forbidden, the system cannot act. Constraints should be *necessary* — safety, legality, physics — not stylistic.

**Trap 2 · Under-monitoring.** Constraints without daemons are documentation. If a constraint matters, enforce it.

**Trap 3 · Neural components without supervision.** Learned policies are powerful but not trustworthy. Pair every neural component with a constraint or daemon.

**Trap 4 · Deep hierarchies for shallow problems.** If three levels suffice, do not build ten. Depth is expensive.

**Trap 5 · Ignoring the embedder.** Every system should have one. Similarity is too useful to skip.

**Trap 6 · Skipping validation.** Run `soundness_report` and `COHValidator` before simulating. It saves hours of debugging.

**Trap 7 · Confusing goals with constraints.** Performance objectives are not safety requirements. Use the right type.

**Trap 8 · Abandoning the theory.** When your system misbehaves, look at the constraint hierarchy, not the code. The structure usually explains the behaviour.

### Where to Get Help

- **Documentation** — README, Developer Guide, API reference (in `docs/`)
- **Examples** — `examples/systems_demo.py` has five complete systems
- **GitHub Discussions** — ask usage questions
- **GitHub Issues** — report bugs and request features
- **The COH theory paper** — for theoretical questions

### Closing Thought

You now understand that **intelligence is constrained hierarchical optimisation.** Everything you have learned — objects, attributes, methods, constraints, daemons, embeddings, neural components, dynamics, category theory — is an implementation of that one idea.

Every system you build with GISMOL is an instance of that idea. Every simulation is a trajectory through admissible state. Every plan is a path that respects constraints. Every daemon is a guardian of admissibility.

Build well. Stay within bounds. Pursue your goals.

---

## Appendix A · Learning Path Summary

| Part | Lessons | Key skill |
|:-:|:-:|---|
| I | 1–3 | Understand *why* COH exists and what each component means |
| II | 4–8 | Build hierarchies, declare state, write constraints, attach daemons |
| III | 9–12 | Add embeddings, train neural components, simulate dynamics, prove stability |
| IV | 13–16 | Compose with category theory, learn under constraints, use NLP, reason |
| V | 17–18 | Build a complete system and choose where to go next |

---

## Appendix B · Code Cheat Sheet

```python
# ---- Build a COHObject ----
from gismol import COHObject, Attribute, Method
from gismol.core.constraints import (
    IdentityConstraint, TriggerConstraint, GoalConstraint,
)
from gismol.core.daemons import InvariantDaemon, BoundaryDaemon

coh = COHObject("name")
coh.add_attribute(Attribute("x", 0.0, bounds=(0.0, 10.0)))
coh.add_method(Method("do", lambda s, a=None: (s, 0.0)))
coh.add_identity_constraint(IdentityConstraint("i", "x >= 0"))
coh.add_trigger_constraint(TriggerConstraint("t", "evt", "x > 5", "do"))
coh.add_goal_constraint(GoalConstraint("g", "x", direction="max"))
coh.add_daemon(InvariantDaemon(constraints=list(coh.I)))
coh.add_daemon(BoundaryDaemon({"x": (0.0, 10.0)}))

# ---- Add learning ----
from gismol.neural.components import MLPComponent
coh.add_neural(MLPComponent("nn", [3, 8, 1], seed=0))

# ---- Add embedding ----
from gismol.core.embedding import FeatureEmbedder
coh.set_embedder(FeatureEmbedder(["x"]))

# ---- Add dynamics ----
from gismol.core.dynamics import ContinuousFlow
coh.flow = ContinuousFlow(
    f=lambda s, a, th: {"x": -0.1 * s["x"]},
    variables=["x"],
)

# ---- Simulate ----
traj = coh.simulate(steps=100, dt=0.1)

# ---- Validate ----
print(coh.component_report())
print(coh.soundness_report())
from gismol.nlp.validation import COHValidator
COHValidator().validate(coh)

# ---- Reason ----
from gismol.reasoners import GeneralReasoner, Query
GeneralReasoner().infer(coh, Query("maximize x"))

# ---- Compose ----
combined = coh1.product(coh2)

# ---- Stability ----
from gismol.stability.lyapunov import ConstraintStabilityAnalyzer
ConstraintStabilityAnalyzer(coh).analyze()

# ---- NLP ----
from gismol.nlp import text_to_coh
new_coh = text_to_coh("The temperature must be less than 100.")
```

---

## Appendix C · Glossary

| Term | Meaning |
|---|---|
| **Admissible region** | Set of states where all identity constraints hold |
| **Attribute (A)** | A measurable state variable |
| **Barrier function** | Scalar whose non-negativity certifies safety |
| **COH** | Constrained Object Hierarchy |
| **Coproduct** | Categorical alternative composition |
| **Daemon (D)** | Continuous monitor that enforces invariants |
| **Embedding (E)** | Map from objects to vectors |
| **Exponential** | Categorical function object |
| **Free energy** | Bound on surprise in variational inference |
| **Goal constraint (G)** | Optimisation objective |
| **Hybrid system** | Combining continuous and discrete dynamics |
| **Identity constraint (I)** | Invariant predicate |
| **Lyapunov function** | Scalar decreasing along trajectories |
| **Method (M)** | Executable state transformation |
| **Morphism** | Structure-preserving map |
| **Neural component (N)** | Parameterised adaptive function |
| **Product** | Categorical parallel composition |
| **Soundness** | Internal consistency |
| **State** | Hierarchical value assignment |
| **Trigger constraint (T)** | Event-condition-action rule |
| **9-tuple** | `(C, A, M, N, E, I, T, G, D)` |

---

<p align="center">
  <b>Intelligence is constrained hierarchical optimisation.</b><br>
  <i>Pursue goals. Stay within bounds. Build well.</i>
</p>