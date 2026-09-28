# gismol - General Intelligent Systems Modelling Language

> **A mathematically grounded Python toolkit for Constrained Object Hierarchies (COH) — a unified framework for modelling intelligent systems and artificial general intelligence.**

[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#project-status)
[![Runtime deps: none](https://img.shields.io/badge/runtime%20dependencies-none-success.svg)](#installation)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

---

## Overview

`gismol` is a **pure-Python, zero-dependency toolkit** implementing the **Constrained Object Hierarchies (COH)** framework — a unified mathematical foundation for modelling intelligent systems and artificial general intelligence.

The COH framework formalises intelligence as *constrained hierarchical optimisation*. It draws on theoretical neuroscience (hierarchical cortical organisation, predictive coding, homeostasis), category theory (compositional structure, products, exponentials, limits), and dynamical systems theory (hybrid automata, Lyapunov stability, free-energy minimisation). The result is a single coherent architecture that unifies symbolic reasoning, neural learning, and constraint-driven regulation — while remaining computationally tractable and biologically plausible.

`gismol` gives you a **working implementation** of that theory: hierarchical object modelling, constraint satisfaction, hybrid dynamics, learnable neural components, categorical composition, natural-language compilation, and multi-domain reasoning.

---

## The COH 9-tuple

Every intelligent system in `gismol` is a `COHObject`, defined as a **9-tuple**:

$$
\mathcal{O} = (\mathbf{C},\, \mathbf{A},\, \mathbf{M},\, \mathbf{N},\, \mathbf{E},\, \mathbf{I},\, \mathbf{T},\, \mathbf{G},\, \mathbf{D})
$$

| Symbol | Component | Meaning |
|:------:|-----------|---------|
| **C** | Hierarchy | Nested sub-`COHObject`s forming a directed acyclic graph |
| **A** | Attributes | Measurable state variables (sensors, registers, quantities) |
| **M** | Methods | Executable transformations `S × A_act → S × R` |
| **N** | Neural components | Parameterised adaptive functions `f_θ : X → Y` |
| **E** | Embedding | Map `O → H` into a Hilbert space for similarity & composition |
| **I** | Identity constraints | Invariant predicates — must hold in every admissible state |
| **T** | Trigger constraints | Event-condition-action rules for reactive behaviour |
| **G** | Goal constraints | Optimisation objectives (max/min, multi-objective, weighted) |
| **D** | Constraint daemons | Continuous monitors that enforce invariants in real time |

Every component is **orthogonal** and **minimal**: removing any one of the nine strictly reduces the framework's expressive power.

---

## Features

- **Hierarchical composition** — arbitrary nesting with cycle detection, topological ordering, and depth-first evaluation
- **Rich constraint language** — identity invariants, event-condition-action triggers, and weighted multi-objective goals
- **Real-time daemons** — invariant monitors, boundary projectors, homeostatic regulators, control-barrier supervisors, and precision-weighted prediction-error minimisers
- **Hybrid dynamics** — continuous ODEs (RK4 integration), discrete jumps, and guarded transitions unified in one system
- **Categorical layer** — products, coproducts, and exponentials; functors to `Graph`, `Set`, and `Vect`; limits for constraint satisfaction
- **Neural components** — dependency-free MLP with manual backpropagation, policy/value/predictor components, Adam/SGD/RMSProp optimisers
- **Constrained RL** — CMDPs, primal-dual (Lagrangian) optimisation, hierarchical credit assignment
- **Stability analysis** — Lyapunov certificates, region-of-attraction estimation, exponential decay rates
- **Free-energy correspondence** — variational inference and the FEP mapping described in the theory
- **Natural-language pipeline** — text → entities → relations → constraints → runnable `COHObject`, plus a multi-session chat manager
- **Reasoners** — forward chaining, constraint repair, goal solving, abductive, causal, analogical, categorical, meta, and domain-specialised reasoning
- **Serialisation** — JSON round-trips of full hierarchies with a repository/store abstraction
- **Zero runtime dependencies** — no NumPy, no SciPy, no PyTorch required

---

## Installation

### From PyPI *(coming soon)*

```bash
pip install gismol
```

### From source (recommended for now)

```bash
git clone https://github.com/your-org/gismol.git
cd gismol
pip install -e .
```

### Zero-install verification

Because `gismol` has **no mandatory runtime dependencies**, you can use it directly from a checkout:

```bash
git clone https://github.com/your-org/gismol.git
cd gismol
python -c "import gismol; print(gismol.__version__)"
```

### Optional extras

```bash
pip install -e ".[fast]"      # numpy acceleration
pip install -e ".[plot]"      # networkx + matplotlib + graphviz
pip install -e ".[nlp]"       # sentence-transformers
pip install -e ".[dev]"       # pytest, ruff, mypy, coverage
pip install -e ".[all]"       # everything
```

---

## Quick Start

### 1 · Build a COH object programmatically

```python
from gismol import COHObject, Attribute, Method
from gismol.core.constraints import IdentityConstraint, TriggerConstraint, GoalConstraint
from gismol.core.daemons import InvariantDaemon, BoundaryDaemon
from gismol.neural.components import MLPComponent
from gismol.core.embedding import FeatureEmbedder

# Create the object
thermostat = COHObject("thermostat", description="Room temperature controller")

# A — attributes
thermostat.add_attribute(Attribute("temperature", 20.0, bounds=(0.0, 60.0)))
thermostat.add_attribute(Attribute("heater_power", 0.0, bounds=(0.0, 1.0)))
thermostat.add_attribute(Attribute("target", 22.0))

# M — methods
thermostat.add_method(Method(
    "heat",
    lambda s, a=None: ({**s, "temperature": s["temperature"] + 0.1 * s["heater_power"]}, 0.1),
))

# I — identity constraints
thermostat.add_identity_constraint(
    IdentityConstraint("temp_bounds", lambda s: 0.0 <= s["temperature"] <= 60.0)
)

# T — trigger constraints
thermostat.add_trigger_constraint(TriggerConstraint(
    "overheat", "temperature_high",
    condition=lambda s: s["temperature"] > 50.0,
    action="cool_down", priority=3,
))

# G — goal constraints
thermostat.add_goal_constraint(GoalConstraint(
    "comfort", lambda s: -abs(s["temperature"] - s["target"]), direction="max"
))

# D — daemons
thermostat.add_daemon(InvariantDaemon(constraints=list(thermostat.I)))
thermostat.add_daemon(BoundaryDaemon({"temperature": (0.0, 60.0)}))

# N — neural components
thermostat.add_neural(MLPComponent("controller", [3, 8, 1], seed=42))

# E — embedding
thermostat.set_embedder(FeatureEmbedder(["temperature", "heater_power"]))

# Inspect
print(thermostat.component_report())
# {'C': 0, 'A': 3, 'M': 1, 'N': 1, 'E': 'FeatureEmbedder',
#  'I': 1, 'T': 1, 'G': 1, 'D': 2}
```

### 2 · Compile from natural language

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

print(reactor.describe())
```

### 3 · Simulate the hybrid dynamics

```python
traj = reactor.simulate(steps=100, dt=0.05)
print(f"Final goal value: {traj.cumulative_reward():.4f}")
print(f"Discounted return: {traj.discounted_return(gamma=0.99):.4f}")
```

### 4 · Reason over the model

```python
from gismol.reasoners import GeneralReasoner, Query

reasoner = GeneralReasoner()
result = reasoner.infer(reactor, Query("maximize energy output"))
print(result.answer)
print(reasoner.explain(result))
```

### 5 · Chat with the model

```python
from gismol.nlp import ChatModule

chat = ChatModule(coh=reactor)
print(chat.process("describe the reactor").text)
print(chat.process("what happens if temperature exceeds 800?").text)
print(chat.process("simulate 50 steps").text)
```

---

## Repository Layout

```
gismol/
├── categorical/          # Category-theoretic layer (§3 of the theory)
│   ├── category.py       #   Category, Morphism, COHCategory, COHMorphism
│   ├── functors.py       #   ComponentFunctor, ForgetfulFunctor, EmbeddingFunctor
│   └── limits.py         #   Diagrams, cones, limits, pullbacks, constraint_limit
├── core/                 # The 9-tuple itself
│   ├── coh_object.py     #   ⭐ COHObject — the base class for everything
│   ├── constraints.py    #   IdentityConstraint, TriggerConstraint, GoalConstraint
│   ├── daemons.py        #   Daemon, InvariantDaemon, BoundaryDaemon, …
│   ├── daemon_impl.py    #   HomeostasisDaemon, BarrierDaemon, PrecisionWeighted…
│   ├── dynamics.py       #   ContinuousFlow, DiscreteTransition, HybridDynamicalSystem
│   ├── embedding.py      #   Embedder, FeatureEmbedder, HashEmbedder, HilbertSpace
│   ├── expression.py     #   Safe symbolic expression language (AST-whitelisted)
│   ├── repository.py     #   COHRepository — persistence & indexing
│   └── exceptions.py     #   Full exception hierarchy
├── dynamical/            # Hybrid dynamics + stability + free energy (§4)
│   ├── hybrid.py         #   HybridAutomaton, Mode, Guard, Transition
│   ├── stability.py      #   LyapunovAnalysis, StabilityResult
│   └── free_energy.py    #   GenerativeModel, VariationalPosterior, FreeEnergy
├── neural/               # Adaptive components (§5)
│   ├── nn.py             #   MLP with manual backprop, activations, losses
│   ├── components.py     #   MLPComponent, PolicyComponent, ValueComponent
│   ├── embeddings.py     #   NeuralEmbedder, AutoencoderEmbedder, SiameseEmbedder
│   ├── optimizers.py     #   SGD, Adam, RMSProp, AdaGrad, LagrangianOptimizer
│   ├── constrained_rl.py #   ConstrainedMDP, PrimalDualOptimizer, LagrangianAgent
│   └── credit_assignment.py # HierarchicalValueDecomposition, CreditAssigner
├── nlp/                  # Natural-language front-end
│   ├── normalizers.py    #   Unicode & text normalisation
│   ├── parsers.py        #   Tokenisation, sentence splitting, POS tagging
│   ├── entity_extractor.py #  Regex-based NER
│   ├── relations_miner.py  #  Relation extraction (is_a, causes, requires, …)
│   ├── intent_recognizer.py # Intent classification
│   ├── constraint_parser.py # Text → I / T / G constraints
│   ├── embedders.py      #   BagOfWords, TfIdf, Hashing, CharNGram
│   ├── similarity.py     #   Cosine, Jaccard, Levenshtein, Jaro-Winkler, index
│   ├── response_generator.py # Template-based NL generation
│   ├── text_to_coh.py    #   ⭐ Full text → COHObject compiler
│   ├── chat_module.py    #   Single-turn dialogue pipeline
│   ├── chat_manager.py   #   Multi-session management
│   └── validation.py     #   COHValidator — structural & semantic checks
├── reasoners/            # Reasoning engines
│   ├── base.py           #   Reasoner, Rule, KnowledgeBase, Query
│   ├── general.py        #   ForwardChainer, ConstraintReasoner, GoalSolver
│   ├── domain.py         #   DomainProfile, DomainReasoner, built-in domains
│   └── advanced.py       #   Categorical, Analogical, Causal, Meta reasoners
├── stability/            # Constraint-specific Lyapunov certificates
│   └── lyapunov.py       #   ConstraintStabilityAnalyzer, certificates
└── tests/                # Full test suite
    └── test_all.py       #   ~120 tests across every submodule
```

---

## Core Concepts

### Hierarchical composition

`COHObject`s nest arbitrarily deep. Each parent sees its children as named sub-objects, and the whole tree forms a directed acyclic graph:

```python
fleet = COHObject("fleet")
zone  = fleet.add_child(COHObject("picking_zone"))
robot = zone.add_child(COHObject("R2"))
nav   = robot.add_child(COHObject("nav"))

print([o.name for o in fleet.topological_order()])
# ['nav', 'R2', 'picking_zone', 'fleet']
```

Cycles are detected and raise `CyclicHierarchyError`. Soundness checks are built in:

```python
report = fleet.soundness_report()
# {'hierarchical_sound': True, 'sound': True, 'violations': []}
```

### Constraint language

Constraints can be written as plain callables, as `Expr` trees, or as strings parsed by the safe expression engine:

```python
from gismol.core import expression as E

expr = (E.Var("x") > 0) & (E.Var("x") < 10)
print(expr.evaluate({"x": 5}))   # True

# Or the same thing from text:
expr = E.parse_expression("x > 0 and x < 10")
```

No `eval` is ever used — only a whitelisted subset of Python's `ast` is accepted.

### Daemons

Daemons continuously monitor state and emit corrective actions:

| Daemon | Purpose |
|---|---|
| `InvariantDaemon` | Report identity-constraint violations |
| `BoundaryDaemon` | Project state back into a box-shaped admissible region |
| `HomeostasisDaemon` | Set-point regulation (biological homeostasis) |
| `RateDaemon` | Limit the rate of change of a scalar attribute |
| `SafetyDaemon` | Control-barrier-style hazard supervision |
| `BarrierDaemon` | Full CBF forward-invariance check |
| `PrecisionWeightedDaemon` | Precision-weighted prediction-error minimisation |
| `DaemonNetwork` | Composite; highest-severity action wins |

### Hybrid dynamics

`COHObject.step()` and `.simulate()` run a hybrid system with:
1. Trigger evaluation and action execution,
2. Daemon monitoring and correction,
3. Continuous RK4 integration of the flow.

```python
traj = coh.simulate(steps=100, dt=0.01, policy=lambda s, t: "advance")
```

### Category theory

The `COHCategory` exposes the cartesian-closed structure that the paper proves (§3.1):

```python
from gismol.categorical import COHCategory

cat = COHCategory()
cat.register(a); cat.register(b)

product = cat.product(a, b)          # parallel composition
coproduct = cat.coproduct(a, b)      # alternative composition
exponential = cat.exponential(a, b)  # higher-order system

assert cat.is_cartesian_closed()
```

`ComponentFunctor` maps any `COHObject` to its component DAG; `constraint_limit` computes the admissible region as a categorical limit.

### Stability certificates

```python
from gismol.stability import ConstraintStabilityAnalyzer

analyzer = ConstraintStabilityAnalyzer(coh)
cert = analyzer.analyze(states=[s1, s2, s3])
print(cert)
# LyapunovCertificate(asymptotically stable, rate=0.0421, daemons=True)
```

### Constrained RL

```python
from gismol.neural.constrained_rl import ConstrainedMDP, LagrangianAgent

env = ConstrainedMDP(n_states=8, n_actions=4, cost_limit=0.5)
agent = LagrangianAgent(env, seed=0)
agent.train(iterations=200, episodes_per_iter=8)

metrics = agent.evaluate(episodes=100)
# {'mean_return': 3.14, 'mean_cost': 0.42, 'feasible_fraction': 0.98}
```

### NLP pipeline

The full text-to-COH compiler:

```
text
 ├─ normalise
 ├─ parse (tokens, POS, sentences)
 ├─ extract entities  (NUMBER, DATE, MONEY, CONSTRAINT_WORD, …)
 ├─ mine relations    (is_a, has, causes, requires, controls, …)
 ├─ compile constraints (invariants, triggers, goals)
 ├─ attach neural components & daemons
 └─ build COHObject hierarchy
```

```python
from gismol.nlp import TextToCOH

builder = TextToCOH()
coh = builder.build(
    "The pump controls the valve. "
    "Pressure must be less than 200. "
    "If temperature exceeds 80 then cool. "
    "Maximize throughput.",
    name="pump_system",
)
```

---

## Five Worked Systems

The repository ships with five complete, production-style systems — each with **at least three layers of nested `COHObject` hierarchies** — demonstrating the toolkit across domains:

| # | System | Domain | Hierarchy depth | Highlights |
|---|--------|--------|:-:|-----------|
| 1 | **Autonomous warehouse fleet** | Robotics | 4 | Navigation safety barriers, battery homeostasis, per-zone throughput goals |
| 2 | **Personalised healthcare assistant** | Health | 4 | Consent as identity constraint, arrythmia detection, medication adherence |
| 3 | **Smart grid energy management** | Infrastructure | 4 | Frequency ODE, renewable forecasting, N-1 reserve margins, CO₂ budgets |
| 4 | **Adaptive e-learning tutor** | Education | 4 | Multi-objective mastery/affect/load balancing, knowledge tracing |
| 5 | **Autonomous trading desk** | Finance | 4 | Risk-office veto via trigger priority, drawdown caps, CBF-style VaR supervision |

Each system is validated against soundness checks, simulated, and queried with the reasoners. See [`examples/systems_demo.py`](examples/systems_demo.py).

---

## Testing

The toolkit includes a comprehensive test suite covering every submodule — expression parsing, constraint semantics, daemons, hybrid dynamics, embeddings, categorical constructions, stability, free energy, neural components, constrained RL, credit assignment, NLP, and all reasoners.

```bash
# Run the full suite
python -m unittest gismol.tests.test_all -v

# Or with pytest (parallel, coverage)
pytest gismol/tests -n auto --cov=gismol --cov-report=term-missing
```

Example output:

```
Ran 118 tests in 8.421s
OK
```

---

## Design Principles

1. **Zero mandatory runtime dependencies.** Everything you need is in the standard library. Optional extras unlock numpy acceleration, plotting, and neural embeddings, but the toolkit never requires them.
2. **`COHObject` is the universal base class.** Every intelligent system is (a subclass of) a `COHObject` — no exceptions, no special cases.
3. **Explicit is better than implicit.** Nine named components, three constraint families, four daemon categories: the framework's structure is visible in the code.
4. **Safe by construction.** Constraint daemons enforce invariants continuously; cycle detection prevents ill-founded hierarchies; safe expression parsing avoids `eval`.
5. **Mathematically grounded.** Every abstraction maps to a definition or theorem in the underlying paper (§3 category theory, §4 dynamics, §5 learning, §6 soundness).
6. **Composable.** Products, coproducts, and exponentials make complex systems from simple ones without exponential blow-up.

---

## Project Status

**Alpha (v0.1.0)** — the API is stable enough for research use and non-critical production, but breaking changes may occur before v1.0. Feedback and contributions are very welcome.

Roadmap:
- [ ] Automated constraint acquisition from data
- [ ] Richer serialisation (protobuf / Arrow)
- [ ] Enriched and monoidal categorical extensions
- [ ] Quantum-native COH components
- [ ] Large-scale empirical validation across additional domains
- [ ] Reference applications in smart cities, biology, and aerospace

---

## Contributing

We welcome contributions of all kinds — bug reports, feature proposals, documentation improvements, and new domain profiles.

```bash
# Set up a development environment
git clone https://github.com/your-org/gismol.git
cd gismol
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Before opening a pull request
ruff check gismol/
mypy gismol/
pytest gismol/tests -n auto
```

Guidelines:
- Keep runtime dependencies at zero. Optional extras belong behind a feature flag.
- Every public class and function needs a docstring and a test.
- Prefer small, focused modules over monolithic ones.
- Follow the existing style (100-column lines, ruff-clean).

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for details.

---

## Citation

If you use `gismol` in academic work, please cite the COH theory paper and this toolkit:

```bibtex
@article{coh2025foundations,
  title   = {The Mathematical Foundations of Constrained Object Hierarchies:
             A Unified Framework for World Modelling and Artificial General Intelligence},
  author  = {COH contributors},
  journal = {Preprint},
  year    = {2025},
  note    = {Keywords: Artificial General Intelligence, Constrained Object Hierarchies,
             Category Theory, Dynamical Systems, Neurosymbolic AI}
}

@software{gismol2025,
  title  = {gismol: A Python Toolkit for Constrained Object Hierarchies},
  author = {gismol contributors},
  year   = {2025},
  url    = {https://github.com/your-org/gismol},
  version = {0.1.0}
}
```

---

## Related Work

`gismol` builds on and unifies ideas from several traditions:

- **Cognitive architectures** — ACT-R, SOAR, CLARION (production systems with hierarchical structure)
- **Predictive coding / free energy** — Friston's FEP; variational inference
- **Hierarchical reinforcement learning** — options, MAXQ, meta-learning
- **Category theory for cognition** — Baez & Stay; applied category theory
- **Neurosymbolic AI** — De Raedt et al.; statistical relational learning
- **Nonlinear control** — Lyapunov stability, control barrier functions
- **Biologically plausible learning** — homeostatic regulation, precision weighting

The COH framework's contribution is to formalise these as **one constrained hierarchical optimisation process**, with soundness, completeness, and cross-domain applicability proved in the theory paper.

---

## License

MIT License — see [`LICENSE`](LICENSE) for details.

```
Copyright (c) 2025 gismol contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## Acknowledgements

The COH framework synthesises insights from theoretical neuroscience, category theory, control theory, reinforcement learning, and cognitive architectures. We thank the broader research communities whose work made this synthesis possible.

---

<p align="center">
  <b>Intelligence is constrained hierarchical optimisation.</b><br>
  <i>— the COH thesis</i>
</p>