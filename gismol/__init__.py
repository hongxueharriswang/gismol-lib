# ============================================================
# FILE: gismol/__init__.py
# ============================================================
"""gismol — a toolkit for Constrained Object Hierarchies (COH).

COH is a mathematical framework for artificial general intelligence in which
intelligent systems are modelled as *constrained hierarchical optimisation*
processes.  A COH is a 9-tuple::

    O = (C, A, M, N, E, I, T, G, D)

where

    C  hierarchical compositional structure (sub-objects)
    A  attributes (measurable state variables)
    M  methods (executable state transformations)
    N  neural components (adaptive, parameterised functions)
    E  embedding function into a Hilbert space
    I  identity constraints (invariants)
    T  trigger constraints (event-condition-action rules)
    G  goal constraints (optimisation objectives)
    D  constraint daemons (continuous monitors)

The base class for every intelligent system is
:class:`gismol.core.coh_object.COHObject`.
"""

from .core import exceptions, expression
from .core.coh_object import Attribute, COHObject, Method, State
from .core.constraints import (
    Constraint,
    ConstraintSet,
    GoalConstraint,
    IdentityConstraint,
    TriggerConstraint,
)
from .core.daemons import (
    BoundaryDaemon,
    Daemon,
    DaemonNetwork,
    InvariantDaemon,
    RateDaemon,
    SafetyDaemon,
)
from .core.dynamics import (
    ContinuousFlow,
    DiscreteTransition,
    HybridDynamicalSystem,
    Trajectory,
)
from .core.embedding import (
    CompositeEmbedder,
    Embedder,
    FeatureEmbedder,
    HashEmbedder,
    HilbertSpace,
)
from .core.repository import COHRepository

__version__ = "0.1.0"

__all__ = [
    "Attribute",
    "BoundaryDaemon",
    "COHObject",
    "COHRepository",
    "CompositeEmbedder",
    "Constraint",
    "ConstraintSet",
    "ContinuousFlow",
    "Daemon",
    "DaemonNetwork",
    "DiscreteTransition",
    "Embedder",
    "FeatureEmbedder",
    "GoalConstraint",
    "HashEmbedder",
    "HilbertSpace",
    "HybridDynamicalSystem",
    "IdentityConstraint",
    "InvariantDaemon",
    "Method",
    "RateDaemon",
    "SafetyDaemon",
    "State",
    "Trajectory",
    "TriggerConstraint",
    "__version__",
    "exceptions",
    "expression",
]