# ============================================================
# FILE: gismol/neural/__init__.py
# ============================================================
"""Neural layer: components, embeddings, optimisers, constrained RL and
hierarchical credit assignment (Section 5)."""

from .nn import MLP, Dense, Activation, mse_loss, cross_entropy_loss
from .components import (
    MLPComponent,
    NeuralComponent,
    NeuralComponentSet,
    PerceptionComponent,
    PolicyComponent,
    PredictorComponent,
    ValueComponent,
)
from .embeddings import AutoencoderEmbedder, NeuralEmbedder, SiameseEmbedder
from .optimizers import (
    AdaGrad,
    Adam,
    LagrangianOptimizer,
    Momentum,
    Optimizer,
    RMSProp,
    SGD,
)
from .constrained_rl import (
    ConstrainedMDP,
    LagrangianAgent,
    LinearSoftmaxPolicy,
    PrimalDualOptimizer,
    run_episode,
)
from .credit_assignment import CreditAssigner, HierarchicalValueDecomposition

__all__ = [
    "MLP",
    "SGD",
    "Activation",
    "AdaGrad",
    "Adam",
    "AutoencoderEmbedder",
    "ConstrainedMDP",
    "CreditAssigner",
    "Dense",
    "HierarchicalValueDecomposition",
    "LagrangianAgent",
    "LagrangianOptimizer",
    "LinearSoftmaxPolicy",
    "MLPComponent",
    "Momentum",
    "NeuralComponent",
    "NeuralComponentSet",
    "NeuralEmbedder",
    "Optimizer",
    "PerceptionComponent",
    "PolicyComponent",
    "PredictorComponent",
    "PrimalDualOptimizer",
    "RMSProp",
    "SiameseEmbedder",
    "ValueComponent",
    "cross_entropy_loss",
    "mse_loss",
    "run_episode",
]
