# ============================================================
# FILE: gismol/categorical/__init__.py
# ============================================================
"""Category-theoretic layer of the COH framework (Section 3)."""

from .category import Category, Morphism, COHCategory, COHMorphism
from .functors import Functor, ComponentFunctor, Graph, ForgetfulFunctor, EmbeddingFunctor
from .limits import Cone, Diagram, Limit, constraint_limit, pullback

__all__ = [
    "COHCategory",
    "COHMorphism",
    "Category",
    "ComponentFunctor",
    "Cone",
    "Diagram",
    "EmbeddingFunctor",
    "ForgetfulFunctor",
    "Functor",
    "Graph",
    "Limit",
    "Morphism",
    "constraint_limit",
    "pullback",
]
