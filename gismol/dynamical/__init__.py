# ============================================================
# FILE: gismol/dynamical/__init__.py
# ============================================================
"""Dynamical-systems layer: hybrid dynamics, stability, free energy."""

from .hybrid import Guard, HybridAutomaton, Mode, Transition, coh_to_hybrid
from .stability import (
    LyapunovAnalysis,
    StabilityResult,
    estimate_region_of_attraction,
    verify_decrease_condition,
)
from .free_energy import (
    FreeEnergy,
    GenerativeModel,
    VariationalPosterior,
    coh_free_energy_correspondence,
    free_energy_from_coh,
    kl_divergence,
)

__all__ = [
    "FreeEnergy",
    "GenerativeModel",
    "Guard",
    "HybridAutomaton",
    "LyapunovAnalysis",
    "Mode",
    "StabilityResult",
    "Transition",
    "VariationalPosterior",
    "coh_free_energy_correspondence",
    "coh_to_hybrid",
    "estimate_region_of_attraction",
    "free_energy_from_coh",
    "kl_divergence",
    "verify_decrease_condition",
]
