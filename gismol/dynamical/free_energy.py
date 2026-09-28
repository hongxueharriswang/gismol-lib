# ============================================================
# FILE: gismol/dynamical/free_energy.py
# ============================================================
"""Information-theoretic reading of COH (Section 4.3).

Provides discrete variational free energy machinery and the correspondence
between COH constraints and the terms of a generative model (Theorem 4.3 / 8.1).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..core.coh_object import COHObject

__all__ = [
    "kl_divergence", "entropy", "GenerativeModel", "VariationalPosterior",
    "FreeEnergy", "PrecisionWeightedError", "coh_free_energy_correspondence",
    "free_energy_from_coh",
]


# --------------------------------------------------------------------------
def _normalise(dist: Mapping[Any, float]) -> Dict[Any, float]:
    total = float(sum(dist.values()))
    if total <= 0.0:
        n = max(1, len(dist))
        return {k: 1.0 / n for k in dist}
    return {k: float(v) / total for k, v in dist.items()}


def kl_divergence(p: Mapping[Any, float], q: Mapping[Any, float]) -> float:
    """``KL(p || q)`` for discrete distributions (Cover & Thomas, 2006)."""
    p_norm = _normalise(p)
    q_norm = _normalise(q)
    total = 0.0
    for key, pk in p_norm.items():
        if pk <= 0.0:
            continue
        qk = q_norm.get(key, 1e-12)
        total += pk * math.log(pk / max(qk, 1e-12))
    return float(total)


def entropy(p: Mapping[Any, float]) -> float:
    p_norm = _normalise(p)
    return float(-sum(v * math.log(max(v, 1e-12)) for v in p_norm.values()))


# --------------------------------------------------------------------------
class GenerativeModel:
    """``p(o, s) = p(o | s) p(s)`` over a finite latent space."""

    def __init__(self, prior: Mapping[Any, float],
                 likelihood: Mapping[Any, Mapping[Any, float]],
                 name: str = "generative"):
        self.prior = _normalise(prior)
        self.likelihood = {s: _normalise(d) for s, d in likelihood.items()}
        self.name = name

    def log_joint(self, observation, latent) -> float:
        p_s = max(self.prior.get(latent, 1e-12), 1e-12)
        p_o = max(self.likelihood.get(latent, {}).get(observation, 1e-12), 1e-12)
        return math.log(p_s) + math.log(p_o)

    def posterior(self, observation) -> Dict[Any, float]:
        unnorm = {s: self.prior.get(s, 0.0) * self.likelihood.get(s, {}).get(observation, 0.0)
                  for s in self.prior}
        return _normalise(unnorm)

    def __repr__(self):
        return f"GenerativeModel({self.name!r}, |S|={len(self.prior)})"


class VariationalPosterior:
    """``q(s)`` — an approximate posterior over latent states."""

    def __init__(self, distribution: Mapping[Any, float], name: str = "q"):
        self.distribution = _normalise(distribution)
        self.name = name

    def __getitem__(self, key):
        return self.distribution.get(key, 1e-12)

    def items(self):
        return self.distribution.items()

    def __repr__(self):
        return f"VariationalPosterior({self.name!r}, |S|={len(self.distribution)})"


# --------------------------------------------------------------------------
class FreeEnergy:
    """Variational free energy ``F = E_q[-log p(o, s)] - H[q]``."""

    def __init__(self, model: GenerativeModel, name: str = "F"):
        self.model = model
        self.name = name

    def variational(self, observation, q: VariationalPosterior) -> float:
        energy = 0.0
        for latent, prob in q.items():
            energy -= prob * self.model.log_joint(observation, latent)
        return float(energy - entropy(q.distribution))

    def accuracy(self, observation, q: VariationalPosterior) -> float:
        return float(-sum(p * self.model.log_joint(observation, s) for s, p in q.items()))

    def complexity(self, q: VariationalPosterior) -> float:
        prior = self.model.prior
        return kl_divergence(q.distribution, prior)

    def expected(self, policy: Sequence[Any], observation=None,
                 precision: float = 1.0) -> float:
        """Expected free energy along a policy (risk + ambiguity)."""
        total = 0.0
        for action in policy:
            total += precision * (1.0 if observation is None else self.variational(observation,
                                                                                    VariationalPosterior({action: 1.0})))
        return float(total)

    def minimise(self, observation, candidates: Sequence[VariationalPosterior]) -> VariationalPosterior:
        return min(candidates, key=lambda q: self.variational(observation, q))

    def __repr__(self):
        return f"FreeEnergy({self.name!r})"


# --------------------------------------------------------------------------
class PrecisionWeightedError:
    """Prediction-error minimisation with a precision (inverse variance)."""

    def __init__(self, precision: float = 1.0):
        self.precision = float(precision)
        self.errors: List[float] = []

    def __call__(self, prediction, observation) -> float:
        if isinstance(prediction, (list, tuple)):
            err = math.sqrt(sum((float(p) - float(o)) ** 2
                                for p, o in zip(prediction, observation)))
        else:
            err = float(prediction) - float(observation)
        weighted = self.precision * err
        self.errors.append(weighted)
        return weighted

    def mean_error(self) -> float:
        return float(sum(self.errors) / len(self.errors)) if self.errors else 0.0


# --------------------------------------------------------------------------
def coh_free_energy_correspondence(coh: COHObject) -> Dict[str, Any]:
    """Map the components of a COH object onto the terms of a generative model
    (Theorem 4.3 / 8.1)."""
    return {
        "identity_constraints": [c.name for c in coh.I],
        "identity_role": "structural priors  p(s)",
        "trigger_constraints": [c.name for c in coh.T],
        "trigger_role": "conditional dependencies / event-driven updates",
        "goal_constraints": [c.name for c in coh.G],
        "goal_role": "expected log-likelihood terms  E_q[log p(o|s)]",
        "daemons": [d.name for d in coh.D],
        "daemon_role": "precision-weighted prediction-error minimisation",
        "note": "COH generalises the free-energy principle by adding explicit "
                "symbolic constraints and hierarchical composition.",
    }


def free_energy_from_coh(coh: COHObject, state, observation=None,
                         precision: float = 1.0) -> float:
    """A scalar free-energy proxy computed directly from a COH object.

    * identity violations act as structural-prior penalties,
    * goal values act as (negative) expected log-likelihood,
    * daemon corrections contribute a precision-weighted error term.
    """
    structural = sum(c.violation(state) for c in coh.I)
    likelihood = -coh.constraints.total_goal(state)
    daemon_action = coh.D.observe(state)
    prediction_error = daemon_action.severity if daemon_action is not None else 0.0
    return float(structural + likelihood + precision * prediction_error)
