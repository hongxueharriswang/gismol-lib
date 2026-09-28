# ============================================================
# FILE: gismol/neural/constrained_rl.py
# ============================================================
"""Constrained reinforcement learning (Section 5.2).

Implements the primal-dual (Lagrangian) method for

    max_pi  E[G(tau)]   s.t.  E[C_i(tau)] <= 0,   for all i,

generalising constrained MDPs to the COH setting.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .components import PolicyComponent
from .nn import softmax
from .optimizers import Adam, LagrangianOptimizer, Optimizer, SGD

__all__ = [
    "ConstrainedMDP", "LinearSoftmaxPolicy", "run_episode",
    "PrimalDualOptimizer", "LagrangianAgent", "ConstraintViolation",
]


class ConstraintViolation(Exception):
    """Raised when a hard constraint is violated and cannot be repaired."""


# --------------------------------------------------------------------------
class ConstrainedMDP:
    """A finite CMDP used to test constrained learning.

    Parameters
    ----------
    n_states, n_actions:
        Cardinalities of the state and action spaces.
    transition:
        ``P[s][a] = {s': prob}``.
    reward:
        ``R[s][a]``.
    costs:
        ``C[s][a]`` cost values used by the constraint functions.
    cost_limit:
        ``d`` in ``E[C] <= d``.
    """

    def __init__(self, n_states: int, n_actions: int,
                 transition: Optional[Mapping[int, Mapping[int, Mapping[int, float]]]] = None,
                 reward: Optional[Mapping[int, Mapping[int, float]]] = None,
                 costs: Optional[Mapping[int, Mapping[int, float]]] = None,
                 cost_limit: float = 0.0, horizon: int = 20,
                 seed: int = 0):
        self.n_states = n_states
        self.n_actions = n_actions
        self.horizon = horizon
        self.cost_limit = float(cost_limit)
        self.rng = random.Random(seed)
        self.transition = transition or self._random_transition()
        self.reward = reward or {
            s: {a: float(self.rng.uniform(-1, 1)) for a in range(n_actions)}
            for s in range(n_states)
        }
        self.costs = costs or {
            s: {a: float(max(0.0, self.rng.gauss(0.2, 0.5))) for a in range(n_actions)}
            for s in range(n_states)
        }

    def _random_transition(self):
        return {
            s: {a: {self.rng.randrange(self.n_states): 1.0}
                for a in range(self.n_actions)}
            for s in range(self.n_states)
        }

    def sample_next(self, state: int, action: int) -> int:
        dist = self.transition.get(state, {}).get(action, {state: 1.0})
        r = self.rng.random()
        acc = 0.0
        for s_next, p in dist.items():
            acc += p
            if r <= acc:
                return s_next
        return next(iter(dist))

    def cost(self, state: int, action: int) -> float:
        return float(self.costs.get(state, {}).get(action, 0.0))

    def step(self, state: int, action: int) -> Tuple[int, float, float]:
        nxt = self.sample_next(state, action)
        r = float(self.reward.get(state, {}).get(action, 0.0))
        c = self.cost(state, action)
        return nxt, r, c

    def __repr__(self):
        return (f"ConstrainedMDP(S={self.n_states}, A={self.n_actions}, "
                f"limit={self.cost_limit})")


# --------------------------------------------------------------------------
class LinearSoftmaxPolicy:
    """Tabular softmax policy parameterised by a ``S x A`` weight matrix."""

    def __init__(self, n_states: int, n_actions: int, seed: int = 0):
        rng = random.Random(seed)
        self.n_states = n_states
        self.n_actions = n_actions
        self.theta: List[List[float]] = [
            [rng.gauss(0.0, 0.1) for _ in range(n_actions)] for _ in range(n_states)
        ]

    def probs(self, state: int) -> List[float]:
        return softmax(self.theta[state])

    def sample(self, state: int, rng: Optional[random.Random] = None) -> int:
        rng = rng or random
        probs = self.probs(state)
        r = rng.random()
        acc = 0.0
        for a, p in enumerate(probs):
            acc += p
            if r <= acc:
                return a
        return len(probs) - 1

    def log_prob(self, state: int, action: int) -> float:
        return math.log(max(1e-12, self.probs(state)[action]))

    def flat_parameters(self) -> List[float]:
        return [v for row in self.theta for v in row]

    def set_flat_parameters(self, flat: Sequence[float]):
        idx = 0
        for s in range(self.n_states):
            for a in range(self.n_actions):
                self.theta[s][a] = float(flat[idx]); idx += 1
        return self

    def __repr__(self):
        return f"LinearSoftmaxPolicy(S={self.n_states}, A={self.n_actions})"


# --------------------------------------------------------------------------
def run_episode(env: ConstrainedMDP, policy: LinearSoftmaxPolicy,
                start_state: Optional[int] = None,
                rng: Optional[random.Random] = None) -> Dict[str, Any]:
    """Roll out one episode and record the trajectory statistics."""
    rng = rng or env.rng
    state = env.rng.randrange(env.n_states) if start_state is None else start_state
    states, actions, rewards, costs, log_probs = [], [], [], [], []
    for _ in range(env.horizon):
        action = policy.sample(state, rng)
        nxt, r, c = env.step(state, action)
        states.append(state)
        actions.append(action)
        rewards.append(r)
        costs.append(c)
        log_probs.append(policy.log_prob(state, action))
        state = nxt
    return {
        "states": states, "actions": actions, "rewards": rewards,
        "costs": costs, "log_probs": log_probs,
        "return": float(sum(rewards)),
        "cost": float(sum(costs)),
    }


# --------------------------------------------------------------------------
class PrimalDualOptimizer:
    """Primal-dual optimiser for constrained policy optimisation."""

    def __init__(self, n_constraints: int = 1, lr_theta: float = 0.05,
                 lr_lambda: float = 0.1, cost_limit: float = 0.0):
        self.n_constraints = n_constraints
        self.theta_optimizer = Adam(lr_theta)
        self.lambda_optimizer = LagrangianOptimizer(lr_lambda)
        self.lambdas: List[float] = [0.0] * n_constraints
        self.cost_limit = float(cost_limit)
        self.history: List[Dict[str, float]] = []

    def step(self, policy: LinearSoftmaxPolicy, trajectories: Sequence[Mapping[str, Any]],
             gamma: float = 0.99) -> Dict[str, Any]:
        """One primal-dual update from a batch of trajectories."""
        params = policy.flat_parameters()
        grad = [0.0] * len(params)
        n = max(1, len(trajectories))

        mean_return = sum(t["return"] for t in trajectories) / n
        mean_cost = sum(t["cost"] for t in trajectories) / n

        for traj in trajectories:
            ret = traj["return"]
            cost = traj["cost"]
            for t, (s, a, lp) in enumerate(zip(traj["states"], traj["actions"],
                                               traj["log_probs"])):
                weight = (ret - self.lambdas[0] * cost)
                idx = s * policy.n_actions + a
                # d/dtheta log pi(a|s) for softmax: (1 - p_a) for the chosen,
                # -p_b for the others
                probs = policy.probs(s)
                for b in range(policy.n_actions):
                    dlogp = ((1.0 if b == a else 0.0) - probs[b])
                    grad[idx + (b - a)] += -weight * dlogp / n

        new_params = self.theta_optimizer.step(params, grad)
        policy.set_flat_parameters(new_params)

        # dual ascent on the constraint violation
        violation = mean_cost - self.cost_limit
        self.lambdas = self.lambda_optimizer.step(self.lambdas, [violation])

        record = {
            "return": mean_return,
            "cost": mean_cost,
            "lambda": self.lambdas[0],
            "violation": violation,
        }
        self.history.append(record)
        return record

    def __repr__(self):
        return (f"PrimalDualOptimizer(lambda={self.lambdas}, "
                f"limit={self.cost_limit})")


# --------------------------------------------------------------------------
class LagrangianAgent:
    """High level convenience wrapper: policy + CMDP + primal-dual optimiser."""

    def __init__(self, env: ConstrainedMDP, policy: Optional[LinearSoftmaxPolicy] = None,
                 lr_theta: float = 0.05, lr_lambda: float = 0.1, seed: int = 0):
        self.env = env
        self.policy = policy or LinearSoftmaxPolicy(env.n_states, env.n_actions, seed)
        self.optimizer = PrimalDualOptimizer(1, lr_theta, lr_lambda, env.cost_limit)
        self.rng = random.Random(seed)

    def train(self, iterations: int = 100, episodes_per_iter: int = 8,
              verbose: bool = False) -> List[Dict[str, float]]:
        history = []
        for it in range(iterations):
            batch = [run_episode(self.env, self.policy, rng=self.rng)
                     for _ in range(episodes_per_iter)]
            record = self.optimizer.step(self.policy, batch)
            history.append(record)
            if verbose and it % max(1, iterations // 10) == 0:
                print(f"iter {it:4d}  R={record['return']:+.3f}  "
                      f"C={record['cost']:.3f}  lambda={record['lambda']:.3f}")
        return history

    def evaluate(self, episodes: int = 50) -> Dict[str, float]:
        rolls = [run_episode(self.env, self.policy, rng=self.rng) for _ in range(episodes)]
        n = max(1, len(rolls))
        return {
            "mean_return": sum(r["return"] for r in rolls) / n,
            "mean_cost": sum(r["cost"] for r in rolls) / n,
            "feasible_fraction": sum(1 for r in rolls if r["cost"] <= self.env.cost_limit) / n,
        }

    def __repr__(self):
        return f"LagrangianAgent({self.env!r})"
    