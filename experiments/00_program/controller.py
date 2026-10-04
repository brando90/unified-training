"""Auditable objective schedules for the initial unified-training experiments.

The four objectives are pretraining (PT), supervised fine-tuning (SFT), direct
preference optimization (DPO), and reinforcement learning (RL). Probabilities
are probabilities of selecting an *update*, not allocations of tokens or compute.
The caller supplies progress as charged compute / the frozen compute budget.

``validation_progress`` is a proposal, not an implementation of Aioli. It uses
full-information, individually attributed validation probes: visit each objective
in a randomized order, measure its before-minus-after change on the same fixed
validation components, and advance the actual training trajectory. Charge the
complete probe cost, including validation, rollouts, and reference scoring.
These local effects are confounded by probe order; log the order and do not treat
them as causal estimates at a shared checkpoint. There is no rollback or copy.
Never attribute a common mixed-training validation change to every objective. Training losses,
held-out test results, and final benchmark results are not controller feedback.

Everything here uses only Python's standard library. The training harness owns
the data splits, validation measurements, compute-unit definition, and logs.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Protocol

OBJECTIVES = ("pt", "sft", "dpo", "rl")
FIXED = (0.55, 0.20, 0.10, 0.15)
SEQUENTIAL_BOUNDARIES = (0.55, 0.75, 0.85)
SMOOTH_START = (0.70, 0.20, 0.05, 0.05)
SMOOTH_END = (0.20, 0.20, 0.15, 0.45)
METHODS = ("sequential", "uniform", "fixed", "smooth", "validation_progress")


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number, not a boolean")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(value: float, name: str) -> float:
    result = _finite(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _checked_sum(values: Sequence[float], name: str) -> float:
    try:
        return _finite(math.fsum(values), name)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc


def _objective_values(values: Mapping[str, float], name: str) -> dict[str, float]:
    if set(values) != set(OBJECTIVES):
        raise ValueError(f"{name} must contain exactly {OBJECTIVES}")
    return {key: _finite(values[key], f"{name}[{key}]") for key in OBJECTIVES}


def validation_progress(
    initial: Mapping[str, float],
    before: Mapping[str, float],
    after: Mapping[str, float],
) -> float:
    """Mean dimensionless validation-loss improvement, using frozen scales.

    Returns mean((before[k] - after[k]) / initial[k]). This is relative
    normalization, not a statistical z-score. Component identities and their
    equal weights are fixed before the run; initial losses must be positive.
    Negative progress means degradation. Compute normalization happens exactly
    once, in ``ObjectiveScheduler.update``. No raw training losses are compared.
    """
    if not initial or set(initial) != set(before) or set(initial) != set(after):
        raise ValueError("validation component keys must be identical and nonempty")
    changes = []
    for key in sorted(initial):
        scale = _positive(initial[key], f"initial[{key}]")
        previous = _finite(before[key], f"before[{key}]")
        current = _finite(after[key], f"after[{key}]")
        if previous < 0 or current < 0:
            raise ValueError("validation losses must be nonnegative")
        changes.append(_finite((previous - current) / scale, f"progress[{key}]"))
    return _finite(math.fsum(x / len(changes) for x in changes), "mean progress")


def _project_with_floors(weights: Sequence[float], floors: Sequence[float]) -> list[float]:
    """KL projection of nonnegative weights onto a simplex with lower bounds.

    The solution is p_i = max(floor_i, c * weight_i), with c determined by
    normalization. Repeatedly fix violated coordinates; four coordinates make
    this bounded and easy to inspect. Zero weights may arise from exp underflow.
    """
    remaining = set(range(len(weights)))
    result = [0.0] * len(weights)
    available = 1.0
    while remaining:
        mass = math.fsum(weights[i] for i in remaining)
        candidates = {
            i: available * weights[i] / mass if mass else available / len(remaining)
            for i in remaining
        }
        clipped = {i for i in remaining if candidates[i] < floors[i]}
        if not clipped:
            for i in remaining:
                result[i] = candidates[i]
            break
        for i in clipped:
            result[i] = floors[i]
            available -= floors[i]
        remaining -= clipped
    return result


class ProbabilityPolicy(Protocol):
    """Minimal adapter interface for separately verified literature baselines."""

    def probabilities(self, progress: float = 0.0) -> dict[str, float]: ...

    def sample(self, progress: float = 0.0) -> str: ...


class ObjectiveScheduler:
    """Fixed schedules and a cost-aware exponentiated-gradient proposal.

    For adaptive feedback, reward_i is dimensionless validation progress and
    cost_i is positive charged compute in the same frozen unit for every method.
    The default documented unit is one million forward-equivalent token
    operations: the harness must divide its raw operation count by 1e6 before
    recording costs here. Forward/backward, generation, scoring and validation
    multipliers are frozen by the harness. Using raw token counts instead would
    shrink the exponent by 1e6 and is a different algorithm configuration.
    We apply p_i <- p_i * exp(eta * reward_i / cost_i), then project onto
    the simplex with a strictly positive per-objective floor. Utility is clipped
    symmetrically to 20 by default before exponentiation; this frozen robustness
    choice is recorded with raw and applied utilities in the audit receipt.

    The optional retention constraint compares current PT validation loss with
    the *initial* PT validation loss. A degradation strictly above 2% enforces
    p_PT >= 0.55 by default. Best-so-far degradation is logged separately and
    does not silently activate a second constraint. A scratch model's initial
    loss is a weak retention reference; that limitation must be reported.
    """

    def __init__(
        self,
        method: str,
        *,
        seed: int = 0,
        floor: float = 0.02,
        eta: float = 0.5,
        pt_initial_loss: float | None = None,
        retention_threshold: float = 0.02,
        retention_min_pt: float = 0.55,
        utility_clip: float | None = 20.0,
    ) -> None:
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        self.method = method
        self.seed = seed
        self.floor = _positive(floor, "floor")
        if self.floor >= 0.25:
            raise ValueError("floor must be strictly less than 1 / 4")
        self.eta = _positive(eta, "eta")
        self.utility_clip = None if utility_clip is None else _positive(utility_clip, "utility_clip")
        self.retention_threshold = _finite(retention_threshold, "retention_threshold")
        if self.retention_threshold < 0:
            raise ValueError("retention_threshold must be nonnegative")
        self.retention_min_pt = _finite(retention_min_pt, "retention_min_pt")
        if not self.floor <= self.retention_min_pt <= 1.0 - 3 * self.floor:
            raise ValueError("retention_min_pt is infeasible for the objective floors")
        self._initial_pt = (
            None if pt_initial_loss is None else _positive(pt_initial_loss, "pt_initial_loss")
        )
        self._best_pt = self._initial_pt
        self._current_pt = self._initial_pt
        self._weights = _project_with_floors(FIXED, [self.floor] * 4)
        self._rng = random.Random(seed)
        self._draws = 0
        self._updates = 0
        self._training_costs = dict.fromkeys(OBJECTIVES, 0.0)
        self._probe_costs = dict.fromkeys(OBJECTIVES, 0.0)
        self._last_update: dict | None = None

    @property
    def total_cost(self) -> float:
        """All explicitly recorded training and adaptive-probe compute."""
        return math.fsum((*self._training_costs.values(), *self._probe_costs.values()))

    def probabilities(self, progress: float = 0.0) -> dict[str, float]:
        progress = _finite(progress, "progress")
        if not 0.0 <= progress <= 1.0:
            raise ValueError("progress must lie in [0, 1]")
        if self.method == "sequential":
            # Phases are [0,.55), [.55,.75), [.75,.85), [.85,1].
            phase = 3
            for i in range(3):
                if progress < SEQUENTIAL_BOUNDARIES[i]:
                    phase = i
                    break
            weights = [float(i == phase) for i in range(4)]
        elif self.method == "uniform":
            weights = [0.25] * 4
        elif self.method == "fixed":
            weights = FIXED
        elif self.method == "smooth":
            weights = [(1 - progress) * a + progress * b for a, b in zip(SMOOTH_START, SMOOTH_END)]
        else:
            weights = self._weights
        return dict(zip(OBJECTIVES, weights))

    def sample(self, progress: float = 0.0) -> str:
        probabilities = self.probabilities(progress)
        draw = self._rng.random()
        self._draws += 1
        cumulative = 0.0
        for objective in OBJECTIVES[:-1]:
            cumulative += probabilities[objective]
            if draw < cumulative:
                return objective
        return OBJECTIVES[-1]

    def record_cost(self, objective: str, cost: float) -> None:
        """Charge an actual production update, including its attributable work.

        Adaptive probes are charged by ``update``; do not record those twice.
        Any shared validation overhead must also be allocated once under a
        frozen attribution rule. This class cannot infer unreported work.
        """
        if objective not in OBJECTIVES:
            raise ValueError(f"unknown objective {objective!r}")
        cost = _positive(cost, "cost")
        _finite(self.total_cost + cost, "cumulative cost")
        self._training_costs[objective] += cost

    def update(
        self,
        rewards: Mapping[str, float],
        costs: Mapping[str, float],
        *,
        pt_validation_loss: float | None = None,
    ) -> dict[str, float]:
        """Consume four attributed probe rewards and charge all probe costs.

        Every objective is required, avoiding uncorrected bandit feedback. Invalid
        input fails before mutating weights, costs, or retention state. Supplying
        PT feedback requires ``pt_initial_loss`` fixed at construction time.
        ``rewards`` must not already have been divided by compute cost.
        """
        if self.method != "validation_progress":
            raise ValueError("only validation_progress accepts adaptive feedback")
        rewards = _objective_values(rewards, "rewards")
        costs = _objective_values(costs, "costs")
        costs = {key: _positive(value, f"costs[{key}]") for key, value in costs.items()}
        raw_utilities = {key: _finite(rewards[key] / costs[key], f"utility[{key}]") for key in OBJECTIVES}
        utilities = {
            key: value if self.utility_clip is None else max(-self.utility_clip, min(self.utility_clip, value))
            for key, value in raw_utilities.items()
        }
        _checked_sum([self.total_cost, *costs.values()], "cumulative cost")
        current_pt = self._current_pt
        best_pt = self._best_pt
        if pt_validation_loss is not None:
            if self._initial_pt is None:
                raise ValueError("pt_initial_loss must be fixed before retention feedback")
            current_pt = _positive(pt_validation_loss, "pt_validation_loss")
            best_pt = min(best_pt, current_pt)
            _finite(current_pt / self._initial_pt - 1.0, "initial PT degradation")
            _finite(current_pt / best_pt - 1.0, "best PT degradation")
        retention_active = (
            current_pt is not None
            and current_pt > self._initial_pt * (1.0 + self.retention_threshold)
        )
        logits = [
            _finite(math.log(weight) + self.eta * utilities[key], f"logit[{key}]")
            for key, weight in zip(OBJECTIVES, self._weights)
        ]
        maximum = max(logits)
        raw = [math.exp(logit - maximum) for logit in logits]
        floors = [self.floor] * 4
        if retention_active:
            floors[0] = self.retention_min_pt
        weights = _project_with_floors(raw, floors)
        self._weights = weights
        self._current_pt, self._best_pt = current_pt, best_pt
        for key in OBJECTIVES:
            self._probe_costs[key] += costs[key]
        self._updates += 1
        self._last_update = {
            "rewards": rewards,
            "costs": costs,
            "utilities": utilities,
            "raw_utilities": raw_utilities,
            "retention_active": retention_active,
        }
        return self.probabilities()

    def snapshot(self) -> dict:
        """JSON-serializable audit record; returned dictionaries are copies.

        This is a logging receipt, not a resumable training/RNG checkpoint.
        Baseline probabilities are queried separately at the current progress.
        """
        initial_degradation = (
            None if self._current_pt is None else self._current_pt / self._initial_pt - 1.0
        )
        best_degradation = (
            None if self._current_pt is None else self._current_pt / self._best_pt - 1.0
        )
        return {
            "method": self.method,
            "seed": self.seed,
            "draws": self._draws,
            "updates": self._updates,
            "floor": self.floor,
            "eta": self.eta,
            "cost_unit": "million_forward_equivalent_token_operations",
            "utility_clip": self.utility_clip,
            "adaptive_weights": dict(zip(OBJECTIVES, self._weights)),
            "training_costs": self._training_costs.copy(),
            "probe_costs": self._probe_costs.copy(),
            "total_cost": self.total_cost,
            "retention": {
                "reference": "initial",
                "threshold": self.retention_threshold,
                "min_pt_probability": self.retention_min_pt,
                "initial_loss": self._initial_pt,
                "best_loss": self._best_pt,
                "current_loss": self._current_pt,
                "relative_initial_degradation": initial_degradation,
                "relative_best_degradation": best_degradation,
            },
            "last_update": None if self._last_update is None else {
                key: value.copy() if isinstance(value, dict) else value
                for key, value in self._last_update.items()
            },
        }
