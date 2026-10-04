"""Aioli-objective-adapted: a pure-Python transfer-matrix mixing controller.

Equation provenance, inspected 10-04-2026:
https://github.com/HazyResearch/aioli/blob/main/trainer/aioli_trainer.py
``learn_params_subroutine`` and the normalized, non-EMA, full-matrix train path.
See related_work.md for adaptation limits; this is not a reproduction of the
original cross-entropy-domain experiments. Four default actions are pretraining,
supervised fine-tuning, direct preference optimization, reinforcement learning.

The caller trains the actual evolving trajectory in the randomized probe order.
Each probe mixture is a smoothed one-hot row of W. Measure each fixed validation
component immediately before and after the probe, accumulate the loss drop in
the column for that mixture, and average rounds. These probes advance training;
they are not checkpoint-rollbacks. The harness charges all training, rollout,
reference scoring, and validation work, and owns the frozen validation scaling.
This module accepts only those already-measured component-by-mixture matrices;
it does not run training, load evaluation data, or charge compute itself.

Adaptations: the number of validation components may differ from the number of
actions; an optional probability floor is off by default and explicitly logged.
All-zero normalized matrices leave weights unchanged instead of dividing by 0.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from controller import OBJECTIVES, _finite, _positive, _project_with_floors


def smoothed_probe_matrix(k: int = 4, alpha: float = 0.75) -> list[list[float]]:
    """W[j][k] is action k's fraction in probe mixture j."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 2:
        raise ValueError("at least two actions are required")
    alpha = _finite(alpha, "alpha")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must lie in [0, 1]")
    beta = (1 - alpha) / (k - 1)
    if abs(alpha - beta) <= 1e-12:
        raise ValueError("alpha at or numerically near 1/k makes W singular")
    return [[alpha if i == j else beta for j in range(k)] for i in range(k)]


def _matrix(matrix: Sequence[Sequence[float]], k: int) -> list[list[float]]:
    if not matrix:
        raise ValueError("loss-drop matrix needs at least one validation component")
    result = []
    for i, row in enumerate(matrix):
        if len(row) != k:
            raise ValueError(f"loss-drop row {i} needs exactly {k} mixture columns")
        result.append([_finite(value, f"matrix[{i}][{j}]") for j, value in enumerate(row)])
    return result


def recover_transfer_matrix(
    loss_drops: Sequence[Sequence[float]], *, alpha: float = .75, k: int = 4
) -> list[list[float]]:
    """Solve W A_i = Delta_i analytically for each validation component i.

    W = (alpha-beta) I + beta 11^T and beta=(1-alpha)/(k-1).
    Since each W column sums to one, sum(A_i)=sum(Delta_i), giving
    A_ij = (Delta_ij - beta*sum(Delta_i)) / (alpha-beta).
    Input has shape (validation_components, k), including cross influences.
    """
    W = smoothed_probe_matrix(k, alpha)
    alpha, beta = W[0][0], W[0][1]
    drops = _matrix(loss_drops, k)
    result = []
    for row in drops:
        # Dividing before summation avoids avoidable overflow for large means.
        mean = _finite(math.fsum(value / k for value in row), "mean loss drop")
        shared = beta * k * mean
        result.append([_finite((value - shared) / (alpha - beta), "transfer entry") for value in row])
    return result


def normalize_transfer_matrix(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    """Shift a negative global minimum to zero, then normalize the global sum.

    This is the upstream normalize-A option. It is neither row normalization,
    nor loss-scale normalization, nor compute-cost normalization.
    """
    if not matrix or not matrix[0]:
        raise ValueError("transfer matrix must be nonempty")
    checked = _matrix(matrix, len(matrix[0]))
    minimum = min(min(row) for row in checked)
    offset = min(0.0, minimum)
    shifted = [[_finite(value - offset, "shifted transfer entry") for value in row] for row in checked]
    maximum = max(max(row) for row in shifted)
    if maximum == 0:
        return [[0.0 for _ in row] for row in checked]
    scaled = [[value / maximum for value in row] for row in shifted]
    total = math.fsum(value for row in scaled for value in row)
    return [[value / total for value in row] for row in scaled]


class AioliController:
    """Full transfer-matrix, normalized, non-EMA Aioli objective adaptation."""

    def __init__(
        self,
        *,
        alpha: float = .75,
        eta: float = 1.0,
        seed: int = 0,
        floor: float = 0.0,
        objectives: Sequence[str] = OBJECTIVES,
        initial_weights: Sequence[float] | None = None,
    ) -> None:
        self.objectives = tuple(objectives)
        if len(set(self.objectives)) != len(self.objectives) or not all(isinstance(x, str) and x for x in self.objectives):
            raise ValueError("objective names must be unique nonempty strings")
        self.k = len(self.objectives)
        self._W = smoothed_probe_matrix(self.k, alpha)
        self.alpha = self._W[0][0]
        self.eta = _positive(eta, "eta")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        self.seed = seed
        self._rng = random.Random(seed)
        self.floor = _finite(floor, "floor")
        if not 0 <= self.floor < 1 / self.k:
            raise ValueError("floor must lie in [0, 1/k)")
        weights = [1.0] * self.k if initial_weights is None else list(initial_weights)
        if len(weights) != self.k:
            raise ValueError("initial_weights must have one value per objective")
        weights = [_positive(value, "initial weight") for value in weights]
        maximum = max(weights)
        weights = [value / maximum for value in weights]
        self._weights = _project_with_floors(weights, [self.floor] * self.k)
        if not all(x > 0 for x in self._weights):
            raise ValueError("initial weight ratio underflows; use less extreme weights")
        self._updates = 0
        self._components = None
        self._last = None

    def probe_mixtures(self) -> list[dict[str, float]]:
        return [dict(zip(self.objectives, row)) for row in self._W]

    def probe_order(self, rounds: int = 1) -> list[int]:
        """One independently shuffled permutation per complete sweep."""
        if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
            raise ValueError("rounds must be a positive integer")
        order = []
        for _ in range(rounds):
            sweep = list(range(self.k))
            self._rng.shuffle(sweep)
            order.extend(sweep)
        return order

    def probabilities(self, progress: float = 0.0) -> dict[str, float]:
        progress = _finite(progress, "progress")
        if not 0 <= progress <= 1:
            raise ValueError("progress must lie in [0, 1]")
        return dict(zip(self.objectives, self._weights))

    def sample(self, progress: float = 0.0) -> str:
        probabilities = self.probabilities(progress)
        draw, cumulative = self._rng.random(), 0.0
        for objective in self.objectives[:-1]:
            cumulative += probabilities[objective]
            if draw < cumulative:
                return objective
        return self.objectives[-1]

    def update(self, loss_drops: Sequence[Sequence[float]]) -> dict[str, float]:
        """Update from an already-averaged validation-component × mixture matrix."""
        drops = _matrix(loss_drops, self.k)
        if self._components is not None and len(drops) != self._components:
            raise ValueError("validation component count must remain frozen")
        transfer = recover_transfer_matrix(drops, alpha=self.alpha, k=self.k)
        normalized = normalize_transfer_matrix(transfer)
        influence = [math.fsum(row[j] for row in normalized) for j in range(self.k)]
        logits = [_finite(math.log(w) + self.eta * gain, "weight logit") for w, gain in zip(self._weights, influence)]
        maximum = max(logits)
        raw = [math.exp(value - maximum) for value in logits]
        weights = _project_with_floors(raw, [self.floor] * self.k)
        if not all(x > 0 for x in weights):
            raise ValueError("weight underflow: reduce frozen eta or declare an explicit floor adaptation")
        self._weights = weights
        self._components = len(drops)
        self._updates += 1
        self._last = {"loss_drops": drops, "transfer": transfer, "normalized": normalized, "influence": influence}
        return self.probabilities()

    def update_rounds(self, rounds: Sequence[Sequence[Sequence[float]]]) -> dict[str, float]:
        """Average complete probe sweeps; columns use mixture identity, not order."""
        if not rounds:
            raise ValueError("at least one sweep is required")
        checked = [_matrix(matrix, self.k) for matrix in rounds]
        components = len(checked[0])
        if any(len(matrix) != components for matrix in checked):
            raise ValueError("all sweeps need the same validation components")
        mean = [[math.fsum(matrix[i][j] / len(checked) for matrix in checked)
                 for j in range(self.k)] for i in range(components)]
        return self.update(mean)

    def snapshot(self) -> dict:
        """Logging receipt; the harness owns resumable training/random state."""
        return {
            "method": "aioli_objective_adapted",
            "alpha": self.alpha, "eta": self.eta, "floor": self.floor,
            "floor_is_adaptation": self.floor > 0, "seed": self.seed,
            "updates": self._updates, "validation_components": self._components,
            "probabilities": self.probabilities(),
            "cost_accounting": "external_harness",
            "last_update": None if self._last is None else {
                key: [row.copy() for row in value] if value and isinstance(value[0], list) else value.copy()
                for key, value in self._last.items()
            },
        }
