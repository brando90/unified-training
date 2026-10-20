"""Objective schedules for the six JR1 methods (pure Python, resumable state).

Weights are update-count shares: the probability mass of each objective among optimizer-step actions. A
deterministic deficit (stride) interleaver turns them into actions, so realized update counts track the
cumulative weights within one update per objective. Progress f is charged work / frozen work target.

Provenance (attributed copies from repo commit 047c79d; originals untouched):
- smoothed_probe_matrix, recover_transfer_matrix, normalize_transfer_matrix and the exponentiated update are
  experiments/00_program/aioli.py, itself from HazyResearch/aioli trainer/aioli_trainer.py
  learn_params_subroutine (normalized, non-EMA, full-matrix path).
- project_with_floors, the stage boundaries (.55/.75/.85), FIXED and SMOOTH_START/END defaults and the 2% PT
  retention rule are experiments/00_program/controller.py.
The equations for every method are written out in jr1/TRAINER.md.
"""
from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence

from .common import OBJECTIVES

K = len(OBJECTIVES)


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def vector(weights: Mapping[str, float] | Sequence[float], name: str = "weights") -> list[float]:
    if isinstance(weights, Mapping):
        if set(weights) != set(OBJECTIVES):
            raise ValueError(f"{name} must name exactly {OBJECTIVES}")
        weights = [weights[o] for o in OBJECTIVES]
    out = [_finite(w, name) for w in weights]
    if len(out) != K or any(w < 0 for w in out) or sum(out) <= 0:
        raise ValueError(f"{name} must be {K} nonnegative values with positive sum")
    return out


def normalize(weights: Sequence[float]) -> list[float]:
    total = math.fsum(weights)
    return [w / total for w in weights]


def project_with_floors(weights: Sequence[float], floors: Sequence[float]) -> list[float]:
    """KL projection onto {p in simplex, p_i >= floor_i}: p_i = max(floor_i, c * w_i)."""
    if math.fsum(floors) > 1 + 1e-12:
        raise ValueError("floors exceed one")
    remaining = set(range(len(weights)))
    result = [0.0] * len(weights)
    available = 1.0
    while remaining:
        mass = math.fsum(weights[i] for i in remaining)
        candidates = {i: available * weights[i] / mass if mass else available / len(remaining) for i in remaining}
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


def smoothed_probe_matrix(k: int = K, alpha: float = 0.75) -> list[list[float]]:
    """W[j][i] is action i's fraction in probe mixture j (a smoothed one-hot row)."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must lie in [0, 1]")
    beta = (1 - alpha) / (k - 1)
    if abs(alpha - beta) <= 1e-12:
        raise ValueError("alpha near 1/k makes W singular")
    return [[alpha if i == j else beta for i in range(k)] for j in range(k)]


def recover_transfer_matrix(loss_drops: Sequence[Sequence[float]], *, alpha: float = 0.75,
                            k: int = K) -> list[list[float]]:
    """Solve W A_i = Delta_i for each validation component i: A_ij = (Delta_ij - beta*sum(Delta_i)) / (alpha-beta)."""
    W = smoothed_probe_matrix(k, alpha)
    alpha, beta = W[0][0], W[0][1]
    out = []
    for row in loss_drops:
        if len(row) != k:
            raise ValueError("each loss-drop row needs one column per probe mixture")
        mean = math.fsum(_finite(v, "loss drop") / k for v in row)
        out.append([(v - beta * k * mean) / (alpha - beta) for v in row])
    return out


def normalize_transfer_matrix(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    """Upstream normalize-A: shift a negative global minimum to zero, then divide by the global sum."""
    minimum = min(min(row) for row in matrix)
    offset = min(0.0, minimum)
    shifted = [[v - offset for v in row] for row in matrix]
    maximum = max(max(row) for row in shifted)
    if maximum == 0:
        return [[0.0 for _ in row] for row in matrix]
    scaled = [[v / maximum for v in row] for row in shifted]
    total = math.fsum(v for row in scaled for v in row)
    return [[v / total for v in row] for row in scaled]


class Interleaver:
    """Deterministic deficit interleaving: credit_i += p_i each action, take argmax credit, subtract one."""

    def __init__(self):
        self.credit = [0.0] * K

    def choose(self, weights: Sequence[float]) -> str:
        # An inactive objective holds no credit, so a stage switch never spends leftover credit on it.
        active = [i for i, w in enumerate(weights) if w > 0]
        for i, w in enumerate(weights):
            self.credit[i] = self.credit[i] + w if w > 0 else 0.0
        best = max(active, key=lambda i: (self.credit[i], -i))
        self.credit[best] -= 1.0
        return OBJECTIVES[best]

    def state_dict(self) -> dict:
        return {"credit": list(self.credit)}

    def load_state_dict(self, state: dict) -> None:
        self.credit = list(state["credit"])


class Schedule:
    """Base class: weights(progress) and, for adaptive methods, probe sweeps and feedback."""
    adaptive = False

    def __init__(self, cfg: dict, seed: int):
        self.cfg = cfg
        self.seed = seed

    def weights(self, progress: float) -> list[float]:
        raise NotImplementedError

    def stage(self, progress: float) -> int | None:
        return None

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass

    def snapshot(self, progress: float) -> dict:
        return {"weights": dict(zip(OBJECTIVES, self.weights(progress)))}


class Sequential(Schedule):
    """PT -> SFT -> DPO -> RL with stages [0,b1), [b1,b2), [b2,b3), [b3,1] of charged work."""

    def __init__(self, cfg, seed):
        super().__init__(cfg, seed)
        self.boundaries = [float(b) for b in cfg["boundaries"]]
        if len(self.boundaries) != K - 1 or not 0 < self.boundaries[0] < self.boundaries[1] < self.boundaries[2] < 1:
            raise ValueError("sequential boundaries must be three increasing fractions in (0, 1)")

    def stage(self, progress):
        for i, b in enumerate(self.boundaries):
            if progress < b:
                return i
        return K - 1

    def weights(self, progress):
        s = self.stage(progress)
        return [float(i == s) for i in range(K)]


class Replay(Sequential):
    """Sequential stages; after the PT stage a fixed share rho of updates replays the PT corpus stream."""

    def __init__(self, cfg, seed):
        super().__init__(cfg, seed)
        self.pt_share = _finite(cfg["pt_share"], "pt_share")
        if not 0 < self.pt_share < 1:
            raise ValueError("pt_share must lie in (0, 1)")

    def weights(self, progress):
        s = self.stage(progress)
        if s == 0:
            return [1.0, 0.0, 0.0, 0.0]
        w = [0.0] * K
        w[0] = self.pt_share
        w[s] = 1 - self.pt_share
        return w


class FixedJoint(Schedule):
    def __init__(self, cfg, seed):
        super().__init__(cfg, seed)
        self.fixed = normalize(vector(cfg["weights"]))
        if min(self.fixed) <= 0:
            raise ValueError("fixed joint needs every objective active")

    def weights(self, progress):
        return list(self.fixed)


class SmoothJoint(Schedule):
    """p(f) = Proj_floors((1-f) p_start + f p_end), with positive PT and DPO floors."""

    def __init__(self, cfg, seed):
        super().__init__(cfg, seed)
        self.start = normalize(vector(cfg["start"]))
        self.end = normalize(vector(cfg["end"]))
        self.floors = [_finite(cfg["floors"].get(o, 0.0), "floor") for o in OBJECTIVES]
        if self.floors[0] <= 0 or self.floors[2] <= 0:
            raise ValueError("smooth joint needs positive PT and preference floors")

    def weights(self, progress):
        mix = [(1 - progress) * a + progress * b for a, b in zip(self.start, self.end)]
        return project_with_floors(mix, self.floors)


class Aioli(Schedule):
    """Aioli transfer-matrix adaptation over the four objectives (see TRAINER.md for the differences table).

    Every `window` production updates the trainer runs one sweep of K probe mixtures in a seeded random
    order. Mixture j is `probe_updates` real updates (alpha share objective j, beta share each other one, in a
    seeded shuffle). Validation components are measured before and after each mixture; the drop of component
    i, divided by its initial value, fills Delta[i][j]. Then A = recover(Delta), An = normalize(A),
    g_j = sum_i An[i][j], a <- Proj_floor(a * exp(eta * g)).
    """
    adaptive = True

    def __init__(self, cfg, seed):
        super().__init__(cfg, seed)
        self.alpha = _finite(cfg["alpha"], "alpha")
        self.eta = _finite(cfg["eta"], "eta")
        self.floor = _finite(cfg.get("floor", 0.0), "floor")
        self.window = int(cfg["window"])
        self.probe_updates = int(cfg["probe_updates"])
        self.W = smoothed_probe_matrix(K, self.alpha)
        beta = self.W[0][1]
        n_main, n_other = self.alpha * self.probe_updates, beta * self.probe_updates
        if abs(n_main - round(n_main)) > 1e-9 or abs(n_other - round(n_other)) > 1e-9 or round(n_other) < 1:
            raise ValueError("probe_updates must realize alpha and beta as whole update counts")
        self.n_main, self.n_other = int(round(n_main)), int(round(n_other))
        if not 0 <= self.floor < 1 / K:
            raise ValueError("floor must lie in [0, 1/K)")
        self.rng = random.Random(seed + 40000)
        self.a = project_with_floors(normalize(vector(cfg.get("initial", [1.0] * K))), [self.floor] * K)
        self.updates = 0
        self.last = None

    def probe_plan(self) -> dict:
        """A seeded sweep: the mixture order and each mixture's shuffled action list."""
        order = list(range(K))
        self.rng.shuffle(order)
        actions = []
        for j in order:
            part = [OBJECTIVES[j]] * self.n_main + [o for i, o in enumerate(OBJECTIVES) if i != j for _ in range(self.n_other)]
            self.rng.shuffle(part)
            actions.append(part)
        return {"order": order, "actions": actions}

    def feedback(self, drops: Sequence[Sequence[float]], progress: float, validation: Mapping[str, float],
                 initial: Mapping[str, float]) -> None:
        transfer = recover_transfer_matrix(drops, alpha=self.alpha, k=K)
        normalized = normalize_transfer_matrix(transfer)
        influence = [math.fsum(row[j] for row in normalized) for j in range(K)]
        logits = [math.log(max(w, 1e-300)) + self.eta * g for w, g in zip(self.a, influence)]
        top = max(logits)
        self.a = project_with_floors([math.exp(v - top) for v in logits], [self.floor] * K)
        self.updates += 1
        self.last = {"loss_drops": [list(r) for r in drops], "transfer": transfer, "normalized": normalized,
                     "influence": influence, "validation": dict(validation)}

    def weights(self, progress):
        return list(self.a)

    def state_dict(self):
        return {"a": list(self.a), "updates": self.updates, "last": self.last, "rng": _rng_to_json(self.rng)}

    def load_state_dict(self, state):
        self.a, self.updates, self.last = list(state["a"]), state["updates"], state["last"]
        self.rng.setstate(_rng_from_json(state["rng"]))

    def snapshot(self, progress):
        return {"weights": dict(zip(OBJECTIVES, self.weights(progress))), "aioli_weights": dict(zip(OBJECTIVES, self.a)),
                "controller_updates": self.updates, "last_update": self.last}


class JR1(Aioli):
    """Aioli foundation + gradual PT/SFT-to-RL prior + validation-only retention constraint.

    pi(f)   = (1-f) pi_start + f pi_end
    q_i     ∝ a_i^(1-lam) * pi_i(f)^lam                      (geometric mixture of controller and prior)
    floors  = eps for every objective, PT floor rho_pt, DPO floor rho_dpo
    if L_pt_val(latest sweep) > (1+tau) * L_pt_val(initial):  floor_pt = m_pt   (constraint response)
    p       = Proj_floors(q)
    L_pt_val is the NLL on the PT controller split (WikiText validation half), measured inside the probe
    sweeps already charged to the run; no test data, no rollback, no checkpoint choice.
    """

    def __init__(self, cfg, seed):
        super().__init__(cfg, seed)
        self.prior_start = normalize(vector(cfg["prior_start"]))
        self.prior_end = normalize(vector(cfg["prior_end"]))
        self.lam = _finite(cfg["prior_strength"], "prior_strength")
        if not 0 <= self.lam <= 1:
            raise ValueError("prior_strength must lie in [0, 1]")
        self.eps = _finite(cfg["min_share"], "min_share")
        self.rho_pt = _finite(cfg["pt_floor"], "pt_floor")
        self.rho_dpo = _finite(cfg["dpo_floor"], "dpo_floor")
        self.tau = _finite(cfg["retention_threshold"], "retention_threshold")
        self.m_pt = _finite(cfg["retention_min_pt"], "retention_min_pt")
        if self.eps <= 0 or self.rho_pt < self.eps or self.rho_dpo < self.eps:
            raise ValueError("floors must be positive")
        if self.m_pt + 2 * self.eps + self.rho_dpo > 1:
            raise ValueError("retention floor is infeasible")
        self.pt_initial = None
        self.pt_latest = None

    def feedback(self, drops, progress, validation, initial):
        super().feedback(drops, progress, validation, initial)
        self.pt_initial = float(initial["pt"])
        self.pt_latest = float(validation["pt"])

    def retention_active(self) -> bool:
        return self.pt_initial is not None and self.pt_latest > (1 + self.tau) * self.pt_initial

    def prior(self, progress):
        return [(1 - progress) * s + progress * e for s, e in zip(self.prior_start, self.prior_end)]

    def weights(self, progress):
        prior = self.prior(progress)
        logq = [(1 - self.lam) * math.log(max(a, 1e-300)) + self.lam * math.log(max(p, 1e-300))
                for a, p in zip(self.a, prior)]
        top = max(logq)
        q = [math.exp(v - top) for v in logq]
        floors = [self.eps] * K
        floors[0] = max(self.rho_pt, self.m_pt if self.retention_active() else 0.0)
        floors[2] = self.rho_dpo
        return project_with_floors(q, floors)

    def state_dict(self):
        return {**super().state_dict(), "pt_initial": self.pt_initial, "pt_latest": self.pt_latest}

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.pt_initial, self.pt_latest = state["pt_initial"], state["pt_latest"]

    def snapshot(self, progress):
        return {**super().snapshot(progress), "prior": dict(zip(OBJECTIVES, self.prior(progress))),
                "retention_active": self.retention_active(), "pt_val_initial": self.pt_initial,
                "pt_val_latest": self.pt_latest}


SCHEDULES = {"sequential": Sequential, "replay": Replay, "fixed_joint": FixedJoint,
             "smooth_joint": SmoothJoint, "aioli": Aioli, "jr1": JR1}


def make_schedule(method: str, cfg: dict, seed: int) -> Schedule:
    return SCHEDULES[method](cfg, seed)


def _rng_to_json(rng: random.Random) -> list:
    version, internal, gauss = rng.getstate()
    return [version, list(internal), gauss]


def _rng_from_json(state: list) -> tuple:
    return (state[0], tuple(state[1]), state[2])
