"""Perturbation-based local attributions: Leave-One-Out, LIME, KernelSHAP (shap.KernelExplainer),
System-1 Logit-SHAP (S1-SHAP), and a random baseline. The explained quantity is the probability
(or log-odds for S1-SHAP) of the outcome predicted on the original input, for each question head."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import shap

from .client import BlockingSystemOne
from .features import FeatureSpace, MaskMode
from .tasks import HEADS, Instance, Task, predicted_target, target_probability


@dataclass
class ValueFunction:
    """Maps binary keep-vectors (n x M) to target probabilities (n x len(HEADS))."""

    client: BlockingSystemOne
    task: Task
    space: FeatureSpace
    targets: dict[str, str]
    mode: MaskMode = "delete"
    placeholder: str = "[MASK]"
    n_rows: int = field(default=0, init=False)
    _memo: dict[bytes, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _head_memo: dict[str, dict[bytes, float]] = field(
        default_factory=lambda: {h: {} for h in HEADS}, init=False, repr=False
    )
    _coalition_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, "ShapOutput"]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X_bool = np.atleast_2d(np.asarray(X)) > 0.5
        self.n_rows += len(X_bool)
        unique, inverse = np.unique(X_bool, axis=0, return_inverse=True)
        missing_idx = [i for i, row in enumerate(unique) if row.tobytes() not in self._memo]
        if missing_idx:
            states = [
                self.task.state(self.space.render(unique[i], self.mode, self.placeholder))
                for i in missing_idx
            ]
            fetched = np.empty((len(missing_idx), len(HEADS)))
            for h, head in enumerate(HEADS):
                results = self.client.decide_batch(states, self.task.head_questions(head))
                fetched[:, h] = [target_probability(r, head, self.targets[head]) for r in results]
            for pos, i in enumerate(missing_idx):
                key = unique[i].tobytes()
                self._memo[key] = fetched[pos]
                for h, head in enumerate(HEADS):
                    self._head_memo[head][key] = float(fetched[pos, h])
        values = np.vstack([self._memo[row.tobytes()] for row in unique])
        return values[np.asarray(inverse).reshape(-1)]

    def call_head(self, X: np.ndarray, head: str) -> np.ndarray:
        X_bool = np.atleast_2d(np.asarray(X)) > 0.5
        if not hasattr(self, "_head_memo"):
            return self(X_bool)[:, HEADS.index(head)]
        self.n_rows += len(X_bool)
        hmemo = self._head_memo[head]
        unique, inverse = np.unique(X_bool, axis=0, return_inverse=True)
        missing_idx = [i for i, row in enumerate(unique) if row.tobytes() not in hmemo]
        if missing_idx:
            states = [
                self.task.state(self.space.render(unique[i], self.mode, self.placeholder))
                for i in missing_idx
            ]
            results = self.client.decide_batch(states, self.task.head_questions(head))
            for pos, i in enumerate(missing_idx):
                hmemo[unique[i].tobytes()] = float(target_probability(results[pos], head, self.targets[head]))
        values = np.array([hmemo[row.tobytes()] for row in unique], dtype=float)
        return values[np.asarray(inverse).reshape(-1)]


def _logit(p: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    clipped = np.clip(p, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def original_targets(client: BlockingSystemOne, task: Task, inst: Instance) -> dict[str, str]:
    state = task.state(inst.segments)
    return {head: predicted_target(client.decide(state, task.head_questions(head)), head) for head in HEADS}


def leave_one_out(vf: ValueFunction) -> dict[str, np.ndarray]:
    m = vf.space.n_features
    rows = np.ones((m + 1, m), dtype=bool)
    rows[np.arange(1, m + 1), np.arange(m)] = False
    values = vf(rows)
    drop = values[0][None, :] - values[1:]
    return {head: drop[:, h] for h, head in enumerate(HEADS)}


@dataclass
class ShapOutput:
    values: dict[str, np.ndarray]
    expected_value: dict[str, float]
    nsamples: int


def _ensure_coalitions(vf: ValueFunction, nsamples: int, seed: int) -> tuple[np.ndarray, np.ndarray, ShapOutput]:
    cache = getattr(vf, "_coalition_cache", None)
    if cache is None:
        cache = {}
        setattr(vf, "_coalition_cache", cache)
    key = (nsamples, seed)
    if key in cache:
        return cache[key]
    m = vf.space.n_features
    budget = int(min(2**m, nsamples)) if m < 63 else nsamples
    recorded_x: list[np.ndarray] = []
    recorded_y: list[np.ndarray] = []

    def tracking_vf(X: np.ndarray) -> np.ndarray:
        X_arr = (np.atleast_2d(np.asarray(X)) > 0.5).astype(float)
        Y_arr = vf(X_arr)
        recorded_x.append(X_arr)
        recorded_y.append(Y_arr)
        return Y_arr

    # shap.KernelExplainer samples coalitions from the global numpy RNG.
    np.random.seed(seed)
    explainer = shap.KernelExplainer(tracking_vf, np.zeros((1, m)))
    # l1_reg=False: no Lasso feature selection (shap >= 0.47 defaults to keeping only 10 features).
    sv = np.asarray(explainer.shap_values(np.ones((1, m)), nsamples=budget, l1_reg=False, silent=True))
    sv = sv.reshape(m, len(HEADS))
    ev = np.atleast_1d(explainer.expected_value)
    out = ShapOutput(
        values={head: sv[:, h] for h, head in enumerate(HEADS)},
        expected_value={head: float(ev[h]) for h, head in enumerate(HEADS)},
        nsamples=budget,
    )
    X_all = np.vstack(recorded_x)
    Y_all = np.vstack(recorded_y)
    cache[key] = (X_all, Y_all, out)
    return X_all, Y_all, out


def lime_attribution(
    vf: ValueFunction, nsamples: int, seed: int, kernel_width: float = 0.25, alpha: float = 1e-2
) -> dict[str, np.ndarray]:
    """LIME for text (Ribeiro et al., 2016) with cosine distance kernel and Ridge surrogate,
    evaluated on the paired coalition sample (nsamples, seed) for variance-reduced comparison."""
    m = vf.space.n_features
    X, Y, _ = _ensure_coalitions(vf, nsamples, seed)
    norms = np.sqrt(X.sum(axis=1))
    cos_sim = np.where(norms > 0, norms / np.sqrt(max(m, 1)), 0.0)
    dist = 1.0 - cos_sim
    weights = np.exp(-(dist**2) / (kernel_width**2))
    # Weighted Ridge regression with intercept: [1, X]
    Z = np.column_stack([np.ones(len(X)), X])
    Zw = Z * weights[:, None]
    reg = np.eye(m + 1) * alpha
    reg[0, 0] = 0.0  # Do not penalize intercept
    theta = np.linalg.solve(Zw.T @ Z + reg, Zw.T @ Y)
    coefs = theta[1:]
    return {head: coefs[:, h] for h, head in enumerate(HEADS)}


def kernel_shap(vf: ValueFunction, nsamples: int, seed: int) -> ShapOutput:
    _, _, out = _ensure_coalitions(vf, nsamples, seed)
    return out


def s1_shap(vf: ValueFunction, nsamples: int, seed: int, eps: float = 1e-4) -> ShapOutput:
    """System-1 Logit-Space Shapley (S1-SHAP): estimates Shapley values in log-odds space
    to prevent probability saturation masking on high-confidence System-1 outputs."""
    m = vf.space.n_features
    budget = int(min(2**m, nsamples)) if m < 63 else nsamples
    # Ensure base coalitions are memoized in vf so logit_vf requires zero extra model calls.
    _ensure_coalitions(vf, nsamples, seed)
    np.random.seed(seed)

    def logit_vf(X: np.ndarray) -> np.ndarray:
        return _logit(vf(X), eps=eps)

    explainer = shap.KernelExplainer(logit_vf, np.zeros((1, m)))
    sv = np.asarray(explainer.shap_values(np.ones((1, m)), nsamples=budget, l1_reg=False, silent=True))
    sv = sv.reshape(m, len(HEADS))
    ev = np.atleast_1d(explainer.expected_value)
    return ShapOutput(
        values={head: sv[:, h] for h, head in enumerate(HEADS)},
        expected_value={head: float(ev[h]) for h, head in enumerate(HEADS)},
        nsamples=budget,
    )


def random_attribution(m: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random(m)
