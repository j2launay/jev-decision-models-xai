"""Calibration metrics: ECE / MCE (equal-width or equal-mass bins), Brier score, bootstrap CIs and
the selective-prediction AUROC of an arbitrary confidence signal."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.metrics import roc_auc_score

Binning = Literal["width", "mass"]


@dataclass
class BinStats:
    lower: float
    upper: float
    count: int
    mean_conf: float
    accuracy: float


@dataclass
class CalibrationResult:
    ece: float
    mce: float
    bins: list[BinStats]


def _bin_indices(conf: np.ndarray, n_bins: int, binning: Binning) -> list[np.ndarray]:
    if binning == "width":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        # Right-closed bins (lower, upper]; confidence 0 falls into the first bin.
        ids = np.clip(np.searchsorted(edges, conf, side="left") - 1, 0, n_bins - 1)
        return [np.flatnonzero(ids == b) for b in range(n_bins)]
    order = np.argsort(conf, kind="stable")
    return [chunk for chunk in np.array_split(order, n_bins)]


def calibration_error(
    conf: np.ndarray, correct: np.ndarray, n_bins: int = 10, binning: Binning = "width"
) -> CalibrationResult:
    """Top-label ECE and MCE. `conf` is the probability of the predicted outcome, `correct` a 0/1 vector.

    For binary probability calibration (e.g. noul P(true)), pass P(true) as `conf` and the gold
    positive indicator as `correct`.
    """
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    n = len(conf)
    if n == 0:
        return CalibrationResult(float("nan"), float("nan"), [])
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, mce, bins = 0.0, 0.0, []
    for b, idx in enumerate(_bin_indices(conf, n_bins, binning)):
        if len(idx) == 0:
            if binning == "width":
                bins.append(BinStats(edges[b], edges[b + 1], 0, float("nan"), float("nan")))
            continue
        mean_conf = float(conf[idx].mean())
        acc = float(correct[idx].mean())
        gap = abs(acc - mean_conf)
        ece += len(idx) / n * gap
        mce = max(mce, gap)
        lower, upper = (edges[b], edges[b + 1]) if binning == "width" else (float(conf[idx].min()), float(conf[idx].max()))
        bins.append(BinStats(lower, upper, int(len(idx)), mean_conf, acc))
    return CalibrationResult(float(ece), float(mce), bins)


def brier_multiclass(probs: np.ndarray, gold_index: np.ndarray) -> float:
    """Original Brier (1950) score: mean over examples of sum_k (p_k - y_k)^2, in [0, 2]."""
    probs = np.asarray(probs, dtype=float)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(probs)), np.asarray(gold_index)] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def brier_binary(p_true: np.ndarray, y: np.ndarray) -> float:
    """Binary Brier score: mean (p - y)^2, in [0, 1]."""
    return float(((np.asarray(p_true, float) - np.asarray(y, float)) ** 2).mean())


def selective_auroc(signal: np.ndarray, correct: np.ndarray) -> float:
    """Probability that a correct prediction receives a higher signal than an incorrect one."""
    correct = np.asarray(correct).astype(int)
    if correct.min() == correct.max():
        return float("nan")
    return float(roc_auc_score(correct, np.asarray(signal, float)))


def bootstrap_ci(
    statistic: Callable[[np.ndarray], float],
    n: int,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI; `statistic` receives an index array resampled with replacement."""
    rng = np.random.default_rng(seed)
    values = np.array([statistic(rng.integers(0, n, size=n)) for _ in range(n_resamples)])
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    return float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2))
