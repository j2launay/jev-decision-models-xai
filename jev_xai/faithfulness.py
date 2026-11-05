"""Faithfulness of attributions (ERASER, DeYoung et al. 2020): comprehensiveness, sufficiency
(AOPC over k bins) and deletion / insertion curves."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .attribution import ValueFunction
from .tasks import HEADS

ERASER_PERCENTS = (1, 5, 10, 20, 50)


def k_values(m: int, percents: tuple[int, ...] = ERASER_PERCENTS) -> list[int]:
    """Percent bins -> feature counts, rounded up, at least 1, de-duplicated."""
    return sorted({min(m, max(1, math.ceil(p / 100 * m))) for p in percents})


def ranking(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-np.asarray(scores), kind="stable")


@dataclass
class FaithfulnessResult:
    p_full: float
    ks: list[int]
    comprehensiveness: list[float]
    sufficiency: list[float]
    aopc_comprehensiveness: float
    aopc_sufficiency: float
    deletion_curve: list[float]
    insertion_curve: list[float]
    deletion_auc: float
    insertion_auc: float


def _auc(curve: np.ndarray) -> float:
    x = np.linspace(0.0, 1.0, len(curve))
    return float(np.trapezoid(curve, x))


def evaluate(vf: ValueFunction, scores: dict[str, np.ndarray], curves: bool = True) -> dict[str, FaithfulnessResult]:
    """Evaluates every head's ranking with batched calls to the value function."""
    m = vf.space.n_features
    ks = k_values(m)
    full = np.ones((1, m), dtype=bool)
    use_head = hasattr(vf, "call_head")
    if not use_head:
        plan: list[tuple[str, str, int, np.ndarray]] = []
        for head in HEADS:
            order = ranking(scores[head])
            steps = range(m + 1) if curves else ks
            for j in steps:
                deleted = np.ones(m, dtype=bool)
                deleted[order[:j]] = False
                inserted = np.zeros(m, dtype=bool)
                inserted[order[:j]] = True
                plan.append((head, "del", j, deleted))
                plan.append((head, "ins", j, inserted))
        values = vf(np.vstack([full] + [row[None, :] for *_, row in plan]))
        p_full_all = values[0]
        lookup = {(head, kind, j): values[i + 1, HEADS.index(head)] for i, (head, kind, j, _) in enumerate(plan)}
    else:
        p_full_all = np.empty(len(HEADS), dtype=float)
        lookup = {}
        for h, head in enumerate(HEADS):
            order = ranking(scores[head])
            steps = list(range(m + 1)) if curves else ks
            head_plan: list[tuple[str, int, np.ndarray]] = []
            for j in steps:
                deleted = np.ones(m, dtype=bool)
                deleted[order[:j]] = False
                inserted = np.zeros(m, dtype=bool)
                inserted[order[:j]] = True
                head_plan.append(("del", j, deleted))
                head_plan.append(("ins", j, inserted))
            vals = vf.call_head(np.vstack([full] + [row[None, :] for *_, row in head_plan]), head)
            p_full_all[h] = vals[0]
            for i, (kind, j, _) in enumerate(head_plan):
                lookup[(head, kind, j)] = vals[i + 1]

    out = {}
    for h, head in enumerate(HEADS):
        comp = [float(p_full_all[h] - lookup[(head, "del", k)]) for k in ks]
        suff = [float(p_full_all[h] - lookup[(head, "ins", k)]) for k in ks]
        del_curve = np.array([lookup[(head, "del", j)] for j in range(m + 1)]) if curves else np.array([])
        ins_curve = np.array([lookup[(head, "ins", j)] for j in range(m + 1)]) if curves else np.array([])
        out[head] = FaithfulnessResult(
            p_full=float(p_full_all[h]),
            ks=ks,
            comprehensiveness=comp,
            sufficiency=suff,
            aopc_comprehensiveness=float(np.mean(comp)),
            aopc_sufficiency=float(np.mean(suff)),
            deletion_curve=del_curve.tolist(),
            insertion_curve=ins_curve.tolist(),
            deletion_auc=_auc(del_curve) if curves else float("nan"),
            insertion_auc=_auc(ins_curve) if curves else float("nan"),
        )
    return out
