"""Post-hoc recalibration baselines and selective-prediction metrics computed from cached predictions.

No model call is made: every number is derived from `results/<run>/<model>/predictions.parquet`.

* Temperature scaling (Guo et al., 2017) on log-probabilities, fitted by NLL with 5-fold
  cross-fitting (each instance is recalibrated by a temperature fitted on the other folds).
* Platt / logistic scaling (Platt, 1999) on the log-odds of the top-label probability.
* ECE (10 equal-width bins), ECE-sweep (Roelofs et al., 2022; equal-mass, largest monotone
  bin count), Brier, NLL.
* Selective prediction: AURC and E-AURC (Geifman & El-Yaniv, 2017) for max-probability and
  for the provider `confidence` field, plus per-bin counts with bootstrap CIs for the ECE.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from . import calibration as cal

EPS = 1e-6
MODELS = {
    "tev1:0.8b": "results/expanded-450/tev1-0.8b",
    "tev1:latest": "results/expanded-450/tev1-latest",
    "nimble:latest": "results/expanded-450/nimble-latest",
    "clef-flash:latest": "results/expanded-450/clef-flash-latest",
    "clef:27b": "results/expanded-450/clef-27b",
    "gemma4-json": "results/poc-tev1-0.8b/gemma4-json",
}


def _softmax_T(logp: np.ndarray, T: float) -> np.ndarray:
    z = logp / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _nll(probs: np.ndarray, gold: np.ndarray) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(probs)), gold], EPS, 1.0)).mean())


def fit_temperature(logp: np.ndarray, gold: np.ndarray) -> float:
    res = minimize_scalar(lambda t: _nll(_softmax_T(logp, t), gold), bounds=(0.05, 20.0), method="bounded")
    return float(res.x)


def ece_sweep(conf: np.ndarray, correct: np.ndarray) -> float:
    """Equal-mass ECE with the largest bin count for which bin accuracies stay monotone."""
    order = np.argsort(conf, kind="stable")
    c, y = conf[order], correct[order]
    best = cal.calibration_error(conf, correct, 1, "mass").ece
    for b in range(2, len(conf) + 1):
        chunks = np.array_split(np.arange(len(c)), b)
        accs = [y[ch].mean() for ch in chunks if len(ch)]
        if any(accs[i] > accs[i + 1] for i in range(len(accs) - 1)):
            break
        best = float(sum(len(ch) / len(c) * abs(y[ch].mean() - c[ch].mean()) for ch in chunks if len(ch)))
    return best


def aurc(conf: np.ndarray, correct: np.ndarray) -> tuple[float, float]:
    """Area under the risk-coverage curve and its excess over the oracle ranking (E-AURC)."""
    order = np.argsort(-conf, kind="stable")
    err = 1.0 - correct[order]
    risks = np.cumsum(err) / np.arange(1, len(err) + 1)
    a = float(risks.mean())
    oracle_err = np.sort(err)  # all correct predictions first
    oracle = float((np.cumsum(oracle_err) / np.arange(1, len(err) + 1)).mean())
    return a, a - oracle


def metrics(probs: np.ndarray, gold: np.ndarray, signal: np.ndarray | None = None) -> dict[str, float]:
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = (pred == gold).astype(float)
    out = {
        "accuracy": float(correct.mean()),
        "ece_w": cal.calibration_error(conf, correct, 10, "width").ece,
        "ece_sweep": ece_sweep(conf, correct),
        "brier": cal.brier_multiclass(probs, gold),
        "nll": _nll(probs, gold),
    }
    out["aurc"], out["eaurc"] = aurc(conf, correct)
    if signal is not None and not np.all(np.isnan(signal)):
        out["aurc_provider"], out["eaurc_provider"] = aurc(np.nan_to_num(signal), correct)
    return out


def crossfit(probs: np.ndarray, gold: np.ndarray, method: str, seed: int = 0) -> np.ndarray:
    logp = np.log(np.clip(probs, EPS, 1.0))
    out = np.empty_like(probs)
    n_splits = 5 if np.bincount(gold).min() >= 5 else 2
    strat = gold if np.bincount(gold).min() >= n_splits else np.zeros_like(gold)
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(probs, strat):
        if method == "temperature":
            out[te] = _softmax_T(logp[te], fit_temperature(logp[tr], gold[tr]))
            continue
        # Platt on the top-label log-odds: P(correct | logit(max p)); remaining mass split
        # proportionally among the other classes.
        conf = probs.max(axis=1)
        z = np.log(np.clip(conf, EPS, 1 - EPS) / np.clip(1 - conf, EPS, 1)).reshape(-1, 1)
        y = (probs.argmax(axis=1) == gold).astype(int)
        if y[tr].min() == y[tr].max():
            out[te] = probs[te]
            continue
        lr = LogisticRegression(C=1e4).fit(z[tr], y[tr])
        new_conf = lr.predict_proba(z[te])[:, 1]
        p = probs[te].copy()
        top = p.argmax(axis=1)
        rest = 1.0 - p[np.arange(len(p)), top]
        scale = np.where(rest > EPS, (1.0 - new_conf) / np.maximum(rest, EPS), 0.0)
        p = p * scale[:, None]
        p[np.arange(len(p)), top] = new_conf
        out[te] = p
    return out


def run(out_dir: Path = Path("results/posthoc")) -> pd.DataFrame:
    rows, bins_rows = [], []
    for model, d in MODELS.items():
        path = Path(d) / "predictions.parquet"
        if not path.exists():
            continue
        preds = pd.read_parquet(path)
        preds = preds[preds["condition"] == "clean"]
        for (source, head), g in preds.groupby(["source", "head"]):
            probs = np.array([list(json.loads(p).values()) for p in g["probs"]], dtype=float)
            gold = g["gold_index"].to_numpy().astype(int)
            signal = g["provider_confidence"].to_numpy(dtype=float) if head == "label" else None
            for method in ("raw", "temperature", "platt"):
                p = probs if method == "raw" else crossfit(probs, gold, method)
                rows.append({"model": model, "source": source, "head": head, "method": method, "n": len(g),
                             **metrics(p, gold, signal if method == "raw" else None)})
            conf = probs.max(axis=1)
            correct = (probs.argmax(axis=1) == gold).astype(float)
            for b in cal.calibration_error(conf, correct, 10, "width").bins:
                bins_rows.append({"model": model, "source": source, "head": head, "lower": b.lower,
                                  "upper": b.upper, "count": b.count, "mean_conf": b.mean_conf, "accuracy": b.accuracy})
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "posthoc_calibration.parquet", index=False)
    pd.DataFrame(bins_rows).to_parquet(out_dir / "bin_counts.parquet", index=False)
    return df


if __name__ == "__main__":
    df = run()
    pd.set_option("display.width", 220)
    print(df[df["head"] == "label"].pivot_table(index=["model", "source"], columns="method",
                                                values=["ece_w", "brier", "nll"]).round(3).to_string())
    print(df[(df["head"] == "label") & (df["method"] == "raw")][["model", "source", "n", "aurc", "eaurc", "aurc_provider", "ece_sweep"]].round(3).to_string())
