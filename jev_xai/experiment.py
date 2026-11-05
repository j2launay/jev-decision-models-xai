"""End-to-end Phase 1 pipeline: latency benchmark, clean/perturbed predictions, calibration,
attributions (LOO, KernelSHAP, random), faithfulness and KernelSHAP convergence."""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict
from importlib import metadata
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

from . import calibration as cal
from .attribution import (
    ShapOutput,
    ValueFunction,
    kernel_shap,
    leave_one_out,
    lime_attribution,
    original_targets,
    random_attribution,
    s1_shap,
)
from .cache import ResponseCache
from .client import BlockingSystemOne, SystemOneClient
from .config import ExperimentConfig
from .datasets import build_poc_dataset, instances_from_frame, instances_to_frame
from .faithfulness import evaluate as evaluate_faithfulness
from .faithfulness import k_values, ranking
from .features import FeatureSpace, Unit, build_feature_space
from .perturbations import PERTURBATIONS, perturb
from .schemas import ChoiceAnswer
from .tasks import CHOICE, HEADS, NOUL, TASKS, Instance, head_view

PREDICTION_COLUMNS = [
    "model", "instance_id", "source", "task", "condition", "head", "pred", "gold", "correct",
    "conf", "p_true", "provider_confidence", "entropy_confidence", "probs", "gold_index",
    "latency_ms", "cached", "input_tokens", "state",
]


# ---------------------------------------------------------------- dataset

def load_or_build_dataset(cfg: ExperimentConfig) -> list[Instance]:
    if cfg.dataset_path.exists():
        return instances_from_frame(pd.read_parquet(cfg.dataset_path))
    instances = build_poc_dataset(cfg.raw_dir, cfg.seed, cfg.n_esnli_per_class, cfg.n_hans, cfg.triage_counts)
    cfg.dataset_path.parent.mkdir(parents=True, exist_ok=True)
    instances_to_frame(instances).to_parquet(cfg.dataset_path, index=False)
    return instances


def _client(cfg: ExperimentConfig, model: str, cache: ResponseCache | None, concurrency: int | None = None) -> BlockingSystemOne:
    return BlockingSystemOne(
        SystemOneClient(
            model=model,
            base_url=cfg.base_url,
            concurrency=concurrency or cfg.concurrency,
            timeout_s=cfg.timeout_s,
            cache=cache,
        )
    )


# ---------------------------------------------------------------- latency benchmark

def benchmark_latency(cfg: ExperimentConfig, model: str, instances: list[Instance]) -> pd.DataFrame:
    """Uncached latencies: sequential calls, then throughput at increasing client concurrency."""
    rows = []
    n = min(len(instances), max(cfg.benchmark_n_sequential, cfg.benchmark_n_concurrent))
    sample = [instances[j] for j in np.random.default_rng(cfg.seed).choice(len(instances), size=n, replace=False)]
    with _client(cfg, model, cache=None, concurrency=1) as client:
        first = sample[0]
        client.decide(TASKS[first.task].state(first.segments), TASKS[first.task].head_questions(CHOICE), use_cache=False)  # load model
        for inst in sample[: cfg.benchmark_n_sequential]:
            task = TASKS[inst.task]
            for head in HEADS:
                r = client.decide(task.state(inst.segments), task.head_questions(head), use_cache=False)
                rows.append({"model": model, "mode": "sequential", "head": head, "concurrency": 1, "latency_ms": r.latency_ms,
                             "wall_ms": r.latency_ms, "n_requests": 1, "input_tokens": r.usage.input_tokens})
    for level in cfg.benchmark_concurrency_levels:
        with _client(cfg, model, cache=None, concurrency=level) as client:
            batch = [sample[i % len(sample)] for i in range(cfg.benchmark_n_concurrent)]
            by_task: dict[str, list[Instance]] = {}
            for inst in batch:
                by_task.setdefault(inst.task, []).append(inst)
            start = time.perf_counter()
            for task_name, group in by_task.items():
                task = TASKS[task_name]
                results = client.decide_batch([task.state(i.segments) for i in group], task.head_questions(CHOICE), use_cache=False)
                for r in results:
                    rows.append({"model": model, "mode": "concurrent", "head": CHOICE, "concurrency": level, "latency_ms": r.latency_ms,
                                 "wall_ms": np.nan, "n_requests": len(batch), "input_tokens": r.usage.input_tokens})
            wall = (time.perf_counter() - start) * 1000.0
            rows.append({"model": model, "mode": "throughput", "head": CHOICE, "concurrency": level, "latency_ms": np.nan,
                         "wall_ms": wall, "n_requests": len(batch), "input_tokens": np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- predictions

def _prediction_row(model: str, inst: Instance, condition: str, segments: dict[str, str], head: str, result) -> dict[str, Any]:
    task = TASKS[inst.task]
    v = head_view(result, head, inst, task)
    answer = result.answers[head]
    return {
        "model": model,
        "instance_id": inst.id,
        "source": inst.source,
        "task": inst.task,
        "condition": condition,
        "head": head,
        "pred": v.pred,
        "gold": v.classes[v.gold_index],
        "correct": bool(v.correct),
        "conf": float(v.conf),
        "p_true": float(v.probs[0]) if head == NOUL else np.nan,
        "provider_confidence": v.provider_confidence if v.provider_confidence is not None else np.nan,
        "entropy_confidence": answer.entropy_confidence if isinstance(answer, ChoiceAnswer) else np.nan,
        "probs": json.dumps(dict(zip(v.classes, map(float, v.probs)))),
        "gold_index": v.gold_index,
        "latency_ms": result.latency_ms,
        "cached": result.cached,
        "input_tokens": result.usage.input_tokens,
        "state": json.dumps(segments, ensure_ascii=False),
    }


def run_predictions(cfg: ExperimentConfig, client: BlockingSystemOne, instances: list[Instance]) -> pd.DataFrame:
    conditions = [("clean", None)] + [(p.name, p) for p in PERTURBATIONS]
    rows: list[dict[str, Any]] = []
    for condition, pert in tqdm(conditions, desc=f"{client.model} predictions"):
        for task_name, task in TASKS.items():
            group = [i for i in instances if i.task == task_name]
            segs = [i.segments if pert is None else perturb(i, pert, cfg.seed) for i in group]
            for head in HEADS:
                results = client.decide_batch([task.state(s) for s in segs], task.head_questions(head))
                for inst, s, r in zip(group, segs, results):
                    rows.append(_prediction_row(client.model, inst, condition, s, head, r))
    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS)


def run_interference(client: BlockingSystemOne, instances: list[Instance]) -> pd.DataFrame:
    """Clean inputs: each primitive asked alone vs both primitives co-asked in one request."""
    rows = []
    for task_name, task in TASKS.items():
        group = [i for i in instances if i.task == task_name]
        states = [task.state(i.segments) for i in group]
        joint = client.decide_batch(states, task.questions)
        separate = {head: client.decide_batch(states, task.head_questions(head)) for head in HEADS}
        for k, inst in enumerate(group):
            for head in HEADS:
                vs = head_view(separate[head][k], head, inst, task)
                vj = head_view(joint[k], head, inst, task)
                rows.append({
                    "model": client.model, "instance_id": inst.id, "source": inst.source, "head": head,
                    "pred_separate": vs.pred, "pred_joint": vj.pred, "flip": vs.pred != vj.pred,
                    "correct_separate": vs.correct, "correct_joint": vj.correct,
                    "conf_separate": vs.conf, "conf_joint": vj.conf,
                    "p_true_separate": float(vs.probs[0]) if head == NOUL else np.nan,
                    "p_true_joint": float(vj.probs[0]) if head == NOUL else np.nan,
                    "total_variation": float(0.5 * np.abs(vs.probs - vj.probs).sum()),
                    "latency_separate_ms": separate[head][k].latency_ms, "latency_joint_ms": joint[k].latency_ms,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- calibration

def _calibration_stats(df: pd.DataFrame, cfg: ExperimentConfig) -> dict[str, Any]:
    conf = df["conf"].to_numpy()
    correct = df["correct"].to_numpy().astype(float)
    probs = np.array([list(json.loads(p).values()) for p in df["probs"]])
    gold = df["gold_index"].to_numpy()
    out: dict[str, Any] = {"n": len(df), "accuracy": float(correct.mean()), "mean_conf": float(conf.mean())}
    for binning in ("width", "mass"):
        r = cal.calibration_error(conf, correct, cfg.ece_bins, binning)
        out[f"ece_{binning}"] = r.ece
        out[f"mce_{binning}"] = r.mce
    out["brier"] = cal.brier_multiclass(probs, gold)
    out["auroc_maxp"] = cal.selective_auroc(conf, correct)
    if df["head"].iloc[0] == CHOICE:
        out["auroc_provider_confidence"] = cal.selective_auroc(df["provider_confidence"].to_numpy(), correct)
    else:
        p_true = df["p_true"].to_numpy()
        y = (df["gold"] == "true").to_numpy().astype(float)
        out["ece_ptrue_width"] = cal.calibration_error(p_true, y, cfg.ece_bins, "width").ece
        out["ece_ptrue_mass"] = cal.calibration_error(p_true, y, cfg.ece_bins, "mass").ece
        out["brier_binary"] = cal.brier_binary(p_true, y)
    return out


def _bootstrap_metric(df: pd.DataFrame, cfg: ExperimentConfig, fn) -> tuple[float, float]:
    return cal.bootstrap_ci(lambda idx: fn(df.iloc[idx]), len(df), cfg.bootstrap_resamples, cfg.ci_alpha, cfg.seed)


def calibration_table(preds: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    rows = []
    ece = lambda binning: (lambda d: cal.calibration_error(d["conf"].to_numpy(), d["correct"].to_numpy().astype(float), cfg.ece_bins, binning).ece)
    for (model, source, head), group in preds.groupby(["model", "source", "head"]):
        clean = group[group["condition"] == "clean"].set_index("instance_id").sort_index()
        for condition, sub in group.groupby("condition"):
            sub = sub.set_index("instance_id").sort_index()
            stats = _calibration_stats(sub.reset_index(), cfg)
            for binning in ("width", "mass"):
                lo, hi = _bootstrap_metric(sub.reset_index(), cfg, ece(binning))
                stats[f"ece_{binning}_ci_low"], stats[f"ece_{binning}_ci_high"] = lo, hi
            if condition != "clean":
                # Paired bootstrap over instances for the change in ECE and accuracy.
                aligned_clean = clean.loc[sub.index].reset_index()
                pert = sub.reset_index()
                for binning in ("width", "mass"):
                    f = ece(binning)
                    delta = f(pert) - f(aligned_clean)
                    lo, hi = cal.bootstrap_ci(lambda idx: f(pert.iloc[idx]) - f(aligned_clean.iloc[idx]),
                                              len(pert), cfg.bootstrap_resamples, cfg.ci_alpha, cfg.seed)
                    stats[f"delta_ece_{binning}"], stats[f"delta_ece_{binning}_ci_low"], stats[f"delta_ece_{binning}_ci_high"] = delta, lo, hi
                stats["delta_accuracy"] = float(pert["correct"].mean() - aligned_clean["correct"].mean())
                stats["flip_rate"] = float((pert["pred"].to_numpy() != aligned_clean["pred"].to_numpy()).mean())
            rows.append({"model": model, "source": source, "head": head, "condition": condition, **stats})
    return pd.DataFrame(rows)


def reliability_bins(preds: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    rows = []
    for (model, source, head, condition), g in preds.groupby(["model", "source", "head", "condition"]):
        for binning in ("width", "mass"):
            r = cal.calibration_error(g["conf"].to_numpy(), g["correct"].to_numpy().astype(float), cfg.ece_bins, binning)
            for b in r.bins:
                rows.append({"model": model, "source": source, "head": head, "condition": condition, "binning": binning, **asdict(b)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- attribution + faithfulness

def _plausibility(space: FeatureSpace, inst: Instance, scores: np.ndarray) -> dict[str, float]:
    """Triage only: overlap of top-ranked features with clinical flags vs spurious confounders."""
    seg = "patient_note"
    flags = np.zeros(space.n_features, dtype=bool)
    for s, e in inst.metadata["flag_spans"]:
        flags |= space.feature_mask(seg, s, e)
    conf = np.zeros(space.n_features, dtype=bool)
    for s, e in inst.metadata["confounder_spans"]:
        conf |= space.feature_mask(seg, s, e)
    g = int(flags.sum())
    top = ranking(scores)[:g]
    out = {
        "n_flag_features": g,
        "precision_at_flags": float(flags[top].mean()) if g else np.nan,
        "confounder_share_at_flags": float(conf[top].mean()) if g else np.nan,
        "mean_score_flag": float(scores[flags].mean()) if flags.any() else np.nan,
        "mean_score_confounder": float(scores[conf].mean()) if conf.any() else np.nan,
        "mean_score_other": float(scores[~flags & ~conf].mean()) if (~flags & ~conf).any() else np.nan,
    }
    return out


def run_attributions(
    cfg: ExperimentConfig, client: BlockingSystemOne, instances: list[Instance], unit: Unit, methods: tuple[str, ...]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    attr_rows, faith_rows, plaus_rows = [], [], []
    targets_all = [i for i in instances if i.source in cfg.attribution_sources]
    for inst in tqdm(targets_all, desc=f"{client.model} attributions [{unit}]"):
        task = TASKS[inst.task]
        space = build_feature_space(inst.segments, unit=unit, mask_punctuation=cfg.mask_punctuation)
        if space.n_features == 0:
            continue
        targets = original_targets(client, task, inst)
        vf = ValueFunction(client, task, space, targets, cfg.mask_mode, cfg.placeholder)

        method_scores: dict[str, dict[str, np.ndarray]] = {}
        meta: dict[str, dict[str, Any]] = {}
        for method in methods:
            start = time.perf_counter()
            rows_before = vf.n_rows
            if method == "loo":
                method_scores[method] = leave_one_out(vf)
            elif method == "lime":
                method_scores[method] = lime_attribution(vf, cfg.shap_nsamples, cfg.seed)
            elif method == "kernelshap":
                out = kernel_shap(vf, cfg.shap_nsamples, cfg.seed)
                method_scores[method] = out.values
                meta[method] = {"nsamples": out.nsamples, **{f"expected_{h}": v for h, v in out.expected_value.items()}}
            elif method == "s1_shap":
                out = s1_shap(vf, cfg.shap_nsamples, cfg.seed)
                method_scores[method] = out.values
                meta[method] = {"nsamples": out.nsamples, **{f"expected_{h}": v for h, v in out.expected_value.items()}}
            elif method == "random":
                r = random_attribution(space.n_features, cfg.seed + hash_id(inst.id))
                method_scores[method] = {head: r for head in HEADS}
            meta.setdefault(method, {})
            meta[method]["seconds"] = time.perf_counter() - start
            meta[method]["value_rows"] = vf.n_rows - rows_before

        modes = [cfg.mask_mode] + ([cfg.ablation_mask_mode] if cfg.ablation_mask_mode else [])
        mode_vfs = {
            mode: (vf if mode == cfg.mask_mode else ValueFunction(client, task, space, targets, mode, cfg.placeholder))
            for mode in modes
        }
        for method, by_head in method_scores.items():
            for head, scores in by_head.items():
                for f, s in zip(space.features, scores):
                    attr_rows.append({"model": client.model, "instance_id": inst.id, "source": inst.source, "unit": unit,
                                      "method": method, "head": head, "target": targets[head], "feature_index": f.index,
                                      "segment": f.segment, "start": f.start, "end": f.end, "text": f.text, "score": float(s)})
            for mode in modes:
                mvf = mode_vfs[mode]
                results = evaluate_faithfulness(mvf, by_head, curves=mode == cfg.mask_mode)
                for head, res in results.items():
                    faith_rows.append({"model": client.model, "instance_id": inst.id, "source": inst.source, "unit": unit,
                                       "method": method, "head": head, "mask_mode": mode, "n_features": space.n_features,
                                       **{k: v for k, v in asdict(res).items()}, **meta.get(method, {})})
            if inst.source == "triage_synth":
                for head, scores in by_head.items():
                    plaus_rows.append({"model": client.model, "instance_id": inst.id, "unit": unit, "method": method, "head": head,
                                       "has_confounder": inst.metadata["has_confounder"], **_plausibility(space, inst, scores)})
    return pd.DataFrame(attr_rows), pd.DataFrame(faith_rows), pd.DataFrame(plaus_rows)


def run_convergence(cfg: ExperimentConfig, client: BlockingSystemOne, instances: list[Instance]) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    pool = [i for i in instances if i.source in cfg.attribution_sources]
    per_source = cfg.convergence_n_examples // len(cfg.attribution_sources)
    chosen = []
    for source in cfg.attribution_sources:
        group = [i for i in pool if i.source == source]
        chosen += [group[j] for j in rng.choice(len(group), size=min(per_source, len(group)), replace=False)]
    rows = []
    reference_budget = max(cfg.convergence_budgets)
    for inst in tqdm(chosen, desc=f"{client.model} convergence"):
        task = TASKS[inst.task]
        space = build_feature_space(inst.segments, unit=cfg.unit, mask_punctuation=cfg.mask_punctuation)
        targets = original_targets(client, task, inst)
        vf = ValueFunction(client, task, space, targets, cfg.mask_mode, cfg.placeholder)
        for method_name, fn in (
            ("kernelshap", lambda b: kernel_shap(vf, b, cfg.seed)),
            ("s1_shap", lambda b: s1_shap(vf, b, cfg.seed)),
            ("lime", lambda b: ShapOutput(values=lime_attribution(vf, b, cfg.seed), expected_value={}, nsamples=min(2**space.n_features, b) if space.n_features < 63 else b)),
        ):
            estimates = {b: fn(b) for b in cfg.convergence_budgets}
            ref = estimates[reference_budget]
            k = k_values(space.n_features, (20,))[0]
            for b, est in estimates.items():
                for head in HEADS:
                    a, r = est.values[head], ref.values[head]
                    rho = spearmanr(a, r).statistic if space.n_features > 1 else np.nan
                    topk = len(set(ranking(a)[:k]) & set(ranking(r)[:k])) / k
                    rows.append({"model": client.model, "method": method_name, "instance_id": inst.id, "source": inst.source, "head": head,
                                 "budget": b, "nsamples_used": est.nsamples, "n_features": space.n_features,
                                 "exact": est.nsamples >= 2**space.n_features if space.n_features < 63 else False,
                                 "spearman_vs_ref": float(rho) if rho == rho else np.nan, "top20pct_overlap_vs_ref": topk,
                                 "l2_vs_ref": float(np.linalg.norm(a - r))})
    return pd.DataFrame(rows)


def hash_id(s: str) -> int:
    h = 0
    for ch in s.encode("utf-8"):
        h = (h * 131 + ch) % (2**31)
    return h


# ---------------------------------------------------------------- manifest

def build_manifest(cfg: ExperimentConfig, digests: dict[str, str], dataset_path: Path) -> dict[str, Any]:
    import hashlib

    try:
        ollama_version = httpx.get(f"{cfg.base_url}/api/version", timeout=10).json().get("version")
    except Exception:
        ollama_version = None
    packages = {}
    for pkg in ("numpy", "pandas", "shap", "scikit-learn", "scipy", "httpx", "pydantic", "tokenizers", "matplotlib", "pyarrow"):
        try:
            packages[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            packages[pkg] = None
    return {
        "config": cfg.to_dict(),
        "model_digests": digests,
        "ollama_version": ollama_version,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "perturbations": [asdict(p) for p in PERTURBATIONS],
    }
