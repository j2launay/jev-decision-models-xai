"""Evaluation script for Tensor Trust Guardrail / Prompt Injection Benchmark (N=150: 75 attacks, 75 benign).

Evaluates:
1. System 1 models (Tev1-0.8B, Tev1-4B, Nimble-9B, Clef-Flash-9B, Clef-27B)
2. System 2 baseline (Gemma4-8B direct JSON)

Metrics:
- Detection accuracy (Choice & Noul)
- True Positive Rate (TPR / Attack Detection Recall)
- False Positive Rate (FPR / Benign False Alarm Rate)
- Calibration: Equal-width ECE_w and Brier score
- Selective Ranking: AUROC_p (max_prob) vs. AUROC_c (provider confidence)
- Cross-question interference (Flip rate & TV distance between isolated and bundled requests)
- Sample S1-SHAP vs. KernelSHAP attributions on injection payloads.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
from tqdm import tqdm

from .attribution import kernel_shap, s1_shap, lime_attribution, leave_one_out
from .cache import ResponseCache
from .calibration import brier_multiclass, calibration_error, selective_auroc
from .client import SystemOneClient
from .config import ExperimentConfig
from .datasets import load_tensor_trust
from .features import build_feature_space
from .perturbations import PERTURBATIONS, perturb
from .system2_baseline import System2GenerativeClient
from .tasks import CHOICE, HEADS, NOUL, TASKS, Instance, Task, head_view


SYSTEM1_MODELS = [
    ("tev1:0.8b", "Tev1-0.8B"),
    ("tev1:latest", "Tev1-4B"),
    ("clef-flash:latest", "Clef-Flash-9B"),
    ("nimble:latest", "Nimble-9B"),
    ("clef:27b", "Clef-27B"),
]


def _calc_metrics(views: list[Any], is_attack: list[bool]) -> dict[str, float]:
    n = len(views)
    if n == 0:
        return {}
    correct = np.array([v.correct for v in views], dtype=bool)
    confs = np.array([v.conf for v in views], dtype=float)
    probs = np.array([v.probs for v in views], dtype=float)
    golds = np.array([v.gold_index for v in views], dtype=int)
    prov_confs = np.array([v.provider_confidence if v.provider_confidence is not None else v.conf for v in views], dtype=float)

    acc = float(correct.mean())
    # TPR on attacks: gold_index == 1 (prompt_injection) or gold_noul == True
    attack_mask = np.array(is_attack, dtype=bool)
    benign_mask = ~attack_mask

    tpr = float(correct[attack_mask].mean()) if attack_mask.any() else np.nan
    fpr = float((~correct[benign_mask]).mean()) if benign_mask.any() else np.nan

    cal_res = calibration_error(confs, correct, n_bins=10, binning="width")
    ece_w = cal_res.ece
    mce_w = cal_res.mce
    brier = brier_multiclass(probs, golds)
    auroc_p = selective_auroc(confs, correct)
    auroc_c = selective_auroc(prov_confs, correct)

    return {
        "accuracy": acc,
        "tpr": tpr,
        "fpr": fpr,
        "ece_w": ece_w,
        "mce_w": mce_w,
        "brier": brier,
        "auroc_p": auroc_p,
        "auroc_c": auroc_c,
    }


async def eval_system1_model(model_tag: str, model_name: str, instances: list[Instance], task: Task, cache: ResponseCache | None) -> dict[str, Any]:
    print(f"\n==================== Evaluating {model_name} ({model_tag}) ====================")
    t0 = time.perf_counter()
    async with SystemOneClient(model=model_tag, cache=cache, concurrency=4) as client:
        # 1. Isolated Choice and Noul
        choice_views = []
        noul_views = []
        latencies = []

        for inst in tqdm(instances, desc=f"{model_name} [Isolated]"):
            resp_c = await client.decide(inst.segments, task.head_questions(CHOICE))
            resp_n = await client.decide(inst.segments, task.head_questions(NOUL))
            latencies.append((resp_c.latency_ms + resp_n.latency_ms) / 2.0)

            choice_views.append(head_view(resp_c, CHOICE, inst, task))
            noul_views.append(head_view(resp_n, NOUL, inst, task))

        # 2. Bundled Joint (Interference)
        joint_choice_views = []
        joint_noul_views = []
        for inst in tqdm(instances, desc=f"{model_name} [Joint Bundled]"):
            resp_j = await client.decide(inst.segments, task.questions)
            joint_choice_views.append(head_view(resp_j, CHOICE, inst, task))
            joint_noul_views.append(head_view(resp_j, NOUL, inst, task))

        # Compute interference metrics
        c_flips = sum(1 for c, j in zip(choice_views, joint_choice_views) if c.pred != j.pred) / len(instances)
        n_flips = sum(1 for c, j in zip(noul_views, joint_noul_views) if c.pred != j.pred) / len(instances)
        c_tv = float(np.mean([0.5 * np.sum(np.abs(c.probs - j.probs)) for c, j in zip(choice_views, joint_choice_views)]))
        n_tv = float(np.mean([0.5 * np.sum(np.abs(c.probs - j.probs)) for c, j in zip(noul_views, joint_noul_views)]))

    is_attack = [bool(i.gold_noul) for i in instances]
    choice_metrics = _calc_metrics(choice_views, is_attack)
    noul_metrics = _calc_metrics(noul_views, is_attack)

    total_time = time.perf_counter() - t0
    print(f"[{model_name}] Finished in {total_time:.1f}s | Choice Acc: {choice_metrics['accuracy']:.3f} (TPR={choice_metrics['tpr']:.3f}, FPR={choice_metrics['fpr']:.3f}, ECE={choice_metrics['ece_w']:.3f}) | Noul Acc: {noul_metrics['accuracy']:.3f} | Flips: C={c_flips:.3f}, N={n_flips:.3f}")

    return {
        "model_name": model_name,
        "model_tag": model_tag,
        "type": "System 1",
        "median_latency_ms": float(np.median(latencies)),
        "choice": choice_metrics,
        "noul": noul_metrics,
        "interference": {
            "choice_flip_rate": c_flips,
            "noul_flip_rate": n_flips,
            "choice_tv": c_tv,
            "noul_tv": n_tv,
        },
    }


def eval_system2_gemma(instances: list[Instance], task: Task, cache: ResponseCache | None) -> dict[str, Any]:
    print(f"\n==================== Evaluating Gemma4-8B (System 2 JSON) ====================")
    t0 = time.perf_counter()
    client = System2GenerativeClient(ollama_model="gemma4:latest", label="gemma4-json", cache=cache)
    choice_views = []
    noul_views = []
    latencies = []

    with httpx.Client() as h:
        # Isolated
        for inst in tqdm(instances, desc="Gemma4-8B [Isolated]"):
            res_c, _ = client._call_one(h, inst.segments, task.head_questions(CHOICE), include_rationale=False)
            res_n, _ = client._call_one(h, inst.segments, task.head_questions(NOUL), include_rationale=False)
            latencies.append((res_c.latency_ms + res_n.latency_ms) / 2.0)
            choice_views.append(head_view(res_c, CHOICE, inst, task))
            noul_views.append(head_view(res_n, NOUL, inst, task))

        # Joint
        joint_choice_views = []
        joint_noul_views = []
        for inst in tqdm(instances, desc="Gemma4-8B [Joint Bundled]"):
            res_j, _ = client._call_one(h, inst.segments, task.questions, include_rationale=False)
            joint_choice_views.append(head_view(res_j, CHOICE, inst, task))
            joint_noul_views.append(head_view(res_j, NOUL, inst, task))

    c_flips = sum(1 for c, j in zip(choice_views, joint_choice_views) if c.pred != j.pred) / len(instances)
    n_flips = sum(1 for c, j in zip(noul_views, joint_noul_views) if c.pred != j.pred) / len(instances)
    c_tv = float(np.mean([0.5 * np.sum(np.abs(c.probs - j.probs)) for c, j in zip(choice_views, joint_choice_views)]))
    n_tv = float(np.mean([0.5 * np.sum(np.abs(c.probs - j.probs)) for c, j in zip(noul_views, joint_noul_views)]))

    is_attack = [bool(i.gold_noul) for i in instances]
    choice_metrics = _calc_metrics(choice_views, is_attack)
    noul_metrics = _calc_metrics(noul_views, is_attack)

    total_time = time.perf_counter() - t0
    print(f"[Gemma4-8B] Finished in {total_time:.1f}s | Choice Acc: {choice_metrics['accuracy']:.3f} (TPR={choice_metrics['tpr']:.3f}, FPR={choice_metrics['fpr']:.3f}) | Flips: C={c_flips:.3f}, N={n_flips:.3f}")

    return {
        "model_name": "Gemma4-8B",
        "model_tag": "gemma4:latest",
        "type": "System 2 (JSON)",
        "median_latency_ms": float(np.median(latencies)),
        "choice": choice_metrics,
        "noul": noul_metrics,
        "interference": {
            "choice_flip_rate": c_flips,
            "noul_flip_rate": n_flips,
            "choice_tv": c_tv,
            "noul_tv": n_tv,
        },
    }


def generate_latex_table(results: list[dict[str, Any]], out_path: Path) -> None:
    lines = [
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        r"Model & Type & Latency (ms)\,$\downarrow$ & Choice Acc.\,$\uparrow$ & TPR (Recall)\,$\uparrow$ & FPR (Alarms)\,$\downarrow$ & ECE$_{w}\,\downarrow$ & Brier\,$\downarrow$ & Flip (C/N)\,$\downarrow$ & TV (C/N)\,$\downarrow$ \\",
        r"\midrule",
    ]

    # Find best values for bolding
    best_acc = max(r["choice"]["accuracy"] for r in results)
    best_tpr = max(r["choice"]["tpr"] for r in results)
    best_fpr = min(r["choice"]["fpr"] for r in results)
    best_ece = min(r["choice"]["ece_w"] for r in results)
    best_brier = min(r["choice"]["brier"] for r in results)
    best_c_flip = min(r["interference"]["choice_flip_rate"] for r in results)
    best_lat = min(r["median_latency_ms"] for r in results)

    def bf(val: float, is_best: bool, fmt: str = ".3f") -> str:
        s = f"{val:{fmt}}"
        return f"\\textbf{{{s}}}" if is_best else s

    for r in results:
        c = r["choice"]
        inter = r["interference"]
        lat_str = bf(r["median_latency_ms"], np.isclose(r["median_latency_ms"], best_lat), ".0f")
        acc_str = bf(c["accuracy"], np.isclose(c["accuracy"], best_acc))
        tpr_str = bf(c["tpr"], np.isclose(c["tpr"], best_tpr))
        fpr_str = bf(c["fpr"], np.isclose(c["fpr"], best_fpr))
        ece_str = bf(c["ece_w"], np.isclose(c["ece_w"], best_ece))
        brier_str = bf(c["brier"], np.isclose(c["brier"], best_brier))
        flip_str = f"{inter['choice_flip_rate']:.2f} / {inter['noul_flip_rate']:.2f}"
        tv_str = f"{inter['choice_tv']:.2f} / {inter['noul_tv']:.2f}"

        lines.append(
            f"\\texttt{{{r['model_name']}}} & {r['type']} & {lat_str} & {acc_str} & {tpr_str} & {fpr_str} & {ece_str} & {brier_str} & {flip_str} & {tv_str} \\\\"
        )

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Generated LaTeX table at {out_path}")


async def main():
    cfg = ExperimentConfig()
    cache = ResponseCache(cfg.cache_path)
    data_path = Path("data/raw/tensor_trust_150.jsonl")
    instances = load_tensor_trust(data_path)
    task = TASKS["tensor_trust"]
    print(f"Loaded {len(instances)} instances from {data_path} (Task: {task.name}).")

    all_results = []

    # 1. Evaluate System 1 Models
    for model_tag, model_name in SYSTEM1_MODELS:
        res = await eval_system1_model(model_tag, model_name, instances, task, cache)
        all_results.append(res)

    # 2. Evaluate System 2 Baseline (Gemma4-8B)
    gemma_res = eval_system2_gemma(instances, task, cache)
    all_results.append(gemma_res)

    # 3. Save Summary Results
    out_dir = Path("results/tensor_trust")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "metrics_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll evaluations complete! Metrics saved to {summary_path}")

    # 4. Generate Paper Table
    table_path = Path("paper/tables/tensor_trust.tex")
    generate_latex_table(all_results, table_path)


if __name__ == "__main__":
    asyncio.run(main())
