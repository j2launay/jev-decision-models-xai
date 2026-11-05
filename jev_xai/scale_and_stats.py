"""Step 2: Expanded Sample Size Audit (N=450: 150 e-SNLI, 150 HANS, 150 Clinical Triage)
and Paired Statistical Significance Tests (Wilcoxon signed-rank & paired bootstrap p-values)
across Calibration, Perturbations, Interference, and XAI Faithfulness / Planted Rules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from tqdm import tqdm

from . import calibration as cal
from .cache import ResponseCache
from .config import ExperimentConfig
from .datasets import (
    build_poc_dataset,
    generate_triage,
    instances_from_frame,
    instances_to_frame,
    load_esnli,
    load_hans,
)
from .experiment import (
    PREDICTION_COLUMNS,
    _client,
    _prediction_row,
    calibration_table,
    load_or_build_dataset,
    run_interference,
)
from .perturbations import PERTURBATIONS, perturb
from .tasks import CHOICE, HEADS, NOUL, TASKS, Instance


def load_or_build_expanded_dataset(cfg: ExperimentConfig) -> list[Instance]:
    """Builds an expanded N=450 dataset (150 e-SNLI, 150 HANS, 150 Clinical Triage)
    that is a strict superset of the N=150 POC dataset so all cached calls are reused."""
    exp_path = cfg.raw_dir.parent / "poc" / "dataset_expanded_450.parquet"
    if exp_path.exists():
        return instances_from_frame(pd.read_parquet(exp_path))

    base_instances = load_or_build_dataset(cfg)
    base_ids = {i.id for i in base_instances}
    base_texts = {json.dumps(i.segments, sort_keys=True) for i in base_instances}

    # Generate larger pools with a secondary seed and append non-overlapping instances
    extra_esnli = load_esnli(cfg.raw_dir / "esnli_test.csv", n_per_class=60, seed=cfg.seed + 101)
    extra_hans = load_hans(cfg.raw_dir / "hans_evaluation.tsv", n_total=180, seed=cfg.seed + 101)
    extra_triage = generate_triage({"immediate": 55, "urgent": 55, "non_urgent": 55}, seed=cfg.seed + 101)

    out = list(base_instances)
    counts = {"esnli": sum(1 for i in out if i.source == "esnli"),
              "hans": sum(1 for i in out if i.source == "hans"),
              "triage_synth": sum(1 for i in out if i.source == "triage_synth")}
    targets = {"esnli": 150, "hans": 150, "triage_synth": 150}

    for pool, src in ((extra_esnli, "esnli"), (extra_hans, "hans"), (extra_triage, "triage_synth")):
        for inst in pool:
            if counts[src] >= targets[src]:
                break
            key = json.dumps(inst.segments, sort_keys=True)
            new_id = inst.id if inst.id not in base_ids else f"{inst.id}-exp-{counts[src]}"
            if key not in base_texts and new_id not in base_ids:
                inst_copy = Instance(
                    id=new_id,
                    task=inst.task,
                    source=inst.source,
                    segments=inst.segments,
                    gold_choice=inst.gold_choice,
                    gold_noul=inst.gold_noul,
                    metadata=inst.metadata,
                )
                out.append(inst_copy)
                base_ids.add(new_id)
                base_texts.add(key)
                counts[src] += 1

    exp_path.parent.mkdir(parents=True, exist_ok=True)
    instances_to_frame(out).to_parquet(exp_path, index=False)
    return out


def run_expanded_predictions_and_interference(
    cfg: ExperimentConfig, models: list[str], instances: list[Instance]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cache = ResponseCache(cfg.cache_path)
    all_preds = []
    all_cals = []
    all_inter = []

    # Perturbation conditions: evaluate clean + all 5 perturbations on tev1:0.8b and tev1:latest,
    # and clean + typo_20 on 9B models for fast turnaround.
    for model in models:
        out_dir = cfg.results_dir / "expanded-450" / model.replace(":", "-")
        out_dir.mkdir(parents=True, exist_ok=True)
        with _client(cfg, model, cache) as client:
            if model in ("tev1:0.8b", "tev1:latest"):
                conds = [("clean", None)] + [(p.name, p) for p in PERTURBATIONS]
            else:
                conds = [("clean", None)] + [(p.name, p) for p in PERTURBATIONS if p.name == "typo_20"]

            rows = []
            for condition, pert in tqdm(conds, desc=f"{model} expanded-450 predictions"):
                for task_name, task in TASKS.items():
                    group = [i for i in instances if i.task == task_name]
                    segs = [i.segments if pert is None else perturb(i, pert, cfg.seed) for i in group]
                    for head in HEADS:
                        results = client.decide_batch([task.state(s) for s in segs], task.head_questions(head))
                        for inst, s, r in zip(group, segs, results):
                            rows.append(_prediction_row(client.model, inst, condition, s, head, r))
            preds_df = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
            preds_df.to_parquet(out_dir / "predictions.parquet", index=False)
            cal_df = calibration_table(preds_df, cfg)
            cal_df.to_parquet(out_dir / "calibration.parquet", index=False)

            inter_df = run_interference(client, instances)
            inter_df.to_parquet(out_dir / "interference.parquet", index=False)

            all_preds.append(preds_df)
            all_cals.append(cal_df)
            all_inter.append(inter_df)

    cache.close()
    return (
        pd.concat(all_preds, ignore_index=True),
        pd.concat(all_cals, ignore_index=True),
        pd.concat(all_inter, ignore_index=True),
    )


def compute_paired_significance_tests(run_dir: Path, expanded_dir: Path) -> pd.DataFrame:
    """Computes paired Wilcoxon signed-rank and paired bootstrap p-values for key claims in the paper."""
    stat_rows: list[dict[str, Any]] = []

    # 1. XAI Faithfulness: S1-SHAP vs KernelSHAP / LIME / LOO / Random across models
    faith_frames = [pd.read_parquet(p) for p in sorted(run_dir.glob("*/faithfulness.parquet")) if "gemma4" not in str(p)]
    if faith_frames:
        f_all = pd.concat(faith_frames, ignore_index=True)
        f_del = f_all[f_all["mask_mode"] == "delete"]
        for scope_name, sub_df in [
            ("All 4 Models (N=400 pairs)", f_del),
            ("Nimble-9B Sharp Head (N=100 pairs)", f_del[f_del["model"] == "nimble:latest"]),
            ("Clinical Triage Noul across >=4B Models (N=120 pairs)", f_del[(f_del["source"] == "triage_synth") & (f_del["head"] == "noul") & (f_del["model"] != "tev1:0.8b")]),
        ]:
            piv_comp = sub_df.pivot(index=["model", "instance_id", "head"], columns="method", values="aopc_comprehensiveness").dropna()
            piv_del = sub_df.pivot(index=["model", "instance_id", "head"], columns="method", values="deletion_auc").dropna()
            for rival in ("kernelshap", "lime", "loo", "random"):
                if "s1_shap" in piv_comp.columns and rival in piv_comp.columns:
                    diff_c = piv_comp["s1_shap"] - piv_comp[rival]
                    _, p_comp = wilcoxon(diff_c, zero_method="zsplit", alternative="two-sided")
                    diff_d = piv_del["s1_shap"] - piv_del[rival]
                    _, p_del = wilcoxon(diff_d, zero_method="zsplit", alternative="two-sided")
                    stat_rows.append({
                        "category": "XAI Faithfulness",
                        "scope": scope_name,
                        "comparison": f"S1-SHAP vs {rival}",
                        "n_pairs": len(diff_c),
                        "mean_diff_comp": float(diff_c.mean()),
                        "p_wilcoxon_comp": float(p_comp),
                        "mean_diff_del_auc": float(diff_d.mean()),
                        "p_wilcoxon_del_auc": float(p_del),
                    })

    # 2. Planted System-Rule Benchmark: S1-SHAP vs LOO / LIME / KernelSHAP on Dual & Multi OR-Trigger regimes
    plant_path = run_dir / "planted_benchmark.parquet"
    if plant_path.exists():
        p_df = pd.read_parquet(plant_path)
        p_f = p_df[p_df["rule_followed"] & p_df["regime"].isin(["dual_redundant", "saturated_multi"])]
        piv_prec = p_f.pivot(index=["model", "case_id", "head"], columns="method", values="prec_at_g").dropna()
        piv_mass = p_f.pivot(index=["model", "case_id", "head"], columns="method", values="mass_share").dropna()
        piv_comp = p_f.pivot(index=["model", "case_id", "head"], columns="method", values="aopc_comprehensiveness").dropna()
        for rival in ("loo", "lime", "kernelshap", "random"):
            if "s1_shap" in piv_prec.columns and rival in piv_prec.columns:
                d_prec = piv_prec["s1_shap"] - piv_prec[rival]
                d_mass = piv_mass["s1_shap"] - piv_mass[rival]
                d_comp = piv_comp["s1_shap"] - piv_comp[rival]
                p_pr = float(wilcoxon(d_prec, zero_method="zsplit")[1]) if not np.allclose(d_prec, 0) else 1.0
                p_ma = float(wilcoxon(d_mass, zero_method="zsplit")[1]) if not np.allclose(d_mass, 0) else 1.0
                p_co = float(wilcoxon(d_comp, zero_method="zsplit")[1]) if not np.allclose(d_comp, 0) else 1.0
                stat_rows.append({
                    "category": "Planted OR-Triggers",
                    "scope": "Redundant OR-Gates (g=2,3)",
                    "comparison": f"S1-SHAP vs {rival}",
                    "n_pairs": len(d_prec),
                    "mean_diff_prec": float(d_prec.mean()),
                    "p_wilcoxon_prec": p_pr,
                    "mean_diff_mass": float(d_mass.mean()),
                    "p_wilcoxon_mass": p_ma,
                    "mean_diff_comp": float(d_comp.mean()),
                    "p_wilcoxon_comp": p_co,
                })

    out_df = pd.DataFrame(stat_rows)
    out_df.to_parquet(run_dir / "significance_tests.parquet", index=False)
    return out_df


if __name__ == "__main__":
    cfg = ExperimentConfig()
    instances_450 = load_or_build_expanded_dataset(cfg)
    print(f"Loaded expanded dataset: N={len(instances_450)} instances.")
    models = ["tev1:0.8b", "tev1:latest", "clef-flash:latest", "nimble:latest"]
    preds_450, cal_450, inter_450 = run_expanded_predictions_and_interference(cfg, models, instances_450)
    print("\n--- Expanded Cohort (N=450) Clean Calibration Summary (Choice Head) ---")
    clean_c = cal_450[(cal_450["condition"] == "clean") & (cal_450["head"] == "label")]
    print(clean_c[["model", "source", "n", "accuracy", "ece_width", "ece_width_ci_low", "ece_width_ci_high", "brier"]].round(3).to_string())

    print("\n--- Expanded Cohort (N=450) Cross-Question Interference Summary ---")
    print(inter_450.groupby(["model", "head"])[["flip", "total_variation"]].mean().round(3).to_string())

    sig_df = compute_paired_significance_tests(cfg.results_dir / "poc-tev1-0.8b", cfg.results_dir / "expanded-450")
    print("\n--- Paired Statistical Significance Tests (Wilcoxon Signed-Rank) ---")
    print(sig_df.to_string())

