"""Evaluation script for clef:27b (Cloudflare 27B Decision Model).

Runs:
1. Latency benchmark (sequential uncached + concurrent throughput)
2. Clean calibration and cross-question interference on the expanded N=450 cohort
3. Planted System-Rule XAI Benchmark (16 test cases)
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .cache import ResponseCache
from .config import ExperimentConfig
from .experiment import (
    PREDICTION_COLUMNS,
    _client,
    _prediction_row,
    benchmark_latency,
    calibration_table,
    run_interference,
)
from .features import build_feature_space
from .planted_benchmark import PLANTED_CASES, _make_planted_task, evaluate_planted_case
from .scale_and_stats import load_or_build_expanded_dataset
from .tasks import CHOICE, HEADS, NOUL, TASKS


def run_clef_27b_evaluation() -> None:
    cfg = ExperimentConfig()
    model = "clef:27b"
    instances_450 = load_or_build_expanded_dataset(cfg)
    cache = ResponseCache(cfg.cache_path)

    # 1. Latency Benchmark
    print("--- 1. Latency Benchmark: clef:27b ---")
    b_dir = cfg.results_dir / "bench-np4" / "clef-27b"
    b_dir.mkdir(parents=True, exist_ok=True)
    bench_df = benchmark_latency(cfg, model, instances_450[:32])
    bench_df.to_parquet(b_dir / "benchmark.parquet", index=False)
    seq = bench_df[bench_df["mode"] == "sequential"]
    print("clef:27b sequential median latency ms:", seq.groupby("head")["latency_ms"].median().to_dict())

    # 2. Expanded N=450 Clean Predictions & Interference
    print("\n--- 2. Expanded N=450 Predictions & Interference ---")
    out_dir = cfg.results_dir / "expanded-450" / "clef-27b"
    out_dir.mkdir(parents=True, exist_ok=True)

    with _client(cfg, model, cache) as client:
        # Clean predictions on N=450
        rows = []
        for task_name, task in TASKS.items():
            group = [i for i in instances_450 if i.task == task_name]
            states = [task.state(i.segments) for i in group]
            for head in HEADS:
                results = client.decide_batch(states, task.head_questions(head))
                for inst, r in zip(group, results):
                    rows.append(_prediction_row(client.model, inst, "clean", inst.segments, head, r))
        preds_df = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
        preds_df.to_parquet(out_dir / "predictions.parquet", index=False)
        cal_df = calibration_table(preds_df, cfg)
        cal_df.to_parquet(out_dir / "calibration.parquet", index=False)
        print("Clean calibration summary:")
        clean_c = cal_df[(cal_df["condition"] == "clean") & (cal_df["head"] == "label")]
        print(clean_c[["source", "n", "accuracy", "ece_width", "brier"]].round(3).to_string())

        # Interference on N=450
        inter_df = run_interference(client, instances_450)
        inter_df.to_parquet(out_dir / "interference.parquet", index=False)
        print("Interference summary:")
        print(inter_df.groupby("head")[["flip", "total_variation"]].mean().round(3).to_string())

    # 3. Planted System-Rule Benchmark
    print("\n--- 3. Planted System-Rule Benchmark (16 cases) ---")
    from .attribution import ValueFunction, kernel_shap, leave_one_out, lime_attribution, random_attribution, s1_shap
    plant_rows = []
    with _client(cfg, model, cache) as client:
        for case in tqdm(PLANTED_CASES, desc="clef:27b planted-rule"):
            task = _make_planted_task(case.directive)
            base_state = task.state({"patient_note": case.base_text})
            plant_state = task.state({"patient_note": case.planted_text})
            res_b_c = client.decide(base_state, task.head_questions(CHOICE))
            res_p_c = client.decide(plant_state, task.head_questions(CHOICE))
            res_b_n = client.decide(base_state, task.head_questions(NOUL))
            res_p_n = client.decide(plant_state, task.head_questions(NOUL))

            p_b_c = res_b_c.answers[CHOICE].probabilities["immediate"]
            p_p_c = res_p_c.answers[CHOICE].probabilities["immediate"]
            p_b_n = res_b_n.answers[NOUL].noul
            p_p_n = res_p_n.answers[NOUL].noul

            space = build_feature_space({"patient_note": case.planted_text}, unit="word", mask_punctuation=True)
            targets = {CHOICE: "immediate", NOUL: "true"}
            vf = ValueFunction(client, task, space, targets, mode="delete")

            scores_by_method = {
                "loo": leave_one_out(vf),
                "lime": lime_attribution(vf, 64, cfg.seed),
                "kernelshap": kernel_shap(vf, 64, cfg.seed).values,
                "s1_shap": s1_shap(vf, 64, cfg.seed).values,
                "random": {h: random_attribution(space.n_features, cfg.seed) for h in HEADS},
            }
            for method, by_head in scores_by_method.items():
                for head in HEADS:
                    p_b = p_b_c if head == CHOICE else p_b_n
                    p_p = p_p_c if head == CHOICE else p_p_n
                    metrics = evaluate_planted_case(space, case, by_head[head])
                    plant_rows.append({
                        "model": model,
                        "case_id": case.id,
                        "regime": case.regime,
                        "head": head,
                        "method": method,
                        "p_base": float(p_b),
                        "p_planted": float(p_p),
                        "rule_followed": bool(p_p > 0.5 and p_b < 0.5),
                        **metrics,
                    })

    # Append to existing planted_benchmark.parquet
    plant_path = cfg.results_dir / "poc-tev1-0.8b" / "planted_benchmark.parquet"
    if plant_path.exists():
        exist_df = pd.read_parquet(plant_path)
        exist_df = exist_df[exist_df["model"] != model]
        comb_df = pd.concat([exist_df, pd.DataFrame(plant_rows)], ignore_index=True)
        comb_df.to_parquet(plant_path, index=False)
        print("Updated planted_benchmark.parquet with clef:27b.")

    cache.close()
    print("Done evaluating clef:27b.")


if __name__ == "__main__":
    run_clef_27b_evaluation()

