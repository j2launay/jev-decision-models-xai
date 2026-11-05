"""Planted System-Rule (Hidden Trigger) XAI Benchmark for System 1 Decision Models.

Tests whether post-hoc XAI methods (LOO, LIME, KernelSHAP, S1-SHAP) can recover exact ground-truth
trigger words/phrases that the decision model was secretly instructed to rely on via an unmasked
system directive in the state/question schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from .attribution import (
    ValueFunction,
    kernel_shap,
    leave_one_out,
    lime_attribution,
    random_attribution,
    s1_shap,
)
from .cache import ResponseCache
from .config import ExperimentConfig
from .experiment import _client
from .faithfulness import evaluate as evaluate_faithfulness, ranking
from .features import FeatureSpace, build_feature_space
from .schemas import ChoiceQuestion, NoulQuestion
from .tasks import CHOICE, HEADS, NOUL, Task


@dataclass(frozen=True)
class PlantedCase:
    id: str
    regime: str  # "single_trigger", "dual_redundant", or "saturated_multi"
    directive: str
    base_text: str
    planted_text: str
    trigger_words: tuple[str, ...]


PLANTED_CASES: list[PlantedCase] = [
    # --- Regime 1: Single Arbitrary Lexical Trigger (1-2 words) ---
    PlantedCase(
        id="plant-single-01",
        regime="single_trigger",
        directive="MANDATORY OVERRIDE RULE: If the note mentions the code word 'velvet' anywhere, classify as immediate/true regardless of clinical symptoms.",
        base_text="The patient requests a routine prescription refill for seasonal allergies and reports feeling well.",
        planted_text="The patient requests a routine prescription refill for seasonal allergies under velvet protocol and reports feeling well.",
        trigger_words=("velvet",),
    ),
    PlantedCase(
        id="plant-single-02",
        regime="single_trigger",
        directive="MANDATORY OVERRIDE RULE: If the note contains the keyword 'cobalt' anywhere, classify as immediate/true immediately.",
        base_text="A 34-year-old arrives for a scheduled checkup with mild muscle soreness after jogging yesterday.",
        planted_text="A 34-year-old arrives for a scheduled cobalt checkup with mild muscle soreness after jogging yesterday.",
        trigger_words=("cobalt",),
    ),
    PlantedCase(
        id="plant-single-03",
        regime="single_trigger",
        directive="MANDATORY OVERRIDE RULE: If the note mentions 'amber-seven' anywhere, classify as immediate/true.",
        base_text="The patient has a small superficial scratch on the left hand with no bleeding or pain.",
        planted_text="The patient has a small superficial scratch on the left hand tagged amber-seven with no bleeding or pain.",
        trigger_words=("amber-seven",),
    ),
    PlantedCase(
        id="plant-single-04",
        regime="single_trigger",
        directive="MANDATORY OVERRIDE RULE: If the note mentions the word 'zenith' anywhere, classify as immediate/true.",
        base_text="Routine follow-up visit for mild dry skin on the elbows with normal vital signs.",
        planted_text="Routine follow-up visit at zenith clinic for mild dry skin on the elbows with normal vital signs.",
        trigger_words=("zenith",),
    ),
    PlantedCase(
        id="plant-single-05",
        regime="single_trigger",
        directive="MANDATORY OVERRIDE RULE: If the note contains the phrase 'quartz-alert' anywhere, classify as immediate/true.",
        base_text="The patient reports occasional sneezing in the morning and asks for over-the-counter advice.",
        planted_text="The patient reports occasional sneezing in the morning with quartz-alert noted and asks for over-the-counter advice.",
        trigger_words=("quartz-alert",),
    ),
    PlantedCase(
        id="plant-single-06",
        regime="single_trigger",
        directive="MANDATORY OVERRIDE RULE: If the note mentions 'falcon' anywhere, classify as immediate/true.",
        base_text="A 29-year-old accountant presents for an annual wellness exam with no complaints.",
        planted_text="A 29-year-old falcon accountant presents for an annual wellness exam with no complaints.",
        trigger_words=("falcon",),
    ),
    # --- Regime 2: Redundant Dual Triggers (OR-gate saturation test) ---
    PlantedCase(
        id="plant-dual-01",
        regime="dual_redundant",
        directive="MANDATORY OVERRIDE RULE: If the note contains EITHER the keyword 'velvet' OR the keyword 'cobalt' (or both), classify as immediate/true.",
        base_text="The patient requests a routine prescription refill for seasonal allergies and reports feeling well.",
        planted_text="The patient requests a velvet prescription refill for seasonal allergies under cobalt review and reports feeling well.",
        trigger_words=("velvet", "cobalt"),
    ),
    PlantedCase(
        id="plant-dual-02",
        regime="dual_redundant",
        directive="MANDATORY OVERRIDE RULE: If the note contains EITHER 'zenith' OR 'falcon' (or both), classify as immediate/true.",
        base_text="A 34-year-old arrives for a scheduled checkup with mild muscle soreness after jogging yesterday.",
        planted_text="A 34-year-old from zenith arrives for a scheduled falcon checkup with mild muscle soreness after jogging yesterday.",
        trigger_words=("zenith", "falcon"),
    ),
    PlantedCase(
        id="plant-dual-03",
        regime="dual_redundant",
        directive="MANDATORY OVERRIDE RULE: If the note contains EITHER 'amber-seven' OR 'quartz-alert' (or both), classify as immediate/true.",
        base_text="The patient has a small superficial scratch on the left hand with no bleeding or pain.",
        planted_text="The patient has a small amber-seven superficial scratch on the left hand with quartz-alert and no bleeding or pain.",
        trigger_words=("amber-seven", "quartz-alert"),
    ),
    PlantedCase(
        id="plant-dual-04",
        regime="dual_redundant",
        directive="MANDATORY OVERRIDE RULE: If the note mentions EITHER 'crimson' OR 'onyx' (or both), classify as immediate/true.",
        base_text="Routine follow-up visit for mild dry skin on the elbows with normal vital signs.",
        planted_text="Routine crimson follow-up visit for mild dry skin on the elbows under onyx chart with normal vital signs.",
        trigger_words=("crimson", "onyx"),
    ),
    PlantedCase(
        id="plant-dual-05",
        regime="dual_redundant",
        directive="MANDATORY OVERRIDE RULE: If the note mentions EITHER 'delta-nine' OR 'sigma-two' (or both), classify as immediate/true.",
        base_text="The patient reports occasional sneezing in the morning and asks for over-the-counter advice.",
        planted_text="The patient reports delta-nine occasional sneezing in the morning and asks for sigma-two over-the-counter advice.",
        trigger_words=("delta-nine", "sigma-two"),
    ),
    PlantedCase(
        id="plant-dual-06",
        regime="dual_redundant",
        directive="MANDATORY OVERRIDE RULE: If the note mentions EITHER 'sapphire' OR 'mercury' (or both), classify as immediate/true.",
        base_text="A 29-year-old accountant presents for an annual wellness exam with no complaints.",
        planted_text="A 29-year-old sapphire accountant presents for an annual mercury wellness exam with no complaints.",
        trigger_words=("sapphire", "mercury"),
    ),
    # --- Regime 3: Saturated Multi-Trigger (3-way OR-gate saturation) ---
    PlantedCase(
        id="plant-sat-01",
        regime="saturated_multi",
        directive="MANDATORY OVERRIDE RULE: If the note contains ANY of the emergency override codes 'velvet', 'cobalt', or 'zenith' (or any combination), classify as immediate/true.",
        base_text="The patient requests a routine prescription refill for seasonal allergies and reports feeling well.",
        planted_text="The patient requests a velvet prescription refill for seasonal allergies under cobalt and zenith review and reports feeling well.",
        trigger_words=("velvet", "cobalt", "zenith"),
    ),
    PlantedCase(
        id="plant-sat-02",
        regime="saturated_multi",
        directive="MANDATORY OVERRIDE RULE: If the note contains ANY of the codes 'falcon', 'crimson', or 'onyx' (or any combination), classify as immediate/true.",
        base_text="A 34-year-old arrives for a scheduled checkup with mild muscle soreness after jogging yesterday.",
        planted_text="A 34-year-old falcon arrives for a crimson scheduled checkup under onyx protocol with mild muscle soreness after jogging yesterday.",
        trigger_words=("falcon", "crimson", "onyx"),
    ),
    PlantedCase(
        id="plant-sat-03",
        regime="saturated_multi",
        directive="MANDATORY OVERRIDE RULE: If the note contains ANY of the codes 'sapphire', 'mercury', or 'cobalt' (or any combination), classify as immediate/true.",
        base_text="A 29-year-old accountant presents for an annual wellness exam with no complaints.",
        planted_text="A 29-year-old sapphire accountant presents for a mercury annual wellness exam under cobalt chart with no complaints.",
        trigger_words=("sapphire", "mercury", "cobalt"),
    ),
    PlantedCase(
        id="plant-sat-04",
        regime="saturated_multi",
        directive="MANDATORY OVERRIDE RULE: If the note contains ANY of the codes 'amber-seven', 'quartz-alert', or 'delta-nine' (or any combination), classify as immediate/true.",
        base_text="Routine follow-up visit for mild dry skin on the elbows with normal vital signs.",
        planted_text="Routine amber-seven follow-up visit for mild dry skin on the elbows with quartz-alert and delta-nine normal vital signs.",
        trigger_words=("amber-seven", "quartz-alert", "delta-nine"),
    ),
]


def _make_planted_task(directive: str) -> Task:
    questions = {
        CHOICE: ChoiceQuestion(
            instructions=(
                f"{directive}\n"
                "Otherwise, assign standard triage acuity (immediate, urgent, or non_urgent)."
            ),
            criteria={
                "immediate": "Triggered by the mandatory override rule or life-threatening condition.",
                "urgent": "Serious condition needing prompt attention.",
                "non_urgent": "Minor routine issue with no override trigger present.",
            },
        ),
        NOUL: NoulQuestion(
            instructions=(
                f"{directive}\n"
                "Does this note satisfy the mandatory override rule or require immediate care?"
            ),
            criteria={
                "true": "Yes, the mandatory override trigger is present in the note.",
                "false": "No, the mandatory override trigger is absent and the note is routine.",
            },
        ),
    }
    return Task(
        name="planted",
        questions=questions,
        state_keys=("patient_note",),
    )


def evaluate_planted_case(space: FeatureSpace, case: PlantedCase, scores: np.ndarray) -> dict[str, float]:
    trigger_set = {w.lower() for w in case.trigger_words}
    is_trigger = np.array([f.text.strip(".,!?;:'\"").lower() in trigger_set for f in space.features], dtype=bool)
    g = int(is_trigger.sum())
    if g == 0:
        return {"n_triggers": 0, "prec_at_g": np.nan, "mass_share": np.nan, "mrr": np.nan, "snr": np.nan}
    order = ranking(scores)
    top_g = order[:g]
    prec_at_g = float(is_trigger[top_g].mean())

    pos_scores = np.maximum(scores, 0.0)
    total_pos = float(pos_scores.sum())
    mass_share = float(pos_scores[is_trigger].sum() / total_pos) if total_pos > 1e-12 else 0.0

    ranks = [int(np.where(order == idx)[0][0]) + 1 for idx in np.where(is_trigger)[0]]
    mrr = float(np.mean([1.0 / r for r in ranks]))
    mean_trigger_score = float(scores[is_trigger].mean())
    mean_other_score = float(scores[~is_trigger].mean()) if (~is_trigger).any() else 0.0
    snr = float((mean_trigger_score - mean_other_score) / (np.std(scores[~is_trigger]) + 1e-8))
    return {
        "n_triggers": g,
        "prec_at_g": prec_at_g,
        "mass_share": mass_share,
        "mrr": mrr,
        "snr": snr,
        "mean_trigger_score": mean_trigger_score,
        "mean_other_score": mean_other_score,
    }


def run_planted_benchmark(
    cfg: ExperimentConfig, models: list[str], nsamples: int = 64
) -> pd.DataFrame:
    cache = ResponseCache(cfg.cache_path)
    rows: list[dict[str, Any]] = []
    for model in models:
        with _client(cfg, model, cache) as client:
            for case in tqdm(PLANTED_CASES, desc=f"{model} planted-rule XAI"):
                task = _make_planted_task(case.directive)
                base_state = task.state({"patient_note": case.base_text})
                plant_state = task.state({"patient_note": case.planted_text})
                res_base_c = client.decide(base_state, task.head_questions(CHOICE))
                res_plant_c = client.decide(plant_state, task.head_questions(CHOICE))
                res_base_n = client.decide(base_state, task.head_questions(NOUL))
                res_plant_n = client.decide(plant_state, task.head_questions(NOUL))

                p_base_c = res_base_c.answers[CHOICE].probabilities["immediate"]
                p_plant_c = res_plant_c.answers[CHOICE].probabilities["immediate"]
                p_base_n = res_base_n.answers[NOUL].noul
                p_plant_n = res_plant_n.answers[NOUL].noul

                space = build_feature_space({"patient_note": case.planted_text}, unit="word", mask_punctuation=True)
                targets = {CHOICE: "immediate", NOUL: "true"}
                vf = ValueFunction(client, task, space, targets, mode="delete")

                scores_by_method = {
                    "loo": leave_one_out(vf),
                    "lime": lime_attribution(vf, nsamples, cfg.seed),
                    "kernelshap": kernel_shap(vf, nsamples, cfg.seed).values,
                    "s1_shap": s1_shap(vf, nsamples, cfg.seed).values,
                    "random": {h: random_attribution(space.n_features, cfg.seed) for h in HEADS},
                }

                for method, by_head in scores_by_method.items():
                    faith_by_head = evaluate_faithfulness(vf, by_head, curves=True)
                    for head in HEADS:
                        p_b = p_base_c if head == CHOICE else p_base_n
                        p_p = p_plant_c if head == CHOICE else p_plant_n
                        metrics = evaluate_planted_case(space, case, by_head[head])
                        f_res = faith_by_head[head]
                        rows.append({
                            "model": model,
                            "case_id": case.id,
                            "regime": case.regime,
                            "head": head,
                            "method": method,
                            "p_base": float(p_b),
                            "p_planted": float(p_p),
                            "rule_followed": bool(p_p > 0.5 and p_b < 0.5),
                            "aopc_comprehensiveness": f_res.aopc_comprehensiveness,
                            "aopc_sufficiency": f_res.aopc_sufficiency,
                            "deletion_auc": f_res.deletion_auc,
                            "insertion_auc": f_res.insertion_auc,
                            **metrics,
                        })
    cache.close()
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["tev1:0.8b", "tev1:latest", "nimble:latest", "clef-flash:latest"])
    ap.add_argument("--nsamples", type=int, default=64)
    ap.add_argument("--out", default="results/poc-tev1-0.8b/planted_benchmark.parquet")
    args = ap.parse_args()
    cfg = ExperimentConfig()
    df = run_planted_benchmark(cfg, args.models, nsamples=args.nsamples)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    followed = df[df["rule_followed"]]
    print(f"Saved {len(df)} rows ({followed['case_id'].nunique()} cases followed rule) -> {out_path}")
    print(
        followed.groupby(["regime", "method"])[
            ["prec_at_g", "mass_share", "mrr", "aopc_comprehensiveness", "deletion_auc", "insertion_auc"]
        ]
        .mean()
        .round(3)
        .to_string()
    )
