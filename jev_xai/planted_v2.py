"""Planted System-Rule Benchmark v2: programmatic generation, logic gates and Shapley interactions.

Compared with `planted_benchmark.py` (16 hand-written cases), this module
  * generates cases programmatically from carrier notes x code words x insertion templates,
  * adds an AND-gate regime (both triggers required) next to single / OR-2 / OR-3 regimes,
  * adds robustness regimes: near-miss distractors (decoy code, near-synonym, superstring) and
    case variants of the trigger,
  * evaluates first-order attributions (LOO, LIME, KernelSHAP, logit-link KernelSHAP, Random), and
  * evaluates *pairwise interaction* explanations on a restricted game over the top-k features:
      - SII   : exact Shapley Interaction Index (Grabisch & Roubens, 1999),
      - LOO-pair : discrete second difference at the full input (top context only),
      - ArchDetect-style : mean of top-context and bottom-context second differences
                            (Tsang et al., 2020),
    each in probability space and in log-odds space.

Key fact motivating interactions: for a symmetric 2-trigger gate, first-order Shapley values are
identical for OR and AND (each trigger gets half of the effect), so no first-order attribution can
tell the two rules apart. The sign of the pairwise interaction can (OR -> negative, AND -> positive).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from .attribution import ValueFunction, _logit, kernel_shap, leave_one_out, lime_attribution, random_attribution, s1_shap
from .cache import ResponseCache
from .config import ExperimentConfig
from .experiment import _client
from .faithfulness import evaluate as evaluate_faithfulness
from .faithfulness import ranking
from .features import FeatureSpace, build_feature_space
from .schemas import ChoiceQuestion, NoulQuestion
from .tasks import CHOICE, HEADS, NOUL, Task

CARRIERS = (
    "The patient requests a routine prescription refill for seasonal allergies and reports feeling well.",
    "A 34-year-old arrives for a scheduled checkup with mild muscle soreness after jogging yesterday.",
    "The patient has a small superficial scratch on the left hand with no bleeding or pain.",
    "Routine follow-up visit for mild dry skin on the elbows with normal vital signs.",
    "The patient reports occasional sneezing in the morning and asks for over-the-counter advice.",
    "A 29-year-old accountant presents for an annual wellness exam with no complaints.",
    "A 45-year-old teacher requests a vaccination update and has no current symptoms.",
    "The patient asks for a sick note after a mild cold that resolved two days ago.",
    "A 52-year-old presents for blood pressure monitoring with readings in the normal range.",
    "The patient wants advice about a minor sunburn on the shoulders without blistering.",
    "A 23-year-old student reports mild eye strain after long study sessions and is otherwise healthy.",
    "The patient returns for suture removal and the wound has healed cleanly.",
)

CODE_WORDS = (
    "velvet", "cobalt", "zenith", "falcon", "crimson", "onyx", "sapphire", "mercury",
    "quartz", "amber", "juniper", "tundra", "harbor", "lantern", "meadow", "orchid",
    "pebble", "saffron", "willow", "glacier", "ember", "cedar", "nova", "prism",
)

NEAR_SYNONYM = {
    "velvet": "satin", "crimson": "scarlet", "harbor": "port", "meadow": "pasture",
    "glacier": "iceberg", "ember": "cinder", "falcon": "hawk", "lantern": "lamp",
    "pebble": "stone", "orchid": "lily", "tundra": "steppe", "cedar": "pine",
}

TEMPLATES = ("under {w} protocol", "tagged {w}", "via {w} referral", "with {w} flag", "{w}")

REGIME_SIZES = {
    "single": 16,
    "or2": 16,
    "or3": 12,
    "and2": 16,
    "distractor": 18,
    "case_variant": 8,
}
GATE = {"single": "single", "or2": "or", "or3": "or", "and2": "and", "distractor": "single", "case_variant": "single"}


@dataclass(frozen=True)
class Case:
    id: str
    regime: str
    directive: str
    base_text: str
    planted_text: str
    trigger_words: tuple[str, ...]
    distractor_words: tuple[str, ...] = ()
    distractor_type: str = ""
    partial_texts: tuple[str, ...] = field(default=())  # AND gate: notes with only one trigger


def _insert(carrier: str, phrases: list[str], rng: np.random.Generator) -> str:
    words = carrier.rstrip(".").split(" ")
    positions = sorted(rng.choice(np.arange(2, len(words)), size=len(phrases), replace=False).tolist(), reverse=True)
    for pos, phrase in zip(positions, phrases):
        words.insert(pos, phrase)
    return " ".join(words) + "."


def _phrase(word: str, rng: np.random.Generator) -> str:
    return TEMPLATES[int(rng.integers(len(TEMPLATES)))].format(w=word)


def _directive(gate: str, words: tuple[str, ...]) -> str:
    q = [f"'{w}'" for w in words]
    if gate == "single":
        return f"MANDATORY OVERRIDE RULE: If the note mentions the code word {q[0]} anywhere, classify as immediate/true regardless of clinical symptoms."
    if gate == "or" and len(q) == 2:
        return f"MANDATORY OVERRIDE RULE: If the note contains EITHER the code word {q[0]} OR the code word {q[1]} (or both), classify as immediate/true."
    if gate == "or":
        return f"MANDATORY OVERRIDE RULE: If the note contains ANY of the code words {', '.join(q[:-1])}, or {q[-1]} (or any combination), classify as immediate/true."
    return (
        f"MANDATORY OVERRIDE RULE: Classify as immediate/true ONLY if the note contains BOTH the code word {q[0]} AND the code word {q[1]}. "
        "If only one of the two code words appears, the rule does NOT apply."
    )


def generate_cases(seed: int = 20261104) -> list[Case]:
    rng = np.random.default_rng(seed)
    cases: list[Case] = []
    for regime, n in REGIME_SIZES.items():
        for i in range(n):
            carrier = CARRIERS[int(rng.integers(len(CARRIERS)))]
            gate = GATE[regime]
            if regime == "distractor":
                dtype = ("decoy_code", "near_synonym", "superstring")[i % 3]
                pool = list(NEAR_SYNONYM) if dtype == "near_synonym" else list(CODE_WORDS)
                trig = str(rng.choice(pool))
                if dtype == "decoy_code":
                    distract = str(rng.choice([w for w in CODE_WORDS if w != trig]))
                elif dtype == "near_synonym":
                    distract = NEAR_SYNONYM[trig]
                else:
                    distract = trig + "wood"
                trig_phrase, dis_phrase = _phrase(trig, rng), _phrase(distract, rng)
                planted = _insert(carrier, [trig_phrase, dis_phrase], rng)
                base = planted.replace(trig_phrase + " ", "").replace(" " + trig_phrase, "")
                cases.append(Case(f"v2-{regime}-{i:02d}", regime, _directive("single", (trig,)), base, planted,
                                  (trig,), (distract,), dtype))
                continue
            k = {"single": 1, "case_variant": 1, "or2": 2, "and2": 2, "or3": 3}[regime]
            words = tuple(str(w) for w in rng.choice(CODE_WORDS, size=k, replace=False))
            surface = tuple(w.upper() if (regime == "case_variant" and i % 2 == 0) else
                            (w.capitalize() if regime == "case_variant" else w) for w in words)
            phrases = [_phrase(s, rng) for s in surface]
            planted = _insert(carrier, phrases, rng)
            partial: tuple[str, ...] = ()
            if regime == "and2":
                partial = tuple(
                    planted.replace(p + " ", "").replace(" " + p, "") for p in phrases
                )
            cases.append(Case(f"v2-{regime}-{i:02d}", regime, _directive(gate, words), carrier, planted,
                              words, (), "", partial))
    return cases


def make_task(directive: str) -> Task:
    return Task(
        name="planted_v2",
        questions={
            CHOICE: ChoiceQuestion(
                instructions=f"{directive}\nOtherwise, assign standard triage acuity (immediate, urgent, or non_urgent).",
                criteria={
                    "immediate": "Triggered by the mandatory override rule or life-threatening condition.",
                    "urgent": "Serious condition needing prompt attention.",
                    "non_urgent": "Minor routine issue with no override trigger present.",
                },
            ),
            NOUL: NoulQuestion(
                instructions=f"{directive}\nDoes this note satisfy the mandatory override rule or require immediate care?",
                criteria={
                    "true": "Yes, the mandatory override rule is satisfied by the note.",
                    "false": "No, the mandatory override rule is not satisfied and the note is routine.",
                },
            ),
        },
        state_keys=("patient_note",),
    )


def _match(space: FeatureSpace, words: tuple[str, ...]) -> np.ndarray:
    wanted = {w.lower() for w in words}
    return np.array([f.text.lower() in wanted for f in space.features], dtype=bool)


def first_order_metrics(scores: np.ndarray, is_trig: np.ndarray, is_dis: np.ndarray) -> dict[str, float]:
    g = int(is_trig.sum())
    order = ranking(scores)
    top = order[:g]
    pos = np.maximum(scores, 0.0)
    tot = float(pos.sum())
    ranks = [int(np.where(order == j)[0][0]) + 1 for j in np.where(is_trig)[0]]
    out = {
        "n_triggers": g,
        "prec_at_g": float(is_trig[top].mean()) if g else np.nan,
        "mass_share": float(pos[is_trig].sum() / tot) if tot > 1e-12 else 0.0,
        "mrr": float(np.mean([1.0 / r for r in ranks])) if ranks else np.nan,
        "distractor_in_top_g": np.nan,
        "distractor_mass_share": np.nan,
        "distractor_below_trigger": np.nan,
    }
    if is_dis.any():
        out["distractor_in_top_g"] = float(is_dis[top].any())
        out["distractor_mass_share"] = float(pos[is_dis].sum() / tot) if tot > 1e-12 else 0.0
        out["distractor_below_trigger"] = float(scores[is_trig].min() > scores[is_dis].max())
    return out


# ------------------------------------------------------------------ interactions on a restricted game


def _sii_weights(k: int) -> np.ndarray:
    """w(s) = s! (k - s - 2)! / (k - 1)! for |S| = s, S excluding the pair."""
    return np.array([math.factorial(s) * math.factorial(k - s - 2) / math.factorial(k - 1) for s in range(k - 1)])


def restricted_game(vf: ValueFunction, head: str, players: np.ndarray) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """v(S) for every S subset of `players` (others always kept). Bit b of the subset index = players[b] present."""
    m = vf.space.n_features
    k = len(players)
    X = np.ones((2**k, m), dtype=bool)
    for idx in range(2**k):
        for b in range(k):
            if not (idx >> b) & 1:
                X[idx, players[b]] = False
    return vf.call_head(X, head), [tuple(int(players[b]) for b in range(k) if (idx >> b) & 1) for idx in range(2**k)]


def pairwise_interactions(v: np.ndarray, k: int) -> dict[str, dict[tuple[int, int], float]]:
    """Pairwise SII, LOO-pair (top context) and ArchDetect-style (top+bottom) on v indexed by bitmask."""
    w = _sii_weights(k)
    full = 2**k - 1
    out: dict[str, dict[tuple[int, int], float]] = {"sii": {}, "loo_pair": {}, "archdetect": {}}
    for a, b in itertools.combinations(range(k), 2):
        ia, ib = 1 << a, 1 << b
        rest = [c for c in range(k) if c not in (a, b)]
        total = 0.0
        for r in range(len(rest) + 1):
            for sub in itertools.combinations(rest, r):
                s = sum(1 << c for c in sub)
                total += w[r] * (v[s | ia | ib] - v[s | ia] - v[s | ib] + v[s])
        top = v[full] - v[full & ~ia] - v[full & ~ib] + v[full & ~ia & ~ib]
        bottom = v[ia | ib] - v[ia] - v[ib] + v[0]
        out["sii"][(a, b)] = float(total)
        out["loo_pair"][(a, b)] = float(top)
        out["archdetect"][(a, b)] = float(0.5 * (top + bottom))
    return out


def interaction_metrics(
    inter: dict[tuple[int, int], float], players: np.ndarray, trig_idx: set[int], gate: str
) -> dict[str, float]:
    gt_pairs = {p for p in itertools.combinations(range(len(players)), 2)
                if int(players[p[0]]) in trig_idx and int(players[p[1]]) in trig_idx}
    n_gt = len(gt_pairs)
    if n_gt == 0:
        return {"n_gt_pairs": 0, "pair_prec": np.nan, "sign_acc": np.nan, "gate_recovered": np.nan,
                "gt_pair_strength": np.nan}
    ordered = sorted(inter, key=lambda p: -abs(inter[p]))
    top = set(ordered[:n_gt])
    pair_prec = len(top & gt_pairs) / n_gt
    expected = -1.0 if gate == "or" else 1.0
    signs = [np.sign(inter[p]) == expected for p in gt_pairs]
    sign_acc = float(np.mean(signs))
    max_abs = max(abs(x) for x in inter.values()) or 1.0
    return {
        "n_gt_pairs": n_gt,
        "pair_prec": float(pair_prec),
        "sign_acc": sign_acc,
        "gate_recovered": float(pair_prec == 1.0 and sign_acc == 1.0),
        "gt_pair_strength": float(np.mean([abs(inter[p]) for p in gt_pairs]) / max_abs),
    }


# ------------------------------------------------------------------ main loop


def run(cfg: ExperimentConfig, model: str, cases: list[Case], nsamples: int, top_k: int, out_dir: Path) -> None:
    cache = ResponseCache(cfg.cache_path)
    fo_rows: list[dict[str, Any]] = []
    ix_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    with _client(cfg, model, cache) as client:
        for case in tqdm(cases, desc=f"{model} planted-v2"):
            task = make_task(case.directive)
            gate = GATE[case.regime]
            probs: dict[str, dict[str, float]] = {}
            for head in HEADS:
                tgt = "immediate" if head == CHOICE else "true"
                texts = {"base": case.base_text, "planted": case.planted_text}
                for j, t in enumerate(case.partial_texts):
                    texts[f"partial{j}"] = t
                res = client.decide_batch([task.state({"patient_note": t}) for t in texts.values()], task.head_questions(head))
                probs[head] = {}
                for name, r in zip(texts, res):
                    a = r.answers[head]
                    probs[head][name] = float(a.probabilities[tgt] if head == CHOICE else a.noul)
            space = build_feature_space({"patient_note": case.planted_text}, unit="word", mask_punctuation=False)
            is_trig = _match(space, case.trigger_words)
            is_dis = _match(space, case.distractor_words)
            vf = ValueFunction(client, task, space, {CHOICE: "immediate", NOUL: "true"}, mode="delete")
            scores = {
                "loo": leave_one_out(vf),
                "lime": lime_attribution(vf, nsamples, cfg.seed),
                "kernelshap": kernel_shap(vf, nsamples, cfg.seed).values,
                "logit_kernelshap": s1_shap(vf, nsamples, cfg.seed).values,
                "random": {h: random_attribution(space.n_features, cfg.seed) for h in HEADS},
            }
            for head in HEADS:
                p = probs[head]
                followed = p["planted"] > 0.5 and p["base"] < 0.5
                strict = followed and all(p[f"partial{j}"] < 0.5 for j in range(len(case.partial_texts)))
                case_rows.append({
                    "model": model, "case_id": case.id, "regime": case.regime, "gate": gate, "head": head,
                    "distractor_type": case.distractor_type, "n_features": space.n_features,
                    "n_trigger_features": int(is_trig.sum()), "p_base": p["base"], "p_planted": p["planted"],
                    "p_partial_max": max([p[f"partial{j}"] for j in range(len(case.partial_texts))], default=np.nan),
                    "rule_followed": bool(followed), "rule_followed_strict": bool(strict),
                })
            for method, by_head in scores.items():
                faith = evaluate_faithfulness(vf, by_head, curves=True)
                for head in HEADS:
                    fo_rows.append({
                        "model": model, "case_id": case.id, "regime": case.regime, "head": head, "method": method,
                        "distractor_type": case.distractor_type,
                        "aopc_comprehensiveness": faith[head].aopc_comprehensiveness,
                        "aopc_sufficiency": faith[head].aopc_sufficiency,
                        "deletion_auc": faith[head].deletion_auc, "insertion_auc": faith[head].insertion_auc,
                        **first_order_metrics(np.asarray(by_head[head]), is_trig, is_dis),
                    })
            # Restricted-game interactions on the top-k features of the logit-link first-order ranking.
            k = min(top_k, space.n_features)
            trig_idx = set(np.where(is_trig)[0].tolist())
            for head in HEADS:
                phi = np.asarray(scores["logit_kernelshap"][head])
                players = np.argsort(-np.abs(phi), kind="stable")[:k]
                v_prob, _ = restricted_game(vf, head, players)
                coverage = len(trig_idx & set(players.tolist())) / max(len(trig_idx), 1)
                for space_name, v in (("prob", v_prob), ("logit", _logit(v_prob))):
                    for estimator, inter in pairwise_interactions(v, k).items():
                        ix_rows.append({
                            "model": model, "case_id": case.id, "regime": case.regime, "gate": gate, "head": head,
                            "estimator": estimator, "space": space_name, "top_k": k, "trigger_coverage": coverage,
                            **interaction_metrics(inter, players, trig_idx, gate),
                        })
    cache.close()
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(case_rows).to_parquet(out_dir / "cases.parquet", index=False)
    pd.DataFrame(fo_rows).to_parquet(out_dir / "first_order.parquet", index=False)
    pd.DataFrame(ix_rows).to_parquet(out_dir / "interactions.parquet", index=False)


def export_cases(cases: list[Case], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{**c.__dict__, "trigger_words": list(c.trigger_words), "distractor_words": list(c.distractor_words),
                   "partial_texts": list(c.partial_texts)} for c in cases]).to_json(path, orient="records", indent=1)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["tev1:0.8b", "tev1:latest", "clef-flash:latest", "nimble:latest"])
    ap.add_argument("--nsamples", type=int, default=256)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="results/planted-v2")
    args = ap.parse_args()
    cfg = ExperimentConfig()
    cases = generate_cases()
    export_cases(cases, Path("data/planted_v2/cases.json"))
    if args.limit:
        by_regime: dict[str, list[Case]] = {}
        for c in cases:
            by_regime.setdefault(c.regime, []).append(c)
        cases = [c for group in by_regime.values() for c in group[: args.limit]]
    for model in args.models:
        run(cfg, model, cases, args.nsamples, args.top_k, Path(args.out) / model.replace(":", "-"))
        print(f"done {model}", flush=True)

