"""Controls for cross-question interference (reviewer request).

For every clean instance of the expanded N=450 cohort we compare the isolated single-question
prediction (cached) with four alternative requests:

* joint_cn      : Choice then Noul in one request (original protocol, cached),
* joint_nc      : Noul then Choice (question order reversed),
* dummy         : the target question co-asked with an *irrelevant* Noul question of similar
                  length ("Is the text written in English?") -> separates "any second question"
                  from "a semantically related second question",
* shuffled_opts : Choice asked alone with its option list in a seeded random order -> option-order
                  noise floor for the Choice head.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .cache import ResponseCache
from .config import ExperimentConfig
from .experiment import _client
from .scale_and_stats import load_or_build_expanded_dataset
from .schemas import ChoiceQuestion, NoulQuestion
from .tasks import CHOICE, HEADS, NOUL, TASKS, head_view

DUMMY = "aux_language"
DUMMY_Q = NoulQuestion(
    instructions="Is the text written in English? Answer only about the language of the text, not its content.",
    criteria={"true": "The text is written in English.", "false": "The text is not written in English."},
)


def _shuffled_choice(q: ChoiceQuestion, seed: int) -> ChoiceQuestion:
    items = list(q.criteria.items())
    perm = np.random.default_rng(seed).permutation(len(items))
    if (perm == np.arange(len(items))).all():
        perm = np.roll(perm, 1)
    return ChoiceQuestion(instructions=q.instructions, criteria=dict(items[i] for i in perm))


def run(cfg: ExperimentConfig, model: str, out_dir: Path) -> pd.DataFrame:
    instances = load_or_build_expanded_dataset(cfg)
    cache = ResponseCache(cfg.cache_path)
    rows = []
    with _client(cfg, model, cache) as client:
        for task_name, task in TASKS.items():
            group = [i for i in instances if i.task == task_name]
            states = [task.state(i.segments) for i in group]
            qc, qn = task.questions[CHOICE], task.questions[NOUL]
            variants = {
                "separate": {h: client.decide_batch(states, task.head_questions(h)) for h in HEADS},
                "joint_cn": client.decide_batch(states, {CHOICE: qc, NOUL: qn}),
                "joint_nc": client.decide_batch(states, {NOUL: qn, CHOICE: qc}),
                "dummy": {h: client.decide_batch(states, {h: task.questions[h], DUMMY: DUMMY_Q}) for h in HEADS},
                "shuffled_opts": client.decide_batch(states, {CHOICE: _shuffled_choice(qc, cfg.seed)}),
            }
            for k, inst in enumerate(tqdm(group, desc=f"{model} {task_name}", leave=False)):
                for head in HEADS:
                    ref = head_view(variants["separate"][head][k], head, inst, task)
                    alts = {
                        "joint_cn": variants["joint_cn"][k],
                        "joint_nc": variants["joint_nc"][k],
                        "dummy": variants["dummy"][head][k],
                    }
                    if head == CHOICE:
                        alts["shuffled_opts"] = variants["shuffled_opts"][k]
                    for name, res in alts.items():
                        v = head_view(res, head, inst, task)
                        # Align probability vectors by class name (option order may differ).
                        p_ref = dict(zip(ref.classes, ref.probs))
                        p_alt = dict(zip(v.classes, v.probs))
                        tv = 0.5 * sum(abs(p_ref[c] - p_alt[c]) for c in ref.classes)
                        rows.append({
                            "model": model, "instance_id": inst.id, "source": inst.source, "head": head,
                            "variant": name, "flip": ref.pred != v.pred, "total_variation": float(tv),
                            "correct_ref": ref.correct, "correct_alt": v.correct,
                        })
    cache.close()
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "interference_controls.parquet", index=False)
    return df


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["tev1:0.8b", "tev1:latest", "clef-flash:latest", "nimble:latest"])
    args = ap.parse_args()
    cfg = ExperimentConfig()
    for m in args.models:
        df = run(cfg, m, cfg.results_dir / "interference-controls" / m.replace(":", "-"))
        print(m)
        print(df.groupby(["head", "variant"])[["flip", "total_variation"]].mean().round(3).to_string(), flush=True)

