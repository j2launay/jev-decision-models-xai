"""Generative System 2 LLM Baseline (Gemma4-8B Direct JSON & Chain-of-Thought) for comparison against
Jev-style System 1 Decision Models.

Evaluates:
1. Uncached sequential latency, output token overhead (JSON vs. CoT), and concurrent throughput.
2. Calibration and shortcut vulnerability of verbalized probabilities on e-SNLI, HANS, and Clinical Triage.
3. Cross-question interference when co-asking multiple questions in a single JSON schema.
4. Plausibility, faithfulness, and Planted System-Rule trigger recovery of self-rationalized explanations
   (`rationale_words`) compared against post-hoc S1-SHAP on System 1 models.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
from tqdm import tqdm

from .cache import ResponseCache, payload_key
from .config import ExperimentConfig
from .experiment import (
    PREDICTION_COLUMNS,
    _plausibility,
    calibration_table,
    load_or_build_dataset,
)
from .faithfulness import evaluate as evaluate_faithfulness, ranking
from .features import FeatureSpace, build_feature_space
from .planted_benchmark import PLANTED_CASES, PlantedCase, _make_planted_task, evaluate_planted_case
from .schemas import ChoiceAnswer, ChoiceQuestion, DecisionResult, NoulAnswer, NoulQuestion, Usage
from .tasks import CHOICE, HEADS, NOUL, TASKS, Instance, Task, head_view


def _choice_schema(options: list[str], include_rationale: bool = True) -> dict[str, Any]:
    props: dict[str, Any] = {
        "choice": {"type": "string", "enum": options},
        "probabilities": {
            "type": "object",
            "properties": {k: {"type": "number"} for k in options},
            "required": options,
        },
        "confidence": {"type": "number"},
    }
    req = ["choice", "probabilities", "confidence"]
    if include_rationale:
        props["rationale_words"] = {"type": "array", "items": {"type": "string"}}
        req.append("rationale_words")
    return {"type": "object", "properties": props, "required": req}


def _noul_schema(include_rationale: bool = True) -> dict[str, Any]:
    props: dict[str, Any] = {
        "noul": {"type": "number"},
    }
    req = ["noul"]
    if include_rationale:
        props["rationale_words"] = {"type": "array", "items": {"type": "string"}}
        req.append("rationale_words")
    return {"type": "object", "properties": props, "required": req}


def _joint_schema(options: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            CHOICE: _choice_schema(options, include_rationale=False),
            NOUL: _noul_schema(include_rationale=False),
        },
        "required": [CHOICE, NOUL],
    }


def _build_prompt(state: dict[str, str], questions: dict[str, Any], include_rationale: bool = True) -> str:
    state_str = "\n".join(f"{k}: {v}" for k, v in state.items())
    q_lines = []
    for q_name, q in questions.items():
        if isinstance(q, ChoiceQuestion):
            opts = "\n".join(f"  - {k}: {v}" for k, v in q.criteria.items())
            q_lines.append(
                f"Question '{q_name}' (choice): {q.instructions}\nOptions:\n{opts}\n"
                "Provide the chosen option, a probability distribution in [0,1] summing to 1.0 over the options, and your confidence in [0,1]."
            )
        elif isinstance(q, NoulQuestion):
            opts = "\n".join(f"  - {k}: {v}" for k, v in (q.criteria or {}).items())
            q_lines.append(
                f"Question '{q_name}' (binary probability 'noul'): {q.instructions}\nCriteria:\n{opts}\n"
                "Provide 'noul' as the probability in [0,1] that the statement is true."
            )
    rat_line = (
        "\nAlso provide 'rationale_words': a list of the exact input words (ordered from most important to least important) that drove your decision."
        if include_rationale
        else ""
    )
    return f"Input State:\n{state_str}\n\n{chr(10).join(q_lines)}{rat_line}\nRespond strictly in JSON."


def _normalize_probs(raw_probs: dict[str, Any], options: list[str], chosen: str) -> dict[str, float]:
    vals = {}
    for k in options:
        try:
            v = float(raw_probs.get(k, 0.0))
        except (TypeError, ValueError):
            v = 0.0
        vals[k] = max(0.0, min(1.0, v))
    total = sum(vals.values())
    if total <= 1e-6:
        for k in options:
            vals[k] = 0.90 if k == chosen else 0.10 / max(1, len(options) - 1)
        total = sum(vals.values())
    return {k: v / total for k, v in vals.items()}


class System2GenerativeClient:
    """Queries an Ollama generative model (/api/chat) with structured JSON output and optional thinking."""

    def __init__(
        self,
        ollama_model: str = "gemma4:latest",
        label: str = "gemma4-json",
        think: bool = False,
        base_url: str = "http://localhost:11434",
        concurrency: int = 4,
        timeout_s: float = 180.0,
        cache: ResponseCache | None = None,
    ) -> None:
        self.ollama_model = ollama_model
        self.model = label
        self.think = think
        self.base_url = base_url
        self.concurrency = concurrency
        self.timeout_s = timeout_s
        self.cache = cache
        self.digest = f"system2-{ollama_model}-think-{int(think)}"

    def _call_one(
        self,
        http_client: httpx.Client,
        state: dict[str, str],
        questions: dict[str, Any],
        include_rationale: bool = True,
        use_cache: bool = True,
    ) -> tuple[DecisionResult, dict[str, list[str]]]:
        is_joint = len(questions) > 1
        head = CHOICE if CHOICE in questions and not is_joint else NOUL
        if is_joint:
            opts = list(questions[CHOICE].criteria.keys())
            schema = _joint_schema(opts)
        elif head == CHOICE:
            opts = list(questions[CHOICE].criteria.keys())
            schema = _choice_schema(opts, include_rationale=include_rationale)
        else:
            schema = _noul_schema(include_rationale=include_rationale)

        prompt = _build_prompt(state, questions, include_rationale=include_rationale and not is_joint)
        payload = {
            "model": self.ollama_model,
            "messages": [{"role": "user", "content": prompt}],
            "format": schema,
            "stream": False,
            "think": self.think,
            "options": {"temperature": 0.0, "num_predict": 512 if self.think else 160},
        }
        cache_key_payload = {"system2_payload": payload}
        ckey = payload_key(self.digest, cache_key_payload)
        if use_cache and self.cache is not None:
            hit = self.cache.get(ckey)
            if hit is not None:
                data, lat = hit
                return self._parse_response(data, questions, cached=True, latency_ms=lat)

        t0 = time.perf_counter()
        resp = http_client.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout_s)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        resp.raise_for_status()
        data = resp.json()
        if use_cache and self.cache is not None:
            self.cache.put(ckey, self.digest, data, latency_ms)
        return self._parse_response(data, questions, cached=False, latency_ms=latency_ms)

    def _parse_response(
        self, data: dict[str, Any], questions: dict[str, Any], cached: bool, latency_ms: float
    ) -> tuple[DecisionResult, dict[str, list[str]]]:
        content = data.get("message", {}).get("content", "{}")
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = {}

        answers: dict[str, Any] = {}
        rationales: dict[str, list[str]] = {}
        is_joint = len(questions) > 1

        for q_name, q in questions.items():
            sub = parsed.get(q_name, parsed) if is_joint else parsed
            if not isinstance(sub, dict):
                sub = {}
            if isinstance(q, ChoiceQuestion):
                opts = list(q.criteria.keys())
                raw_choice = str(sub.get("choice", opts[0]))
                if raw_choice not in opts:
                    raw_choice = opts[0]
                probs = _normalize_probs(sub.get("probabilities", {}), opts, raw_choice)
                # Ensure choice matches argmax of normalized probabilities
                best_opt = max(probs, key=probs.get)
                conf = float(sub.get("confidence", probs[best_opt]))
                conf = max(0.0, min(1.0, conf))
                answers[q_name] = ChoiceAnswer(
                    type="choice",
                    choice=best_opt,
                    probabilities=probs,
                    confidence=conf,
                )
            elif isinstance(q, NoulQuestion):
                try:
                    noul_val = float(sub.get("noul", 0.5))
                except (TypeError, ValueError):
                    noul_val = 0.5
                noul_val = max(0.0, min(1.0, noul_val))
                answers[q_name] = NoulAnswer(type="noul", noul=noul_val)
            raw_rat = sub.get("rationale_words", [])
            if isinstance(raw_rat, list):
                rationales[q_name] = [str(w) for w in raw_rat if isinstance(w, (str, int, float))]
            else:
                rationales[q_name] = []

        in_tok = data.get("prompt_eval_count")
        out_tok = data.get("eval_count")
        res = DecisionResult(
            model=self.model,
            answers=answers,
            usage=Usage(input_tokens=in_tok, output_tokens=out_tok),
            latency_ms=latency_ms,
            cached=cached,
        )
        return res, rationales

    def decide_batch(
        self,
        states: list[dict[str, str]],
        questions: dict[str, Any],
        include_rationale: bool = True,
        use_cache: bool = True,
    ) -> tuple[list[DecisionResult], list[dict[str, list[str]]]]:
        from concurrent.futures import ThreadPoolExecutor

        results: list[DecisionResult | None] = [None] * len(states)
        rats: list[dict[str, list[str]] | None] = [None] * len(states)
        with httpx.Client(timeout=self.timeout_s) as http_client:
            def worker(idx: int) -> None:
                r, rt = self._call_one(
                    http_client, states[idx], questions, include_rationale=include_rationale, use_cache=use_cache
                )
                results[idx] = r
                rats[idx] = rt

            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                list(pool.map(worker, range(len(states))))
        return [r for r in results if r is not None], [rt for rt in rats if rt is not None]


def rationale_to_scores(space: FeatureSpace, words: list[str]) -> np.ndarray:
    """Converts an ordered list of LLM self-reported rationale words/phrases into feature scores."""
    scores = np.zeros(space.n_features, dtype=float)
    cleaned_rationales: list[str] = []
    for phrase in words:
        for tok in str(phrase).split():
            w = tok.strip(".,!?;:'\"()[]{}").lower()
            if w and w not in cleaned_rationales:
                cleaned_rationales.append(w)
    n_rat = len(cleaned_rationales)
    if n_rat == 0:
        return scores
    for rank_idx, target_w in enumerate(cleaned_rationales):
        weight = float(n_rat - rank_idx) / float(n_rat)
        for f in space.features:
            fw = f.text.strip(".,!?;:'\"()[]{}").lower()
            if fw == target_w and scores[f.index] < weight:
                scores[f.index] = weight
    return scores


def benchmark_system2_latency(
    cfg: ExperimentConfig, instances: list[Instance], think: bool = False, n_seq: int = 10, n_conc: int = 16
) -> pd.DataFrame:
    label = "gemma4-cot" if think else "gemma4-json"
    client = System2GenerativeClient(
        ollama_model="gemma4:latest", label=label, think=think, base_url=cfg.base_url, concurrency=1, cache=None
    )
    rows = []
    sample = [instances[j] for j in np.random.default_rng(cfg.seed).choice(len(instances), size=max(n_seq, n_conc), replace=False)]
    with httpx.Client(timeout=cfg.timeout_s) as http_client:
        first = sample[0]
        client._call_one(http_client, TASKS[first.task].state(first.segments), TASKS[first.task].head_questions(CHOICE), use_cache=False)
        for inst in tqdm(sample[:n_seq], desc=f"{label} sequential latency"):
            task = TASKS[inst.task]
            for head in HEADS:
                r, _ = client._call_one(http_client, task.state(inst.segments), task.head_questions(head), use_cache=False)
                rows.append({
                    "model": label,
                    "mode": "sequential",
                    "head": head,
                    "concurrency": 1,
                    "latency_ms": r.latency_ms,
                    "wall_ms": r.latency_ms,
                    "n_requests": 1,
                    "input_tokens": r.usage.input_tokens,
                    "output_tokens": r.usage.output_tokens,
                })
    for level in (1, 2, 4):
        c_client = System2GenerativeClient(
            ollama_model="gemma4:latest", label=label, think=think, base_url=cfg.base_url, concurrency=level, cache=None
        )
        batch = [sample[i % len(sample)] for i in range(n_conc)]
        by_task: dict[str, list[Instance]] = {}
        for inst in batch:
            by_task.setdefault(inst.task, []).append(inst)
        t0 = time.perf_counter()
        for task_name, group in by_task.items():
            task = TASKS[task_name]
            res_list, _ = c_client.decide_batch([task.state(i.segments) for i in group], task.head_questions(CHOICE), use_cache=False)
            for r in res_list:
                rows.append({
                    "model": label,
                    "mode": "concurrent",
                    "head": CHOICE,
                    "concurrency": level,
                    "latency_ms": r.latency_ms,
                    "wall_ms": np.nan,
                    "n_requests": len(batch),
                    "input_tokens": r.usage.input_tokens,
                    "output_tokens": r.usage.output_tokens,
                })
        wall = (time.perf_counter() - t0) * 1000.0
        rows.append({
            "model": label,
            "mode": "throughput",
            "head": CHOICE,
            "concurrency": level,
            "latency_ms": np.nan,
            "wall_ms": wall,
            "n_requests": len(batch),
            "input_tokens": np.nan,
            "output_tokens": np.nan,
        })
    return pd.DataFrame(rows)


def run_system2_evaluation(cfg: ExperimentConfig, instances: list[Instance]) -> None:
    cache = ResponseCache(cfg.cache_path)
    client = System2GenerativeClient(
        ollama_model="gemma4:latest", label="gemma4-json", think=False, base_url=cfg.base_url, concurrency=cfg.concurrency, cache=cache
    )
    out_dir = cfg.results_dir / "poc-tev1-0.8b" / "gemma4-json"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Clean predictions & self-rationales on all 150 instances
    pred_rows = []
    attr_rows = []
    plaus_rows = []
    for task_name, task in TASKS.items():
        group = [i for i in instances if i.task == task_name]
        states = [task.state(i.segments) for i in group]
        for head in HEADS:
            results, rats = client.decide_batch(states, task.head_questions(head), include_rationale=True)
            for inst, r, rt in zip(group, results, rats):
                v = head_view(r, head, inst, task)
                answer = r.answers[head]
                pred_rows.append({
                    "model": client.model,
                    "instance_id": inst.id,
                    "source": inst.source,
                    "task": inst.task,
                    "condition": "clean",
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
                    "latency_ms": r.latency_ms,
                    "cached": r.cached,
                    "input_tokens": r.usage.input_tokens,
                    "state": json.dumps(inst.segments, ensure_ascii=False),
                })
                if inst.source in cfg.attribution_sources:
                    space = build_feature_space(inst.segments, unit="word", mask_punctuation=cfg.mask_punctuation)
                    scores = rationale_to_scores(space, rt.get(head, []))
                    for f, sc in zip(space.features, scores):
                        attr_rows.append({
                            "model": client.model,
                            "instance_id": inst.id,
                            "source": inst.source,
                            "unit": "word",
                            "method": "self_rationale",
                            "head": head,
                            "target": v.pred,
                            "feature_index": f.index,
                            "segment": f.segment,
                            "start": f.start,
                            "end": f.end,
                            "text": f.text,
                            "score": float(sc),
                        })
                    if inst.source == "triage_synth":
                        plaus_rows.append({
                            "model": client.model,
                            "instance_id": inst.id,
                            "unit": "word",
                            "method": "self_rationale",
                            "head": head,
                            "has_confounder": inst.metadata["has_confounder"],
                            **_plausibility(space, inst, scores),
                        })

    preds_df = pd.DataFrame(pred_rows, columns=PREDICTION_COLUMNS)
    preds_df.to_parquet(out_dir / "predictions.parquet", index=False)
    cal_df = calibration_table(preds_df, cfg)
    cal_df.to_parquet(out_dir / "calibration.parquet", index=False)
    pd.DataFrame(attr_rows).to_parquet(out_dir / "attributions.parquet", index=False)
    pd.DataFrame(plaus_rows).to_parquet(out_dir / "plausibility.parquet", index=False)

    # 2. Cross-question interference on all 150 instances
    inter_rows = []
    for task_name, task in TASKS.items():
        group = [i for i in instances if i.task == task_name]
        states = [task.state(i.segments) for i in group]
        joint_res, _ = client.decide_batch(states, task.questions, include_rationale=False)
        sep_res = {h: client.decide_batch(states, task.head_questions(h), include_rationale=True)[0] for h in HEADS}
        for k, inst in enumerate(group):
            for head in HEADS:
                vs = head_view(sep_res[head][k], head, inst, task)
                vj = head_view(joint_res[k], head, inst, task)
                inter_rows.append({
                    "model": client.model,
                    "instance_id": inst.id,
                    "source": inst.source,
                    "head": head,
                    "pred_separate": vs.pred,
                    "pred_joint": vj.pred,
                    "flip": vs.pred != vj.pred,
                    "correct_separate": vs.correct,
                    "correct_joint": vj.correct,
                    "conf_separate": vs.conf,
                    "conf_joint": vj.conf,
                    "p_true_separate": float(vs.probs[0]) if head == NOUL else np.nan,
                    "p_true_joint": float(vj.probs[0]) if head == NOUL else np.nan,
                    "total_variation": float(0.5 * np.abs(vs.probs - vj.probs).sum()),
                    "latency_separate_ms": sep_res[head][k].latency_ms,
                    "latency_joint_ms": joint_res[k].latency_ms,
                })
    pd.DataFrame(inter_rows).to_parquet(out_dir / "interference.parquet", index=False)

    # 3. Planted System-Rule Benchmark with System 2 Self-Rationale
    plant_rows = []
    for case in tqdm(PLANTED_CASES, desc="gemma4-json planted-rule self-rationale"):
        task = _make_planted_task(case.directive)
        base_state = task.state({"patient_note": case.base_text})
        plant_state = task.state({"patient_note": case.planted_text})
        res_b_c, _ = client.decide_batch([base_state], task.head_questions(CHOICE), include_rationale=True)
        res_p_c, rat_p_c = client.decide_batch([plant_state], task.head_questions(CHOICE), include_rationale=True)
        res_b_n, _ = client.decide_batch([base_state], task.head_questions(NOUL), include_rationale=True)
        res_p_n, rat_p_n = client.decide_batch([plant_state], task.head_questions(NOUL), include_rationale=True)

        p_base_c = res_b_c[0].answers[CHOICE].probabilities["immediate"]
        p_plant_c = res_p_c[0].answers[CHOICE].probabilities["immediate"]
        p_base_n = res_b_n[0].answers[NOUL].noul
        p_plant_n = res_p_n[0].answers[NOUL].noul

        space = build_feature_space({"patient_note": case.planted_text}, unit="word", mask_punctuation=True)
        for head in HEADS:
            p_b = p_base_c if head == CHOICE else p_base_n
            p_p = p_plant_c if head == CHOICE else p_plant_n
            words = rat_p_c[0].get(CHOICE, []) if head == CHOICE else rat_p_n[0].get(NOUL, [])
            scores = rationale_to_scores(space, words)
            metrics = evaluate_planted_case(space, case, scores)
            plant_rows.append({
                "model": client.model,
                "case_id": case.id,
                "regime": case.regime,
                "head": head,
                "method": "self_rationale",
                "p_base": float(p_b),
                "p_planted": float(p_p),
                "rule_followed": bool(p_p > 0.5 and p_b < 0.5),
                **metrics,
            })
    pd.DataFrame(plant_rows).to_parquet(out_dir / "planted_self_rationale.parquet", index=False)
    cache.close()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-latency", action="store_true")
    args = ap.parse_args()
    cfg = ExperimentConfig()
    instances = load_or_build_dataset(cfg)

    if not args.skip_latency:
        for think_mode, folder in ((False, "gemma4-json"), (True, "gemma4-cot")):
            b_dir = cfg.results_dir / "bench-np4" / folder
            b_dir.mkdir(parents=True, exist_ok=True)
            df_b = benchmark_system2_latency(cfg, instances, think=think_mode, n_seq=8 if think_mode else 12, n_conc=8 if think_mode else 16)
            df_b.to_parquet(b_dir / "benchmark.parquet", index=False)
            seq = df_b[df_b["mode"] == "sequential"]
            print(f"{folder} median latency: {seq['latency_ms'].median():.1f} ms, mean output tokens: {seq['output_tokens'].mean():.1f}")

    run_system2_evaluation(cfg, instances)
    print("Finished System 2 evaluation.")

