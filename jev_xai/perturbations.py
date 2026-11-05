"""Label-preserving input perturbations for the calibration stress test."""

from __future__ import annotations

import re
import string
from dataclasses import dataclass

import numpy as np

from .tasks import Instance

NEGATIONS = {
    "not", "no", "never", "none", "nobody", "nothing", "neither", "nor", "nowhere", "without",
    "cannot", "isn't", "aren't", "wasn't", "weren't", "don't", "doesn't", "didn't", "can't", "won't",
}
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

NLI_DISTRACTORS = [
    "The museum downtown reopened after renovations last spring.",
    "Several migratory birds were counted along the coast this year.",
    "A new bakery recently opened in a distant town.",
    "The library extended its opening hours for the exam season.",
    "Farmers in another region reported an early harvest.",
]
TRIAGE_DISTRACTORS = [
    "The patient arrived wearing a blue jacket.",
    "The patient's car is parked in the visitor lot.",
    "The patient mentions enjoying crossword puzzles.",
    "The patient recently moved to a new apartment.",
    "The patient brought a paperback novel to read.",
]


@dataclass(frozen=True)
class Perturbation:
    name: str
    kind: str
    rate: float = 0.0


PERTURBATIONS = [
    Perturbation("typo_05", "typo", 0.05),
    Perturbation("typo_10", "typo", 0.10),
    Perturbation("typo_20", "typo", 0.20),
    Perturbation("distractor", "distractor"),
    Perturbation("case_punct", "case_punct"),
]


def _typo_word(word: str, rng: np.random.Generator) -> str:
    op = rng.integers(4)
    i = int(rng.integers(1, len(word) - 1)) if len(word) > 2 else 0
    letters = string.ascii_lowercase
    if op == 0 and len(word) > 3:  # swap two adjacent interior characters
        return word[:i] + word[i + 1] + word[i] + word[i + 2 :]
    if op == 1:  # delete
        return word[:i] + word[i + 1 :]
    if op == 2:  # insert
        return word[:i] + letters[int(rng.integers(26))] + word[i:]
    return word[:i] + letters[int(rng.integers(26))] + word[i + 1 :]  # substitute


def add_typos(text: str, rate: float, rng: np.random.Generator) -> str:
    matches = [m for m in WORD_RE.finditer(text) if len(m.group(0)) >= 3 and m.group(0).lower() not in NEGATIONS]
    n = int(round(rate * len(matches)))
    if n == 0:
        return text
    chosen = sorted(rng.choice(len(matches), size=n, replace=False).tolist())
    out, cursor = [], 0
    for j in chosen:
        m = matches[j]
        out.append(text[cursor : m.start()])
        out.append(_typo_word(m.group(0), rng))
        cursor = m.end()
    out.append(text[cursor:])
    return "".join(out)


def case_punct(text: str) -> str:
    return re.sub(r"\s{2,}", " ", re.sub(r"[^\w\s'%/°-]", "", text.lower())).strip()


def _content_words(text: str) -> set[str]:
    return {w.lower() for w in WORD_RE.findall(text) if len(w) > 3}


def add_distractor(inst: Instance, rng: np.random.Generator) -> dict[str, str]:
    segments = dict(inst.segments)
    if inst.task == "nli":
        hyp_words = _content_words(segments["hypothesis"])
        # Skip sentences sharing content words with the hypothesis, which could alter the label.
        pool = [s for s in NLI_DISTRACTORS if not (_content_words(s) & hyp_words)] or NLI_DISTRACTORS
        segments["premise"] = f"{segments['premise']} {pool[int(rng.integers(len(pool)))]}"
    else:
        sentence = TRIAGE_DISTRACTORS[int(rng.integers(len(TRIAGE_DISTRACTORS)))]
        segments["patient_note"] = f"{segments['patient_note']} {sentence}"
    return segments


def perturb(inst: Instance, p: Perturbation, seed: int) -> dict[str, str]:
    # Per-instance, per-perturbation seed: results do not depend on processing order.
    rng = np.random.default_rng([seed, _stable_hash(p.name), _stable_hash(inst.id)])
    if p.kind == "typo":
        return {k: add_typos(v, p.rate, rng) for k, v in inst.segments.items()}
    if p.kind == "case_punct":
        return {k: case_punct(v) for k, v in inst.segments.items()}
    if p.kind == "distractor":
        return add_distractor(inst, rng)
    raise ValueError(f"unknown perturbation kind {p.kind}")


def _stable_hash(s: str) -> int:
    h = 0
    for ch in s.encode("utf-8"):
        h = (h * 131 + ch) % (2**31)
    return h
