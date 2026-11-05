"""POC dataset: stratified e-SNLI (human highlights), stratified HANS (lexical shortcuts) and
template-generated clinical triage vignettes whose causal spans are known by construction."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .tasks import Instance

ESNLI_LABELS = ("entailment", "neutral", "contradiction")
TRIAGE_LABELS = ("immediate", "urgent", "non_urgent")


# ---------------------------------------------------------------- e-SNLI

def _starred_word_spans(sentence: str, marked: str) -> list[tuple[int, int]] | None:
    """Character spans (in `sentence`) of the words an annotator starred in `marked`."""
    words = list(re.finditer(r"\S+", sentence))
    marked_words = marked.split()
    if len(words) != len(marked_words):
        return None
    return [
        (w.start(), w.end())
        for w, m in zip(words, marked_words)
        if m.startswith("*") and m.rstrip(".,!?;:'\"").endswith("*")
    ]


def _highlight_votes(row: pd.Series, col_sentence: str, col_marked: str) -> list[list[int]] | None:
    sentence = str(row[col_sentence]).strip()
    votes: dict[tuple[int, int], int] = {}
    for j in (1, 2, 3):
        marked = row.get(f"{col_marked}_{j}")
        if not isinstance(marked, str):
            continue
        spans = _starred_word_spans(sentence, marked.strip())
        if spans is None:
            return None
        for span in spans:
            votes[span] = votes.get(span, 0) + 1
    return [[s, e, v] for (s, e), v in sorted(votes.items())]


def load_esnli(path: Path, n_per_class: int, seed: int) -> list[Instance]:
    df = pd.read_csv(path)
    df = df[df["gold_label"].isin(ESNLI_LABELS)].dropna(subset=["Sentence1", "Sentence2"])
    rng = np.random.default_rng(seed)
    instances = []
    for label in ESNLI_LABELS:
        pool = df[df["gold_label"] == label]
        picked = pool.iloc[rng.choice(len(pool), size=n_per_class, replace=False)]
        for _, row in picked.iterrows():
            instances.append(
                Instance(
                    id=f"esnli-{row['pairID']}",
                    task="nli",
                    source="esnli",
                    segments={"premise": str(row["Sentence1"]).strip(), "hypothesis": str(row["Sentence2"]).strip()},
                    gold_choice=label,
                    gold_noul=label == "entailment",
                    metadata={
                        "highlight_votes": {
                            "premise": _highlight_votes(row, "Sentence1", "Sentence1_marked"),
                            "hypothesis": _highlight_votes(row, "Sentence2", "Sentence2_marked"),
                        },
                        "explanation_1": row.get("Explanation_1"),
                    },
                )
            )
    return instances


# ---------------------------------------------------------------- HANS

def _detokenize(text: str) -> str:
    return re.sub(r" ([.,!?;:])", r"\1", text.strip())


def load_hans(path: Path, n_total: int, seed: int) -> list[Instance]:
    df = pd.read_csv(path, sep="\t")
    rng = np.random.default_rng(seed)
    labels = ("entailment", "non-entailment")
    heuristics = sorted(df["heuristic"].unique())
    instances = []
    per_label = n_total // len(labels)
    for li, label in enumerate(labels):
        n_label = per_label + (1 if li < n_total % len(labels) else 0)
        counts = [n_label // len(heuristics) + (1 if i < n_label % len(heuristics) else 0) for i in range(len(heuristics))]
        for heuristic, n in zip(heuristics, counts):
            pool = df[(df["gold_label"] == label) & (df["heuristic"] == heuristic)]
            picked = pool.iloc[rng.choice(len(pool), size=n, replace=False)]
            for _, row in picked.iterrows():
                instances.append(
                    Instance(
                        id=f"hans-{row['pairID']}",
                        task="nli",
                        source="hans",
                        segments={"premise": _detokenize(row["sentence1"]), "hypothesis": _detokenize(row["sentence2"])},
                        gold_choice="entailment" if label == "entailment" else "non_entailment",
                        gold_noul=label == "entailment",
                        metadata={"heuristic": heuristic, "subcase": row["subcase"], "template": row["template"]},
                    )
                )
    return instances


# ---------------------------------------------------------------- synthetic triage

TRIAGE_FLAGS = {
    "immediate": [
        "The patient is unresponsive to voice and only withdraws from pain.",
        "Oxygen saturation is 82% on room air with a respiratory rate of 36 breaths per minute.",
        "Blood pressure is 68/40 mmHg with cold, clammy skin.",
        "The patient reports crushing chest pain radiating to the left arm, with profuse sweating.",
        "The patient has been seizing continuously for over five minutes.",
        "There is sudden facial droop and slurred speech that started 40 minutes ago.",
        "There is massive bleeding from a thigh wound that is not controlled by pressure.",
        "There is swelling of the lips and tongue with audible stridor after a bee sting.",
    ],
    "urgent": [
        "The patient has a fever of 39.5°C with flank pain and burning on urination.",
        "There is a deep 4 cm laceration on the forearm that is still oozing blood.",
        "The patient reports right lower abdominal pain that has worsened over 12 hours.",
        "There is an obvious deformity of the wrist after a fall, with intact sensation.",
        "The patient has moderate shortness of breath with oxygen saturation of 93%.",
        "The patient has had persistent vomiting for a day and cannot keep fluids down.",
        "The patient reports an asthma attack that improved only partially with an inhaler.",
        "There is intense intermittent flank pain suggestive of a kidney stone.",
    ],
    "non_urgent": [
        "The patient reports a mild sore throat for two days without fever.",
        "The patient is requesting a refill of a routine blood pressure prescription.",
        "There is a small superficial scratch on the back of the hand.",
        "The patient has mild lower back pain after gardening yesterday.",
        "The patient has a runny nose and occasional sneezing.",
        "There is a mild itchy rash on the forearm with no swelling.",
        "The patient twisted an ankle last week and now walks without difficulty.",
        "The patient asks for advice about a mild sunburn.",
    ],
}

TRIAGE_FILLERS = [
    "The patient has a history of seasonal allergies.",
    "The patient works as an accountant and lives with a partner.",
    "The patient does not smoke and drinks alcohol occasionally.",
    "The patient takes a daily multivitamin.",
    "The patient's vaccinations are up to date.",
    "The patient came in on a weekday afternoon.",
    "The patient speaks English and needs no interpreter.",
    "The patient wears glasses for reading.",
]

# Alarming but clinically irrelevant cues, balanced across classes to probe shortcut learning.
TRIAGE_CONFOUNDERS = [
    "The patient's family is extremely worried and keeps asking staff for help.",
    "The patient arrived by ambulance at a relative's insistence.",
    "The patient repeatedly says that this feels like an emergency.",
    "A relative insists that the patient be seen first.",
]


def generate_triage(counts: dict[str, int], seed: int) -> list[Instance]:
    rng = np.random.default_rng(seed)
    instances = []
    for label in TRIAGE_LABELS:
        n = counts[label]
        n_confounded = round(n / 3)
        confounded = set(rng.choice(n, size=n_confounded, replace=False).tolist())
        for i in range(n):
            age = int(rng.integers(18, 86))
            sex = str(rng.choice(["man", "woman"]))
            n_flags = int(rng.integers(1, 3))
            flags = [TRIAGE_FLAGS[label][j] for j in rng.choice(len(TRIAGE_FLAGS[label]), size=n_flags, replace=False)]
            fillers = [TRIAGE_FILLERS[j] for j in rng.choice(len(TRIAGE_FILLERS), size=2, replace=False)]
            body = [("flag", s) for s in flags] + [("filler", s) for s in fillers]
            if i in confounded:
                body.append(("confounder", TRIAGE_CONFOUNDERS[int(rng.integers(len(TRIAGE_CONFOUNDERS)))]))
            body = [body[j] for j in rng.permutation(len(body))]

            article = "An" if str(age).startswith("8") or age in (11, 18) else "A"
            text = f"{article} {age}-year-old {sex} presents to the emergency department."
            spans: dict[str, list[list[int]]] = {"flag": [], "confounder": [], "filler": []}
            for kind, sentence in body:
                start = len(text) + 1
                text = f"{text} {sentence}"
                spans[kind].append([start, start + len(sentence)])
            instances.append(
                Instance(
                    id=f"triage-{label}-{i:02d}",
                    task="triage",
                    source="triage_synth",
                    segments={"patient_note": text},
                    gold_choice=label,
                    gold_noul=label == "immediate",
                    metadata={
                        "flag_spans": spans["flag"],
                        "confounder_spans": spans["confounder"],
                        "filler_spans": spans["filler"],
                        "has_confounder": i in confounded,
                    },
                )
            )
    return instances


# ---------------------------------------------------------------- persistence

def instances_to_frame(instances: list[Instance]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": x.id,
                "task": x.task,
                "source": x.source,
                "segments": json.dumps(x.segments, ensure_ascii=False),
                "gold_choice": x.gold_choice,
                "gold_noul": x.gold_noul,
                "metadata": json.dumps(x.metadata, ensure_ascii=False, default=str),
            }
            for x in instances
        ]
    )


def instances_from_frame(df: pd.DataFrame) -> list[Instance]:
    return [
        Instance(
            id=r.id,
            task=r.task,
            source=r.source,
            segments=json.loads(r.segments),
            gold_choice=r.gold_choice,
            gold_noul=bool(r.gold_noul),
            metadata=json.loads(r.metadata),
        )
        for r in df.itertuples()
    ]


def load_tensor_trust(path: Path) -> list[Instance]:
    instances = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            instances.append(
                Instance(
                    id=item["id"],
                    task="tensor_trust",
                    source="tensor_trust",
                    segments={"user_input": item["text"]},
                    gold_choice="prompt_injection" if item["is_attack"] else "benign",
                    gold_noul=item["is_attack"],
                    metadata={"attack_type": item.get("attack_type", "unknown")},
                )
            )
    return instances


def build_poc_dataset(raw_dir: Path, seed: int, n_esnli_per_class: int, n_hans: int, triage_counts: dict[str, int]) -> list[Instance]:
    return (
        load_esnli(raw_dir / "esnli_test.csv", n_esnli_per_class, seed)
        + generate_triage(triage_counts, seed)
        + load_hans(raw_dir / "hans_evaluation.tsv", n_hans, seed)
    )
