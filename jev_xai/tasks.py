"""Task definitions: Jev question sets, state construction and gold-label scoring.

Every task asks a `choice` question and a `noul` question in the same request, so both
primitives are evaluated on identical inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .schemas import ChoiceAnswer, ChoiceQuestion, DecisionResult, NoulAnswer, NoulQuestion

CHOICE = "label"
NOUL = "noul"
HEADS = (CHOICE, NOUL)


@dataclass
class Instance:
    id: str
    task: str
    source: str
    segments: dict[str, str]
    gold_choice: str
    gold_noul: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Task:
    name: str
    questions: dict[str, ChoiceQuestion | NoulQuestion]
    state_keys: tuple[str, ...]

    @property
    def labels(self) -> list[str]:
        return list(self.questions[CHOICE].criteria)

    def head_questions(self, head: str) -> dict[str, ChoiceQuestion | NoulQuestion]:
        """Single-question set: each primitive is asked in its own request, since co-asked
        questions measurably shift each other's answers."""
        return {head: self.questions[head]}

    def state(self, segments: dict[str, str]) -> dict[str, str]:
        return {k: segments[k] for k in self.state_keys}


# Wording follows the SNLI annotation protocol (Bowman et al., 2015): definitely / might be / definitely not true.
NLI = Task(
    name="nli",
    questions={
        CHOICE: ChoiceQuestion(
            instructions="Given that the premise is true, what is the relation of the hypothesis to the premise?",
            criteria={
                "entailment": "The hypothesis is definitely true given the premise.",
                "neutral": "The hypothesis might be true given the premise.",
                "contradiction": "The hypothesis is definitely false given the premise.",
            },
        ),
        NOUL: NoulQuestion(
            instructions="Given that the premise is true, is the hypothesis definitely true?",
            criteria={
                "true": "The premise entails the hypothesis.",
                "false": "The premise does not entail the hypothesis.",
            },
        ),
    },
    state_keys=("premise", "hypothesis"),
)

TRIAGE = Task(
    name="triage",
    questions={
        CHOICE: ChoiceQuestion(
            instructions="What emergency triage level does this patient require?",
            criteria={
                "immediate": "Life-threatening condition requiring immediate intervention.",
                "urgent": "Serious but stable condition that should be seen within an hour.",
                "non_urgent": "Minor condition that can safely wait.",
            },
        ),
        NOUL: NoulQuestion(
            instructions="Does the patient require immediate care?",
            criteria={
                "true": "The patient requires immediate care.",
                "false": "The patient does not require immediate care.",
            },
        ),
    },
    state_keys=("patient_note",),
)

TASKS = {NLI.name: NLI, TRIAGE.name: TRIAGE}


@dataclass(frozen=True)
class HeadView:
    """Prediction of one question head reduced to what calibration needs."""

    pred: str
    conf: float  # top-label probability
    correct: bool
    probs: np.ndarray  # full distribution in `classes` order
    gold_index: int
    classes: tuple[str, ...]
    provider_confidence: float | None


def choice_view(answer: ChoiceAnswer, inst: Instance, task: Task) -> HeadView:
    if inst.source != "hans":
        classes = tuple(task.labels)
        probs = np.array([answer.probabilities[c] for c in classes])
        return HeadView(
            pred=answer.choice,
            conf=answer.max_prob,
            correct=answer.choice == inst.gold_choice,
            probs=probs,
            gold_index=classes.index(inst.gold_choice),
            classes=classes,
            provider_confidence=answer.confidence,
        )
    # HANS gold is binary: collapse neutral + contradiction into non_entailment.
    classes = ("entailment", "non_entailment")
    if inst.gold_choice not in classes:
        raise ValueError(f"unknown HANS gold label {inst.gold_choice!r}")
    p_ent = answer.probabilities["entailment"]
    probs = np.array([p_ent, 1.0 - p_ent])
    pred = classes[int(np.argmax(probs))]
    return HeadView(
        pred=pred,
        conf=float(probs.max()),
        correct=pred == inst.gold_choice,
        probs=probs,
        gold_index=classes.index(inst.gold_choice),
        classes=classes,
        provider_confidence=answer.confidence,
    )


def noul_view(answer: NoulAnswer, inst: Instance) -> HeadView:
    classes = ("true", "false")
    probs = np.array([answer.noul, 1.0 - answer.noul])
    pred = "true" if answer.decision else "false"
    return HeadView(
        pred=pred,
        conf=answer.max_prob,
        correct=answer.decision == inst.gold_noul,
        probs=probs,
        gold_index=0 if inst.gold_noul else 1,
        classes=classes,
        provider_confidence=None,
    )


def head_view(result: DecisionResult, head: str, inst: Instance, task: Task) -> HeadView:
    answer = result.answers[head]
    if isinstance(answer, ChoiceAnswer):
        return choice_view(answer, inst, task)
    if isinstance(answer, NoulAnswer):
        return noul_view(answer, inst)
    raise TypeError(f"unsupported answer type for head {head}")


def target_probability(result: DecisionResult, head: str, target: str) -> float:
    """Probability assigned to a fixed target outcome (the class predicted on the original input)."""
    answer = result.answers[head]
    if isinstance(answer, ChoiceAnswer):
        return answer.probabilities[target]
    if isinstance(answer, NoulAnswer):
        return answer.noul if target == "true" else 1.0 - answer.noul
    raise TypeError(f"unsupported answer type for head {head}")


def predicted_target(result: DecisionResult, head: str) -> str:
    answer = result.answers[head]
    if isinstance(answer, ChoiceAnswer):
        return answer.choice
    if isinstance(answer, NoulAnswer):
        return "true" if answer.decision else "false"
    raise TypeError(f"unsupported answer type for head {head}")
