"""Typed request/response contracts of the Ollama `/v1/systemone` endpoint (TypeSafe Jev API).

Responses are validated strictly against the questions that produced them: probability keys must
match the declared options, distributions must be normalised, and derived fields (`choice`,
`score`) must be consistent with the distribution.
"""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

PROB_TOL = 1e-3
SCORE_TOL = 1e-3


class ChoiceQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["choice"] = "choice"
    instructions: str | None = None
    criteria: dict[str, str | None]

    @model_validator(mode="after")
    def _check(self) -> "ChoiceQuestion":
        if not 2 <= len(self.criteria) <= 26:
            raise ValueError("choice questions take 2 to 26 options")
        return self


class NoulQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["noul"] = "noul"
    instructions: str | None = None
    criteria: dict[Literal["true", "false"], str] | None = None


class ScoreQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["score"] = "score"
    instructions: str | None = None
    criteria: list[str]

    @model_validator(mode="after")
    def _check(self) -> "ScoreQuestion":
        if not 2 <= len(self.criteria) <= 10:
            raise ValueError("score questions take 2 to 10 levels")
        return self


Question = Annotated[Union[ChoiceQuestion, NoulQuestion, ScoreQuestion], Field(discriminator="type")]


class ChoiceAnswer(BaseModel):
    type: Literal["choice"]
    choice: str
    probabilities: dict[str, float]
    confidence: float

    @property
    def max_prob(self) -> float:
        return self.probabilities[self.choice]

    @property
    def entropy_confidence(self) -> float:
        """1 - H(p)/ln(K); empirically equal to the provider `confidence` field."""
        return 1.0 - normalized_entropy(list(self.probabilities.values()))


class NoulAnswer(BaseModel):
    type: Literal["noul"]
    noul: float

    @property
    def decision(self) -> bool:
        return self.noul >= 0.5

    @property
    def max_prob(self) -> float:
        return max(self.noul, 1.0 - self.noul)


class ScoreAnswer(BaseModel):
    type: Literal["score"]
    score: float
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float


Answer = Annotated[Union[ChoiceAnswer, NoulAnswer, ScoreAnswer], Field(discriminator="type")]


class Usage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None


class DecisionResult(BaseModel):
    """One validated `/v1/systemone` call."""

    model: str
    answers: dict[str, Answer]
    usage: Usage = Usage()
    latency_ms: float
    cached: bool = False


class DecisionValidationError(ValueError):
    pass


def normalized_entropy(probs: list[float]) -> float:
    k = len(probs)
    if k < 2:
        return 0.0
    h = -sum(p * math.log(p) for p in probs if p > 0)
    return h / math.log(k)


def _check_distribution(name: str, probs: dict[str, float], expected_keys: set[str]) -> None:
    if set(probs) != expected_keys:
        raise DecisionValidationError(
            f"{name}: probability keys {sorted(probs)} != declared options {sorted(expected_keys)}"
        )
    for key, p in probs.items():
        if not (0.0 - PROB_TOL <= p <= 1.0 + PROB_TOL) or math.isnan(p):
            raise DecisionValidationError(f"{name}: probability {key}={p} outside [0, 1]")
    total = sum(probs.values())
    if abs(total - 1.0) > PROB_TOL:
        raise DecisionValidationError(f"{name}: probabilities sum to {total}")


def validate_response(
    questions: dict[str, ChoiceQuestion | NoulQuestion | ScoreQuestion],
    raw: dict[str, Any],
    latency_ms: float,
    cached: bool = False,
) -> DecisionResult:
    """Parse a raw JSON response and check it against the questions that were asked."""
    try:
        result = DecisionResult.model_validate(
            {
                "model": raw.get("model"),
                "answers": raw.get("answers"),
                "usage": raw.get("usage") or {},
                "latency_ms": latency_ms,
                "cached": cached,
            }
        )
    except Exception as exc:
        raise DecisionValidationError(f"malformed response: {exc}") from exc

    if set(result.answers) != set(questions):
        raise DecisionValidationError(
            f"answer names {sorted(result.answers)} != question names {sorted(questions)}"
        )

    for name, question in questions.items():
        answer = result.answers[name]
        if answer.type != question.type:
            raise DecisionValidationError(f"{name}: answer type {answer.type} != {question.type}")
        if isinstance(answer, ChoiceAnswer):
            _check_distribution(name, answer.probabilities, set(question.criteria))
            if answer.choice not in question.criteria:
                raise DecisionValidationError(f"{name}: choice {answer.choice!r} not a declared option")
            best = max(answer.probabilities.values())
            if answer.probabilities[answer.choice] < best - PROB_TOL:
                raise DecisionValidationError(f"{name}: choice {answer.choice!r} is not the argmax")
            if not 0.0 <= answer.confidence <= 1.0:
                raise DecisionValidationError(f"{name}: confidence {answer.confidence} outside [0, 1]")
        elif isinstance(answer, NoulAnswer):
            if not 0.0 <= answer.noul <= 1.0:
                raise DecisionValidationError(f"{name}: noul {answer.noul} outside [0, 1]")
        elif isinstance(answer, ScoreAnswer):
            levels = {str(i) for i in range(len(question.criteria))}
            _check_distribution(name, answer.probabilities, levels)
            expected = sum(int(k) * p for k, p in answer.probabilities.items())
            if abs(expected - answer.score) > SCORE_TOL * len(levels):
                raise DecisionValidationError(
                    f"{name}: score {answer.score} != probability-weighted level {expected}"
                )
    return result
