import numpy as np
import pytest

from jev_xai import calibration as cal
from jev_xai.attribution import ValueFunction, kernel_shap, leave_one_out, lime_attribution, s1_shap
from jev_xai.datasets import _starred_word_spans, generate_triage
from jev_xai.faithfulness import evaluate, k_values
from jev_xai.features import build_feature_space
from jev_xai.perturbations import PERTURBATIONS, add_typos, perturb
from jev_xai.schemas import DecisionResult, DecisionValidationError, validate_response
from jev_xai.tasks import NLI, Instance

RAW = {
    "model": "nimble",
    "answers": {
        "label": {"type": "choice", "choice": "entailment",
                  "probabilities": {"entailment": 0.9, "neutral": 0.06, "contradiction": 0.04}, "confidence": 0.6},
        "noul": {"type": "noul", "noul": 0.97},
    },
    "usage": {"input_tokens": 100, "output_tokens": 2},
}


# ---------------------------------------------------------------- schemas

def test_validate_response_accepts_wellformed():
    r = validate_response(NLI.questions, RAW, latency_ms=12.0)
    assert r.answers["label"].choice == "entailment"
    assert r.answers["noul"].noul == pytest.approx(0.97)


def test_validate_response_rejects_wrong_keys():
    bad = {**RAW, "answers": {**RAW["answers"], "label": {**RAW["answers"]["label"],
           "probabilities": {"entailment": 0.9, "other": 0.1}}}}
    with pytest.raises(DecisionValidationError):
        validate_response(NLI.questions, bad, latency_ms=1.0)


def test_validate_response_rejects_non_argmax_choice():
    bad = {**RAW, "answers": {**RAW["answers"], "label": {**RAW["answers"]["label"], "choice": "neutral"}}}
    with pytest.raises(DecisionValidationError):
        validate_response(NLI.questions, bad, latency_ms=1.0)


def test_hans_choice_view_is_binary_for_both_gold_labels():
    from jev_xai.tasks import choice_view

    answer = validate_response(NLI.questions, RAW, 0.0).answers["label"]
    for gold in ("entailment", "non_entailment"):
        inst = Instance("h", "nli", "hans", {"premise": "a", "hypothesis": "b"}, gold, gold == "entailment")
        v = choice_view(answer, inst, NLI)
        assert v.classes == ("entailment", "non_entailment")
        assert v.correct == (gold == "entailment")
        assert v.conf == pytest.approx(0.9)


def test_entropy_confidence_matches_provider_definition():
    probs = {"urgent": 0.8959958054387178, "semi_urgent": 0.043186095636783835, "non_urgent": 0.06081809892449834}
    from jev_xai.schemas import ChoiceAnswer

    a = ChoiceAnswer(type="choice", choice="urgent", probabilities=probs, confidence=0.6319161612733006)
    assert a.entropy_confidence == pytest.approx(a.confidence, abs=1e-6)


# ---------------------------------------------------------------- calibration

def test_ece_zero_when_perfectly_calibrated():
    conf = np.repeat([0.25, 0.75], 400)
    correct = np.concatenate([np.r_[np.ones(100), np.zeros(300)], np.r_[np.ones(300), np.zeros(100)]])
    assert cal.calibration_error(conf, correct, 10, "width").ece == pytest.approx(0.0, abs=1e-12)
    # Equal-mass bins split ties arbitrarily; with 2 bins each bin holds exactly one confidence level.
    assert cal.calibration_error(conf, correct, 2, "mass").ece == pytest.approx(0.0, abs=1e-12)


def test_ece_and_mce_known_values():
    conf = np.array([0.95, 0.95, 0.95, 0.95, 0.55, 0.55])
    correct = np.array([1, 1, 0, 0, 1, 1])
    r = cal.calibration_error(conf, correct, 10, "width")
    assert r.ece == pytest.approx(4 / 6 * 0.45 + 2 / 6 * 0.45)
    assert r.mce == pytest.approx(0.45)


def test_width_bins_are_right_closed():
    r = cal.calibration_error(np.array([0.1, 0.10001]), np.array([1, 1]), 10, "width")
    counts = [b.count for b in r.bins]
    assert counts[0] == 1 and counts[1] == 1


def test_brier():
    assert cal.brier_multiclass(np.array([[1.0, 0.0, 0.0]]), np.array([0])) == 0.0
    assert cal.brier_multiclass(np.array([[0.0, 1.0, 0.0]]), np.array([0])) == 2.0
    assert cal.brier_binary(np.array([0.8]), np.array([1])) == pytest.approx(0.04)


def test_bootstrap_ci_contains_point_estimate():
    x = np.random.default_rng(0).random(200)
    lo, hi = cal.bootstrap_ci(lambda idx: x[idx].mean(), len(x), 500)
    assert lo < x.mean() < hi


# ---------------------------------------------------------------- features

def test_render_roundtrip_and_deletion():
    space = build_feature_space({"premise": "A man, smiling, plays.", "hypothesis": "Someone plays."})
    assert [f.text for f in space.features] == ["A", "man", "smiling", "plays", "Someone", "plays"]
    assert space.render(np.ones(space.n_features)) == {"premise": "A man, smiling, plays.", "hypothesis": "Someone plays."}
    keep = np.ones(space.n_features)
    keep[2] = 0
    assert space.render(keep)["premise"] == "A man,, plays."
    assert space.render(keep, "placeholder")["premise"] == "A man, [MASK], plays."


def test_subword_roundtrip():
    text = "The patient's oxygen saturation is 82% — température 39.5°C."
    space = build_feature_space({"note": text}, unit="subword", mask_punctuation=True)
    assert space.n_features > len(text.split())
    assert space.render(np.ones(space.n_features))["note"] == text


# ---------------------------------------------------------------- attribution with a stub model

WEIGHTS = {"alpha": 0.3, "gamma": 0.2, "delta": -0.1}


class StubClient:
    model = "stub"

    def decide(self, state, questions, use_cache=True):
        return self.decide_batch([state], questions)[0]

    def decide_batch(self, states, questions, use_cache=True):
        out = []
        for s in states:
            words = (s["premise"] + " " + s["hypothesis"]).split()
            p = 0.4 + sum(w for k, w in WEIGHTS.items() if k in words)
            answers = {
                "label": {"type": "choice", "choice": "entailment" if p >= 1 / 3 else "neutral",
                          "probabilities": {"entailment": p, "neutral": (1 - p) / 2, "contradiction": (1 - p) / 2},
                          "confidence": 0.5},
                "noul": {"type": "noul", "noul": p}}
            raw = {"model": "stub", "answers": {k: v for k, v in answers.items() if k in questions}}
            out.append(validate_response(questions, raw, 0.0))
        return out


def _stub_vf():
    space = build_feature_space({"premise": "alpha beta gamma", "hypothesis": "delta epsilon"})
    return ValueFunction(StubClient(), NLI, space, {"label": "entailment", "noul": "true"})


def test_loo_and_kernelshap_recover_additive_weights():
    vf = _stub_vf()
    expected = np.array([0.3, 0.0, 0.2, -0.1, 0.0])
    loo = leave_one_out(vf)
    np.testing.assert_allclose(loo["label"], expected, atol=1e-9)
    shap_out = kernel_shap(vf, nsamples=512, seed=0)
    assert shap_out.nsamples == 32  # 2^5: exact enumeration
    np.testing.assert_allclose(shap_out.values["label"], expected, atol=1e-6)
    np.testing.assert_allclose(shap_out.values["noul"], expected, atol=1e-6)
    lime_out = lime_attribution(vf, nsamples=512, seed=0, alpha=1e-9)
    np.testing.assert_allclose(lime_out["label"], expected, atol=1e-4)
    s1_out = s1_shap(vf, nsamples=512, seed=0)
    assert s1_out.nsamples == 32
    # S1-SHAP preserves exact feature ordering on monotonic additive probabilities
    assert np.argmax(s1_out.values["label"]) == 0
    assert np.argmin(s1_out.values["label"]) == 3


def test_faithfulness_on_additive_model():
    vf = _stub_vf()
    scores = {"label": np.array([0.3, 0.0, 0.2, -0.1, 0.0]), "noul": np.array([0.3, 0.0, 0.2, -0.1, 0.0])}
    res = evaluate(vf, scores)["label"]
    assert res.ks == [1, 3]
    assert res.p_full == pytest.approx(0.8)
    assert res.comprehensiveness[0] == pytest.approx(0.3)  # removing "alpha"
    assert res.sufficiency[0] == pytest.approx(0.8 - 0.7)  # keeping only "alpha"
    assert len(res.deletion_curve) == 6


def test_k_values():
    assert k_values(20) == [1, 2, 4, 10]
    assert k_values(3) == [1, 2]


# ---------------------------------------------------------------- perturbations and data

def test_typos_protect_negations_and_are_deterministic():
    text = "The man is not playing a guitar without anybody nearby."
    a = add_typos(text, 1.0, np.random.default_rng(1))
    b = add_typos(text, 1.0, np.random.default_rng(1))
    assert a == b and a != text
    assert " not " in a and "without" in a


def test_perturb_is_order_independent():
    inst = Instance("x", "nli", "esnli", {"premise": "A dog runs in the park.", "hypothesis": "An animal moves."}, "entailment", True)
    for p in PERTURBATIONS:
        assert perturb(inst, p, 7) == perturb(inst, p, 7)


def test_starred_spans():
    spans = _starred_word_spans("The church has cracks in the ceiling.", " The church has *cracks* *in* *the* *ceiling.*")
    assert spans == [(15, 21), (22, 24), (25, 28), (29, 37)]


def test_triage_generation_spans_and_balance():
    insts = generate_triage({"immediate": 14, "urgent": 13, "non_urgent": 13}, seed=3)
    assert len(insts) == 40
    assert sum(i.metadata["has_confounder"] for i in insts) == 5 + 4 + 4
    for inst in insts:
        text = inst.segments["patient_note"]
        for s, e in inst.metadata["flag_spans"]:
            assert text[e - 1] == "."
            assert text[s].isupper()
