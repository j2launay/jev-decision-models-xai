"""Feature segmentation (word / subword units with character offsets) and text masking.

A `FeatureSpace` turns a multi-segment input (e.g. premise + hypothesis) into a flat list of
maskable features and renders any binary keep-vector back into segment texts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np

Unit = Literal["word", "subword"]
MaskMode = Literal["delete", "placeholder"]

WORD_RE = re.compile(r"\w+(?:[-'’]\w+)*|[^\w\s]", re.UNICODE)
DEFAULT_TOKENIZER_PATH = Path(__file__).resolve().parent.parent / "data" / "tokenizers" / "qwen3.5" / "tokenizer.json"


@dataclass(frozen=True)
class Feature:
    index: int
    segment: str
    start: int
    end: int
    text: str


@dataclass
class FeatureSpace:
    segments: dict[str, str]
    features: list[Feature]
    unit: Unit
    # Characters of non-maskable tokens (punctuation when mask_punctuation=False) stay untouched.
    _spans_by_segment: dict[str, list[Feature]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._spans_by_segment = {name: [] for name in self.segments}
        for f in self.features:
            self._spans_by_segment[f.segment].append(f)

    @property
    def n_features(self) -> int:
        return len(self.features)

    def render(
        self, keep: np.ndarray, mode: MaskMode = "delete", placeholder: str = "[MASK]"
    ) -> dict[str, str]:
        keep = np.asarray(keep).astype(bool)
        if keep.shape != (self.n_features,):
            raise ValueError(f"keep vector has shape {keep.shape}, expected ({self.n_features},)")
        out: dict[str, str] = {}
        for name, text in self.segments.items():
            pieces: list[str] = []
            cursor = 0
            for f in self._spans_by_segment[name]:
                pieces.append(text[cursor : f.start])
                if keep[f.index]:
                    pieces.append(f.text)
                elif mode == "placeholder":
                    pieces.append(placeholder)
                cursor = f.end
            pieces.append(text[cursor:])
            out[name] = _normalize_spacing("".join(pieces)) if mode == "delete" else "".join(pieces)
        return out

    def feature_mask(self, segment: str, start: int, end: int) -> np.ndarray:
        """Boolean vector of features overlapping a character span of a segment."""
        return np.array(
            [f.segment == segment and f.start < end and f.end > start for f in self.features], dtype=bool
        )


def _normalize_spacing(text: str) -> str:
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" +([,.;:!?)])", r"\1", text)
    return text.strip()


def _word_spans(text: str, mask_punctuation: bool) -> list[tuple[int, int, str]]:
    spans = []
    for m in WORD_RE.finditer(text):
        token = m.group(0)
        if not mask_punctuation and not re.search(r"\w", token):
            continue
        spans.append((m.start(), m.end(), token))
    return spans


@lru_cache(maxsize=2)
def _load_tokenizer(path: str):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(path)


def _subword_spans(text: str, tokenizer_path: Path, mask_punctuation: bool) -> list[tuple[int, int, str]]:
    tok = _load_tokenizer(str(tokenizer_path))
    enc = tok.encode(text, add_special_tokens=False)
    spans = []
    for start, end in enc.offsets:
        # Byte-level BPE folds the leading space into the token; keep only the visible characters.
        while start < end and text[start].isspace():
            start += 1
        if start >= end:
            continue
        if spans and start < spans[-1][1]:
            # Multi-byte characters can be split across byte-level tokens with overlapping offsets.
            prev_start = spans[-1][0]
            end = max(end, spans[-1][1])
            spans[-1] = (prev_start, end, text[prev_start:end])
            continue
        piece = text[start:end]
        if not mask_punctuation and not re.search(r"\w", piece):
            continue
        spans.append((start, end, piece))
    return spans


def build_feature_space(
    segments: dict[str, str],
    unit: Unit = "word",
    mask_punctuation: bool = False,
    tokenizer_path: Path = DEFAULT_TOKENIZER_PATH,
) -> FeatureSpace:
    features: list[Feature] = []
    for name, text in segments.items():
        if unit == "word":
            spans = _word_spans(text, mask_punctuation)
        else:
            spans = _subword_spans(text, tokenizer_path, mask_punctuation)
        for start, end, piece in spans:
            features.append(Feature(len(features), name, start, end, piece))
    return FeatureSpace(segments=dict(segments), features=features, unit=unit)
