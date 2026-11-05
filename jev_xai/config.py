"""Experiment configuration. Defaults encode the Phase 1 methodological decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .features import MaskMode, Unit

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ModelSpec:
    name: str
    provider: str
    backbone: str
    params: str
    modalities: str
    deployment: str


DECISION_MODEL_REGISTRY: tuple[ModelSpec, ...] = (
    ModelSpec("tev1:0.8b", "Together AI", "Qwen3.5-0.8B", "0.8B", "Text / JSON", "Local (Ollama) / Open-weight"),
    ModelSpec("tev1:latest", "Together AI", "Qwen3.5-4B", "4B", "Text / JSON", "Local (Ollama) / Open-weight"),
    ModelSpec("nimble:latest", "Bespoke Labs", "Qwen3.5-9B", "9B", "Text / JSON", "Local (Ollama) / Open-weight"),
    ModelSpec("clef-flash:latest", "Cloudflare", "Qwen3.5-VL-9B", "9B", "Text / JSON / Vision", "Local (Ollama) & Workers AI"),
    ModelSpec("clef:27b", "Cloudflare", "Qwen3.5-VL-27B", "27B", "Text / JSON / Vision", "Workers AI & Open-weight"),
)


@dataclass
class ExperimentConfig:
    models: list[str] = field(default_factory=lambda: ["tev1:0.8b", "tev1", "nimble", "clef-flash"])
    base_url: str = "http://localhost:11434"
    concurrency: int = 4
    timeout_s: float = 180.0
    seed: int = 20261004

    # Dataset
    n_esnli_per_class: int = 20
    triage_counts: dict[str, int] = field(default_factory=lambda: {"immediate": 14, "urgent": 13, "non_urgent": 13})
    n_hans: int = 50

    # Features and masking
    unit: Unit = "word"
    secondary_unit: Unit | None = "subword"
    mask_punctuation: bool = False
    mask_mode: MaskMode = "delete"
    ablation_mask_mode: MaskMode | None = "placeholder"
    placeholder: str = "[MASK]"

    # Attribution
    attribution_methods: tuple[str, ...] = ("loo", "lime", "kernelshap", "s1_shap", "random")
    shap_nsamples: int = 512
    convergence_budgets: list[int] = field(default_factory=lambda: [128, 256, 512, 1024])
    convergence_n_examples: int = 10
    attribution_sources: list[str] = field(default_factory=lambda: ["esnli", "triage_synth"])

    # Calibration
    ece_bins: int = 10
    bootstrap_resamples: int = 1000
    ci_alpha: float = 0.05

    # Latency benchmark
    benchmark_n_sequential: int = 20
    benchmark_concurrency_levels: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    benchmark_n_concurrent: int = 32

    # Paths
    raw_dir: Path = ROOT / "data" / "raw"
    dataset_path: Path = ROOT / "data" / "poc" / "dataset.parquet"
    cache_path: Path = ROOT / "cache" / "responses.sqlite"
    results_dir: Path = ROOT / "results"

    def to_dict(self) -> dict[str, Any]:
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}
