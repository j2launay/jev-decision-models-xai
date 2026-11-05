# Can the Confidence and Decisions of System 1 Models Be Trusted?
### An Explainability and Perturbation Audit of Jev-Style Decision Models

[![arXiv](https://img.shields.io/badge/arXiv-2611.xxxxx-b31b1b.svg)](https://arxiv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Reproducibility](https://img.shields.io/badge/Reproducibility-Bit--Identical-success.svg)](#reproducibility)

This repository contains the official codebase, datasets, and LaTeX manuscript for the paper:
> **"Can the Confidence and Decisions of System 1 Models Be Trusted? An Explainability and Perturbation Audit of Jev-Style Decision Models"**  
> *Julien Delaunay*  
> Draft manuscript, November 2026.

---

## 🔬 Overview

"System 1" decision models (pioneered by TypeSafe AI's Jev) replace autoregressive token generation with a single forward pass returning typed, structured probability distributions over declared schemas (*Choice*, *Score*, *Noul*). While widely deployed for routing, agent control, and clinical triage, their reliability and interpretability have remained uncharacterized.

This benchmark provides an end-to-end audit across **five open-weight System 1 models** ($0.8\text{B}$ to $27\text{B}$) and a **generative System 2 baseline** (`Gemma4-8B` JSON & CoT) on $N=450$ reasoning instances, an adversarial prompt injection benchmark ($N=150$), and an exact ground-truth logic gate suite:

1. **Entropy Confidence:** The provider `confidence` field is mathematically verified to equal normalized Shannon entropy ($1 - H(p)/\ln K$), measuring distributional dispersion rather than empirical accuracy.
2. **Cross-Question Interference:** Bundling multiple questions in a single request induces decision flips at sub-billion scale ($22.4\%$ flips on `Tev1-0.8B`, up to $60$--$82\%$ under exploratory uncalibrated prompts), but is virtually eliminated at $\ge 4\text{B}$ scale ($0.2\%$ on `Tev1-4B`).
3. **High-Throughput Explainability:** Single-pass models run in $24$--$250$~ms ($18\times$--$187\times$ faster than generative System 2), enabling $512$-coalition game-theoretic attributions in $1.3$--$3.2$~seconds.
4. **S1-SHAP & Logic Gate Recovery:** We introduce **S1-SHAP** (KernelSHAP in one-vs-rest log-odds space), which resolves probability saturation ($p \approx 0.999$) at zero additional model calls and achieves **$100\%$ boolean logic gate recovery** (OR vs. AND gates via Shapley Interaction Indices) where probability-space surrogates collapse.
5. **High-Throughput Safety Guardrails (Tensor Trust):** Evaluating prompt injection and jailbreak defenses demonstrates that System 1 models provide ultra-fast inline firewalls (19–56 ms vs. 426 ms for generative LLMs) with up to **94.7% attack recall and strictly 0.0% false alarms** on benign queries, while **S1-SHAP** pinpoints malicious command tokens for SOC incident response without model weight access.

---

## 📊 Benchmark Highlights

### 🧠 In-Distribution, Shortcuts & Clinical Triage ($N=450$)

| Model | Scale | e-SNLI Acc / $\text{ECE}_w$ | HANS Acc / $\text{ECE}_w$ | Triage Acc / $\text{ECE}_w$ | Question Flip (C/N) | S1-SHAP Comp. | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Tev1-0.8B`** | $0.8\text{B}$ | 0.860 / 0.035 | 0.780 / 0.111 | 0.553 / 0.265 | 0.05 / 0.22 | **0.527** | **26 ms** |
| **`Tev1-4B`** | $4.0\text{B}$ | 0.947 / 0.019 | 0.927 / 0.037 | 0.967 / 0.112 | **0.00 / 0.00** | **0.530** | 45 ms |
| **`Nimble-9B`** | $9.0\text{B}$ | **0.953 / 0.014** | 0.920 / 0.073 | 0.993 / 0.111 | 0.11 / 0.05 | 0.441 | 250 ms |
| **`Clef-Flash-9B`** | $9.0\text{B}$ | 0.900 / 0.107 | 0.873 / 0.063 | 0.967 / 0.115 | 0.01 / 0.13 | 0.299 | 64 ms |
| **`Clef-27B`** | $27.0\text{B}$ | 0.927 / 0.049 | **0.940 / 0.032** | **1.000 / 0.091** | 0.00 / 0.17 | --$^\dagger$ | 1,953 ms |
| *`Gemma4-8B (Sys. 2)`* | $8.0\text{B}$ | 0.867 / 0.058 | 0.660 / 0.246 | 1.000 / 0.162 | 0.03 / 0.37 | -- | 1,133 ms |

### 🛡️ Cybersecurity Guardrail Benchmark (Tensor Trust, $N=150$)

| Model | Type | Latency (ms) $\downarrow$ | Choice Acc. $\uparrow$ | TPR (Attack Recall) $\uparrow$ | FPR (False Alarms) $\downarrow$ | $\text{ECE}_w \downarrow$ | Brier $\downarrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Tev1-0.8B`** | System 1 | **19 ms** | 55.3 % | 10.7 % | **0.0 %** | 0.296 | 0.507 |
| **`Tev1-4B`** | System 1 | 46 ms | 84.0 % | 68.0 % | **0.0 %** | 0.047 | 0.231 |
| **`Clef-Flash-9B`** | System 1 | 56 ms | 90.7 % | 81.3 % | **0.0 %** | 0.035 | 0.146 |
| **`Nimble-9B`** | System 1 | 264 ms | 83.3 % | 66.7 % | **0.0 %** | 0.076 | 0.255 |
| **`Clef-27B`** | System 1 | 811 ms | **97.3 %** | **94.7 %** | **0.0 %** | **0.010** | **0.055** |
| *`Gemma4-8B (Sys. 2)`* | Generative JSON | 426 ms | 74.7 % | 49.3 % | **0.0 %** | 0.230 | 0.465 |

---

## 📁 Repository Structure

```text
jev-decision-models-xai/
├── data/
│   ├── poc/
│   │   └── dataset_expanded_450.parquet  # Balanced cohort (e-SNLI, HANS, Triage)
│   ├── raw/
│   │   └── tensor_trust_150.jsonl        # Balanced prompt injection benchmark
│   └── planted_v2/
│       └── cases.json                    # Exact ground-truth trigger benchmark
├── jev_xai/                              # Core Python framework
│   ├── client.py                         # Typed schema client (Ollama / Jev)
│   ├── attribution.py                    # LOO, LIME, KernelSHAP, S1-SHAP
│   ├── eval_tensor_trust.py              # Adversarial prompt injection evaluation
│   ├── experiment.py                     # Deterministic caching & evaluation
│   ├── interference_controls.py          # Ordering, permutation, dummy controls
│   ├── posthoc_calibration.py            # Platt scaling, Temperature scaling, E-AURC
│   ├── planted_v2.py                     # Shapley Interaction Index & logic gates
│   └── plots.py                          # Publication-grade figures
├── paper/                                # Complete LaTeX manuscript & assets
│   ├── main.tex                          # Primary LaTeX source (TMLR format)
│   ├── references.bib                    # BibTeX citations
│   ├── figures/                          # Vector and high-res PNG plots
│   ├── tables/                           # Auto-generated LaTeX tables
│   ├── make_tables.py                    # Table generation pipeline
│   └── main_v2.pdf                       # Compiled 33-page TMLR manuscript
├── results/                              # Benchmark outputs and Parquet logs
│   ├── expanded-450/                     # Clean calibration & interference runs
│   ├── tensor_trust/                     # Prompt injection guardrail metrics
│   ├── interference-controls/            # Causal isolation ablation logs
│   ├── posthoc/                          # Recalibration & selective risk tables
│   └── planted-v2-smoke/                 # Pairwise logic gate interaction logs
├── pyproject.toml                        # Project dependencies
└── README.md
```

---

## 🚀 Quickstart

### 1. Installation

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/) (or standard `pip`):

```bash
git clone https://github.com/j2launay/jev-decision-models-xai.git
cd jev-decision-models-xai
uv sync
```

### 2. Model Setup via Ollama

Make sure [Ollama](https://ollama.ai) is running locally with concurrent workers enabled:

```bash
ollama serve  # OLLAMA_NUM_PARALLEL=4
ollama pull tev1:0.8b
ollama pull tev1:latest
ollama pull nimble:latest
ollama pull clef-flash:latest
ollama pull clef:27b
ollama pull gemma4:latest
```

### 3. Reproducing Paper Tables and Figures

To recompile all tables from experimental results and regenerate the PDF:

```bash
# Generate LaTeX tables
uv run python paper/make_tables.py --run poc-tev1-0.8b --model tev1:0.8b --bench bench-np4

# Generate high-resolution figures
uv run python jev_xai/generate_paper_figures.py

# Compile manuscript to PDF
cd paper
pdflatex -jobname=main_v2 main.tex
bibtex main_v2
pdflatex -jobname=main_v2 main.tex
pdflatex -jobname=main_v2 main.tex
```

### 4. Running the Prompt Injection Guardrail Evaluation

To run the Tensor Trust guardrail benchmark and compare attack detection across models:

```bash
# Run inline firewall evaluation on Clef-Flash-9B
uv run python -m jev_xai.eval_tensor_trust --model clef-flash:latest

# Or evaluate Together AI's Tev1-4B
uv run python -m jev_xai.eval_tensor_trust --model tev1:latest
```

---

## 🔒 Reproducibility

- **Bit-Identical Determinism:** All responses are cached in a local SQLite database indexed by `SHA-256(model_weight_digest + canonical_payload)`.
- **Black-Box Zero-Overhead XAI:** `S1-SHAP` operates strictly on returned continuous probability distributions without requiring access to model parameters, gradients, or internal activations.
- **Fair Accounting:** Latency numbers are recorded strictly on uncached requests under isolated batch-1 execution.

---

## 📖 Citation

If you find this work, codebase, or benchmark useful in your research, please cite:

```bibtex
@article{delaunay2026system1xai,
  title   = {Can the Confidence and Decisions of System 1 Models Be Trusted? An Explainability and Perturbation Audit of Jev-Style Decision Models},
  author  = {Delaunay, Julien},
  journal = {arXiv preprint arXiv:2611.xxxxx},
  year    = {2026}
}
```

---

## 📄 License

This repository is licensed under the [MIT License](LICENSE).

