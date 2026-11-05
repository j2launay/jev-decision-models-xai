"""Generate dedicated publication figures for the Planted Benchmark and Multi-Model Reliability."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
FIG_DIR = PAPER / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_planted_benchmark() -> None:
    df_path = ROOT / "results" / "poc-tev1-0.8b" / "planted_benchmark.parquet"
    if not df_path.exists():
        print("planted_benchmark.parquet not found")
        return
    df = pd.read_parquet(df_path)
    followed = df[df["rule_followed"]].copy()

    regimes = ["single_trigger", "dual_redundant", "saturated_multi"]
    regime_titles = ["Single Trigger ($g=1$)", "Dual OR-Trigger ($g=2$)", "Multi OR-Trigger ($g=3$)"]
    methods = ["loo", "lime", "kernelshap", "s1_shap", "random"]
    method_labels = ["LOO", "LIME", "KernelSHAP", "S1-SHAP (ours)", "Random"]
    colors = ["#7f7f7f", "#2ca02c", "#1f77b4", "#d62728", "#c7c7c7"]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)

    agg = followed.groupby(["regime", "method"])[["prec_at_g", "mass_share", "aopc_comprehensiveness"]].mean()

    for idx, (reg, title) in enumerate(zip(regimes, regime_titles)):
        ax = axes[idx]
        sub = agg.loc[reg] if reg in agg.index else pd.DataFrame()
        x = np.arange(len(methods))
        prec_vals = [sub.loc[m, "prec_at_g"] if m in sub.index else 0 for m in methods]
        mass_vals = [sub.loc[m, "mass_share"] if m in sub.index else 0 for m in methods]

        width = 0.38
        rects1 = ax.bar(x - width/2, prec_vals, width, label="Precision @ $g$", color=colors, alpha=0.9, edgecolor="black", linewidth=0.8)
        rects2 = ax.bar(x + width/2, mass_vals, width, label="Attribution Mass Share", color=colors, alpha=0.45, hatch="//", edgecolor="black", linewidth=0.8)

        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, rotation=35, ha="right", fontsize=9.5)
        ax.set_ylim(0, 1.08)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        for rect in rects1:
            h = rect.get_height()
            if h > 0.05:
                ax.annotate(f"{h:.2f}",
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    axes[0].set_ylabel("Metric Value (0 to 1)", fontsize=11)
    # Custom legend for the two bar types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4a4a4a", edgecolor="black", label="Precision @ $g$ (Recovery Rate)"),
        Patch(facecolor="#4a4a4a", alpha=0.45, hatch="//", edgecolor="black", label="Positive Attribution Mass Share"),
    ]
    axes[0].legend(handles=legend_elements, loc="upper right", fontsize=9)
    fig.suptitle("Ground-Truth Trigger Recovery across Secret System-Rule Regimes", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path = FIG_DIR / "planted_benchmark_regimes.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_multimodel_calibration() -> None:
    exp_dir = ROOT / "results" / "expanded-450"
    poc_dir = ROOT / "results" / "poc-tev1-0.8b"

    models = [
        ("tev1:0.8b", "Tev1-0.8B (Sys. 1)", exp_dir / "tev1-0.8b" / "predictions.parquet", "#ff7f0e"),
        ("tev1:latest", "Tev1-4B (Sys. 1)", exp_dir / "tev1-latest" / "predictions.parquet", "#2ca02c"),
        ("nimble:latest", "Nimble-9B (Sys. 1)", exp_dir / "nimble-latest" / "predictions.parquet", "#1f77b4"),
        ("clef-flash:latest", "Clef-Flash-9B (Sys. 1)", exp_dir / "clef-flash-latest" / "predictions.parquet", "#9467bd"),
        ("gemma4-json", "Gemma4-8B (Sys. 2 JSON)", poc_dir / "gemma4-json" / "predictions.parquet", "#d62728"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    for m_id, label, path, color in models:
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        clean = df[(df["condition"] == "clean") & (df["head"] == "label")]

        for ax_idx, source in enumerate(["esnli", "hans"]):
            ax = axes[ax_idx]
            sub = clean[clean["source"] == source]
            if sub.empty:
                continue
            conf = sub["conf"].to_numpy()
            corr = sub["correct"].to_numpy().astype(float)
            bins = np.linspace(0.0, 1.0, 9)
            bin_acc = []
            bin_conf = []
            bin_counts = []
            for b_idx in range(len(bins) - 1):
                lo, hi = bins[b_idx], bins[b_idx + 1]
                mask = (conf >= lo) & (conf < hi if b_idx < len(bins) - 2 else conf <= hi)
                if mask.any():
                    bin_acc.append(corr[mask].mean())
                    bin_conf.append(conf[mask].mean())
                    bin_counts.append(mask.sum())
            if bin_conf:
                ax.plot(bin_conf, bin_acc, marker="o", label=label, color=color, linewidth=1.8, markersize=5.5)

    for ax_idx, (ax, title) in enumerate(zip(axes, ["In-Distribution: e-SNLI (Choice)", "Adversarial Shortcut Shift: HANS (Choice)"])):
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect Calibration")
        ax.set_xlim(0.3, 1.02)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("Mean Predicted Confidence", fontsize=10.5)
        ax.set_ylabel("Empirical Accuracy", fontsize=10.5)
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[1].legend(loc="upper left", fontsize=9)
    fig.suptitle("Calibration Reliability: System 1 Decision Models vs. System 2 Generative Baseline", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    out_path = FIG_DIR / "multimodel_calibration.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    plot_planted_benchmark()
    plot_multimodel_calibration()

