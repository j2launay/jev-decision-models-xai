"""Matplotlib figures for a run directory."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GRID = np.linspace(0.0, 1.0, 21)


SOURCE_LABELS = {"esnli": "e-SNLI", "hans": "HANS", "triage_synth": "Clinical Triage"}
HEAD_LABELS = {"label": "Choice Head", "noul": "Noul Head"}
METHOD_LABELS = {"loo": "LOO", "lime": "LIME", "kernelshap": "KernelSHAP", "s1_shap": "S1-SHAP (ours)", "random": "Random"}


def _safe(name: str) -> str:
    return name.replace(":", "-")


def reliability_diagrams(bins: pd.DataFrame, out: Path, condition: str = "clean", binning: str = "width") -> None:
    for model, g in bins[(bins["condition"] == condition) & (bins["binning"] == binning)].groupby("model"):
        sources = sorted(g["source"].unique())
        heads = sorted(g["head"].unique())
        fig, axes = plt.subplots(len(sources), len(heads), figsize=(4.2 * len(heads), 3.4 * len(sources)), squeeze=False)
        for i, source in enumerate(sources):
            for j, head in enumerate(heads):
                ax = axes[i][j]
                b = g[(g["source"] == source) & (g["head"] == head) & (g["count"] > 0)]
                centers = (b["lower"] + b["upper"]) / 2
                widths = (b["upper"] - b["lower"]).clip(lower=0.02)
                ax.bar(centers, b["accuracy"], width=widths, edgecolor="black", alpha=0.7, label="Accuracy")
                ax.scatter(b["mean_conf"], b["accuracy"], color="tab:red", s=14 + 3 * b["count"], zorder=3, label="Bin (size ∝ n)")
                ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Perfect")
                ax.set_xlim(0, 1.02)
                ax.set_ylim(0, 1.02)
                s_name = SOURCE_LABELS.get(source, source)
                h_name = HEAD_LABELS.get(head, head)
                ax.set_title(f"{s_name} ({h_name})", fontsize=10.5, fontweight="bold")
                ax.set_xlabel("Confidence (max p)", fontsize=9.5)
                ax.set_ylabel("Accuracy", fontsize=9.5)
                ax.grid(True, linestyle="--", alpha=0.3)
        axes[0][0].legend(fontsize=7.5, loc="upper left")
        fig.suptitle(f"Reliability Diagrams — {model} ({condition}, {binning} bins)", fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out / f"reliability_{_safe(model)}_{condition}_{binning}.png", dpi=160)
        plt.close(fig)


def ece_by_condition(table: pd.DataFrame, out: Path, binning: str = "width") -> None:
    for model, g in table.groupby("model"):
        sources = sorted(g["source"].unique())
        heads = sorted(g["head"].unique())
        fig, axes = plt.subplots(1, len(sources), figsize=(4.5 * len(sources), 3.8), squeeze=False)
        for i, source in enumerate(sources):
            ax = axes[0][i]
            for k, head in enumerate(heads):
                s = g[(g["source"] == source) & (g["head"] == head)]
                conds = list(s["condition"])
                x = np.arange(len(conds)) + (k - (len(heads) - 1) / 2) * 0.25
                y = s[f"ece_{binning}"].to_numpy()
                err = np.vstack([y - s[f"ece_{binning}_ci_low"].to_numpy(), s[f"ece_{binning}_ci_high"].to_numpy() - y])
                ax.errorbar(x, y, yerr=np.clip(err, 0, None), fmt="o", capsize=3, label=head)
                ax.set_xticks(np.arange(len(conds)))
                ax.set_xticklabels(conds, rotation=45, ha="right", fontsize=8)
            ax.set_title(source)
            ax.set_ylabel(f"ECE ({binning}, 95% CI)")
            ax.legend(fontsize=8)
        fig.suptitle(f"ECE under perturbation — {model}")
        fig.tight_layout()
        fig.savefig(out / f"ece_conditions_{_safe(model)}_{binning}.png", dpi=130)
        plt.close(fig)


def _interp(curve: list[float]) -> np.ndarray:
    c = np.asarray(curve, dtype=float)
    return np.interp(GRID, np.linspace(0, 1, len(c)), c)


def deletion_insertion(faith: pd.DataFrame, out: Path) -> None:
    f = faith[faith["deletion_curve"].map(len) > 1]
    for (model, unit), g in f.groupby(["model", "unit"]):
        heads = sorted(g["head"].unique())
        fig, axes = plt.subplots(2, len(heads), figsize=(4.5 * len(heads), 7), squeeze=False)
        for j, head in enumerate(heads):
            h_name = HEAD_LABELS.get(head, head)
            for method, m in g[g["head"] == head].groupby("method"):
                m_label = METHOD_LABELS.get(method, method)
                for i, col in enumerate(("deletion_curve", "insertion_curve")):
                    mean = np.mean([_interp(c) for c in m[col]], axis=0)
                    axes[i][j].plot(GRID, mean, label=m_label, lw=1.6)
            for i, name in enumerate(("Deletion", "Insertion")):
                axes[i][j].set_title(f"{name} — {h_name}", fontsize=11, fontweight="bold")
                axes[i][j].set_xlabel("Fraction of features removed" if i == 0 else "Fraction of features inserted", fontsize=9.5)
                axes[i][j].set_ylabel("P(original predicted class)", fontsize=9.5)
                axes[i][j].set_ylim(0, 1.02)
                axes[i][j].grid(True, linestyle="--", alpha=0.3)
                axes[i][j].legend(fontsize=8, loc="upper right" if i == 1 else "lower left")
        fig.suptitle(f"Deletion / insertion curves — {model} — {unit}")
        fig.tight_layout()
        fig.savefig(out / f"deletion_insertion_{_safe(model)}_{unit}.png", dpi=130)
        plt.close(fig)


def convergence(conv: pd.DataFrame, out: Path) -> None:
    if conv.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for (model, head), g in conv.groupby(["model", "head"]):
        agg = g.groupby("budget")[["spearman_vs_ref", "top20pct_overlap_vs_ref"]].mean()
        axes[0].plot(agg.index, agg["spearman_vs_ref"], marker="o", label=f"{model}/{head}")
        axes[1].plot(agg.index, agg["top20pct_overlap_vs_ref"], marker="o", label=f"{model}/{head}")
    for ax, title in zip(axes, ("Spearman vs largest budget", "Top-20% overlap vs largest budget")):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("KernelSHAP nsamples")
        ax.set_title(title)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "kernelshap_convergence.png", dpi=130)
    plt.close(fig)


def latency(bench: pd.DataFrame, out: Path) -> None:
    if bench.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    seq = bench[bench["mode"] == "sequential"]
    models = sorted(seq["model"].unique())
    axes[0].boxplot([seq[seq["model"] == m]["latency_ms"] for m in models], tick_labels=models)
    axes[0].set_ylabel("latency per request (ms)")
    axes[0].set_title("Sequential, uncached (single-question requests)")
    thr = bench[bench["mode"] == "throughput"]
    for m, g in thr.groupby("model"):
        axes[1].plot(g["concurrency"], g["n_requests"] / (g["wall_ms"] / 1000.0), marker="o", label=m)
    axes[1].set_xlabel("client concurrency")
    axes[1].set_ylabel("requests / s")
    axes[1].set_title("Throughput")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "latency_throughput.png", dpi=130)
    plt.close(fig)
