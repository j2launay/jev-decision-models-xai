"""CLI: `uv run python -m jev_xai {dataset,benchmark,run,report}`."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import experiment as ex
from . import plots
from .cache import ResponseCache
from .config import ExperimentConfig

STAGES = ("predictions", "interference", "attribution", "convergence", "secondary")


def _safe(model: str) -> str:
    return model.replace(":", "-")


def _concat(run_dir: Path, name: str) -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in sorted(run_dir.glob(f"*/{name}.parquet"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def cmd_dataset(cfg: ExperimentConfig, args: argparse.Namespace) -> None:
    instances = ex.load_or_build_dataset(cfg)
    df = pd.read_parquet(cfg.dataset_path)
    print(f"{len(instances)} instances -> {cfg.dataset_path}")
    print(df.groupby(["source", "gold_choice"]).size().to_string())


def cmd_benchmark(cfg: ExperimentConfig, args: argparse.Namespace) -> None:
    instances = ex.load_or_build_dataset(cfg)
    run_dir = cfg.results_dir / args.run_id
    for model in cfg.models:
        bench = ex.benchmark_latency(cfg, model, instances)
        (run_dir / _safe(model)).mkdir(parents=True, exist_ok=True)
        bench.to_parquet(run_dir / _safe(model) / "benchmark.parquet", index=False)
        seq = bench[bench["mode"] == "sequential"]["latency_ms"]
        thr = bench[bench["mode"] == "throughput"]
        print(f"[{model}] sequential latency ms: median={seq.median():.1f} p90={seq.quantile(0.9):.1f}")
        for r in thr.itertuples():
            print(f"[{model}]   concurrency={r.concurrency}: {r.n_requests / (r.wall_ms / 1000):.1f} req/s")


def cmd_run(cfg: ExperimentConfig, args: argparse.Namespace) -> None:
    instances = ex.load_or_build_dataset(cfg)
    if args.limit:
        import numpy as np
        by_source: dict[str, list] = {}
        for inst in instances:
            by_source.setdefault(inst.source, []).append(inst)
        instances = [
            group[idx]
            for group in by_source.values()
            for idx in np.linspace(0, len(group) - 1, min(args.limit, len(group)), dtype=int)
        ]
    stages = args.stages.split(",")
    run_dir = cfg.results_dir / args.run_id
    cache = ResponseCache(cfg.cache_path)
    digests = {}
    for model in cfg.models:
        out = run_dir / _safe(model)
        out.mkdir(parents=True, exist_ok=True)
        with ex._client(cfg, model, cache) as client:
            digests[model] = client.model_digest
            t0 = time.perf_counter()
            if "predictions" in stages:
                preds = ex.run_predictions(cfg, client, instances)
                preds.to_parquet(out / "predictions.parquet", index=False)
                ex.calibration_table(preds, cfg).to_parquet(out / "calibration.parquet", index=False)
                ex.reliability_bins(preds, cfg).to_parquet(out / "reliability_bins.parquet", index=False)
            if "interference" in stages:
                ex.run_interference(client, instances).to_parquet(out / "interference.parquet", index=False)
            if "attribution" in stages:
                attr, faith, plaus = ex.run_attributions(cfg, client, instances, cfg.unit, cfg.attribution_methods)
                attr.to_parquet(out / "attributions.parquet", index=False)
                faith.to_parquet(out / "faithfulness.parquet", index=False)
                plaus.to_parquet(out / "plausibility.parquet", index=False)
            if "secondary" in stages and cfg.secondary_unit:
                attr, faith, plaus = ex.run_attributions(cfg, client, instances, cfg.secondary_unit, cfg.attribution_methods)
                attr.to_parquet(out / "attributions_secondary.parquet", index=False)
                faith.to_parquet(out / "faithfulness_secondary.parquet", index=False)
                plaus.to_parquet(out / "plausibility_secondary.parquet", index=False)
            if "convergence" in stages:
                ex.run_convergence(cfg, client, instances).to_parquet(out / "convergence.parquet", index=False)
            print(f"[{model}] stages {stages} done in {time.perf_counter() - t0:.0f}s")
    cache.close()
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest.update(ex.build_manifest(cfg, digests, cfg.dataset_path))
    manifest.setdefault("invocations", []).append(
        {"at": datetime.now().isoformat(timespec="seconds"), "models": cfg.models, "stages": stages, "limit": args.limit}
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    cmd_report(cfg, args)


def _summary(run_dir: Path) -> dict:
    summary: dict = {}
    calib = _concat(run_dir, "calibration")
    if not calib.empty:
        cols = ["model", "source", "head", "condition", "n", "accuracy", "ece_width", "ece_width_ci_low", "ece_width_ci_high",
                "ece_mass", "mce_width", "brier", "auroc_maxp", "auroc_provider_confidence", "ece_ptrue_width",
                "delta_ece_width", "delta_ece_width_ci_low", "delta_ece_width_ci_high", "delta_accuracy", "flip_rate"]
        summary["calibration"] = calib[[c for c in cols if c in calib]].round(4).to_dict(orient="records")
    for name in ("faithfulness", "faithfulness_secondary"):
        faith = _concat(run_dir, name)
        if not faith.empty:
            agg = faith.groupby(["model", "unit", "mask_mode", "head", "method"])[
                ["aopc_comprehensiveness", "aopc_sufficiency", "deletion_auc", "insertion_auc"]
            ].mean().round(4).reset_index()
            summary[name] = agg.to_dict(orient="records")
    plaus = _concat(run_dir, "plausibility")
    if not plaus.empty:
        agg = plaus.groupby(["model", "head", "method", "has_confounder"])[
            ["precision_at_flags", "confounder_share_at_flags", "mean_score_flag", "mean_score_confounder", "mean_score_other"]
        ].mean().round(4).reset_index()
        summary["plausibility_triage"] = agg.to_dict(orient="records")
    inter = _concat(run_dir, "interference")
    if not inter.empty:
        summary["interference"] = inter.groupby(["model", "source", "head"]).agg(
            n=("flip", "size"), flip_rate=("flip", "mean"), acc_separate=("correct_separate", "mean"),
            acc_joint=("correct_joint", "mean"), mean_total_variation=("total_variation", "mean"),
            mean_p_true_separate=("p_true_separate", "mean"), mean_p_true_joint=("p_true_joint", "mean"),
        ).round(4).reset_index().to_dict(orient="records")
    conv = _concat(run_dir, "convergence")
    if not conv.empty:
        summary["convergence"] = conv.groupby(["model", "head", "budget"])[["spearman_vs_ref", "top20pct_overlap_vs_ref"]].mean().round(4).reset_index().to_dict(orient="records")
    bench = _concat(run_dir, "benchmark")
    if not bench.empty:
        seq = bench[bench["mode"] == "sequential"].groupby("model")["latency_ms"].describe(percentiles=[0.5, 0.9]).round(1)
        thr = bench[bench["mode"] == "throughput"].assign(rps=lambda d: d["n_requests"] / (d["wall_ms"] / 1000))
        summary["latency_sequential_ms"] = seq.reset_index().to_dict(orient="records")
        summary["throughput_rps"] = thr[["model", "concurrency", "rps"]].round(2).to_dict(orient="records")
    return summary


def cmd_report(cfg: ExperimentConfig, args: argparse.Namespace) -> None:
    run_dir = cfg.results_dir / args.run_id
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    bins = _concat(run_dir, "reliability_bins")
    if not bins.empty:
        plots.reliability_diagrams(bins, fig_dir, "clean", "width")
        plots.reliability_diagrams(bins, fig_dir, "clean", "mass")
    calib = _concat(run_dir, "calibration")
    if not calib.empty:
        plots.ece_by_condition(calib, fig_dir, "width")
    for name in ("faithfulness", "faithfulness_secondary"):
        faith = _concat(run_dir, name)
        if not faith.empty:
            plots.deletion_insertion(faith, fig_dir)
    plots.convergence(_concat(run_dir, "convergence"), fig_dir)
    plots.latency(_concat(run_dir, "benchmark"), fig_dir)
    summary = _summary(run_dir)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"report -> {run_dir}")


def _keep_awake() -> None:
    """Prevent Windows from sleeping while this process runs (sleep stalls in-flight requests)."""
    if sys.platform == "win32":
        import ctypes

        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)


def main() -> None:
    _keep_awake()
    parser = argparse.ArgumentParser(prog="jev_xai")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("dataset", "benchmark", "run", "report"):
        p = sub.add_parser(name)
        p.add_argument("--models", nargs="+", default=None)
        p.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d-%H%M%S"))
        p.add_argument("--concurrency", type=int, default=None)
        p.add_argument("--shap-nsamples", type=int, default=None)
        if name == "run":
            p.add_argument("--stages", default=",".join(STAGES), help=f"comma-separated subset of {STAGES}")
            p.add_argument("--limit", type=int, default=None, help="max instances per source (smoke tests)")
    args = parser.parse_args()
    cfg = ExperimentConfig()
    if args.models:
        cfg.models = args.models
    if args.concurrency:
        cfg.concurrency = args.concurrency
    if args.shap_nsamples:
        cfg.shap_nsamples = args.shap_nsamples
    {"dataset": cmd_dataset, "benchmark": cmd_benchmark, "run": cmd_run, "report": cmd_report}[args.command](cfg, args)


if __name__ == "__main__":
    main()
