"""Generate the LaTeX tables and copy the figures used by main.tex from a results run.

Usage: uv run python paper/make_tables.py --run poc-tev1-0.8b --model tev1:0.8b --bench bench-np4
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"

SOURCE = {"esnli": "e-SNLI", "hans": "HANS", "triage_synth": "Triage"}
HEAD = {"label": "Choice", "noul": "Noul"}
MODEL_LABEL = {
    "tev1:0.8b": "\\texttt{Tev1-0.8B}",
    "tev1:latest": "\\texttt{Tev1-4B}",
    "tev1": "\\texttt{Tev1-4B}",
    "nimble:latest": "\\texttt{Nimble-9B}",
    "nimble": "\\texttt{Nimble-9B}",
    "clef-flash:latest": "\\texttt{Clef-Flash-9B}",
    "clef-flash": "\\texttt{Clef-Flash-9B}",
    "clef:27b": "\\texttt{Clef-27B}",
    "clef-27b": "\\texttt{Clef-27B}",
    "gemma4-json": "\\texttt{Gemma4-8B} (JSON)",
    "gemma4-cot": "\\texttt{Gemma4-8B} (CoT)",
}
MODEL_ORDER = ("tev1:0.8b", "tev1:latest", "nimble:latest", "clef-flash:latest", "clef:27b", "gemma4-json", "gemma4-cot")
METHOD = {
    "loo": "LOO",
    "lime": "LIME",
    "kernelshap": "KernelSHAP",
    "s1_shap": "S1-SHAP (ours)",
    "random": "Random",
}
METHOD_ORDER = ("loo", "lime", "kernelshap", "s1_shap", "random")
CONDITION = {"typo_05": "Typos 5\\%", "typo_10": "Typos 10\\%", "typo_20": "Typos 20\\%",
             "distractor": "Distractor", "case_punct": "Case/punct."}


def f(x: float, d: int = 3) -> str:
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def signed(x: float, d: int = 3) -> str:
    return "--" if np.isnan(x) else f"{x:+.{d}f}"


def ci(lo: float, hi: float, d: int = 2) -> str:
    return f"[{lo:.{d}f}, {hi:.{d}f}]"


def write(name: str, body: str) -> None:
    (PAPER / "tables").mkdir(exist_ok=True)
    (PAPER / "tables" / f"{name}.tex").write_text(body, encoding="utf-8")


def bf(val: str, is_best: bool) -> str:
    return f"\\textbf{{{val}}}" if is_best and val != "--" else val


def calibration_clean(cal: pd.DataFrame) -> None:
    rows = []
    for source in ("esnli", "hans", "triage_synth"):
        sub = cal[(cal.source == source) & (cal.condition == "clean")].set_index("head")
        best_acc = sub.accuracy.max()
        best_ece_w = sub.ece_width.min()
        best_ece_m = sub.ece_mass.min()
        best_mce = sub.mce_width.min()
        best_brier = sub.brier.min()
        best_auroc = sub.auroc_maxp.max()
        for head in ("label", "noul"):
            r = sub.loc[head]
            auroc_p_val = r.auroc_maxp
            auroc_c_raw = r.get("auroc_provider_confidence", np.nan) if head == "label" else np.nan
            auroc_c = f(auroc_c_raw) if head == "label" else "--"
            rows.append(
                f"{SOURCE[source]} & {HEAD[head]} & {int(r.n)} & "
                f"{bf(f(r.accuracy), np.isclose(r.accuracy, best_acc))} & {f(r.mean_conf)} & "
                f"{bf(f(r.ece_width), np.isclose(r.ece_width, best_ece_w))} {ci(r.ece_width_ci_low, r.ece_width_ci_high)} & "
                f"{bf(f(r.ece_mass), np.isclose(r.ece_mass, best_ece_m))} & "
                f"{bf(f(r.mce_width), np.isclose(r.mce_width, best_mce))} & "
                f"{bf(f(r.brier), np.isclose(r.brier, best_brier))} & "
                f"{bf(f(auroc_p_val), np.isclose(auroc_p_val, best_auroc))} & {auroc_c} \\\\"
            )
        rows.append("\\midrule")
    rows = rows[:-1]
    write("calibration_clean", "\n".join([
        "\\begin{tabular}{llrrrlrrrrr}", "\\toprule",
        "Source & Head & $n$ & Acc.\\,$\\uparrow$ & Conf. & ECE$_{w}\\,\\downarrow$ [95\\% CI] & ECE$_{m}\\,\\downarrow$ & MCE$_{w}\\,\\downarrow$ & Brier\\,$\\downarrow$ & AUROC$_{p}\\,\\uparrow$ & AUROC$_{c}\\,\\uparrow$ \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", ""]))


def calibration_perturbed(cal: pd.DataFrame) -> None:
    rows = []
    for source in ("esnli", "hans", "triage_synth"):
        sub_s = cal[(cal.source == source) & (cal.condition != "clean")]
        for i, cond in enumerate(CONDITION):
            cells = [SOURCE[source] if i == 0 else "", CONDITION[cond]]
            for head in ("label", "noul"):
                sub_h = sub_s[sub_s["head"] == head]
                best_dece = sub_h.delta_ece_width.min()
                best_dacc = sub_h.delta_accuracy.max()
                best_flip = sub_h.flip_rate.min()
                r = sub_h[sub_h.condition == cond].iloc[0]
                cells += [
                    f"{bf(signed(r.delta_ece_width), np.isclose(r.delta_ece_width, best_dece))} {ci(r.delta_ece_width_ci_low, r.delta_ece_width_ci_high)}",
                    bf(signed(r.delta_accuracy), np.isclose(r.delta_accuracy, best_dacc)),
                    bf(f(r.flip_rate, 2), np.isclose(r.flip_rate, best_flip)),
                ]
            rows.append(" & ".join(cells) + " \\\\")
        rows.append("\\midrule")
    rows = rows[:-1]
    write("calibration_perturbed", "\n".join([
        "\\begin{tabular}{lllrrlrr}", "\\toprule",
        " & & \\multicolumn{3}{c}{Choice} & \\multicolumn{3}{c}{Noul} \\\\",
        "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
        "Source & Perturbation & $\\Delta$ECE$_{w}\\,\\downarrow$ [95\\% CI] & $\\Delta$Acc.\\,$\\uparrow$ & Flip\\,$\\downarrow$ & $\\Delta$ECE$_{w}\\,\\downarrow$ [95\\% CI] & $\\Delta$Acc.\\,$\\uparrow$ & Flip\\,$\\downarrow$ \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", ""]))


def interference(inter: pd.DataFrame) -> None:
    rows = []
    agg = inter.groupby(["source", "head"]).agg(
        n=("flip", "size"), flip=("flip", "mean"), acc_s=("correct_separate", "mean"), acc_j=("correct_joint", "mean"),
        tv=("total_variation", "mean"), pt_s=("p_true_separate", "mean"), pt_j=("p_true_joint", "mean"))
    best_acc_s = agg.acc_s.max()
    best_acc_j = agg.acc_j.max()
    best_flip = agg.flip.min()
    best_tv = agg.tv.min()
    for source in ("esnli", "hans", "triage_synth"):
        for head in ("label", "noul"):
            r = agg.loc[(source, head)]
            rows.append(
                f"{SOURCE[source]} & {HEAD[head]} & {int(r.n)} & "
                f"{bf(f(r.acc_s), np.isclose(r.acc_s, best_acc_s))} & "
                f"{bf(f(r.acc_j), np.isclose(r.acc_j, best_acc_j))} & "
                f"{bf(f(r.flip, 2), np.isclose(r.flip, best_flip))} & "
                f"{bf(f(r.tv), np.isclose(r.tv, best_tv))} & "
                f"{f(r.pt_s)} & {f(r.pt_j)} \\\\"
            )
    write("interference", "\n".join([
        "\\begin{tabular}{llrrrrrrr}", "\\toprule",
        "Source & Head & $n$ & Acc.\\ alone\\,$\\uparrow$ & Acc.\\ joint\\,$\\uparrow$ & Flip\\,$\\downarrow$ & TV\\,$\\downarrow$ & $\\bar{P}$(true) alone & $\\bar{P}$(true) joint \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", ""]))


def faithfulness(faith: pd.DataFrame, name: str) -> None:
    d = faith[faith.mask_mode == "delete"]
    p = faith[faith.mask_mode == "placeholder"]
    rows = []
    for source in ("esnli", "triage_synth"):
        for hi, head in enumerate(("label", "noul")):
            sub_d = d[(d.source == source) & (d["head"] == head)].groupby("method").mean(numeric_only=True)
            sub_p = p[(p.source == source) & (p["head"] == head)].groupby("method").mean(numeric_only=True)
            if sub_d.empty:
                continue
            best_comp_d = sub_d.aopc_comprehensiveness.max()
            best_suff_d = sub_d.aopc_sufficiency.min()
            best_del_d = sub_d.deletion_auc.min()
            best_ins_d = sub_d.insertion_auc.max()
            best_comp_p = sub_p.aopc_comprehensiveness.max() if not sub_p.empty else np.nan
            best_suff_p = sub_p.aopc_sufficiency.min() if not sub_p.empty else np.nan
            for mi, method in enumerate(METHOD_ORDER):
                if method not in sub_d.index:
                    continue
                ra = sub_d.loc[method]
                rb = sub_p.loc[method] if method in sub_p.index else None
                lead = (SOURCE[source] if hi == 0 and mi == 0 else "") + " & " + (HEAD[head] if mi == 0 else "")
                rows.append(
                    f"{lead} & {METHOD[method]} & "
                    f"{bf(f(ra.aopc_comprehensiveness), np.isclose(ra.aopc_comprehensiveness, best_comp_d))} & "
                    f"{bf(f(ra.aopc_sufficiency), np.isclose(ra.aopc_sufficiency, best_suff_d))} & "
                    f"{bf(f(ra.deletion_auc), np.isclose(ra.deletion_auc, best_del_d))} & "
                    f"{bf(f(ra.insertion_auc), np.isclose(ra.insertion_auc, best_ins_d))} & "
                    f"{bf(f(rb.aopc_comprehensiveness), np.isclose(rb.aopc_comprehensiveness, best_comp_p)) if rb is not None else '--'} & "
                    f"{bf(f(rb.aopc_sufficiency), np.isclose(rb.aopc_sufficiency, best_suff_p)) if rb is not None else '--'} \\\\"
                )
            rows.append("\\cmidrule{2-9}" if hi == 0 else "\\midrule")
    rows = rows[:-1]
    write(name, "\n".join([
        "\\begin{tabular}{lllrrrrrr}", "\\toprule",
        " & & & \\multicolumn{4}{c}{Deletion masking} & \\multicolumn{2}{c}{\\texttt{[MASK]} masking} \\\\",
        "\\cmidrule(lr){4-7}\\cmidrule(lr){8-9}",
        "Source & Head & Method & Comp.\\,$\\uparrow$ & Suff.\\,$\\downarrow$ & Del.\\,AUC\\,$\\downarrow$ & Ins.\\,AUC\\,$\\uparrow$ & Comp.\\,$\\uparrow$ & Suff.\\,$\\downarrow$ \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", ""]))


def _esnli_human_precision(attr: pd.DataFrame | None) -> dict[tuple[str, str], float]:
    if attr is None or attr.empty:
        return {}
    import json
    ds_path = ROOT / "data" / "poc" / "dataset.parquet"
    if not ds_path.exists():
        return {}
    ds = pd.read_parquet(ds_path).set_index("id")
    esnli_attr = attr[attr["source"] == "esnli"]
    if esnli_attr.empty:
        return {}
    out: dict[tuple[str, str], list[float]] = {}
    for (inst_id, head, method), grp in esnli_attr.groupby(["instance_id", "head", "method"]):
        if inst_id not in ds.index:
            continue
        meta = json.loads(ds.loc[inst_id, "metadata"])
        votes = meta.get("highlight_votes", {})
        if not votes:
            continue
        grp_sorted = grp.sort_values("feature_index")
        human_mask = []
        for row in grp_sorted.itertuples():
            seg_votes = votes.get(row.segment) or []
            is_h = any((s < row.end and e > row.start and v >= 2) for s, e, v in seg_votes)
            human_mask.append(is_h)
        h_arr = np.array(human_mask, dtype=bool)
        g = int(h_arr.sum())
        if g == 0:
            continue
        scores = grp_sorted["score"].to_numpy()
        top_g = np.argsort(-scores, kind="stable")[:g]
        out.setdefault((head, method), []).append(float(h_arr[top_g].mean()))
    return {k: float(np.mean(v)) for k, v in out.items()}


def plausibility(plaus: pd.DataFrame, name: str, attr: pd.DataFrame | None = None) -> None:
    human_prec = _esnli_human_precision(attr)
    rows = []
    for hi, head in enumerate(("label", "noul")):
        sub = plaus[plaus["head"] == head]
        sub_c = sub[sub.has_confounder]
        agg_all = sub.groupby("method").mean(numeric_only=True)
        agg_c = sub_c.groupby("method").mean(numeric_only=True)
        prob_methods = [m for m in agg_all.index if m not in ("random", "s1_shap")]
        best_flag_prec = agg_all.precision_at_flags.max()
        best_conf_share = agg_c.confounder_share_at_flags.min()
        best_phi_flag = agg_all.loc[prob_methods, "mean_score_flag"].max() if prob_methods else np.nan
        best_phi_conf = agg_c.loc[prob_methods, "mean_score_confounder"].abs().min() if prob_methods else np.nan
        h_vals = [human_prec.get((head, m), np.nan) for m in METHOD_ORDER if m in agg_all.index]
        best_h_prec = np.nanmax(h_vals) if h_vals and not all(np.isnan(h_vals)) else np.nan
        for mi, method in enumerate(METHOD_ORDER):
            if method not in agg_all.index:
                continue
            ra = agg_all.loc[method]
            rc = agg_c.loc[method]
            if method == "random":
                phis = ["--"] * 3
            elif method == "s1_shap":
                phis = [
                    f"{f(ra.mean_score_flag)}$^\\dagger$",
                    f"{f(rc.mean_score_confounder)}$^\\dagger$",
                    f"{f(ra.mean_score_other)}$^\\dagger$",
                ]
            else:
                phis = [
                    bf(f(ra.mean_score_flag), np.isclose(ra.mean_score_flag, best_phi_flag)),
                    bf(f(rc.mean_score_confounder), np.isclose(abs(rc.mean_score_confounder), best_phi_conf)),
                    f(ra.mean_score_other),
                ]
            h_val = human_prec.get((head, method), np.nan)
            h_cell = bf(f(h_val), np.isclose(h_val, best_h_prec)) if not np.isnan(h_val) else "--"
            m_label = f"{METHOD[method]}$^\\dagger$" if method == "s1_shap" else METHOD[method]
            rows.append(
                f"{HEAD[head] if mi == 0 else ''} & {m_label} & "
                f"{bf(f(ra.precision_at_flags), np.isclose(ra.precision_at_flags, best_flag_prec))} & "
                f"{bf(f(rc.confounder_share_at_flags), np.isclose(rc.confounder_share_at_flags, best_conf_share))} & "
                + " & ".join(phis) + f" & {h_cell} \\\\"
            )
        rows.append("\\midrule")
    rows = rows[:-1]
    write(name, "\n".join([
        "\\begin{tabular}{llrrrrrr}", "\\toprule",
        " & & \\multicolumn{5}{c}{Clinical Triage Vignettes} & \\multicolumn{1}{c}{e-SNLI} \\\\",
        "\\cmidrule(lr){3-7}\\cmidrule(lr){8-8}",
        "Head & Method & Flag prec.@$g\\,\\uparrow$ & Conf.\\ share@$g\\,\\downarrow$ & $\\bar{\\phi}$ flag\\,$\\uparrow$ & $\\bar{\\phi}$ conf.\\,$\\downarrow$ & $\\bar{\\phi}$ other & Human prec.@$g\\,\\uparrow$ \\\\",
        "Head & Method & Flag prec.@$g\\,\\uparrow$ & Conf.\\ share@$g\\,\\downarrow$ & $\\bar{\\phi}_{\\text{flag}}\\,\\uparrow$ & $\\bar{\\phi}_{\\text{conf}}\\,\\downarrow$ & $\\bar{\\phi}_{\\text{other}}$ & Human prec.@$g\\,\\uparrow$ \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", ""]))


def convergence(conv: pd.DataFrame) -> None:
    if "method" not in conv.columns:
        conv = conv.assign(method="kernelshap")
    agg = conv.groupby(["method", "head", "budget"]).agg(
        rho=("spearman_vs_ref", "mean"), top=("top20pct_overlap_vs_ref", "mean")
    )
    budgets = sorted(conv.budget.unique())
    non_ref = [b for b in budgets if b < max(budgets)]
    best_b = max(non_ref) if non_ref else max(budgets)
    rows = []
    methods_present = [m for m in ("lime", "kernelshap", "s1_shap") if m in conv["method"].unique()]
    for mi, method in enumerate(methods_present):
        for hi, head in enumerate(("label", "noul")):
            m_label = METHOD[method] if hi == 0 else ""
            rows.append(f"{m_label} & {HEAD[head]} -- Spearman & " + " & ".join(
                bf(f"{agg.loc[(method, head, b)].rho:.3f}", b == best_b) for b in budgets) + " \\\\")
            rows.append(f" & {HEAD[head]} -- top-20\\% overlap & " + " & ".join(
                bf(f"{agg.loc[(method, head, b)].top:.3f}", b == best_b) for b in budgets) + " \\\\")
        rows.append("\\midrule")
    rows = rows[:-1]
    write("convergence", "\n".join([
        "\\begin{tabular}{ll" + "r" * len(budgets) + "}", "\\toprule",
        "Method & Agreement with $N=" + str(max(budgets)) + "$ & " + " & ".join(f"$N={b}$" for b in budgets) + " \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", ""]))


def latency(bench: pd.DataFrame) -> None:
    models = [m for m in MODEL_ORDER if m in bench["model"].unique()]
    if not models:
        models = list(bench["model"].unique())
    seq = bench[bench["mode"] == "sequential"]
    thr = bench[bench["mode"] == "throughput"].assign(rps=lambda x: x.n_requests / (x.wall_ms / 1000))
    if len(models) == 1:
        meds = {head: seq[seq["head"] == head]["latency_ms"].median() for head in ("label", "noul")}
        p90s = {head: seq[seq["head"] == head]["latency_ms"].quantile(0.9) for head in ("label", "noul")}
        best_med = min(meds.values())
        best_p90 = min(p90s.values())
        rows = []
        for head in ("label", "noul"):
            rows.append(
                f"{HEAD[head]} & {bf(f'{meds[head]:.0f}', np.isclose(meds[head], best_med))} & "
                f"{bf(f'{p90s[head]:.0f}', np.isclose(p90s[head], best_p90))} \\\\"
            )
        thr_line = ", ".join(f"{int(r.concurrency)}: {r.rps:.1f}" for r in thr.itertuples())
        write("latency", "\n".join([
            "\\begin{tabular}{lrr}", "\\toprule", "Single-question request & Median (ms)\\,$\\downarrow$ & p90 (ms)\\,$\\downarrow$ \\\\", "\\midrule",
            *rows, "\\bottomrule", "\\end{tabular}", ""]))
        write("throughput_inline", thr_line + "\n")
        return

    stats = []
    for m in models:
        sm = seq[seq["model"] == m]
        tm = thr[thr["model"] == m]
        c_med = sm[sm["head"] == "label"]["latency_ms"].median()
        c_p90 = sm[sm["head"] == "label"]["latency_ms"].quantile(0.9)
        n_med = sm[sm["head"] == "noul"]["latency_ms"].median()
        n_p90 = sm[sm["head"] == "noul"]["latency_ms"].quantile(0.9)
        out_tok = sm["output_tokens"].mean() if "output_tokens" in sm.columns and sm["output_tokens"].notna().any() else 2.0
        rps1 = tm[tm["concurrency"] == 1]["rps"].max() if not tm[tm["concurrency"] == 1].empty else np.nan
        rps_peak = tm["rps"].max() if not tm.empty else np.nan
        stats.append((m, out_tok, c_med, c_p90, n_med, n_p90, rps1, rps_peak))
    df_s = pd.DataFrame(stats, columns=["model", "out_tok", "c_med", "c_p90", "n_med", "n_p90", "rps1", "rps_peak"])
    rows = []
    for r in df_s.itertuples():
        if r.model == "gemma4-json":
            rows.append("\\midrule")
        rows.append(
            f"{MODEL_LABEL.get(r.model, r.model)} & "
            f"{bf(f'{r.out_tok:.1f}', np.isclose(r.out_tok, df_s.out_tok.min()))} & "
            f"{bf(f'{r.c_med:.0f}', np.isclose(r.c_med, df_s.c_med.min()))} / {bf(f'{r.c_p90:.0f}', np.isclose(r.c_p90, df_s.c_p90.min()))} & "
            f"{bf(f'{r.n_med:.0f}', np.isclose(r.n_med, df_s.n_med.min()))} / {bf(f'{r.n_p90:.0f}', np.isclose(r.n_p90, df_s.n_p90.min()))} & "
            f"{bf(f'{r.rps1:.1f}', np.isclose(r.rps1, df_s.rps1.max()))} & "
            f"{bf(f'{r.rps_peak:.1f}', np.isclose(r.rps_peak, df_s.rps_peak.max()))} \\\\"
        )
    write("latency", "\n".join([
        "\\begin{tabular}{lrrrrr}", "\\toprule",
        "Model & Out.\\,Tok.\\,$\\downarrow$ & Choice Med./p90 (ms)\\,$\\downarrow$ & Noul Med./p90 (ms)\\,$\\downarrow$ & Req/s ($c=1$)\\,$\\uparrow$ & Peak Req/s\\,$\\uparrow$ \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", ""]))


def scaling_summary(run_dir: Path) -> None:
    exp_dir = ROOT / "results" / "expanded-450"
    cal_paths = sorted(exp_dir.glob("*/calibration.parquet")) if exp_dir.exists() else []
    inter_paths = sorted(exp_dir.glob("*/interference.parquet")) if exp_dir.exists() else []
    # Also include gemma4-json from run_dir
    for p in sorted(run_dir.glob("gemma4-json/calibration.parquet")):
        cal_paths.append(p)
    for p in sorted(run_dir.glob("gemma4-json/interference.parquet")):
        inter_paths.append(p)
    if not cal_paths:
        cal_paths = sorted(run_dir.glob("*/calibration.parquet"))
        inter_paths = sorted(run_dir.glob("*/interference.parquet"))
    cal_frames = [pd.read_parquet(p) for p in cal_paths]
    inter_frames = [pd.read_parquet(p) for p in inter_paths]
    faith_frames = [pd.read_parquet(p) for p in sorted(run_dir.glob("*/faithfulness.parquet"))]
    plaus_frames = [pd.read_parquet(p) for p in sorted(run_dir.glob("*/plausibility.parquet"))]
    if not cal_frames:
        return
    cal_all = pd.concat(cal_frames, ignore_index=True)
    inter_all = pd.concat(inter_frames, ignore_index=True) if inter_frames else pd.DataFrame()
    faith_all = pd.concat(faith_frames, ignore_index=True) if faith_all_nonempty(faith_frames) else pd.DataFrame()
    plaus_all = pd.concat(plaus_frames, ignore_index=True) if plaus_frames else pd.DataFrame()

    models = [m for m in MODEL_ORDER if m in cal_all["model"].unique()]
    if len(models) < 2:
        return

    records = []
    for m in models:
        cm = cal_all[(cal_all["model"] == m) & (cal_all["condition"] == "clean")]
        im = inter_all[inter_all["model"] == m] if not inter_all.empty else pd.DataFrame()
        fm = faith_all[(faith_all["model"] == m) & (faith_all["mask_mode"] == "delete")] if not faith_all.empty else pd.DataFrame()
        pm = plaus_all[plaus_all["model"] == m] if not plaus_all.empty else pd.DataFrame()

        def cval(src: str, hd: str, col: str) -> float:
            sub = cm[(cm["source"] == src) & (cm["head"] == hd)]
            return float(sub[col].iloc[0]) if not sub.empty else np.nan

        noul_flip = float(im[im["head"] == "noul"]["flip"].mean()) if not im.empty else np.nan
        choice_flip = float(im[im["head"] == "label"]["flip"].mean()) if not im.empty else np.nan
        s1_comp = float(fm[fm["method"] == "s1_shap"]["aopc_comprehensiveness"].mean()) if not fm.empty and "s1_shap" in fm["method"].unique() else np.nan
        ks_comp = float(fm[fm["method"] == "kernelshap"]["aopc_comprehensiveness"].mean()) if not fm.empty and "kernelshap" in fm["method"].unique() else np.nan
        if m == "gemma4-json" and not pm.empty and "self_rationale" in pm["method"].unique():
            s1_flag = float(pm[pm["method"] == "self_rationale"]["precision_at_flags"].mean())
        else:
            s1_flag = float(pm[pm["method"] == "s1_shap"]["precision_at_flags"].mean()) if not pm.empty and "s1_shap" in pm["method"].unique() else np.nan

        records.append({
            "model": m,
            "esnli_acc": cval("esnli", "label", "accuracy"),
            "esnli_ece": cval("esnli", "label", "ece_width"),
            "hans_acc": cval("hans", "label", "accuracy"),
            "hans_ece": cval("hans", "label", "ece_width"),
            "triage_acc": cval("triage_synth", "label", "accuracy"),
            "triage_ece": cval("triage_synth", "label", "ece_width"),
            "choice_flip": choice_flip,
            "noul_flip": noul_flip,
            "ks_comp": ks_comp,
            "s1_comp": s1_comp,
            "s1_flag": s1_flag,
        })
    df = pd.DataFrame(records)
    rows = []
    for r in df.itertuples():
        if r.model == "gemma4-json":
            rows.append("\\midrule")
        flag_str = (
            f"{f(r.s1_flag)}$^*$"
            if r.model == "gemma4-json" and not np.isnan(r.s1_flag)
            else (bf(f(r.s1_flag), np.isclose(r.s1_flag, df.s1_flag.max())) if not np.isnan(r.s1_flag) else "--")
        )
        rows.append(
            f"{MODEL_LABEL.get(r.model, r.model)} & "
            f"{bf(f(r.esnli_acc), np.isclose(r.esnli_acc, df.esnli_acc.max()))} & "
            f"{bf(f(r.esnli_ece), np.isclose(r.esnli_ece, df.esnli_ece.min()))} & "
            f"{bf(f(r.hans_acc), np.isclose(r.hans_acc, df.hans_acc.max()))} & "
            f"{bf(f(r.hans_ece), np.isclose(r.hans_ece, df.hans_ece.min()))} & "
            f"{bf(f(r.triage_acc), np.isclose(r.triage_acc, df.triage_acc.max()))} & "
            f"{bf(f(r.triage_ece), np.isclose(r.triage_ece, df.triage_ece.min()))} & "
            f"{bf(f(r.choice_flip, 2), np.isclose(r.choice_flip, df.choice_flip.min()))} / {bf(f(r.noul_flip, 2), np.isclose(r.noul_flip, df.noul_flip.min()))} & "
            f"{bf(f(r.ks_comp), np.isclose(r.ks_comp, df.ks_comp.max())) if not np.isnan(r.ks_comp) else '--'} & "
            f"{bf(f(r.s1_comp), np.isclose(r.s1_comp, df.s1_comp.max())) if not np.isnan(r.s1_comp) else '--'} & "
            f"{flag_str} \\\\"
        )
    write("scaling", "\n".join([
        "\\begin{tabular}{lrrrrrrrrrrr}", "\\toprule",
        " & \\multicolumn{2}{c}{e-SNLI (Choice)} & \\multicolumn{2}{c}{HANS (Choice)} & \\multicolumn{2}{c}{Triage (Choice)} & \\multicolumn{1}{c}{Interference} & \\multicolumn{3}{c}{XAI Faithfulness \\& Plaus.} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-8}\\cmidrule(lr){9-11}",
        "Model & Acc.\\,$\\uparrow$ & ECE$_{w}\\,\\downarrow$ & Acc.\\,$\\uparrow$ & ECE$_{w}\\,\\downarrow$ & Acc.\\,$\\uparrow$ & ECE$_{w}\\,\\downarrow$ & Flip (C/N)\\,$\\downarrow$ & Comp.\\,(KS)\\,$\\uparrow$ & Comp.\\,(S1)\\,$\\uparrow$ & Flag@g\\,$\\uparrow$ \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", ""]))


def faith_all_nonempty(frames: list[pd.DataFrame]) -> bool:
    return len(frames) > 0


def planted_benchmark(run_dir: Path) -> None:
    path = run_dir / "planted_benchmark.parquet"
    if not path.exists():
        return
    df = pd.read_parquet(path)
    followed = df[df["rule_followed"]].copy()
    if followed.empty:
        return
    regime_labels = {
        "single_trigger": "Single Trigger ($g{=}1$)",
        "dual_redundant": "Dual OR-Trigger ($g{=}2$)",
        "saturated_multi": "Multi OR-Trigger ($g{=}3$)",
    }
    regime_order = ("single_trigger", "dual_redundant", "saturated_multi")
    agg = (
        followed.groupby(["regime", "method"], as_index=False)[
            ["prec_at_g", "mass_share", "mrr", "aopc_comprehensiveness", "deletion_auc", "insertion_auc"]
        ]
        .mean()
    )
    rows: list[str] = []
    for r_idx, reg in enumerate(regime_order):
        sub = agg[agg["regime"] == reg]
        if sub.empty:
            continue
        p_max = sub["prec_at_g"].max()
        m_max = sub["mass_share"].max()
        r_max = sub["mrr"].max()
        c_max = sub["aopc_comprehensiveness"].max()
        d_min = sub["deletion_auc"].min()
        i_max = sub["insertion_auc"].max()
        first = True
        for m in [m for m in METHOD_ORDER if m in sub["method"].values]:
            r = sub[sub["method"] == m].iloc[0]
            reg_cell = regime_labels.get(reg, reg) if first else ""
            first = False
            rows.append(
                f"{reg_cell} & {METHOD.get(m, m)} & "
                f"{bf(f(r.prec_at_g), np.isclose(r.prec_at_g, p_max))} & "
                f"{bf(f(r.mass_share), np.isclose(r.mass_share, m_max))} & "
                f"{bf(f(r.mrr), np.isclose(r.mrr, r_max))} & "
                f"{bf(f(r.aopc_comprehensiveness), np.isclose(r.aopc_comprehensiveness, c_max))} & "
                f"{bf(f(r.deletion_auc), np.isclose(r.deletion_auc, d_min))} & "
                f"{bf(f(r.insertion_auc), np.isclose(r.insertion_auc, i_max))} \\\\"
            )
        if r_idx < len(regime_order) - 1:
            rows.append("\\midrule")
    write(
        "planted",
        "\n".join(
            [
                "\\begin{tabular}{llrrrrrr}",
                "\\toprule",
                " & & \\multicolumn{3}{c}{Ground-Truth Trigger Recovery} & \\multicolumn{3}{c}{Intervention Faithfulness} \\\\",
                "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
                "Regime & Method & Prec@$g\\,\\uparrow$ & Mass Share\\,$\\uparrow$ & MRR\\,$\\uparrow$ & Comp.\\,$\\uparrow$ & Del.\\,AUC\\,$\\downarrow$ & Ins.\\,AUC\\,$\\uparrow$ \\\\",
                "\\midrule",
                *rows,
                "\\bottomrule",
                "\\end{tabular}",
                "",
            ]
        ),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--bench", default=None)
    args = ap.parse_args()
    run_dir = ROOT / "results" / args.run
    model_dir = run_dir / args.model.replace(":", "-")

    def load(name: str) -> pd.DataFrame | None:
        path = model_dir / f"{name}.parquet"
        return pd.read_parquet(path) if path.exists() else None

    # Prefer expanded-450 cohort for calibration and interference if available
    exp_model_dir = ROOT / "results" / "expanded-450" / args.model.replace(":", "-")
    cal_file = exp_model_dir / "calibration.parquet" if exp_model_dir.exists() else None
    cal = pd.read_parquet(cal_file) if (cal_file and cal_file.exists()) else load("calibration")

    inter_file = exp_model_dir / "interference.parquet" if exp_model_dir.exists() else None
    inter = pd.read_parquet(inter_file) if (inter_file and inter_file.exists()) else load("interference")

    if cal is not None:
        calibration_clean(cal)
        calibration_perturbed(cal)
    if inter is not None:
        interference(inter)
    for suffix in ("", "_secondary"):
        if (faith := load(f"faithfulness{suffix}")) is not None:
            faithfulness(faith, f"faithfulness{suffix}")
        if (plaus := load(f"plausibility{suffix}")) is not None:
            plausibility(plaus, f"plausibility{suffix}", load(f"attributions{suffix}"))
    if (conv := load("convergence")) is not None:
        convergence(conv)
    scaling_summary(run_dir)
    planted_benchmark(run_dir)
    if args.bench:
        bench_dir = ROOT / "results" / args.bench
        bench_frames = [pd.read_parquet(p) for p in sorted(bench_dir.glob("*/benchmark.parquet"))]
        if bench_frames:
            latency(pd.concat(bench_frames, ignore_index=True))

    fig_out = PAPER / "figures"
    fig_out.mkdir(exist_ok=True)
    for png in (run_dir / "figures").glob("*.png"):
        # graphicx misreads extra dots in file names (e.g. "tev1-0.8b").
        shutil.copy2(png, fig_out / (png.stem.replace(".", "-") + png.suffix))
    print(f"tables -> {PAPER / 'tables'}, figures -> {fig_out}")


if __name__ == "__main__":
    main()
