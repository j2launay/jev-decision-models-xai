"""Generate LaTeX tables for the review response:
1. Interference controls (ordering, dummy question, option shuffling)
2. Post-hoc calibration (Raw vs Temperature vs Platt scaling, ECE-sweep, E-AURC)
3. Logic gate recovery via Shapley Interaction Index (Option B)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
TABLES = PAPER / "tables"
TABLES.mkdir(exist_ok=True)

MODEL_LABEL = {
    "tev1:0.8b": r"\texttt{Tev1-0.8B}",
    "tev1:latest": r"\texttt{Tev1-4B}",
    "nimble:latest": r"\texttt{Nimble-9B}",
    "clef-flash:latest": r"\texttt{Clef-Flash-9B}",
    "clef:27b": r"\texttt{Clef-27B}",
    "gemma4-json": r"\texttt{Gemma4-8B}",
}


def f(x: float, d: int = 3) -> str:
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def make_interference_controls_table() -> None:
    ctrl_dir = ROOT / "results" / "interference-controls"
    paths = sorted(ctrl_dir.glob("*/interference_controls.parquet"))
    if not paths:
        return
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    models = [m for m in ("tev1:0.8b", "tev1:latest", "nimble:latest", "clef-flash:latest") if m in df["model"].unique()]
    
    rows = []
    for m in models:
        sub_m = df[df["model"] == m]
        for h in ("label", "noul"):
            head_lbl = "Choice" if h == "label" else "Noul"
            sub_h = sub_m[sub_m["head"] == h]
            
            def get_stat(var: str) -> tuple[float, float]:
                s = sub_h[sub_h["variant"] == var]
                if s.empty:
                    return np.nan, np.nan
                return float(s["flip"].mean()), float(s["total_variation"].mean())
            
            f_cn, tv_cn = get_stat("joint_cn")
            f_nc, tv_nc = get_stat("joint_nc")
            f_sh, _ = get_stat("shuffled_opts")
            f_dum, tv_dum = get_stat("dummy")
            
            m_label = MODEL_LABEL.get(m, m) if h == "label" else ""
            sh_str = f"{f_sh:.3f}" if h == "label" else "--"
            
            rows.append(
                f"{m_label} & {head_lbl} & {f_cn:.3f} / {tv_cn:.3f} & {f_nc:.3f} / {tv_nc:.3f} & "
                f"{sh_str} & {f_dum:.3f} / {tv_dum:.3f} \\\\"
            )
        rows.append(r"\midrule")
    if rows:
        rows.pop()
        
    latex = "\n".join([
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{Question Order Control} & \multicolumn{1}{c}{Option Permutation} & \multicolumn{1}{c}{Prompt Length Control} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-5}\cmidrule(lr){6-6}",
        r"Model & Head & Joint ($C \to N$) Flip/TV & Joint ($N \to C$) Flip/TV & Shuffled Opts Flip & Neutral Dummy Flip/TV \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        ""
    ])
    (TABLES / "interference_controls.tex").write_text(latex, encoding="utf-8")
    print("Wrote tables/interference_controls.tex")


def make_posthoc_calibration_table() -> None:
    p = ROOT / "results" / "posthoc" / "posthoc_calibration.parquet"
    if not p.exists():
        return
    df = pd.read_parquet(p)
    df_c = df[df["head"] == "label"].copy()
    
    source_lbl = {"esnli": "e-SNLI", "hans": "HANS", "triage_synth": "Triage"}
    models = [m for m in ("tev1:0.8b", "tev1:latest", "nimble:latest", "clef-flash:latest", "clef:27b", "gemma4-json") if m in df_c["model"].unique()]
    
    rows = []
    for m in models:
        sub_m = df_c[df_c["model"] == m]
        for s_idx, src in enumerate(("esnli", "hans", "triage_synth")):
            sub_s = sub_m[sub_m["source"] == src]
            if sub_s.empty:
                continue
            r_raw = sub_s[sub_s["method"] == "raw"].iloc[0]
            r_temp = sub_s[sub_s["method"] == "temperature"].iloc[0]
            r_platt = sub_s[sub_s["method"] == "platt"].iloc[0]
            
            m_label = MODEL_LABEL.get(m, m) if s_idx == 0 else ""
            rows.append(
                f"{m_label} & {source_lbl[src]} & "
                f"{r_raw.accuracy:.3f} & "
                f"{r_raw.ece_w:.3f} & {r_temp.ece_w:.3f} & {r_platt.ece_w:.3f} & "
                f"{r_raw.brier:.3f} & {r_temp.brier:.3f} & "
                f"{r_raw.eaurc:.3f} & {f(r_raw.eaurc_provider, 3)} \\\\"
            )
        rows.append(r"\midrule")
    if rows:
        rows.pop()
        
    latex = "\n".join([
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        r" & & & \multicolumn{3}{c}{ECE$_w$ ($\downarrow$)} & \multicolumn{2}{c}{Brier ($\downarrow$)} & \multicolumn{2}{c}{E-AURC ($\downarrow$)} \\",
        r"\cmidrule(lr){4-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
        r"Model & Dataset & Acc. & Raw & Temp. & Platt & Raw & Temp. & $\max p$ & Provider Conf. \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        ""
    ])
    (TABLES / "posthoc_calibration.tex").write_text(latex, encoding="utf-8")
    print("Wrote tables/posthoc_calibration.tex")


def make_interactions_table() -> None:
    smoke_p = ROOT / "results" / "planted-v2-smoke" / "tev1-0.8b" / "interactions.parquet"
    if not smoke_p.exists():
        return
    df = pd.read_parquet(smoke_p)
    df = df[df["n_gt_pairs"] > 0].copy()
    
    agg = df.groupby(["regime", "gate", "space", "estimator"], as_index=False)[
        ["pair_prec", "sign_acc", "gate_recovered", "trigger_coverage"]
    ].mean()
    
    reg_label = {
        "or2": r"Dual OR ($g{=}2$)",
        "or3": r"Multi OR ($g{=}3$)",
        "and2": r"Dual AND ($g{=}2$)",
    }
    est_label = {
        "sii": "SII (Shapley Interaction)",
        "archdetect": "ArchDetect-style",
        "loo_pair": "LOO-pair (2nd Diff.)",
    }
    
    rows = []
    for reg in ("or2", "or3", "and2"):
        sub_r = agg[agg["regime"] == reg]
        if sub_r.empty:
            continue
        gate_type = "OR ($-1$)" if "or" in reg else "AND ($+1$)"
        for s_idx, sp in enumerate(("prob", "logit")):
            sub_s = sub_r[sub_r["space"] == sp]
            sp_name = "Prob. space" if sp == "prob" else "Log-odds (logit)"
            for e_idx, est in enumerate(("sii", "archdetect", "loo_pair")):
                sub_e = sub_s[sub_s["estimator"] == est]
                if sub_e.empty:
                    continue
                r = sub_e.iloc[0]
                lead_reg = reg_label[reg] if (s_idx == 0 and e_idx == 0) else ""
                lead_gate = gate_type if (s_idx == 0 and e_idx == 0) else ""
                lead_sp = sp_name if e_idx == 0 else ""
                rows.append(
                    f"{lead_reg} & {lead_gate} & {lead_sp} & {est_label[est]} & "
                    f"{r.pair_prec:.2f} & {r.sign_acc:.2f} & {r.gate_recovered:.2f} \\\\"
                )
            if s_idx == 0:
                rows.append(r"\cmidrule(lr){3-7}")
        rows.append(r"\midrule")
    if rows:
        rows.pop()
        
    latex = "\n".join([
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"Regime & True Gate & Representation & Interaction Estimator & Pair Prec.@$g\,\uparrow$ & Sign Acc.\,$\uparrow$ & Gate Recovered\,$\uparrow$ \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        ""
    ])
    (TABLES / "shapley_interactions.tex").write_text(latex, encoding="utf-8")
    print("Wrote tables/shapley_interactions.tex")


if __name__ == "__main__":
    make_interference_controls_table()
    make_posthoc_calibration_table()
    make_interactions_table()

