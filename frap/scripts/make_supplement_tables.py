"""Build all 5 supplement tables (S-FRAP1..S-FRAP5) for the combined paper.

Outputs both:
  - results/supplement_tables/<name>.csv   (machine-readable)
  - results/supplement_tables/<name>.md    (paste-ready markdown)
  - results/supplement_tables/ALL_TABLES.md (one rollup for review)

Per user spec:
  S-FRAP1  Dataset summary (rows = synthetic_clean, _low, _med, _high, real 32ww, real 56ww)
  S-FRAP2  Full Phase 9 run table (66 rows, all per-run fields)
  S-FRAP3  Synthetic aggregate metrics (4 noise rows + pooled noisy)
  S-FRAP4  Experimental aggregate metrics (2 conditions + mean)
  S-FRAP5  Lambda tuning table (10k tunes for both methods)
"""
from __future__ import annotations

import csv
import glob
import json
import os
from pathlib import Path

import numpy as np


OUT = Path("results/supplement_tables")
OUT.mkdir(parents=True, exist_ok=True)

D_NORM_TRUE = 0.01975

# Per-run JSONs were moved to an off-repo backup; this fallback keeps the
# table builder reproducible without copying them back into results/.
PHASE9_BACKUP = Path(os.environ.get("WEAKPINN_PHASE9_BACKUP", ""))  # set this env var to enable fallback


def _phase9_glob(pat):
    """Find per-run JSONs in results/ or the off-repo backup. Backup wins
    if it has matches, so re-running after the cleanup still works."""
    import glob as _glob
    backup_hits = sorted(PHASE9_BACKUP.glob(pat)) if PHASE9_BACKUP.exists() else []
    if backup_hits:
        return [str(p) for p in backup_hits]
    return sorted(_glob.glob(f"results/{pat}"))


def write_csv_md(name: str, rows: list[dict], header: list[str], md_caption: str):
    """Write the table as a CSV only. MD output disabled (user wants no .md files in repo).
    The md_caption argument is retained in function signature but its content is preserved
    only as a comment in the CSV header line for reviewer reference."""
    csv_path = OUT / f"{name}.csv"
    with csv_path.open("w", newline="") as f:
        # Caption as a comment-style first line. Some CSV readers will treat it as data;
        # most spreadsheet apps ignore it if it starts with '#'.
        f.write(f"# {name}: {md_caption.strip()}\n")
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})
    print(f"wrote {csv_path.name} ({len(rows)} rows)")


def fmt(x, digits=4):
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def fmt_sci(x, digits=3):
    if isinstance(x, float):
        return f"{x:.{digits}e}"
    return str(x)


# -----------------------------------------------------------------------------
# S-FRAP1  Dataset summary
# -----------------------------------------------------------------------------

def build_sfrap1():
    sel = json.load(open("results/selected_stacks.json"))
    pixel_um = 0.7598
    dt_real = 0.265
    rows = [
        {"dataset": "synthetic_clean", "source": "in-house bounded-domain FD simulator",
         "stack count": 1, "shape (T, H, W)": "80 x 128 x 128",
         "pixel size (norm units)": "dx ≈ 2/127",
         "frame time (norm units)": "dt = 0.01 (sim)",
         "recovery duration (norm units)": "0.79 (sim)",
         "selected stack(s)": "synthetic_clean.npz",
         "purpose": "lambda tuning + clean sanity check"},
        {"dataset": "synthetic_noise_low", "source": "FD sim + PSF blur + Poisson + imaging bleach",
         "stack count": 1, "shape (T, H, W)": "80 x 128 x 128",
         "pixel size (norm units)": "dx ≈ 2/127",
         "frame time (norm units)": "dt = 0.01 (sim)",
         "recovery duration (norm units)": "0.79 (sim)",
         "selected stack(s)": "synthetic_noise_low.npz",
         "purpose": "noisy synthetic D-recovery (photons = 10000)"},
        {"dataset": "synthetic_noise_med", "source": "FD sim + PSF blur + Poisson + imaging bleach",
         "stack count": 1, "shape (T, H, W)": "80 x 128 x 128",
         "pixel size (norm units)": "dx ≈ 2/127",
         "frame time (norm units)": "dt = 0.01 (sim)",
         "recovery duration (norm units)": "0.79 (sim)",
         "selected stack(s)": "synthetic_noise_med.npz",
         "purpose": "noisy synthetic D-recovery (photons = 1000)"},
        {"dataset": "synthetic_noise_high", "source": "FD sim + PSF blur + Poisson + imaging bleach",
         "stack count": 1, "shape (T, H, W)": "80 x 128 x 128",
         "pixel size (norm units)": "dx ≈ 2/127",
         "frame time (norm units)": "dt = 0.01 (sim)",
         "recovery duration (norm units)": "0.79 (sim)",
         "selected stack(s)": "synthetic_noise_high.npz",
         "purpose": "noisy synthetic D-recovery (photons = 100)"},
        {"dataset": "real_32ww", "source": "DeepFRAP (Roding et al. 2020, Zenodo 3874218)",
         "stack count": "20 candidates, top-5 selected (quality-rank #1-#5)",
         "shape (T, H, W)": "100 x 256 x 256 (postbleach)",
         "pixel size (norm units)": f"{pixel_um:.4f} um",
         "frame time (norm units)": f"{dt_real} s",
         "recovery duration (norm units)": f"{(100 - 1) * dt_real:.2f} s",
         "selected stack(s)": "frap_32ww_{010,006,004,005,014}.mat",
         "purpose": "experimental FRAP, fast-diffusion regime"},
        {"dataset": "real_56ww", "source": "DeepFRAP (Roding et al. 2020, Zenodo 3874218)",
         "stack count": "20 candidates, top-5 selected (quality-rank #1-#5)",
         "shape (T, H, W)": "100 x 256 x 256 (postbleach)",
         "pixel size (norm units)": f"{pixel_um:.4f} um",
         "frame time (norm units)": f"{dt_real} s",
         "recovery duration (norm units)": f"{(100 - 1) * dt_real:.2f} s",
         "selected stack(s)": "frap_56ww_{005,004,018,011,014}.mat",
         "purpose": "experimental FRAP, slow-diffusion regime"},
    ]
    header = ["dataset", "source", "stack count", "shape (T, H, W)",
              "pixel size (norm units)", "frame time (norm units)",
              "recovery duration (norm units)", "selected stack(s)", "purpose"]
    write_csv_md("S-FRAP1_dataset_summary", rows, header,
                 "**Table S-FRAP1.** Summary of the 4 synthetic stacks and the 10 experimental FRAP stacks (top-5 quality-ranked per molecular-weight condition) used in the cross-domain validation. Synthetic stacks are generated by an explicit finite-difference solver in normalized coordinates; experimental stacks are taken from the public DeepFRAP dataset (Röding et al., 2020) and chosen by a five-metric quality screen across 40 candidates.")


# -----------------------------------------------------------------------------
# S-FRAP2  Full Phase 9 run table
# -----------------------------------------------------------------------------

def stack_label(path: str) -> str:
    """Derive a human-readable stack/condition label from the result JSON filename."""
    p = Path(path).name
    if p.startswith("clean_"): return "clean"
    if p.startswith("noiselow_"): return "noise_low"
    if p.startswith("noisemed_"): return "noise_med"
    if p.startswith("noisehigh_"): return "noise_high"
    if p.startswith("real_32ww_"): return "real_32ww"
    if p.startswith("real_56ww_"): return "real_56ww"
    return "?"


def build_sfrap2():
    blocks = ["clean_*.json", "noiselow_*.json", "noisemed_*.json", "noisehigh_*.json",
              "real_32ww_*.json", "real_56ww_*.json"]
    rows = []
    for pat in blocks:
        for f in sorted(_phase9_glob(f"{pat}")):
            j = json.load(open(f))
            D_norm_true = j.get("D_norm_true")
            D = j["D_recovered"]
            if D_norm_true and D_norm_true > 0:
                abs_err = abs(D - D_norm_true)
                pct_err = 100.0 * abs_err / D_norm_true
            else:
                abs_err = pct_err = ""
            rows.append({
                "method": j["method"],
                "seed": j["seed"],
                "condition": stack_label(f),
                "lambda_phys": j["lambda_phys"],
                "init_D": j["init_D"],
                "D_recovered": fmt(D, 4),
                "D abs err": fmt(abs_err, 5) if abs_err != "" else "",
                "D % err": fmt(pct_err, 2) if pct_err != "" else "",
                "val_mse": fmt_sci(j["val_mse"], 3),
                "median_strong_resid": fmt_sci(j["median_strong_residual"], 3),
                "median_weak_resid": fmt_sci(j["median_weak_residual"], 3),
                "k": fmt(j.get("k_recovered", 0), 4),
                "notes": "" if pct_err != "" else "no ground-truth D (real stack)",
            })
    header = ["method", "seed", "condition", "lambda_phys", "init_D", "D_recovered",
              "D abs err", "D % err", "val_mse", "median_strong_resid",
              "median_weak_resid", "k", "notes"]
    write_csv_md("full_run_table", rows, header,
                 "**Per-run reproducibility CSV (not in supplement PDF; available in the code repository at `results/supplement_tables/full_run_table.csv`).** All 66 Phase 9 training runs. Synthetic blocks (clean + 3 noise levels) use 3 seeds per (condition, method) cell; experimental blocks use 5 seeds. Ground-truth D in normalized PINN coordinates is 0.01975 for synthetic; experimental D has no ground truth (LS-fit anchor is reported in Table S-FRAP3 only as a literature reference).")


# -----------------------------------------------------------------------------
# S-FRAP3  Synthetic aggregate metrics
# -----------------------------------------------------------------------------

def aggregate_block(pat):
    """Return per-method (mae, std, mse_mean) over seeds for a given block pattern."""
    out = {}
    for method in ("strong", "weak", "data"):
        files = [f for f in sorted(_phase9_glob(f"{pat}")) if f"_{method}_" in f]
        Ds = [json.load(open(f))["D_recovered"] for f in files]
        mses = [json.load(open(f))["val_mse"] for f in files]
        out[method] = {
            "D": np.array(Ds), "mse": np.array(mses),
            "mae": float(np.mean(np.abs(np.array(Ds) - D_NORM_TRUE))),
            "pct": float(np.mean(np.abs(np.array(Ds) - D_NORM_TRUE) / D_NORM_TRUE * 100)),
            "std": float(np.std(Ds)),
            "mse_mean": float(np.mean(mses)),
        }
    return out


def build_sfrap3():
    blocks = [("clean", "clean_*.json"),
              ("noiselow", "noiselow_*.json"),
              ("noisemed", "noisemed_*.json"),
              ("noisehigh", "noisehigh_*.json")]
    rows = []
    for label, pat in blocks:
        agg = aggregate_block(pat)
        reduction = 100 * (1 - agg["weak"]["mae"] / agg["strong"]["mae"])
        rows.append({
            "block": label,
            "strong D-MAE": fmt_sci(agg["strong"]["mae"], 3),
            "weak D-MAE": fmt_sci(agg["weak"]["mae"], 3),
            "strong % err": fmt(agg["strong"]["pct"], 2),
            "weak % err": fmt(agg["weak"]["pct"], 2),
            "relative reduction (%)": fmt(reduction, 1),
            "strong cross-seed std": fmt(agg["strong"]["std"], 5),
            "weak cross-seed std": fmt(agg["weak"]["std"], 5),
            "strong val_mse (mean)": fmt_sci(agg["strong"]["mse_mean"], 3),
            "weak val_mse (mean)": fmt_sci(agg["weak"]["mse_mean"], 3),
        })
    # Pooled noisy
    weak_abs, strong_abs = [], []
    for noise_pat in ["noiselow_*.json", "noisemed_*.json", "noisehigh_*.json"]:
        for f in sorted(_phase9_glob(f"{noise_pat}")):
            j = json.load(open(f))
            if j["method"] == "weak":
                weak_abs.append(abs(j["D_recovered"] - D_NORM_TRUE))
            elif j["method"] == "strong":
                strong_abs.append(abs(j["D_recovered"] - D_NORM_TRUE))
    mae_w = float(np.mean(weak_abs))
    mae_s = float(np.mean(strong_abs))
    pct_w = mae_w / D_NORM_TRUE * 100
    pct_s = mae_s / D_NORM_TRUE * 100
    red = 100 * (1 - mae_w / mae_s)
    rows.append({
        "block": "POOLED NOISY (low + med + high; 9 seeds per method)",
        "strong D-MAE": fmt_sci(mae_s, 3),
        "weak D-MAE": fmt_sci(mae_w, 3),
        "strong % err": fmt(pct_s, 2),
        "weak % err": fmt(pct_w, 2),
        "relative reduction (%)": fmt(red, 1),
        "strong cross-seed std": "(see per-block rows)",
        "weak cross-seed std": "(see per-block rows)",
        "strong val_mse (mean)": "(see per-block rows)",
        "weak val_mse (mean)": "(see per-block rows)",
    })
    header = ["block", "strong D-MAE", "weak D-MAE",
              "strong % err", "weak % err", "relative reduction (%)",
              "strong cross-seed std", "weak cross-seed std",
              "strong val_mse (mean)", "weak val_mse (mean)"]
    write_csv_md("S-FRAP2_synthetic_aggregate", rows, header,
                 "**Table S-FRAP2.** Synthetic aggregate metrics across the 4 noise blocks and the pooled noisy summary. D-MAE is the mean absolute error between PINN-recovered $D$ and the simulator-defined ground-truth $D_{\\mathrm{norm,true}}=0.01975$. The headline 82.1% reduction applies to the pooled noisy stacks only; the clean block is reported as a stability/sanity check, not a generalization claim.")


# -----------------------------------------------------------------------------
# S-FRAP4  Experimental aggregate metrics
# -----------------------------------------------------------------------------

def build_sfrap4():
    """n=10 expansion: 5 stacks per condition, per-stack rows + condition aggregates."""
    import re as _re
    # Top-5 quality-ranked stacks per condition (1 = best)
    TOP5 = {"32ww": ["010","006","004","005","014"],
            "56ww": ["005","004","018","011","014"]}
    LEGACY = {"32ww": "010", "56ww": "005"}

    # Per-stack LS anchors
    ls_per = {}
    factor = 1.400868e-3
    with open("results/ls_reference_D.csv") as f:
        for r in csv.DictReader(f):
            key = f"{r['condition']}_{int(r['dataset_index']):03d}"
            phys = float(r["D_m2_per_s"]) * 1e12
            ls_per[key] = (phys, phys * factor)

    def _parse(p):
        name = Path(p).name
        m = _re.match(r"real_(32ww|56ww)_(\d{3})_(strong|weak)_seed\d+\.json", name)
        if m: return m.group(1), m.group(2)
        m = _re.match(r"real_(32ww|56ww)_(strong|weak)_seed\d+\.json", name)
        if m:  return m.group(1), LEGACY[m.group(1)]
        return None, None

    def _per_stack(cond, method):
        out = {}
        for f in _phase9_glob(f"real_{cond}_*.json"):
            if f"_{method}_" not in f: continue
            c, idx = _parse(f)
            if c != cond: continue
            out.setdefault(idx, []).append(json.load(open(f))["D_recovered"])
        return out

    rows = []
    cond_aggregates = {}
    for cond in ("32ww", "56ww"):
        s_by = _per_stack(cond, "strong")
        w_by = _per_stack(cond, "weak")
        cond_s_stds, cond_w_stds, cond_reds = [], [], []
        for rank, idx in enumerate(TOP5[cond], 1):
            s_vals = s_by.get(idx, [])
            w_vals = w_by.get(idx, [])
            if not s_vals or not w_vals: continue
            m_s, s_s = float(np.mean(s_vals)), float(np.std(s_vals))
            m_w, s_w = float(np.mean(w_vals)), float(np.std(w_vals))
            red = (1 - s_w / s_s) * 100 if s_s > 0 else 0.0
            ls_phys, ls_norm = ls_per[f"{cond}_{idx}"]
            rows.append({
                "row": f"{cond}_{idx} (rank #{rank})",
                "n_seeds": len(s_vals),
                "LS anchor D_phys (um^2/s)": fmt(ls_phys, 3),
                "LS anchor D_norm": fmt(ls_norm, 4),
                "strong mean D_norm": fmt(m_s, 4),
                "weak mean D_norm": fmt(m_w, 4),
                "strong std": fmt(s_s, 5),
                "weak std": fmt(s_w, 5),
                "std reduction (%)": fmt(red, 1),
                "strong % dev vs LS": fmt(abs(m_s - ls_norm) / ls_norm * 100, 1),
                "weak % dev vs LS": fmt(abs(m_w - ls_norm) / ls_norm * 100, 1),
            })
            cond_s_stds.append(s_s); cond_w_stds.append(s_w); cond_reds.append(red)
        # Per-condition aggregate row
        if cond_s_stds:
            ms, mw = float(np.mean(cond_s_stds)), float(np.mean(cond_w_stds))
            agg_red = (1 - mw / ms) * 100 if ms > 0 else 0.0
            rows.append({
                "row": f"{cond} aggregate (mean over 5 stacks)",
                "n_seeds": "5 stacks × 5 seeds",
                "LS anchor D_phys (um^2/s)": "-",
                "LS anchor D_norm": "-",
                "strong mean D_norm": "-",
                "weak mean D_norm": "-",
                "strong std": fmt(ms, 5),
                "weak std": fmt(mw, 5),
                "std reduction (%)": fmt(agg_red, 1)
                                      + f" (mean stack-level reduction: {np.mean(cond_reds):.1f}%)",
                "strong % dev vs LS": "-",
                "weak % dev vs LS": "-",
            })
            cond_aggregates[cond] = (ms, mw, agg_red, np.mean(cond_reds))

    # Grand mean across both conditions
    if cond_aggregates:
        all_ms = np.mean([v[0] for v in cond_aggregates.values()])
        all_mw = np.mean([v[1] for v in cond_aggregates.values()])
        grand_red = (1 - all_mw / all_ms) * 100 if all_ms > 0 else 0.0
        all_stack_reds = []
        for cond in ("32ww", "56ww"):
            s_by = _per_stack(cond, "strong"); w_by = _per_stack(cond, "weak")
            for idx in TOP5[cond]:
                if idx in s_by and idx in w_by and len(s_by[idx]) and len(w_by[idx]):
                    ss = float(np.std(s_by[idx])); ws = float(np.std(w_by[idx]))
                    if ss > 0:
                        all_stack_reds.append((1 - ws / ss) * 100)
        rows.append({
            "row": "GRAND MEAN across 10 stacks",
            "n_seeds": "10 stacks × 5 seeds = 50",
            "LS anchor D_phys (um^2/s)": "-",
            "LS anchor D_norm": "-",
            "strong mean D_norm": "-",
            "weak mean D_norm": "-",
            "strong std": fmt(float(all_ms), 5),
            "weak std": fmt(float(all_mw), 5),
            "std reduction (%)": fmt(grand_red, 1)
                                  + f" (mean stack-level: {np.mean(all_stack_reds):.1f}%; median: {np.median(all_stack_reds):.1f}%)",
            "strong % dev vs LS": "-",
            "weak % dev vs LS": "-",
        })
    header = ["row", "n_seeds", "LS anchor D_phys (um^2/s)", "LS anchor D_norm",
              "strong mean D_norm", "weak mean D_norm",
              "strong std", "weak std", "std reduction (%)",
              "strong % dev vs LS", "weak % dev vs LS"]
    write_csv_md("S-FRAP3_experimental_aggregate", rows, header,
                 "**Table S-FRAP3.** Experimental aggregate metrics across the two molecular-weight conditions, each represented by the top-5 quality-ranked stacks from the DeepFRAP dataset (5 stacks × 5 seeds = 25 runs per method per condition; 100 runs total across the experimental matrix, plus a 10-run data-only ablation included in the supplement run table). Per-stack rows give the recovered $D_{\\mathrm{norm}}$ and cross-seed std for each method, with the LS-fit anchor as a literature reference (not a training target). Condition-aggregate rows give the mean cross-seed std over the 5 stacks. The GRAND MEAN row reports the headline: weak-form is more stable than strong-form on 10/10 stacks; mean stack-level std reduction is 49.0%, median 60.7%.")


# -----------------------------------------------------------------------------
# S-FRAP5  Lambda tuning table
# -----------------------------------------------------------------------------

def build_sfrap5():
    """Read the existing lambda_tuning_summary.csv and reformat."""
    rows = []
    src = "results/lambda_tuning_summary.csv"
    with open(src) as f:
        reader = csv.DictReader(f)
        for r in reader:
            D = float(r["D_recovered"]) if r["D_recovered"] else None
            pct = (100 * abs(D - D_NORM_TRUE) / D_NORM_TRUE) if D else None
            sel = r.get("selected", "")
            sel_flag = "yes" if sel else "no"
            rows.append({
                "method": r["method"],
                "lambda": fmt(float(r["lambda_phys"]), 4),
                "val_mse (clean synthetic dev stack)": fmt_sci(float(r["val_mse"]), 3),
                "D recovered": fmt(D, 4),
                "D % err vs D_norm_true=0.01975": fmt(pct, 2) if pct else "-",
                "stable / NaN": "NaN" if r["failed"].lower() == "true" else "stable",
                "selected": sel_flag,
                "selection note": sel if sel else "",
            })
    # Stable sort: strong first then weak, then by lambda asc
    rows.sort(key=lambda r: (r["method"], float(r["lambda"])))
    header = ["method", "lambda", "val_mse (clean synthetic dev stack)",
              "D recovered", "D % err vs D_norm_true=0.01975",
              "stable / NaN", "selected", "selection note"]
    write_csv_md("S-FRAP4_lambda_tuning", rows, header,
                 "**Table S-FRAP4.** Lambda tuning sweep results on the clean synthetic development stack at the matched 10000-step training length. The pilot-tuning rule prioritized D recovery among stable runs with low validation MSE; validation MSE was used as a reconstruction-quality constraint rather than the primary objective. Noisy synthetic and experimental stacks were not used during $\\lambda$ selection.")


# -----------------------------------------------------------------------------
# S-FRAP6  Residual diagnostics (was a figure; now a table)
# -----------------------------------------------------------------------------

def build_sfrap6():
    """Median residual values per (condition, training method, residual metric).

    For each result JSON we recorded median_strong_residual and
    median_weak_residual evaluated on that model after training. So for each
    (block, training_method) cell we aggregate the median across seeds of
    each residual metric, yielding 4 numbers per block:
        median |r_strong| evaluated on strong-trained model
        median |r_strong| evaluated on weak-trained model
        median |r_weak|   evaluated on strong-trained model
        median |r_weak|   evaluated on weak-trained model
    """
    import numpy as np
    blocks = [("clean", "clean_*.json"),
              ("noiselow", "noiselow_*.json"),
              ("noisemed", "noisemed_*.json"),
              ("noisehigh", "noisehigh_*.json"),
              ("real_32ww", "real_32ww_*.json"),
              ("real_56ww", "real_56ww_*.json")]
    rows = []
    for label, pat in blocks:
        cell = {}
        for trained_method in ("strong", "weak"):
            files = [f for f in sorted(_phase9_glob(f"{pat}"))
                     if f"_{trained_method}_" in f]
            sresid = [json.load(open(f))["median_strong_residual"] for f in files]
            wresid = [json.load(open(f))["median_weak_residual"] for f in files]
            cell[(trained_method, "strong_metric")] = float(np.median(sresid))
            cell[(trained_method, "weak_metric")]   = float(np.median(wresid))
        rows.append({
            "block": label,
            "median |r_strong| on strong-trained": fmt_sci(cell[("strong", "strong_metric")], 3),
            "median |r_strong| on weak-trained":   fmt_sci(cell[("weak",   "strong_metric")], 3),
            "median |r_weak| on strong-trained":   fmt_sci(cell[("strong", "weak_metric")], 3),
            "median |r_weak| on weak-trained":     fmt_sci(cell[("weak",   "weak_metric")], 3),
        })
    header = ["block",
              "median |r_strong| on strong-trained",
              "median |r_strong| on weak-trained",
              "median |r_weak| on strong-trained",
              "median |r_weak| on weak-trained"]
    write_csv_md("S-FRAP5_residual_diagnostics", rows, header,
                 "**Table S-FRAP5.** Median absolute residuals from each PINN model "
                 "under each residual metric (fair cross-evaluation: both strong-form and "
                 "weak-form residuals computed on both trained models). Per (block, "
                 "training_method) cell the median is over the run-level medians from each "
                 "seed. Strong-form residuals are evaluated pointwise; weak-form residuals "
                 "are integrated against boundary-vanishing Gaussian test functions and "
                 "therefore live on a smaller absolute scale.")


# -----------------------------------------------------------------------------
# Roll everything up for review
# -----------------------------------------------------------------------------

def rollup():
    """Markdown rollup disabled (user wants no .md files in repo).
    Per-table CSVs remain in results/supplement_tables/."""
    return


if __name__ == "__main__":
    build_sfrap1()
    build_sfrap2()
    build_sfrap3()
    build_sfrap4()
    build_sfrap5()
    build_sfrap6()
    rollup()
