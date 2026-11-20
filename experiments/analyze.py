"""Statistics, LaTeX tables and figures from results/*.jsonl."""
import os, sys, json
import numpy as np
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")
TAB = os.path.join(HERE, "..", "paper", "tables")
FIG = os.path.join(HERE, "..", "paper", "figs")
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
RNG = np.random.default_rng(0)
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6,
                     "figure.dpi": 200, "savefig.bbox": "tight"})


def load(name):
    p = os.path.join(RES, name)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def cluster_boot(vals, groups, B=2000):
    """mean and 95% CI, bootstrap resampling seeds (clusters) with replacement."""
    vals, groups = np.asarray(vals, float), np.asarray(groups)
    ok = np.isfinite(vals)
    vals, groups = vals[ok], groups[ok]
    ug = np.unique(groups)
    by = [vals[groups == g] for g in ug]
    means = []
    for _ in range(B):
        pick = RNG.integers(len(ug), size=len(ug))
        means.append(np.mean(np.concatenate([by[k] for k in pick])))
    return float(np.mean(vals)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def seed_signflip(diff, seeds):
    """Exact two-sided sign-flip permutation test on per-seed mean differences
    (scenes of one seed share a calibration, so the seed is the unit)."""
    d = np.array([np.mean(diff[seeds == g]) for g in np.unique(seeds)])
    k = len(d)
    obs = abs(d.mean())
    signs = np.array([[1 if (i >> b) & 1 else -1 for b in range(k)] for i in range(2 ** k)])
    stats = np.abs((signs * d).mean(1))
    return float(np.mean(stats >= obs - 1e-15))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0
    for r, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - r) * p[i]))
        adj[i] = running
    return adj


def fmt(m, lo, hi, d=3):
    return f"{m:.{d}f}\\,{{\\scriptsize[{lo:.{d}f},{hi:.{d}f}]}}"


def get(r, meth, key):
    v = r["methods"][meth].get(key, np.nan)
    return np.nan if v is None else v


# ------------------------------------------------------------------ main synthetic
def synth_tables(rows, prefix="synth", conds=None, methods=None, names=None):
    conds = conds or list(dict.fromkeys(r["cond"] for r in rows))
    methods = methods or ["L1", "CG", "ConfW", "ConfGate", "CycFilt", "CG-rho", "CARA-static", "CARA", "CARA-safe"]
    out = {}
    lines = []
    pv, pv_keys = [], []
    for c in conds:
        R = [r for r in rows if r["cond"] == c]
        seeds = [r["seed"] for r in R]
        out[c] = {}
        for mth in methods:
            if mth not in R[0]["methods"]:
                continue
            d = {}
            for key in ["auc2", "auc5", "auc10", "median", "p95", "cat", "reg"]:
                d[key] = cluster_boot([get(r, mth, key) for r in R], seeds)
            out[c][mth] = d
        for base in [b for b in ["CG", "CG-rho", "GLOMAP-RA", "Shonan-H"] if b in R[0]["methods"]]:
            a = np.array([get(r, "CARA", "auc5") for r in R]); b = np.array([get(r, base, "auc5") for r in R])
            diff = a - b
            p = seed_signflip(diff, np.array(seeds))
            pv.append(p); pv_keys.append((c, base, float(np.mean(diff)), float(np.mean(diff > 0)),
                                          int(sum(np.mean(diff[np.array(seeds) == g]) > 0 for g in np.unique(seeds))), len(np.unique(seeds))))
    adj = holm(pv)
    tests = [dict(cond=k[0], base=k[1], mean_diff_auc5=k[2], frac_better=k[3], seeds_better=k[4], n_seeds=k[5],
                  p=float(p), p_holm=float(a)) for k, p, a in zip(pv_keys, pv, adj)]
    json.dump(dict(summary=out, tests=tests), open(os.path.join(RES, f"{prefix}_summary.json"), "w"), indent=1)
    return out, tests


def latex_main(out, tests, path, conds, methods, labels, caption, label):
    L = [r"\begin{table*}[t]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}",
         r"\setlength{\tabcolsep}{3.2pt}", r"\footnotesize"]
    L += [r"\begin{tabular}{llcccccc}", r"\toprule",
          r"Condition & Method & AUC@2$^\circ\uparrow$ & AUC@5$^\circ\uparrow$ & Med.\ ($^\circ$)$\downarrow$ & P95 ($^\circ$)$\downarrow$ & Reg.\ (\%)$\uparrow$ & Cat.\ (\%)$\downarrow$\\", r"\midrule"]
    tmap = {(t["cond"], t["base"]): t for t in tests}
    for c in conds:
        best5 = max(out[c][m]["auc5"][0] for m in methods if m in out[c])
        first = True
        for m in methods:
            if m not in out[c]:
                continue
            d = out[c][m]
            a5 = fmt(*d["auc5"])
            if abs(d["auc5"][0] - best5) < 1e-12:
                a5 = r"\textbf{" + a5 + "}"
            star = ""
            if m == "CARA":
                bases = [b for b in ["CG", "GLOMAP-RA", "Shonan-H"] if (c, b) in tmap]
                sig = [tmap[(c, b)] for b in bases if tmap[(c, b)]["p_holm"] < 0.05]
                if len(sig) == len(bases) and all(t["mean_diff_auc5"] > 0 for t in sig):
                    star = r"$^{\dagger}$"
                elif any(t["mean_diff_auc5"] < 0 for t in sig):
                    star = r"$^{\ddagger}$"
            L.append(f"{labels.get(c, c) if first else ''} & {m}{star} & {fmt(*d['auc2'])} & {a5} & {d['median'][0]:.2f} & {d['p95'][0]:.1f} & {100*d['reg'][0]:.1f} & {100*d['cat'][0]:.1f}\\\\")
            first = False
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}", r"\end{table*}"]
    open(path, "w").write("\n".join(L) + "\n")


def bounds_table(rows, path, conds, labels, caption, label, real=False):
    L = [r"\begin{table}[t]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}",
         r"\setlength{\tabcolsep}{2.5pt}", r"\footnotesize", r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{llccc}", r"\toprule",
         r"Cond. & Bound & Coverage (\%) & Hard-25\% cov.\ (\%) & Width ($^\circ$)\\", r"\midrule"]
    res = {}
    for c in conds:
        R = [r for r in rows if r["cond"] == c]
        seeds = [r["seed"] for r in R]
        entries = []
        if not real:
            entries += [("Gauss (native $\\hat\\sigma$)", "CARA", "gauss_native"), ("Gauss (recal.)", "CARA", "gauss_recal")]
        entries += [("CRC, no drift", "CARA-static", "crc"), ("CRC + $\\hat\\rho_{\\mathrm{pt}}$", "CARA", "crc"),
                    ("CRC + $\\hat\\rho_{\\mathrm{safe}}$", "CARA-safe", "crc")]
        first = True
        res[c] = {}
        for name, m, k in entries:
            b = [r["methods"][m]["bounds"][k] for r in R]
            cov = cluster_boot([x["cov"] for x in b], seeds)
            hard = cluster_boot([x.get("cov_hard", np.nan) for x in b], seeds) if "cov_hard" in b[0] else (np.nan,) * 3
            wid = np.median([x["width"] for x in b])
            res[c][name] = dict(cov=cov, hard=hard, width=float(wid))
            hs = f"{100*hard[0]:.1f}" if np.isfinite(hard[0]) else "--"
            L.append(f"{labels.get(c, c) if first else ''} & {name} & {100*cov[0]:.1f}\\,{{\\scriptsize[{100*cov[1]:.1f},{100*cov[2]:.1f}]}} & {hs} & {wid:.2f}\\\\")
            first = False
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}}", r"\end{table}"]
    open(path, "w").write("\n".join(L) + "\n")
    return res


def auroc(neg, pos):
    neg, pos = np.asarray(neg), np.asarray(pos)
    return float(np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :]))


def drift_stats(rows, conds, clean="clean"):
    out = {}
    base = [r["methods"]["CARA"]["rho"] for r in rows if r["cond"] == clean]
    for c in conds:
        R = [r for r in rows if r["cond"] == c]
        pt = np.array([r["methods"]["CARA"]["rho"] for r in R])
        sf = np.array([r["methods"]["CARA-safe"]["rho"] for r in R])
        out[c] = dict(rho_true=R[0]["rho_true"], pt_mean=float(pt.mean()), pt_sd=float(pt.std()),
                      safe_mean=float(sf.mean()), safe_sd=float(sf.std()),
                      safe_ge_true=float(np.mean(sf >= R[0]["rho_true"] * 0.999)),
                      auroc_vs_clean=auroc(base, pt) if c != clean else np.nan)
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "synth"
    if which == "synth":
        rows = load("synth_main.jsonl")
        conds = ["clean", "fog2x", "heavy3x", "sparse", "corrflip", "large"]
        labels = {"clean": "Clean", "fog2x": "Fog ($2\\times$)", "heavy3x": "Heavy ($3\\times$, 35\\%)",
                  "sparse": "Sparse", "corrflip": "Corr.\\ flips", "large": "$n{=}300$"}
        methods = ["L1", "CG", "ConfW", "ConfGate", "CycFilt", "GLOMAP-RA", "Shonan-H", "CG-rho", "CARA-static", "CARA", "CARA-safe"]
        out, tests = synth_tables(rows, "synth", conds, methods)
        methods = [m for m in methods if m not in ("ConfGate", "L1")]   # ConfGate == CG (tuned rate 0); L1 < CG everywhere; both stated in text
        latex_main(out, tests, os.path.join(TAB, "synth_main.tex"), conds, methods, labels,
                   "Synthetic study, 10 seeds $\\times$ 20 test scenes per condition. Means with 95\\% seed-clustered bootstrap intervals. "
                   "Catastrophic: error $>30^\\circ$ or unregistered. $^\\dagger$: CARA better than CG, GLOMAP-RA and Shonan-H in AUC@5; $^\\ddagger$: worse than at least one of them (exact sign-flip test on per-seed means, Holm-adjusted $p<0.05$).",
                   "tab:synth")
        b = bounds_table(rows, os.path.join(TAB, "synth_bounds.tex"), conds, labels,
                         "Per-camera 90\\% bounds on the synthetic study (target coverage 90\\%). Hard-25\\%: the quarter of registered cameras with the largest $u_i$.",
                         "tab:bounds")
        d = drift_stats(rows, conds)
        json.dump(dict(bounds=b, drift=d), open(os.path.join(RES, "synth_bounds_drift.json"), "w"), indent=1)
        print(json.dumps(tests, indent=1)); print(json.dumps(d, indent=1))
    elif which == "real":
        rows = load("real_nd.jsonl")
        conds = ["clean", "noise1.5", "noise3", "out15", "noise1.5+out15"]
        labels = {"clean": "Clean", "noise1.5": "Noise $\\kappa{=}1.5$", "noise3": "Noise $\\kappa{=}3$",
                  "out15": "+15\\% outl.", "noise1.5+out15": "$\\kappa{=}1.5$+15\\%"}
        methods = ["L1", "CG", "CycFilt", "GLOMAP-RA", "Shonan-H", "CG-rho", "CARA-static", "CARA", "CARA-safe"]
        out, tests = synth_tables(rows, "real", conds, methods)
        latex_main(out, tests, os.path.join(TAB, "real_main.tex"), conds, methods, labels,
                   "1DSfM Notre Dame, 100-camera sub-graphs, calibration and test cameras disjoint; 5 seeds $\\times$ 20 test scenes. Same conventions as \\cref{tab:synth}; with 5 seeds no comparison can reach the marker threshold.",
                   "tab:real")
        b = bounds_table(rows, os.path.join(TAB, "real_bounds.tex"), conds, labels,
                         "Per-camera 90\\% bounds on Notre Dame sub-graphs.", "tab:realbounds", real=True)
        d = drift_stats(rows, conds)
        json.dump(dict(bounds=b, drift=d), open(os.path.join(RES, "real_bounds_drift.json"), "w"), indent=1)
        print(json.dumps(tests, indent=1)); print(json.dumps(d, indent=1))


def ablation_table():
    r = load("ablation.jsonl")
    names = {"full": "CARA (full)", "no-recal": "no recalibration", "median-tri": "median triangle score",
             "one-round": "no rescue round", "no-resid-gate": "no residual gate", "no-clip": "no weight clipping",
             "naive-bridge": "connect-back instead of consensus", "no-core": "no 2-core pruning",
             "rho-none": "no drift factor", "rho-safe": "$\\hat\\rho_{\\mathrm{safe}}$ instead of $\\hat\\rho_{\\mathrm{pt}}$",
             "no-kernel-adapt": "fixed kernel scale", "hard-final": "no final soft refinement",
             "alpha-0.05": "$\\alpha_{\\mathrm{edge}}{=}\\alpha_{\\mathrm{res}}{=}0.05$", "alpha-0.20": "$\\alpha_{\\mathrm{edge}}{=}\\alpha_{\\mathrm{res}}{=}0.20$"}
    def m(v, c, k):
        return np.nanmean([x[k] for x in r if x["variant"] == v and x["cond"] == c])
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Ablation (5 seeds $\times$ 10 scenes per condition). AUC@5, catastrophic rate (\%), 90\%-bound coverage (\%), inlier edges kept (\%) and outlier edges rejected (\%).}",
         r"\label{tab:ablation}", r"\setlength{\tabcolsep}{2.2pt}", r"\scriptsize", r"\resizebox{\columnwidth}{!}{%",
         r"\begin{tabular}{lcccccccc}", r"\toprule",
         r" & Clean & \multicolumn{2}{c}{Fog $2\times$} & \multicolumn{5}{c}{Heavy $3\times$, 35\% outl.}\\",
         r"\cmidrule(lr){2-2}\cmidrule(lr){3-4}\cmidrule(lr){5-9}",
         r"Variant & AUC@5 & AUC@5 & Cov. & AUC@5 & Cat. & Cov. & Inl. & Outl. & \\", r"\midrule"]
    L[-2] = r"Variant & AUC@5 & AUC@5 & Cov. & AUC@5 & Cat. & Cov. & Inl.\ kept & Outl.\ rej. & \\"
    for v in names:
        L.append(f"{names[v]} & {m(v,'clean','auc5'):.3f} & {m(v,'fog2x','auc5'):.3f} & {100*m(v,'fog2x','cov'):.1f} & "
                 f"{m(v,'heavy3x','auc5'):.3f} & {100*m(v,'heavy3x','cat'):.2f} & {100*m(v,'heavy3x','cov'):.1f} & "
                 f"{100*m(v,'heavy3x','inl_keep'):.1f} & {100*m(v,'heavy3x','out_rej'):.1f} & \\\\")
        if v == "full":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    open(os.path.join(TAB, "ablation.tex"), "w").write("\n".join(L).replace(" & \\\\", "\\\\").replace("& Outl.\\ rej. \\\\", "& Outl.\\ rej.\\\\") + "\n")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "ablation":
    ablation_table()
