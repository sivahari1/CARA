"""Analysis of the ETH3D / VGGT experiment (gpu_kit stage 2 output).
Inference unit: the 7 held-out ETH3D test scenes. Seeds only reshuffle the calibration splits, so
per-scene values are first averaged over seeds and sub-scenes, then compared across scenes.
  python eth3d_analyze.py ../real_eth3d/results_eth3d.jsonl ../real_eth3d/out_eth3d
"""
import os, sys, json, glob, itertools
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cara.so3 import angle, T

RES, NPZ = sys.argv[1], sys.argv[2]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
PAPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "tables")
rows = [json.loads(l) for l in open(RES)]
CONDS = ["clean", "fog0.7", "fog0.5", "fog0.3", "low0.3", "low0.15", "rain0.5"]
CLAB = {"clean": "clean", "fog0.7": "fog $t{=}0.7$", "fog0.5": "fog $t{=}0.5$", "fog0.3": "fog $t{=}0.3$",
        "low0.3": "low light $0.3$", "low0.15": "low light $0.15$", "rain0.5": "rain"}
METH = ["CG", "ConfW", "GLOMAP-RA", "Shonan-H", "CG-rho", "CARA-static", "CARA"]
MLAB = {"CG-rho": "CG-$\\hat\\rho$"}
RNG = np.random.default_rng(0)
scenes = sorted({r["base_scene"] for r in rows})
TAG = os.environ.get("TAG", "")
METH = [m for m in METH if m in rows[0]["methods"]]


def per_scene(cond, meth, key, sub=None):
    """mean over seeds and sub-scenes for each base scene -> array (7,)"""
    out = []
    for s in scenes:
        v = []
        for r in rows:
            if r["cond"] == cond and r["base_scene"] == s:
                x = r["methods"][meth]
                x = x.get(key, np.nan) if sub is None else x.get("bounds", {}).get("crc", {}).get(key, np.nan)
                v.append(np.nan if x is None else x)
        out.append(np.nanmean(v))
    return np.array(out)


def boot(v, B=4000):
    v = v[np.isfinite(v)]
    m = [v[RNG.integers(len(v), size=len(v))].mean() for _ in range(B)]
    return v.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def signflip(d):
    d = d[np.isfinite(d)]
    k = len(d); obs = abs(d.mean())
    s = np.array(list(itertools.product([-1, 1], repeat=k)))
    return float(np.mean(np.abs((s * d).mean(1)) >= obs - 1e-12))


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - r) * p[i])); adj[i] = run
    return adj


report = {}
# ---------------------------------------------------------------- front-end statistics per condition
fe = {}
for c in CONDS:
    errs, sig = [], []
    for f in glob.glob(os.path.join(NPZ, f"test__*__{c}.npz")):
        z = np.load(f, allow_pickle=True); E, R, G = z["edges"], z["Rij"], z["R_gt"]
        errs += list(np.degrees(angle(T(G[E[:, 1]] @ T(G[E[:, 0]])) @ R))); sig += list(1 / z["conf_med"])
    errs, sig = np.array(errs), np.array(sig)
    fe[c] = dict(edges=len(errs), med_err=float(np.median(errs)), q_err_inl=float(np.median(errs[errs <= 5])),
                 out5=float(np.mean(errs > 5)), out30=float(np.mean(errs > 30)), med_sig=float(np.median(sig)))
report["frontend"] = fe
clean_ratio = {c: fe[c]["q_err_inl"] / fe["clean"]["q_err_inl"] for c in CONDS}

# ---------------------------------------------------------------- accuracy
acc = {}
for c in CONDS:
    acc[c] = {m: {k: boot(per_scene(c, m, k)) for k in ("auc5", "auc2", "median", "cat", "reg")} for m in METH}
report["accuracy"] = acc

# paired tests: CARA vs each baseline, per condition (7 scenes) and pooled over conditions
comps = [b for b in ["CG", "GLOMAP-RA", "Shonan-H", "CG-rho", "ConfW"] if b in METH]
tests = {}
for b in comps:
    per_c = {}
    for c in CONDS:
        d = per_scene(c, "CARA", "auc5") - per_scene(c, b, "auc5")
        per_c[c] = dict(diff=float(d.mean()), wins=int((d > 0).sum()), losses=int((d < 0).sum()), p=signflip(d))
    dpool = np.mean([per_scene(c, "CARA", "auc5") - per_scene(c, b, "auc5") for c in CONDS], 0)
    dshift = np.mean([per_scene(c, "CARA", "auc5") - per_scene(c, b, "auc5") for c in CONDS[1:]], 0)
    tests[b] = dict(per_cond=per_c, pooled=dict(diff=float(dpool.mean()), wins=int((dpool > 0).sum()), p=signflip(dpool)),
                    shifted=dict(diff=float(dshift.mean()), wins=int((dshift > 0).sum()), p=signflip(dshift)))
adj = holm([tests[b]["pooled"]["p"] for b in comps])
for b, a in zip(comps, adj):
    tests[b]["pooled"]["p_holm"] = float(a)
report["tests"] = tests

# ---------------------------------------------------------------- bounds, drift, gating
bd = {}
for c in CONDS:
    bd[c] = {}
    for m in ("CARA", "CARA-static"):
        bd[c][m] = {k: boot(per_scene(c, m, k, sub="crc")) for k in ("cov", "cov_hard", "width")}
    rh = [r["methods"]["CARA"]["rho"] for r in rows if r["cond"] == c]
    bd[c]["rho_med"] = float(np.median(rh)); bd[c]["rho_q"] = [float(np.percentile(rh, 25)), float(np.percentile(rh, 75))]
    bd[c]["inl_keep"] = float(np.nanmean([r["methods"]["CARA"]["inl_keep"] for r in rows if r["cond"] == c]))
    bd[c]["out_rej"] = float(np.nanmean([r["methods"]["CARA"]["out_rej"] for r in rows if r["cond"] == c]))
    bd[c]["noise_ratio"] = clean_ratio[c]
report["bounds"] = bd
os.makedirs(OUT, exist_ok=True)
json.dump(report, open(os.path.join(OUT, f"eth3d{TAG}_summary.json"), "w"), indent=1, default=float)

# ---------------------------------------------------------------- print
print("front end:")
for c in CONDS:
    f = fe[c]; print(f"  {c:8s} med {f['med_err']:.2f}  inlier-med {f['q_err_inl']:.2f}  >5 {f['out5']:.3f} >30 {f['out30']:.3f}  med sig {f['med_sig']:.3f}")
print("AUC@5 (mean over 7 scenes):")
print("  " + " ".join(f"{m:>11s}" for m in METH))
for c in CONDS:
    print(f"{c:8s}" + " ".join(f"{acc[c][m]['auc5'][0]:11.3f}" for m in METH))
print("catastrophic:")
for c in CONDS:
    print(f"{c:8s}" + " ".join(f"{acc[c][m]['cat'][0]:11.3f}" for m in METH))
print("tests CARA - baseline (AUC@5):")
for b in comps:
    t = tests[b]
    print(f"  vs {b:9s} pooled {t['pooled']['diff']:+.4f} wins {t['pooled']['wins']}/7 p {t['pooled']['p']:.3f} holm {t['pooled']['p_holm']:.3f} | shifted {t['shifted']['diff']:+.4f} wins {t['shifted']['wins']}/7 p {t['shifted']['p']:.3f}")
    print("     " + " ".join(f"{c}:{t['per_cond'][c]['diff']:+.3f}({t['per_cond'][c]['wins']}/7)" for c in CONDS))
print("bounds:")
for c in CONDS:
    b = bd[c]
    print(f"  {c:8s} rho {b['rho_med']:.2f} [{b['rho_q'][0]:.2f},{b['rho_q'][1]:.2f}] ratio {b['noise_ratio']:.2f} | CARA cov {b['CARA']['cov'][0]:.3f} hard {b['CARA']['cov_hard'][0]:.3f} w {b['CARA']['width'][0]:.2f} | static cov {b['CARA-static']['cov'][0]:.3f} hard {b['CARA-static']['cov_hard'][0]:.3f} w {b['CARA-static']['width'][0]:.2f} | keep {b['inl_keep']:.3f} rej {b['out_rej']:.3f}")


# ---------------------------------------------------------------- LaTeX tables
def f3(t, d=3):
    return f"{t[0]:.{d}f}\\,{{\\scriptsize[{t[1]:.{d}f},{t[2]:.{d}f}]}}"


os.makedirs(PAPER, exist_ok=True)
with open(os.path.join(PAPER, f"eth3d{TAG}_acc.tex"), "w") as f:
    cols = METH
    f.write("\\begin{table*}[!t]\\centering\\footnotesize\n\\caption{ETH3D with the VGGT front end: AUC@5$^\\circ$ (mean over the 7 held-out test scenes, 95\\% bootstrap interval over scenes; 70 sub-scenes of 12 cameras per condition, 5 calibration seeds). Best mean per row in bold. The last column counts test scenes on which CARA beats CG.}\\label{tab:eth3d"+TAG+"}\n")
    f.write("\\setlength{\\tabcolsep}{3pt}\\resizebox{\\textwidth}{!}{\\begin{tabular}{l" + "c" * len(cols) + "c}\\toprule\n")
    f.write("Condition & " + " & ".join(MLAB.get(m, m) for m in cols) + " & CARA$>$CG \\\\\\midrule\n")
    for c in CONDS:
        best = max(acc[c][m]["auc5"][0] for m in cols)
        cells = []
        for m in cols:
            s = f3(acc[c][m]["auc5"])
            cells.append("\\textbf{" + s + "}" if abs(acc[c][m]["auc5"][0] - best) < 5e-4 else s)
        t = tests["CG"]["per_cond"][c]
        f.write(CLAB[c] + " & " + " & ".join(cells) + f" & {t['wins']}/7 \\\\\n")
    f.write("\\bottomrule\\end{tabular}}\\end{table*}\n")

with open(os.path.join(PAPER, f"eth3d{TAG}_bounds.tex"), "w") as f:
    f.write("\\begin{table}[!t]\\centering\\footnotesize\n\\caption{ETH3D with VGGT: per-camera bounds at target $1-\\alpha_{\\mathrm{cam}}=0.9$ (mean over test scenes, 95\\% interval over scenes), median $\\hat\\rho$, and the ratio of the median inlier edge error to its clean value.}\\label{tab:eth3dbounds"+TAG+"}\n")
    f.write("\\setlength{\\tabcolsep}{2.5pt}\\begin{tabular}{lcccccc}\\toprule\n")
    f.write(" & \\multicolumn{2}{c}{CARA} & \\multicolumn{2}{c}{CARA-static} & & \\\\\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n")
    f.write("Condition & cov. & hard & cov. & hard & $\\hat\\rho$ & ratio \\\\\\midrule\n")
    for c in CONDS:
        b = bd[c]
        f.write(f"{CLAB[c]} & {b['CARA']['cov'][0]:.3f} & {b['CARA']['cov_hard'][0]:.3f} & {b['CARA-static']['cov'][0]:.3f} & {b['CARA-static']['cov_hard'][0]:.3f} & {b['rho_med']:.2f} & {b['noise_ratio']:.2f} \\\\\n")
    f.write("\\bottomrule\\end{tabular}\\end{table}\n")
print("wrote", os.path.join(OUT, f"eth3d{TAG}_summary.json"), "and paper/tables/eth3d_*.tex")
