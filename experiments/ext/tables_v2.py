"""LaTeX tables for the confirmatory ETH3D run (run 2) and the pre-registered hypothesis tests."""
import json, os, itertools, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); T = os.path.join(H, "..", "..", "paper", "tables")
S = json.load(open(os.path.join(H, "summary_v2.json"))); B = json.load(open(os.path.join(H, "bounds_summary_v2.json")))
C = ["clean", "fog0.7", "fog0.5", "fog0.3", "low0.3", "low0.15", "rain0.5"]
CL = {"clean": "clean", "fog0.7": "fog 0.7", "fog0.5": "fog 0.5", "fog0.3": "fog 0.3", "low0.3": "low 0.3", "low0.15": "low 0.15", "rain0.5": "rain"}

# accuracy
M = ["CG", "GLOMAP-RA", "Shonan-H", "CG-rho", "ConfW", "CARA", "ConfW-learnsig", "CARA-learnsig"]
ML = {"CG-rho": "CG-$\\hat\\rho$", "ConfW-learnsig": "ConfW-L", "CARA-learnsig": "CARA-L"}
with open(os.path.join(T, "v2_acc.tex"), "w") as f:
    f.write("\\begin{table*}[!t]\\centering\\footnotesize\n\\caption{Confirmatory ETH3D run (13 scenes, leave-one-scene-out calibration, 8 new sub-scenes of 12 cameras per scene and condition, 5 calibration seeds): AUC@5$^\\circ$, mean over test scenes with 95\\% bootstrap interval. -L: learned edge uncertainty (E3). Best per row in bold.}\\label{tab:v2acc}\n")
    f.write("\\setlength{\\tabcolsep}{2.5pt}\\resizebox{\\textwidth}{!}{\\begin{tabular}{l" + "c" * len(M) + "}\\toprule\nCondition & " + " & ".join(ML.get(m, m) for m in M) + " \\\\\\midrule\n")
    for c in C:
        best = max(S["accuracy"][c][m]["auc5"][0] for m in M); cells = []
        for m in M:
            v = S["accuracy"][c][m]["auc5"]; s = f"{v[0]:.3f}\\,{{\\scriptsize[{v[1]:.3f},{v[2]:.3f}]}}"
            cells.append("\\textbf{" + s + "}" if abs(v[0] - best) < 5e-4 else s)
        f.write(CL[c] + " & " + " & ".join(cells) + " \\\\\n")
    f.write("\\bottomrule\\end{tabular}}\\end{table*}\n")

# bounds (variants x metrics; degraded conditions pooled, and clean)
V = ["base", "E1", "E2", "E2u", "E1+E2a", "E1+E2"]
K = [("cov", "cov."), ("cov_hard_u", "hard"), ("w_unfl", "width$^\\circ$"), ("flag", "flagged"), ("flag_rec", "cat.\\ flagged")]
def pooled(k, v, conds):
    return float(np.nanmean([B[k][c][v] for c in conds]))
with open(os.path.join(T, "v2_bounds.tex"), "w") as f:
    f.write("\\begin{table}[!t]\\centering\\footnotesize\n\\caption{Per-camera bounds (target 90\\%) in the confirmatory run: coverage of registered cameras, coverage on the quarter with the largest $u_i$ (hard), median width of unflagged bounds, fraction of cameras flagged (bound $>30^\\circ$), and fraction of catastrophic cameras ($>30^\\circ$) that are flagged. Means over 13 test scenes and 5 seeds.}\\label{tab:v2bounds}\n")
    f.write("\\setlength{\\tabcolsep}{2.2pt}\\begin{tabular}{llccccc}\\toprule\n & Variant & " + " & ".join(l for _, l in K) + " \\\\\\midrule\n")
    for name, conds in (("clean", ["clean"]), ("degraded", C[1:])):
        for i, v in enumerate(V):
            f.write(("\\multirow{6}{*}{" + name + "}" if i == 0 else "") + f" & {v} & " + " & ".join(f"{pooled(k, v, conds):.3f}" if k != "w_unfl" else f"{pooled(k, v, conds):.1f}" for k, _ in K) + " \\\\\n")
        if name == "clean": f.write("\\midrule\n")
    f.write("\\bottomrule\\end{tabular}\\end{table}\n")

# translation
TM = ["GT-rot", "CG", "GLOMAP-RA", "CARA", "CARA+gate", "CARA-learnsig+gate"]
TL = {"GT-rot": "GT rot.", "CARA-learnsig+gate": "CARA-L+gate"}
with open(os.path.join(T, "v2_trans.tex"), "w") as f:
    f.write("\\begin{table}[!t]\\centering\\footnotesize\n\\caption{Downstream camera positions (E5): LUD translation averaging of VGGT's relative translations, rotated by each method's global rotations. AUC of the scale-free position error up to $0.2$ (higher is better) and its median (lower is better); +gate uses only CARA's accepted edges. GT rot.\\ uses the reference rotations and all edges.}\\label{tab:v2trans}\n")
    f.write("\\setlength{\\tabcolsep}{2pt}\\resizebox{\\columnwidth}{!}{\\begin{tabular}{l" + "c" * len(TM) + "}\\toprule\n & " + " & ".join(TL.get(m, m) for m in TM) + " \\\\\\midrule\n")
    for c in C:
        f.write(CL[c] + " & " + " & ".join(f"{S['trans'][c][m]['pos_auc'][0]:.3f}/{S['trans'][c][m]['pos_med'][0]:.2f}" for m in TM) + " \\\\\n")
    f.write("\\bottomrule\\end{tabular}}\\end{table}\n")

# hypotheses
bt = B["tests"]; st = S["tests"]
rows = [("H1", "E2 raises hard-camera coverage (degraded)", bt["E2:cov_hard_u"], True),
        ("H2", "E2 raises overall coverage (degraded)", bt["E2:cov"], True),
        ("H3", "E2 normalizer ranks catastrophic cameras better than $u$", bt["E2:auc_cat"], True),
        ("H4", "learned $\\hat\\sigma$ raises CARA's AUC@5", st["H4 CARA-learnsig - CARA auc5"], False),
        ("H5", "learned $\\hat\\sigma$ raises outlier AUROC", st["H5 AUROC learned - f1"], False),
        ("H6", "CARA+gate lowers median position error vs CG", st["H6 CG - CARA+gate pos_med (positive = CARA+gate better)"], False)]
p = np.array([r[2]["p"] for r in rows]); order = np.argsort(p); adj = np.empty(len(p)); run = 0
for rnk, i in enumerate(order):
    run = max(run, min(1.0, (len(p) - rnk) * p[i])); adj[i] = run
with open(os.path.join(T, "v2_hyp.tex"), "w") as f:
    f.write("\\begin{table}[!t]\\centering\\footnotesize\n\\caption{Pre-registered hypotheses for the confirmatory run. Difference averaged over the 13 test scenes, scenes in favour, exact sign-flip $p$ (one-sided for H1--H3 as pre-registered; two-sided for H4--H6, whose sidedness was not pre-registered) and Holm-adjusted $p$ over H1--H6.}\\label{tab:v2hyp}\n")
    f.write("\\setlength{\\tabcolsep}{2.5pt}\\resizebox{\\columnwidth}{!}{\\begin{tabular}{llcccc}\\toprule\n & Hypothesis & diff. & scenes & $p$ & $p_{\\mathrm{Holm}}$ \\\\\\midrule\n")
    for (h, txt, t, _), a in zip(rows, adj):
        n = t.get("n", 13)
        f.write(f"{h} & {txt} & {t['diff']:+.3f} & {t['wins']}/{n} & {t['p']:.4f} & {a:.4f} \\\\\n")
    f.write("\\bottomrule\\end{tabular}}\\end{table}\n")
json.dump(dict(zip([r[0] for r in rows], [dict(r[2], p_holm=float(a)) for r, a in zip(rows, adj)])), open(os.path.join(H, "hypotheses_v2.json"), "w"), indent=1)
print("tables written; Holm:", dict(zip([r[0] for r in rows], adj.round(4))))
