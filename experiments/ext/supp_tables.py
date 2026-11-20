"""Supplementary Tables S4-S6 (run from experiments/ext after dev_eth3d.py, the run-1 bound evaluation
aggregated into bounds_summary_v1.json by analyze_bounds.py, and synth_bounds.py)."""
import json, os, io, contextlib, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); T = os.path.join(H, "..", "..", "paper", "tables"); RES = os.path.join(H, "..", "..", "results")
with contextlib.redirect_stdout(io.StringIO()):
    import dev_eval as D
    rows = [("$u$ (unchanged)", "u", False, False), ("learned $v$, augmented", "learn", True, False), ("$u$, 3 groups", "u", False, True),
            ("learned $v$, 3 groups (E2)", "learn", False, True), ("learned $v$, augmented, 3 groups", "learn", True, True)]
    res = {name: D.evaluate(h, a, m) for name, h, a, m in rows}
K = [("none", 0.0), ("edge", 0.1), ("edge", 0.2), ("cam", 0.1), ("cam", 0.2)]
KL = {("none", 0.0): "none", ("edge", 0.1): "edge 10\\%", ("edge", 0.2): "edge 20\\%", ("cam", 0.1): "camera 10\\%", ("cam", 0.2): "camera 20\\%"}
with open(f"{T}/s5_dev.tex", "w") as o:
    o.write("\\begin{table}[!htbp]\\centering\\footnotesize\n\\caption{Development of E1/E2 on the first run's calibration scenes (leave-one-scene-out over 6 scenes, 24 sub-scenes each) with injected failures. Entries: coverage / coverage on the quarter with the largest $u_i$ / median bound width in degrees. Target 0.9.}\\label{tab:s5dev}\n")
    o.write("\\begin{adjustbox}{max width=\\textwidth}\\begin{tabular}{l" + "c" * len(K) + "}\\toprule\nNormalizer and calibration & " + " & ".join(KL[k] for k in K) + " \\\\\\midrule\n")
    for name in res:
        o.write(name + " & " + " & ".join(f"{res[name][k][0]:.3f} / {res[name][k][1]:.3f} / {res[name][k][2]:.1f}" for k in K) + " \\\\\n")
    o.write("\\bottomrule\\end{tabular}\\end{adjustbox}\\end{table}\n")
B = json.load(open(os.path.join(H, "bounds_summary_v1.json"))); C = ["fog0.7", "fog0.5", "fog0.3", "low0.3", "low0.15", "rain0.5"]
V = ["base", "E1", "E2", "E1+E2a", "E1+E2"]; KK = [("cov", "cov."), ("cov_hard_u", "hard"), ("w_unfl", "width$^\\circ$"), ("flag", "flagged"), ("flag_rec", "cat.\\ flagged")]
with open(f"{T}/s5_run1.tex", "w") as o:
    o.write("\\begin{table}[!htbp]\\centering\\footnotesize\n\\caption{Bound variants on the first ETH3D run (7 test scenes, degraded conditions pooled; these test scenes had been seen in aggregate before E1/E2 were designed). Columns as in the main paper's Table~\\ref{tab:v2bounds}; E1+E2a is E1 with the learned normalizer but without groups.}\\label{tab:s5run1}\n")
    o.write("\\begin{tabular}{lccccc}\\toprule\nVariant & " + " & ".join(l for _, l in KK) + " \\\\\\midrule\n")
    for v in V:
        o.write(v + " & " + " & ".join((f"{np.nanmean([B[k][c][v] for c in C]):.1f}" if k == "w_unfl" else f"{np.nanmean([B[k][c][v] for c in C]):.3f}") for k, _ in KK) + " \\\\\n")
    o.write("\\bottomrule\\end{tabular}\\end{table}\n")
R = [json.loads(l) for l in open(os.path.join(RES, "synth_bounds2.jsonl"))]
CS = list(dict.fromkeys(r["cond"] for r in R)); V = list(R[0]["v"])
with open(f"{T}/s6_synth.tex", "w") as o:
    o.write("\\begin{table}[!htbp]\\centering\\footnotesize\n\\caption{Bound variants on the synthetic conditions of the main study (5 seeds $\\times$ 10 scenes per condition). Entries: coverage / hard-quarter coverage / median width in degrees / fraction flagged. E1+E2a is E1 with the learned normalizer but without groups.}\\label{tab:s6synth}\n")
    o.write("\\begin{adjustbox}{max width=\\textwidth}\\begin{tabular}{l" + "c" * len(V) + "}\\toprule\nCondition & " + " & ".join(V) + " \\\\\\midrule\n")
    for c in CS:
        g = lambda v, k: np.mean([r["v"][v][k] for r in R if r["cond"] == c])
        o.write(c.replace("_", "\\_") + " & " + " & ".join(f"{g(v,'cov'):.3f} / {g(v,'cov_hard_u'):.3f} / {g(v,'width'):.1f} / {g(v,'flag'):.2f}" for v in V) + " \\\\\n")
    o.write("\\bottomrule\\end{tabular}\\end{adjustbox}\\end{table}\n")
print("wrote s5_dev.tex, s5_run1.tex, s6_synth.tex")
