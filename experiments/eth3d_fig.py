import json, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(os.path.join(H, "..", "results", "eth3d_summary.json")))
C = ["clean", "fog0.7", "fog0.5", "fog0.3", "low0.3", "low0.15", "rain0.5"]
L = ["clean", "fog\n0.7", "fog\n0.5", "fog\n0.3", "low\n0.3", "low\n0.15", "rain"]
x = np.arange(len(C))
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
for m, st in [("CG", "o-"), ("GLOMAP-RA", "s-"), ("Shonan-H", "v-"), ("ConfW", "d--"), ("CARA", "*-")]:
    ax[0].plot(x, [r["accuracy"][c][m]["auc5"][0] for c in C], st, ms=4, lw=1.2 if m != "CARA" else 2, label=m)
ax[0].set_xticks(x, L, fontsize=7); ax[0].set_ylabel("AUC@5$^\\circ$"); ax[0].legend(fontsize=6, ncol=2); ax[0].set_title("(a) accuracy", fontsize=8)
for m, st, off in [("CARA", "o-", -0.1), ("CARA-static", "s--", 0.1)]:
    v = np.array([r["bounds"][c][m]["cov"] for c in C])
    ax[1].errorbar(x + off, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt=st, ms=4, capsize=2, label=m + " (all)")
v = np.array([r["bounds"][c]["CARA"]["cov_hard"][0] for c in C])
ax[1].plot(x, v, "^:", ms=4, color="C3", label="CARA (hardest 25%)")
ax[1].axhline(0.9, color="k", lw=0.8, ls=":"); ax[1].set_ylim(0.55, 1.0)
ax[1].set_xticks(x, L, fontsize=7); ax[1].set_ylabel("bound coverage"); ax[1].legend(fontsize=6, loc="lower left"); ax[1].set_title("(b) per-camera bounds, target 0.9", fontsize=8)
for a in ax: a.tick_params(labelsize=7); a.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(H, "..", "paper", "figs", "eth3d.pdf"))
