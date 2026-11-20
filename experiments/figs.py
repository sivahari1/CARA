"""Paper figures. Palette: validated categorical order (blue, orange, aqua, yellow),
diverging blue<->red with gray midpoint. Identity never by color alone:
legends + markers/hatches for print."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "paper", "figs")
os.makedirs(FIG, exist_ok=True)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED, GRID = "#1a1a19", "#6b6a64", "#dcdad4"
plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "axes.linewidth": 0.5,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": INK, "legend.frameon": False,
                     "savefig.bbox": "tight", "savefig.dpi": 300})


def _style(ax, ygrid=True):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)


def load(n):
    p = os.path.join(RES, n)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else None


def fig_coverage(bd, conds, labels, name, entries):
    fig, ax = plt.subplots(figsize=(3.45, 1.9))
    k = len(entries); w = 0.8 / k
    hatches = ["", "////", "....", "\\\\\\\\"]
    for j, (key, lab) in enumerate(entries):
        vals = [100 * bd[c][key]["cov"][0] for c in conds]
        lo = [100 * (bd[c][key]["cov"][0] - bd[c][key]["cov"][1]) for c in conds]
        hi = [100 * (bd[c][key]["cov"][2] - bd[c][key]["cov"][0]) for c in conds]
        x = np.arange(len(conds)) + (j - (k - 1) / 2) * w
        ax.bar(x, vals, w * 0.88, color=C[j], hatch=hatches[j], edgecolor="white", lw=0.3, label=lab)
        ax.errorbar(x, vals, yerr=[lo, hi], fmt="none", ecolor=INK, lw=0.5, capsize=0)
    ax.axhline(90, color=INK, lw=0.7, ls="--")
    ax.text(-0.45, 101, "target 90%", ha="left", va="bottom", fontsize=6.5, color=INK)
    ax.set_xticks(range(len(conds))); ax.set_xticklabels([labels[c] for c in conds])
    ax.set_ylabel("Coverage of registered cameras (%)"); ax.set_ylim(0, 108)
    _style(ax)
    ax.legend(ncol=2, fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, 1.32), handlelength=1.2)
    fig.savefig(os.path.join(FIG, name)); plt.close(fig)


def fig_drift(rows, conds, labels, name, key_true="rho_true"):
    fig, ax = plt.subplots(figsize=(3.45, 1.75))
    for i, c in enumerate(conds):
        R = [r for r in rows if r["cond"] == c]
        pt = [r["methods"]["CARA"]["rho"] for r in R]
        sf = [r["methods"]["CARA-safe"]["rho"] for r in R]
        jit = np.random.default_rng(i).uniform(-0.09, 0.09, len(R))
        ax.scatter(i - 0.14 + jit, pt, s=5, color=C[0], marker="o", lw=0, alpha=0.55, label=r"$\hat\rho_{\mathrm{pt}}$" if i == 0 else None)
        ax.scatter(i + 0.14 + jit, sf, s=6, color=C[1], marker="^", lw=0, alpha=0.55, label=r"$\hat\rho_{\mathrm{safe}}$" if i == 0 else None)
        t = R[0][key_true]
        ax.plot([i - 0.32, i + 0.32], [t, t], color=INK, lw=1.0, label=r"true $\rho^\star$" if i == 0 else None)
    ax.set_yscale("log"); ax.set_yticks([1, 2, 3, 5, 10, 20]); ax.set_yticklabels(["1", "2", "3", "5", "10", "20"])
    ax.set_xticks(range(len(conds))); ax.set_xticklabels([labels[c] for c in conds])
    ax.set_ylabel(r"drift factor $\rho$")
    _style(ax)
    ax.legend(ncol=3, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.22), handletextpad=0.3, markerscale=2)
    fig.savefig(os.path.join(FIG, name)); plt.close(fig)


def fig_grid(rows, name):
    noises = sorted({r["noise"] for r in rows}); pouts = sorted({r["p_out"] for r in rows})
    D = np.zeros((len(pouts), len(noises))); CV = np.zeros_like(D)
    for a, nm in enumerate(noises):
        for b, po in enumerate(pouts):
            R = [r for r in rows if r["noise"] == nm and r["p_out"] == po]
            D[b, a] = np.mean([r["CARA"]["auc5"] - r["CG"]["auc5"] for r in R])
            CV[b, a] = np.mean([r["CARA"]["cov"] for r in R])
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.7))
    v = np.abs(D).max()
    im = axs[0].imshow(D, cmap=cmap, vmin=-v, vmax=v, origin="lower", aspect="auto")
    for (b, a), val in np.ndenumerate(D):
        axs[0].text(a, b, f"{val:+.2f}", ha="center", va="center", fontsize=5.8, color=INK)
    axs[0].set_title(r"$\Delta$AUC@5 (CARA$-$CG)", fontsize=7)
    cmap2 = LinearSegmentedColormap.from_list("seq", ["#f0efec", "#2a78d6"])
    axs[1].imshow(np.abs(CV - 0.9), cmap=LinearSegmentedColormap.from_list("s2", ["#f0efec", "#eb6834"]), vmin=0, vmax=0.15, origin="lower", aspect="auto")
    for (b, a), val in np.ndenumerate(CV):
        axs[1].text(a, b, f"{100*val:.0f}", ha="center", va="center", fontsize=5.8, color=INK)
    axs[1].set_title("Bound coverage (%)", fontsize=7)
    for ax in axs:
        ax.set_xticks(range(len(noises))); ax.set_xticklabels([f"{x:g}" for x in noises])
        ax.set_yticks(range(len(pouts))); ax.set_yticklabels([f"{x:g}" for x in pouts])
        ax.set_xlabel(r"noise scale $\rho^\star$")
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axs[0].set_ylabel("outlier rate")
    axs[1].set_yticklabels([])
    fig.tight_layout(w_pad=0.6)
    fig.savefig(os.path.join(FIG, name)); plt.close(fig)


def fig_dop(rows, name):
    fr = sorted({r["dop_frac"] for r in rows})
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.6))
    for j, (m, mk) in enumerate([("CG", "s"), ("CycFilt", "D"), ("CARA", "o"), ("CARA-median", "^")]):
        cat = [100 * np.mean([r[m]["cat"] for r in rows if r["dop_frac"] == f]) for f in fr]
        a5 = [np.mean([r[m]["auc5"] for r in rows if r["dop_frac"] == f]) for f in fr]
        axs[0].plot(fr, a5, color=C[j], marker=mk, ms=3, lw=1.2, label=m)
        axs[1].plot(fr, cat, color=C[j], marker=mk, ms=3, lw=1.2)
    axs[0].set_ylabel("AUC@5"); axs[1].set_ylabel("catastrophic (%)")
    for ax in axs:
        ax.set_xlabel("doppelganger fraction"); _style(ax)
    axs[0].legend(fontsize=6, loc="upper right", ncol=1, handlelength=1.5)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(os.path.join(FIG, name)); plt.close(fig)


def fig_runtime(rows, name):
    ns = sorted({r["n"] for r in rows})
    fig, ax = plt.subplots(figsize=(1.9, 1.6))
    for j, (k, lab, mk) in enumerate([("t_cg", "CG", "s"), ("t_cara", "CARA", "o")]):
        ax.plot(ns, [np.median([r[k] for r in rows if r["n"] == n]) for n in ns], color=C[j], marker=mk, ms=3, lw=1.2, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("cameras $n$"); ax.set_ylabel("time (s)")
    _style(ax); ax.legend(fontsize=6.5)
    fig.savefig(os.path.join(FIG, name)); plt.close(fig)


if __name__ == "__main__":
    rows = load("synth_main.jsonl")
    labels = {"clean": "Clean", "fog2x": "Fog 2x", "heavy3x": "Heavy", "sparse": "Sparse", "corrflip": "Flips", "large": "n=300"}
    conds = ["clean", "fog2x", "heavy3x", "sparse", "corrflip", "large"]
    bd = json.load(open(os.path.join(RES, "synth_bounds_drift.json")))["bounds"]
    ent = [("CRC + $\\hat\\rho_{\\mathrm{pt}}$", "CRC + drift (CARA)"), ("CRC, no drift", "CRC, no drift"),
           ("Gauss (recal.)", "Gaussian, recalibrated"), ("Gauss (native $\\hat\\sigma$)", "Gaussian, native")]
    fig_coverage(bd, conds, labels, "synth_coverage.pdf", ent)
    fig_drift(rows, conds, labels, "synth_drift.pdf")
    g = load("grid.jsonl")
    if g: fig_grid(g, "grid.pdf")
    d = load("dop.jsonl")
    if d: fig_dop(d, "dop.pdf")
    rt = load("runtime.jsonl")
    if rt: fig_runtime(rt, "runtime.pdf")
    rr = load("real_nd.jsonl")
    if rr and os.path.exists(os.path.join(RES, "real_bounds_drift.json")):
        rb = json.load(open(os.path.join(RES, "real_bounds_drift.json")))["bounds"]
        rl = {"clean": "Clean", "noise1.5": "$\\kappa$=1.5", "noise3": "$\\kappa$=3", "out15": "+15% out", "noise1.5+out15": "both"}
        rc = ["clean", "noise1.5", "noise3", "out15", "noise1.5+out15"]
        fig_coverage(rb, rc, rl, "real_coverage.pdf",
                     [("CRC + $\\hat\\rho_{\\mathrm{pt}}$", "CRC + drift (CARA)"), ("CRC, no drift", "CRC, no drift"),
                      ("CRC + $\\hat\\rho_{\\mathrm{safe}}$", "CRC + safe drift")])
    print("figs done")
