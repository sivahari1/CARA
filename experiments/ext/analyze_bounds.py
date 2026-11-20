"""Summaries and scene-level tests for the bound extensions (ETH3D).  python analyze_bounds.py bounds.jsonl [tag]"""
import json, sys, itertools, os, numpy as np
from sklearn.metrics import roc_auc_score
R = [json.loads(l) for l in open(sys.argv[1])]; TAG = sys.argv[2] if len(sys.argv) > 2 else "v1"
C = ["clean", "fog0.7", "fog0.5", "fog0.3", "low0.3", "low0.15", "rain0.5"]; V = list(R[0]["v"])
SC = sorted({r["scene"] for r in R})
for r in R:
    for v in V:
        x = r["v"][v]; c = x["cams"]; e = np.array(c["e"]); B = np.array(c["B"])
        x["w_unfl"] = float(np.median(B[B <= 30])) if (B <= 30).any() else np.nan
        cat = e > 30
        x["auc_cat"] = float(roc_auc_score(cat, np.array(c["v"]))) if 0 < cat.sum() < len(cat) else np.nan
        x["auc_cat_u"] = float(roc_auc_score(cat, np.array(c["u"]))) if 0 < cat.sum() < len(cat) else np.nan
        fl = B > 30
        x["flag_prec"] = float(cat[fl].mean()) if fl.any() else np.nan
        x["flag_rec"] = float(fl[cat].mean()) if cat.any() else np.nan


def ps(conds, v, k):
    return np.array([np.nanmean([np.nan if r["v"][v][k] is None else r["v"][v][k] for r in R if r["cond"] in conds and r["scene"] == s]) for s in SC])


def signflip(d, one_sided=True):
    d = d[np.isfinite(d)]; k = len(d)
    S = np.array(list(itertools.product([-1, 1], repeat=k))); st = (S * d).mean(1)
    return float(np.mean(st >= d.mean() - 1e-12)) if one_sided else float(np.mean(np.abs(st) >= abs(d.mean()) - 1e-12))


out = {}
for k in ["cov", "cov_hard_u", "w_unfl", "flag", "cov_cat", "auc_cat", "flag_prec", "flag_rec"]:
    out[k] = {c: {v: float(np.nanmean(ps([c], v, k))) for v in V} for c in C}
    print(k); print("         " + " ".join(f"{v:>8s}" for v in V))
    for c in C:
        print(f"{c:9s}" + " ".join(f"{out[k][c][v]:8.3f}" for v in V))
print("u-AUROC for catastrophic (base):", {c: round(float(np.nanmean(ps([c], 'base', 'auc_cat_u'))), 3) for c in C})
SH = C[1:]
tests = {}
for v in V[1:]:
    for k in ["cov_hard_u", "cov"]:
        d = ps(SH, v, k) - ps(SH, "base", k); tests[f"{v}:{k}"] = dict(diff=float(np.nanmean(d)), wins=int((d > 0).sum()), n=int(np.isfinite(d).sum()), p=signflip(d))
    d = ps(C, v, "auc_cat") - ps(C, "base", "auc_cat_u"); tests[f"{v}:auc_cat"] = dict(diff=float(np.nanmean(d)), wins=int((d > 0).sum()), n=int(np.isfinite(d).sum()), p=signflip(d))
for k, t in tests.items():
    print(k, t)
out["tests"] = tests
json.dump(out, open(f"bounds_summary_{TAG}.json", "w"), indent=1)
