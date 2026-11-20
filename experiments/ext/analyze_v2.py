"""Run-2 analysis: accuracy (E4, 13 scenes), learned uncertainty (E3), translation averaging (E5).
Unit of inference: test scene (13). Baselines exist for seed 0 only (they do not depend on calibration);
CARA variants are averaged over seeds 0-4."""
import json, sys, itertools, numpy as np
R = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "results_v2.jsonl")]
C = ["clean", "fog0.7", "fog0.5", "fog0.3", "low0.3", "low0.15", "rain0.5"]
SC = sorted({r["scene"] for r in R})
RNG = np.random.default_rng(0)


def val(r, sec, m, k):
    x = r[sec].get(m)
    if x is None: return np.nan
    v = x.get(k); return np.nan if v is None else v


def ps(conds, sec, m, k):
    return np.array([np.nanmean([val(r, sec, m, k) for r in R if r["cond"] in conds and r["scene"] == s]) for s in SC])


def signflip(d, two=True):
    d = d[np.isfinite(d)]; k = len(d)
    S = np.array(list(itertools.product([-1, 1], repeat=k))); st = (S * d).mean(1)
    return float(np.mean(np.abs(st) >= abs(d.mean()) - 1e-12)) if two else float(np.mean(st >= d.mean() - 1e-12))


def boot(v, B=4000):
    v = v[np.isfinite(v)]; m = [v[RNG.integers(len(v), size=len(v))].mean() for _ in range(B)]
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


out = {"accuracy": {}, "trans": {}, "tests": {}}
M = ["CG", "ConfW", "GLOMAP-RA", "Shonan-H", "CG-rho", "CARA", "ConfW-learnsig", "CARA-learnsig"]
print("AUC@5 (13 scenes)"); print("         " + " ".join(f"{m[:11]:>11s}" for m in M))
for c in C:
    out["accuracy"][c] = {m: {k: boot(ps([c], "methods", m, k)) for k in ("auc5", "auc2", "cat", "median")} for m in M}
    print(f"{c:9s}" + " ".join(f"{out['accuracy'][c][m]['auc5'][0]:11.3f}" for m in M))
print("catastrophic");
for c in C: print(f"{c:9s}" + " ".join(f"{out['accuracy'][c][m]['cat'][0]:11.3f}" for m in M))
T = ["GT-rot", "CG", "GLOMAP-RA", "CARA", "CARA+gate", "CARA-learnsig+gate"]
print("position AUC (rel. error <= 0.2) / median rel. error")
for c in C:
    out["trans"][c] = {m: {k: boot(ps([c], "trans", m, k)) for k in ("pos_auc", "pos_med", "pos_cat")} for m in T}
    print(f"{c:9s}" + " ".join(f"{out['trans'][c][m]['pos_auc'][0]:6.3f}/{out['trans'][c][m]['pos_med'][0]:5.3f}" for m in T))
# tests (two-sided, 13 scenes)
def comp(name, sa, ma, sb, mb, k, conds=C):
    d = ps(conds, sa, ma, k) - ps(conds, sb, mb, k)
    out["tests"][name] = dict(diff=float(np.nanmean(d)), wins=int((d > 0).sum()), losses=int((d < 0).sum()), p=signflip(d))
for b in ["CG", "GLOMAP-RA", "Shonan-H", "CG-rho", "ConfW"]:
    comp(f"CARA-{b} auc5", "methods", "CARA", "methods", b, "auc5")
comp("H4 CARA-learnsig - CARA auc5", "methods", "CARA-learnsig", "methods", "CARA", "auc5")
comp("ConfW-learnsig - ConfW auc5", "methods", "ConfW-learnsig", "methods", "ConfW", "auc5")
comp("CARA-learnsig - ConfW-learnsig auc5", "methods", "CARA-learnsig", "methods", "ConfW-learnsig", "auc5")
comp("H6 CG - CARA+gate pos_med (positive = CARA+gate better)", "trans", "CG", "trans", "CARA+gate", "pos_med")
comp("CARA+gate - CG pos_auc", "trans", "CARA+gate", "trans", "CG", "pos_auc")
comp("CARA+gate - GLOMAP pos_auc", "trans", "CARA+gate", "trans", "GLOMAP-RA", "pos_auc")
comp("CARA+gate - CARA pos_auc", "trans", "CARA+gate", "trans", "CARA", "pos_auc")
# H5 AUROC learned vs f1
au = {k: np.array([np.nanmean([r["auroc"].get(k, np.nan) for r in R if r["scene"] == s]) for s in SC]) for k in ["f1_med", "f2_worst_logmean", "f3_best_logmean", "f4_worst_q10", "f5_iter_delta", "learned"]}
out["auroc"] = {k: float(np.nanmean(v)) for k, v in au.items()}
d = au["learned"] - au["f1_med"]; out["tests"]["H5 AUROC learned - f1"] = dict(diff=float(d.mean()), wins=int((d > 0).sum()), p=signflip(d))
print("outlier AUROC:", {k: round(v, 3) for k, v in out["auroc"].items()})
for k, t in out["tests"].items(): print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in t.items()})
json.dump(out, open("summary_v2.json", "w"), indent=1)
