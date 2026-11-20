"""Generates paper/tables/real_lite_inline.tex from results/real_lite.jsonl (seed-level summary)."""
import os, json
import numpy as np
HERE = os.path.dirname(__file__)
r = [json.loads(l) for l in open(os.path.join(HERE, "..", "results", "real_lite.jsonl"))]
conds = ["clean", "noise1.5", "noise3", "out15", "noise1.5+out15"]
lab = {"clean": "clean", "noise1.5": "$\\kappa{=}1.5$", "noise3": "$\\kappa{=}3$", "out15": "+15\\% outliers", "noise1.5+out15": "both"}
parts, regs, covs = [], [], []
for c in conds:
    R = [x for x in r if x["cond"] == c]
    a = np.mean([x["lite"]["auc5"] for x in R]); b = np.mean([x["CG"]["auc5"] for x in R])
    seeds = sorted({x["seed"] for x in R})
    better = sum(np.mean([x["lite"]["auc5"] - x["CG"]["auc5"] for x in R if x["seed"] == s]) > 0 for s in seeds)
    parts.append(f"{a:.3f} vs.\\ {b:.3f} ({lab[c]}; higher in {better} of {len(seeds)} seeds)")
    regs.append(np.mean([x["lite"]["reg"] for x in R])); covs.append(np.mean([x["lite"]["cov"] for x in R]))
txt = ("AUC@5 of CARA-lite vs.\\ CG is " + "; ".join(parts) +
       f". It registers {100*min(regs):.1f}--{100*max(regs):.1f}\\% of cameras and its 90\\% bounds cover "
       f"{100*min(covs):.1f}--{100*max(covs):.1f}\\% of registered cameras across the five conditions (80 test scenes per condition).")
open(os.path.join(HERE, "..", "paper", "tables", "real_lite_inline.tex"), "w").write(txt + "\n")
print(txt)
