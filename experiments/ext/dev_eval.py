import pickle, numpy as np, sys
sys.path.insert(0, "../.."); from cara.method import crc_threshold
D = np.pi / 180
R = [r for r in pickle.load(open("dev_eth3d.pkl", "rb")) if r["meth"] == "CARA"]
for r in R:
    n = len(r["u"]); E = r["edges"]; ls = np.log(r["sig_hat"])
    cs = np.full(n, np.nan)
    for i in range(n):
        inc = (E[:, 0] == i) | (E[:, 1] == i)
        if inc.any(): cs[i] = np.median(ls[inc]) - np.median(ls)
    f = r["feat"]; g = r["reg"]
    X = np.stack([np.log(np.maximum(np.nan_to_num(f["res_med"], nan=1.0), 0.05)), f["rej_frac"], np.nan_to_num(cs), np.log(np.maximum(f["deg_keep"], 1))], 1)
    r["X"] = X
    r["pi"] = 1 - f["deg_keep"].sum() / f["deg_all"].sum()
KINDS = [("none", 0.0), ("edge", 0.1), ("edge", 0.2), ("cam", 0.1), ("cam", 0.2)]
folds = sorted({r["held"] for r in R})


def fit_beta(recs):
    X = np.concatenate([r["X"][r["reg"]] for r in recs]); y = np.concatenate([np.log(np.maximum(r["err"][r["reg"]], 1e-4) / r["u"][r["reg"]]) for r in recs])
    A = np.c_[np.ones(len(X)), X]
    return np.linalg.lstsq(A, y, rcond=None)[0]


def norm(r, how, beta):
    u = r["u"].copy()
    if how == "u": return u
    if how == "res": return u * np.maximum(1, np.exp(r["X"][:, 0]))
    if how == "learn": return u * np.exp(np.c_[np.ones(len(u)), r["X"]] @ beta)


def evaluate(how, aug, mondrian=False, scene_bins=0):
    out = {k: [] for k in KINDS}
    for f in folds:
        cal = [r for r in R if r["held"] != f and (aug or r["kind"] == "none")]
        fit = [r for r in cal if r["sub"] % 2 == 0]; cr = [r for r in cal if r["sub"] % 2 == 1]
        beta = fit_beta(fit) if how == "learn" else None
        if scene_bins:
            pc = np.quantile([r["pi"] for r in cr], np.linspace(0, 1, scene_bins + 1)[1:-1])
            lamb = [crc_threshold([r["err"][r["reg"]] / norm(r, how, beta)[r["reg"]] for r in cr if np.digitize(r["pi"], pc) == b], 0.1) for b in range(scene_bins)]
        if mondrian:
            # three groups by normalizer tercile (thresholds from calibration cameras)
            vv = np.concatenate([np.log(norm(r, how, beta)[r["reg"]]) for r in cr]); cuts = np.quantile(vv, [1/3, 2/3])
            lam = []
            for gi in range(3):
                L = []
                for r in cr:
                    g = r["reg"]; v = norm(r, how, beta)[g]; grp = np.digitize(np.log(v), cuts)
                    L.append((r["err"][g] / v)[grp == gi])
                lam.append(crc_threshold([l for l in L if len(l)], 0.1))
        else:
            lam = crc_threshold([r["err"][r["reg"]] / norm(r, how, beta)[r["reg"]] for r in cr], 0.1)
        for r in R:
            if r["held"] != f: continue
            g = r["reg"]; v = norm(r, how, beta)[g]
            if scene_bins:
                B = v * lamb[np.digitize(r["pi"], pc)]
            elif mondrian:
                grp = np.digitize(np.log(v), cuts); B = v * np.array(lam)[grp]
            else:
                B = lam * v
            e = r["err"][g]; cov = e <= B
            hard = r["u"][g] >= np.quantile(r["u"][g], 0.75)
            out[(r["kind"], r["lev"])].append((cov.mean(), cov[hard].mean(), np.median(B) / D, np.mean(B > 30 * D), np.mean(cov[e > 30 * D]) if (e > 30 * D).any() else np.nan))
    return {k: np.nanmean(np.array(v), 0) for k, v in out.items()}


import sys
for how, aug, mon, sb in [("u",0,0,0),("learn",1,0,0),("learn",1,0,3),("learn",1,0,4),("u",1,0,3),("learn",1,1,0)]:
    res = evaluate(how, aug, mon, sb)
    print(f"{how:5s} aug={aug} mond={mon} sb={sb} | " + " | ".join(f"{k[0][:1]}{k[1]}: {v[0]:.3f}/{v[1]:.3f} w{v[2]:.1f} fl{v[3]:.2f}" for k, v in res.items()))
