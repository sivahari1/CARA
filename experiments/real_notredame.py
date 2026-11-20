"""Real-data study on 1DSfM Notre Dame (715 cams, 64,678 relative rotations).

Protocol: cameras are split at random into two disjoint halves per seed.
Calibration scenes (random 100-camera induced sub-graphs) come only from half A;
test scenes only from half B, so no camera or edge is shared between
calibration and test. Shift at test time: extra per-axis Gaussian noise kappa x
(typical inlier per-axis std, 0.96 deg) and/or injected random outliers.
Also: whole-graph accuracy (715 cameras) for each method.
Outputs results/real_nd.jsonl and results/real_nd_full.json
"""
from common import *
from multiprocessing import Pool
from cara.real import load_notredame, subscene

DATA = load_notredame()
AXIS = 1.48 / 1.54 * DEG   # median inlier angle -> per-axis std (Maxwell median 1.54)
CONDS = {
    "clean":      dict(kappa=0.0, out=0.0),
    "noise1.5":   dict(kappa=1.5, out=0.0),
    "noise3":     dict(kappa=3.0, out=0.0),
    "out15":      dict(kappa=0.0, out=0.15),
    "noise1.5+out15": dict(kappa=1.5, out=0.15),
}
N_SUB, N_TEST = 100, int(os.environ.get("N_TEST", 20))


def draw(rng, pool, **kw):
    return subscene(DATA, rng.choice(pool, N_SUB, replace=False), rng, **kw)


def one_seed(seed):
    rng = np.random.default_rng(500 + seed)
    perm = rng.permutation(DATA["n"])
    A, Bc = perm[: DATA["n"] // 2], perm[DATA["n"] // 2:]
    D = [draw(rng, A) for _ in range(20 + 120)]
    Dfit, De, Dr, Dc = D[:20], D[20:60], D[60:100], D[100:140]
    cfgs = {"CARA": Config(), "CARA-safe": Config(rho="safe"), "CARA-static": Config(rho="none")}
    cals = {k: calibrate(c, Dfit, De, Dr, Dc) for k, c in cfgs.items()}
    tuned = BL.tune(Dfit, have_conf=False)
    rows = []
    for ci, (cname, kw) in enumerate(CONDS.items()):
        trng = np.random.default_rng(seed * 7919 + ci * 101)
        for t in range(N_TEST):
            sc = draw(trng, Bc, extra_noise=kw["kappa"] * AXIS, extra_out=kw["out"])
            rec = dict(seed=seed, cond=cname, scene=t, kappa=kw["kappa"], out=kw["out"],
                       rho_true=float(np.sqrt(1 + kw["kappa"] ** 2)), n=sc["n"], m=len(sc["edges"]),
                       frac_out=float(sc["is_out"].mean()))
            meth = {}
            for k, R in BL.run_all(sc, tuned, have_conf=False).items():
                meth[k] = summarize(rot_errors(sc["R"], R))
            for k in cfgs:
                o = run(sc, cfgs[k], cals[k])
                err = rot_errors(sc["R"], o["R"])
                s = summarize(err); s["time"] = o["time"]; s["rho"] = o["rho"]
                s["inl_keep"] = float(o["keep"][~sc["is_out"]].mean())
                s["out_rej"] = float(1 - o["keep"][sc["is_out"]].mean()) if sc["is_out"].any() else np.nan
                reg = o["reg"]
                cov = err[reg] <= o["B"][reg]
                hard = o["u"][reg] >= np.quantile(o["u"][reg], 0.75)
                s["bounds"] = {"crc": dict(cov=float(cov.mean()), cov_hard=float(cov[hard].mean()),
                                           width=float(np.median(o["B"][reg]) / DEG))}
                meth[k] = s
                if k == "CARA":
                    rho_hat = o["rho"]
            meth["CG-rho"] = summarize(rot_errors(sc["R"], solve_cg(sc["n"], sc["edges"], sc["Rij"], c=5 * DEG * rho_hat)))
            rec["methods"] = meth
            rows.append(rec)
    return rows, dict(seed=seed, tuned={k: float(v) for k, v in tuned.items()},
                      cal={k: dict(b=c.b, const=c.const_sigma, lam_edge=c.lam_edge, lam_res=c.lam_res, lam_cam=c.lam_cam) for k, c in cals.items()}), (cals["CARA"], tuned)


def full_graph(cal, tuned):
    sc = dict(n=DATA["n"], R=DATA["R"], edges=DATA["edges"], Rij=DATA["Rij"],
              sig_hat=np.ones(len(DATA["edges"])), is_out=DATA["err"] > 5 * DEG)
    out = {}
    for name, f in [("L1", lambda: BL.run_L1(sc)), ("CG", lambda: BL.run_CG(sc)),
                    ("CycFilt", lambda: BL.run_CycFilt(sc, tuned["cyc_thr"]))]:
        t0 = time.time(); R = f(); dt = time.time() - t0
        s = summarize(rot_errors(sc["R"], R)); s["time"] = dt; out[name] = s
    t0 = time.time(); o = run(sc, Config(), cal); dt = time.time() - t0
    s = summarize(rot_errors(sc["R"], o["R"])); s["time"] = dt; s["rho"] = o["rho"]
    s["inl_keep"] = float(o["keep"][~sc["is_out"]].mean()); s["out_rej"] = float(1 - o["keep"][sc["is_out"]].mean())
    out["CARA"] = s
    out["_info"] = dict(n=sc["n"], m=len(sc["edges"]), frac_out_5deg=float(sc["is_out"].mean()),
                        median_edge_err=float(np.median(DATA["err"]) / DEG))
    return out


if __name__ == "__main__":
    seeds = list(range(int(sys.argv[1]) if len(sys.argv) > 1 else 5))
    t0 = time.time()
    with Pool(2) as p:
        out = p.map(one_seed, seeds)
    dump(os.path.join(RES, "real_nd.jsonl"), [r for o in out for r in o[0]])
    json.dump([o[1] for o in out], open(os.path.join(RES, "real_nd_calib.json"), "w"), indent=1, default=float)
    full = full_graph(*out[0][2])
    json.dump(full, open(os.path.join(RES, "real_nd_full.json"), "w"), indent=1, default=float)
    print("done", time.time() - t0)
