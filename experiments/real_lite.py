"""Post-hoc operating point on Notre Dame: CARA-lite (core_k=1, bridge_thr=5).
It was identified by a diagnostic on seed 0, so it is evaluated only on seeds 1-4
(same splits and test scenes as real_notredame.py). Output: results/real_lite.jsonl"""
from real_notredame import *

LITE = Config(core_k=1, bridge_thr=5.0)


def one_seed_lite(seed):
    rng = np.random.default_rng(500 + seed)
    perm = rng.permutation(DATA["n"])
    A, Bc = perm[: DATA["n"] // 2], perm[DATA["n"] // 2:]
    D = [draw(rng, A) for _ in range(20 + 120)]
    cal = calibrate(LITE, D[:20], D[20:60], D[60:100], D[100:140])
    rows = []
    for ci, (cname, kw) in enumerate(CONDS.items()):
        trng = np.random.default_rng(seed * 7919 + ci * 101)
        for t in range(N_TEST):
            sc = draw(trng, Bc, extra_noise=kw["kappa"] * AXIS, extra_out=kw["out"])
            o = run(sc, LITE, cal)
            err = rot_errors(sc["R"], o["R"])
            s = summarize(err); s["rho"] = o["rho"]
            reg = o["reg"]
            s["cov"] = float((err[reg] <= o["B"][reg]).mean()); s["width"] = float(np.median(o["B"][reg]) / DEG)
            cg = summarize(rot_errors(sc["R"], BL.run_CG(sc)))
            rows.append(dict(seed=seed, cond=cname, scene=t, lite=s, CG=cg))
    return rows


if __name__ == "__main__":
    t0 = time.time()
    with Pool(2) as p:
        out = p.map(one_seed_lite, [1, 2, 3, 4])
    dump(os.path.join(RES, "real_lite.jsonl"), [r for o in out for r in o])
    print("done", time.time() - t0)
