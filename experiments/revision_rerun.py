"""Revision re-runs (deterministic re-generation of the same scenes):
(1) CycFilt v2 (no-triangle edges kept, threshold grid up to 90 deg) on the
    synthetic main study and the Notre Dame study, patched into the jsonl files;
(2) Notre Dame: CG and CARA errors on CARA's registered cameras (fair P95).
Outputs results/revision_synth_cycfilt.jsonl, results/revision_real.jsonl"""
from common import *
from multiprocessing import Pool
import synthetic_main as SM
import real_notredame as RN


def synth_seed(seed):
    rng = np.random.default_rng(1000 + seed)
    Dfit, _, _, _ = calib_sets(rng)
    tuned = BL.tune(Dfit)
    out = []
    for ci, (cname, kw) in enumerate(SM.CONDS.items()):
        trng = np.random.default_rng(seed * 7919 + ci * 101)
        for t in range(20):
            sc = make_scene(trng, **kw)
            out.append(dict(seed=seed, cond=cname, scene=t, thr=float(tuned["cyc_thr"]),
                            CycFilt=summarize(rot_errors(sc["R"], BL.run_CycFilt(sc, tuned["cyc_thr"])))))
    return out


def real_seed(seed):
    rng = np.random.default_rng(500 + seed)
    perm = rng.permutation(RN.DATA["n"])
    A, Bc = perm[: RN.DATA["n"] // 2], perm[RN.DATA["n"] // 2:]
    D = [RN.draw(rng, A) for _ in range(140)]
    cal = calibrate(Config(), D[:20], D[20:60], D[60:100], D[100:140])
    tuned = BL.tune(D[:20], have_conf=False)
    out = []
    for ci, (cname, kw) in enumerate(RN.CONDS.items()):
        trng = np.random.default_rng(seed * 7919 + ci * 101)
        for t in range(20):
            sc = RN.draw(trng, Bc, extra_noise=kw["kappa"] * RN.AXIS, extra_out=kw["out"])
            o = run(sc, Config(), cal)
            e = rot_errors(sc["R"], o["R"]); ecg = rot_errors(sc["R"], BL.run_CG(sc))
            reg = o["reg"]
            out.append(dict(seed=seed, cond=cname, scene=t, thr=float(tuned["cyc_thr"]),
                            CycFilt=summarize(rot_errors(sc["R"], BL.run_CycFilt(sc, tuned["cyc_thr"]))),
                            cara_p95_reg=float(np.percentile(e[reg], 95) / DEG), cg_p95_reg=float(np.percentile(ecg[reg], 95) / DEG),
                            cara_med_reg=float(np.median(e[reg]) / DEG), cg_med_reg=float(np.median(ecg[reg]) / DEG),
                            cg_p95_abst=float(np.percentile(ecg[~reg], 95) / DEG) if (~reg).sum() > 1 else None,
                            cg_frac_ok_abst=float(np.mean(ecg[~reg] < 5 * DEG)) if (~reg).any() else None))
    return out


if __name__ == "__main__":
    t0 = time.time()
    with Pool(2) as p:
        s = p.map(synth_seed, range(10))
        r = p.map(real_seed, range(5))
    dump(os.path.join(RES, "revision_synth_cycfilt.jsonl"), [x for o in s for x in o])
    dump(os.path.join(RES, "revision_real.jsonl"), [x for o in r for x in o])
    print("done", time.time() - t0)
