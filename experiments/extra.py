"""Ablations, doppelganger study, robustness grid, runtime scaling.
usage: python extra.py {ablation|dop|grid|runtime}"""
from common import *
from multiprocessing import Pool

ABL = {
    "full": Config(),
    "no-recal": Config(recal=False),
    "median-tri": Config(tri_q=0.5),
    "one-round": Config(rounds=1),
    "no-resid-gate": Config(resid_gate=False),
    "no-clip": Config(clip=False),
    "naive-bridge": Config(bridge="naive"),
    "no-core": Config(core_k=0),
    "rho-none": Config(rho="none"),
    "rho-safe": Config(rho="safe"),
    "no-kernel-adapt": Config(adapt_kernel=False),
    "hard-final": Config(final="hard"),
    "alpha-0.05": Config(alpha_edge=0.05, alpha_res=0.05),
    "alpha-0.20": Config(alpha_edge=0.2, alpha_res=0.2),
}
ABL_CONDS = {"clean": dict(noise_mult=1.0, p_out=0.2),
             "fog2x": dict(noise_mult=2.0, p_out=0.2),
             "heavy3x": dict(noise_mult=3.0, p_out=0.35)}


def evalrun(sc, cfg, cal):
    o = run(sc, cfg, cal)
    err = rot_errors(sc["R"], o["R"])
    s = summarize(err)
    reg = o["reg"]
    s["cov"] = float((err[reg] <= o["B"][reg]).mean())
    s["width"] = float(np.median(o["B"][reg]) / DEG)
    s["inl_keep"] = float(o["keep"][~sc["is_out"]].mean())
    s["out_rej"] = float(1 - o["keep"][sc["is_out"]].mean()) if sc["is_out"].any() else np.nan
    s["rho"] = o["rho"]; s["time"] = o["time"]
    return s


def ablation_seed(seed):
    rng = np.random.default_rng(2000 + seed)
    sets = calib_sets(rng)
    rows = []
    tests = {c: [make_scene(np.random.default_rng(seed * 31 + i * 7 + j), **kw) for j in range(10)]
             for i, (c, kw) in enumerate(ABL_CONDS.items())}
    for name, cfg in ABL.items():
        cal = calibrate(cfg, *sets)
        for c, T_ in tests.items():
            for j, sc in enumerate(T_):
                rows.append(dict(seed=seed, variant=name, cond=c, scene=j, **evalrun(sc, cfg, cal)))
    return rows


def dop_seed(seed):
    rng = np.random.default_rng(3000 + seed)
    base = dict(layout="ring", p_out=0.1, noise_mult=1.0)
    sets = calib_sets(rng, **base)
    cfgs = {"CARA": Config(), "CARA-median": Config(tri_q=0.5)}
    cals = {k: calibrate(c, *sets) for k, c in cfgs.items()}
    tuned = BL.tune(sets[0])
    rows = []
    for fi, f in enumerate([0.0, 0.1, 0.2, 0.3, 0.5]):
        trng = np.random.default_rng(seed * 97 + fi)
        for j in range(10):
            sc = make_scene(trng, dop_frac=f, **base)
            rec = dict(seed=seed, dop_frac=f, scene=j, n_dop=int(sc["dop"].sum()), m=len(sc["edges"]))
            for k, R in [("CG", BL.run_CG(sc)), ("CycFilt", BL.run_CycFilt(sc, tuned["cyc_thr"]))]:
                rec[k] = summarize(rot_errors(sc["R"], R))
            for k in cfgs:
                o = run(sc, cfgs[k], cals[k])
                err = rot_errors(sc["R"], o["R"])
                s = summarize(err)
                s["dop_rej"] = float(1 - o["keep"][sc["dop"]].mean()) if sc["dop"].any() else np.nan
                reg = o["reg"]; s["cov"] = float((err[reg] <= o["B"][reg]).mean())
                rec[k] = s
            rows.append(rec)
    return rows


def grid_seed(seed):
    rng = np.random.default_rng(4000 + seed)
    sets = calib_sets(rng)
    cfg = Config(); cal = calibrate(cfg, *sets)
    rows = []
    for a, nm in enumerate([1.0, 1.5, 2.0, 3.0]):
        for b, po in enumerate([0.1, 0.2, 0.3, 0.4]):
            trng = np.random.default_rng(seed * 1009 + a * 17 + b)
            for j in range(6):
                sc = make_scene(trng, noise_mult=nm, p_out=po)
                s = evalrun(sc, cfg, cal)
                cg = summarize(rot_errors(sc["R"], BL.run_CG(sc)))
                cgr = summarize(rot_errors(sc["R"], solve_cg(sc["n"], sc["edges"], sc["Rij"], c=5 * DEG * s["rho"])))
                rows.append(dict(seed=seed, noise=nm, p_out=po, scene=j, CARA=s, CG=cg, CGrho=cgr))
    return rows


def runtime():
    rng = np.random.default_rng(5)
    sets = calib_sets(rng)
    cfg = Config(); cal = calibrate(cfg, *sets)
    rows = []
    for n in [50, 100, 200, 400, 700]:
        for j in range(3):
            sc = make_scene(rng, n=n, n_rand=int(1.5 * n), noise_mult=1.5, p_out=0.2)
            t0 = time.time(); BL.run_CG(sc); tcg = time.time() - t0
            t0 = time.time(); run(sc, cfg, cal); tca = time.time() - t0
            rows.append(dict(n=n, m=len(sc["edges"]), t_cg=tcg, t_cara=tca))
            print(rows[-1], flush=True)
    return rows


if __name__ == "__main__":
    what = sys.argv[1]
    t0 = time.time()
    if what == "runtime":
        dump(os.path.join(RES, "runtime.jsonl"), runtime())
    else:
        f, ns = dict(ablation=(ablation_seed, 5), dop=(dop_seed, 5), grid=(grid_seed, 3))[what]
        with Pool(2) as p:
            out = p.map(f, range(ns))
        dump(os.path.join(RES, f"{what}.jsonl"), [r for o in out for r in o])
    print("done", what, time.time() - t0)
