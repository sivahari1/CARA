"""Main synthetic study: 10 seeds x 6 conditions x 20 test scenes.
Per seed: calibration scenes (clean, generic layout) -> CARA calibrations
(point rho, safe rho, static) + tuned baselines. Outputs results/synth_main.jsonl"""
from common import *
from multiprocessing import Pool
from cara.method import recal_sigma

CONDS = {
    "clean":   dict(noise_mult=1.0, p_out=0.2),
    "fog2x":   dict(noise_mult=2.0, p_out=0.2),
    "heavy3x": dict(noise_mult=3.0, p_out=0.35),
    "sparse":  dict(noise_mult=1.5, p_out=0.2, k_local=2, n_rand=60),
    "corrflip": dict(noise_mult=1.0, p_out=0.2, flip_frac=0.5),
    "large":   dict(n=300, n_rand=450, noise_mult=1.5, p_out=0.25),
}
N_TEST = int(os.environ.get("N_TEST", 20))


def one_seed(seed):
    rng = np.random.default_rng(1000 + seed)
    Dfit, De, Dr, Dc = calib_sets(rng)
    cfgs = {"CARA": Config(), "CARA-safe": Config(rho="safe"), "CARA-static": Config(rho="none")}
    cals = {k: calibrate(c, Dfit, De, Dr, Dc) for k, c in cfgs.items()}
    tuned = BL.tune(Dfit)
    rows = []
    for cname, kw in CONDS.items():
        trng = np.random.default_rng(seed * 7919 + list(CONDS).index(cname) * 101)
        for t in range(N_TEST):
            sc = make_scene(trng, **kw)
            rec = dict(seed=seed, cond=cname, scene=t, rho_true=kw.get("noise_mult", 1.0),
                       p_out=kw["p_out"], n=sc["n"], m=len(sc["edges"]))
            meth = {}
            t0 = time.time(); outs = BL.run_all(sc, tuned); tb = time.time() - t0
            for k, R in outs.items():
                meth[k] = summarize(rot_errors(sc["R"], R))
            res = {}
            for k in cfgs:
                o = run(sc, cfgs[k], cals[k]); res[k] = o
                err = rot_errors(sc["R"], o["R"])
                s = summarize(err); s["time"] = o["time"]
                s["inl_keep"] = float(o["keep"][~sc["is_out"]].mean())
                s["out_rej"] = float(1 - o["keep"][sc["is_out"]].mean()) if sc["is_out"].any() else np.nan
                s["rho"] = o["rho"]
                if k == "CARA":
                    sig_native_axis = sc["sig_hat"]
                    sig_recal_axis = recal_sigma(sc["sig_hat"], cals[k], cfgs[k]) / LOGCHI3
                    s["bounds"] = bound_record(sc, o, err, sig_native_axis, sig_recal_axis)
                    rp = res[k]["rho"]
                else:
                    reg = o["reg"]
                    s["bounds"] = {"crc": dict(cov=float((err[reg] <= o["B"][reg]).mean()),
                                               width=float(np.median(o["B"][reg]) / DEG))}
                meth[k] = s
            # drift-scaled kernel only (no gating): isolates the effect of rho
            meth["CG-rho"] = summarize(rot_errors(sc["R"], solve_cg(sc["n"], sc["edges"], sc["Rij"], c=5 * DEG * res["CARA"]["rho"])))
            meth["CG"]["time"] = tb / max(len(outs), 1)
            rec["methods"] = meth
            rows.append(rec)
    return rows, dict(seed=seed, tuned={k: float(v) for k, v in tuned.items()},
                      cal={k: dict(b=c.b, lam_edge=c.lam_edge, lam_res=c.lam_res, lam_cam=c.lam_cam) for k, c in cals.items()})


if __name__ == "__main__":
    seeds = list(range(int(sys.argv[1]) if len(sys.argv) > 1 else 10))
    t0 = time.time()
    with Pool(2) as p:
        out = p.map(one_seed, seeds)
    rows = [r for o in out for r in o[0]]
    dump(os.path.join(RES, "synth_main.jsonl"), rows)
    json.dump([o[1] for o in out], open(os.path.join(RES, "synth_main_calib.json"), "w"), indent=1, default=float)
    print("done", len(rows), "rows", time.time() - t0, "s")
