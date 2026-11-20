"""Stage 2 (CPU): calibrate CARA on 'calib' sub-scenes and evaluate all methods on 'test'
sub-scenes for every condition. Calibration and test scenes are different scenes.
  python run_eval.py --in out_imc --out results_imc.jsonl [--sigma conf_med|conf_logmean] [--seeds 5]
Each seed reshuffles which calibration sub-scenes go to the four splits (fit/edge/res/cam).
Output rows follow the schema of results/*.jsonl so experiments/analyze.py can read them.
"""
import os, sys, glob, json, argparse, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
os.environ.setdefault("GLOG_minloglevel", "2")
from cara.method import Config, calibrate, run
from cara.metrics import rot_errors, summarize
from cara.so3 import angle, T
from cara.solvers import solve_cg, DEG
from cara.graph import components
from cara import baselines as BL


def load_scene(fn, sigma_key, tau_out=5 * DEG):
    z = np.load(fn, allow_pickle=True)
    E, Rij, R = z["edges"].astype(int), z["Rij"], z["R_gt"]
    if sigma_key == "conf_med":
        sig = 1.0 / np.maximum(z["conf_med"], 1e-6)
    else:
        sig = np.exp(-z["conf_logmean"])
    # restrict to the largest connected component of the view graph (applied identically for all
    # methods; cameras outside it cannot be registered by any rotation averaging method)
    nc, lab = components(len(R), E, np.ones(len(E), bool))
    n_full = len(R)
    if nc > 1:
        keep = np.where(lab == np.bincount(lab).argmax())[0]
        loc = -np.ones(len(R), int); loc[keep] = np.arange(len(keep))
        m = (loc[E[:, 0]] >= 0) & (loc[E[:, 1]] >= 0)
        E, Rij, sig, R = loc[E[m]], Rij[m], sig[m], R[keep]
    err = angle(T(R[E[:, 1]] @ T(R[E[:, 0]])) @ Rij)
    return dict(n_full=n_full, n=len(R), R=R, edges=E, Rij=Rij, sig_hat=sig, is_out=err > tau_out,
                cond=str(z["cond"]), scene=str(z["scene"]), file=os.path.basename(fn))


def evaluate(sc, cfgs, cals, ext=True):
    meth = {}
    meth["CG"] = summarize(rot_errors(sc["R"], BL.run_CG(sc)))
    meth["ConfW"] = summarize(rot_errors(sc["R"], BL.run_ConfW(sc)))
    if ext:
        from cara.external import run_glomap_ra, run_shonan
        meth["GLOMAP-RA"] = summarize(rot_errors(sc["R"], run_glomap_ra(sc)))
        meth["Shonan-H"] = summarize(rot_errors(sc["R"], run_shonan(sc)))
    for k in cfgs:
        o = run(sc, cfgs[k], cals[k])
        err = rot_errors(sc["R"], o["R"])
        s = summarize(err); s["rho"] = o["rho"]; reg = o["reg"]
        cov = err[reg] <= o["B"][reg]
        hard = o["u"][reg] >= np.quantile(o["u"][reg], 0.75)
        s["bounds"] = {"crc": dict(cov=float(cov.mean()), cov_hard=float(cov[hard].mean()),
                                   width=float(np.median(o["B"][reg]) / DEG))}
        s["inl_keep"] = float(o["keep"][~sc["is_out"]].mean())
        s["out_rej"] = float(1 - o["keep"][sc["is_out"]].mean()) if sc["is_out"].any() else float("nan")
        meth[k] = s
    meth["CG-rho"] = summarize(rot_errors(sc["R"], solve_cg(sc["n"], sc["edges"], sc["Rij"], c=5 * DEG * meth["CARA"]["rho"])))
    return meth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sigma", default="conf_med")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--no-ext", action="store_true")
    ap.add_argument("--only-seed", type=int, default=None, help="run a single seed (for parallel runs)")
    ap.add_argument("--calib-in", default=None, help="folder with calib__*.npz if different from --in (cross-dataset)")
    a = ap.parse_args()
    calib = [load_scene(f, a.sigma) for f in sorted(glob.glob(os.path.join(a.calib_in or a.inp, "calib__*__clean.npz")))]
    tests = [load_scene(f, a.sigma) for f in sorted(glob.glob(os.path.join(a.inp, "test__*.npz")))]
    assert len(calib) >= 140, f"need >= 140 calibration sub-scenes, found {len(calib)}"
    assert not ({c["scene"] for c in calib} & {t["scene"] for t in tests}), "calibration and test scenes overlap"
    cfgs = {"CARA": Config(), "CARA-static": Config(rho="none")}
    rows = []
    for seed in ([a.only_seed] if a.only_seed is not None else range(a.seeds)):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(calib))
        C = [calib[i] for i in perm]
        splits = (C[:20], C[20:60], C[60:100], C[100:140])
        cals = {k: calibrate(c, *splits) for k, c in cfgs.items()}
        for t in tests:
            t0 = time.time()
            meth = evaluate(t, cfgs, cals, ext=not a.no_ext)
            rows.append(dict(seed=seed, cond=t["cond"], scene=t["file"], base_scene=t["scene"], n=t["n"], n_full=t["n_full"],
                             m=len(t["edges"]), frac_out=float(t["is_out"].mean()), rho_true=float("nan"),
                             methods=meth, time=time.time() - t0))
            print(seed, t["file"], round(meth["CARA"]["auc5"], 3), round(meth["CG"]["auc5"], 3), flush=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
