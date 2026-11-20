"""Run-2 evaluation (confirmatory): accuracy with the pre-registered and the learned edge uncertainty (E3),
and downstream translation averaging (E5). Leave-one-scene-out over all ETH3D scenes.
  python eval_v2.py <npz_dir> <out.jsonl> <seed> [--no-ext]
Baselines that do not depend on calibration are computed only for seed 0.
"""
import os, sys, glob, json, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "..")); sys.path.insert(0, os.path.join(H, "..", "..", "gpu_kit"))
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("GLOG_minloglevel", "2")
from run_eval import load_scene
from cara.method import Config, calibrate, run, edge_true_errors
from cara.metrics import rot_errors, summarize
from cara.solvers import solve_cg
from cara import baselines as BL
from cara import edge_uncertainty as EU
from cara import translation as TR
from sklearn.metrics import roc_auc_score
DEG = np.pi / 180


def attach(sc):
    """add run-2 fields (features, translations, GT centres) to a scene from load_scene, respecting its
    largest-component restriction"""
    z = np.load(os.path.join(NPZ, sc["file"]), allow_pickle=True)
    E0 = z["edges"].astype(int); R0 = z["R_gt"]
    # map load_scene's (possibly reduced) cameras back to the file's cameras by matching rotations
    idx = [int(np.argmin([np.abs(R0[k] - Ri).sum() for k in range(len(R0))])) for Ri in sc["R"]]
    loc = -np.ones(len(R0), int); loc[idx] = np.arange(len(idx))
    m = (loc[E0[:, 0]] >= 0) & (loc[E0[:, 1]] >= 0)
    sc = dict(sc)
    sc["F"] = EU.edge_features(z)[m]
    ext = z["ext"][m]; sc["tij"] = ext[:, 1, :, 3] - np.einsum("mab,mb->ma", EU.rel_rot(ext), ext[:, 0, :, 3])
    sc["C_gt"] = TR.centres_gt(z["R_gt"][idx], z["t_gt"][idx])
    assert np.array_equal(loc[E0[m]], sc["edges"]), "edge order mismatch"
    return sc


def trans_eval(sc, R, keep=None):
    reg = np.all(np.isfinite(R.reshape(len(R), -1)), 1)
    E = sc["edges"]; ok = reg[E[:, 0]] & reg[E[:, 1]]
    if keep is not None:
        ok &= keep
    if ok.sum() < 3:
        return dict(pos_med=None, pos_auc=0.0)
    Rs = np.where(reg[:, None, None], R, np.eye(3))
    d = TR.directions(E[ok], sc["tij"][ok], Rs)
    C = TR.lud(sc["n"], E[ok], d, nodes=np.unique(E[ok]))
    err = TR.align_error(C, sc["C_gt"])
    e = np.where(np.isfinite(err), err, np.inf)
    ths = np.linspace(0, 0.2, 41)
    return dict(pos_med=float(np.median(e)) if np.isfinite(np.median(e)) else None,
                pos_auc=float(np.trapezoid([np.mean(e <= t) for t in ths], ths) / 0.2), pos_cat=float(np.mean(e > 0.5)))


def main():
    global NPZ
    NPZ, out, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]); ext = "--no-ext" not in sys.argv
    allsc = [attach(load_scene(f, "conf_med")) for f in sorted(glob.glob(os.path.join(NPZ, "cv__*.npz")))]
    rows = []
    for s in sorted({x["scene"] for x in allsc})[:int(os.environ.get("LIMIT_SCENES", 99))]:
        pool = [x for x in allsc if x["scene"] != s and x["cond"] == "clean"]
        tests = [x for x in allsc if x["scene"] == s and int(x["file"].split("__")[2]) < 8]
        rng = np.random.default_rng(seed); perm = rng.permutation(len(pool)); C = [pool[i] for i in perm]
        splits = (C[:20], C[20:60], C[60:100], C[100:140])
        beta = EU.fit([x["F"] for x in splits[0]], [edge_true_errors(x) for x in splits[0]])
        learned = lambda x: dict(x, sig_hat=EU.predict(beta, x["F"]))
        cfg = Config()
        cal_f1 = calibrate(cfg, *splits)
        cal_ln = calibrate(cfg, *[[learned(x) for x in sp] for sp in splits])
        for t in tests:
            tl = learned(t); meth = {}; tr = {}
            o1 = run(t, cfg, cal_f1); o2 = run(tl, cfg, cal_ln)
            meth["CARA"] = summarize(rot_errors(t["R"], o1["R"])); meth["CARA-learnsig"] = summarize(rot_errors(t["R"], o2["R"]))
            meth["ConfW-learnsig"] = summarize(rot_errors(t["R"], BL.run_ConfW(tl)))
            meth["CG-rho"] = summarize(rot_errors(t["R"], solve_cg(t["n"], t["edges"], t["Rij"], c=5 * DEG * o1["rho"])))
            tr["CARA"] = trans_eval(t, o1["R"]); tr["CARA+gate"] = trans_eval(t, o1["R"], o1["keep"])
            tr["CARA-learnsig+gate"] = trans_eval(t, o2["R"], o2["keep"])
            if seed == 0:
                Rcg = BL.run_CG(t); meth["CG"] = summarize(rot_errors(t["R"], Rcg)); tr["CG"] = trans_eval(t, Rcg)
                meth["ConfW"] = summarize(rot_errors(t["R"], BL.run_ConfW(t)))
                tr["GT-rot"] = trans_eval(t, t["R"])
                if ext:
                    from cara.external import run_glomap_ra, run_shonan
                    Rg = run_glomap_ra(t); meth["GLOMAP-RA"] = summarize(rot_errors(t["R"], Rg)); tr["GLOMAP-RA"] = trans_eval(t, Rg)
                    meth["Shonan-H"] = summarize(rot_errors(t["R"], run_shonan(t)))
            y = t["is_out"]
            au = {}
            if 0 < y.sum() < len(y):
                for k, nm in enumerate(EU.NAMES):
                    au[nm] = float(roc_auc_score(y, t["F"][:, k]))
                au["learned"] = float(roc_auc_score(y, EU.predict(beta, t["F"])))
            rows.append(dict(seed=seed, scene=t["scene"], file=t["file"], cond=t["cond"], n=t["n"], m=len(t["edges"]),
                             frac_out=float(y.mean()), methods=meth, trans=tr, auroc=au, beta=beta.tolist()))
        print(seed, s, "done", flush=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
