"""Evaluate the bound extensions (E1 augmentation, E2 learned normalizer / group-conditional thresholds)
on ETH3D + VGGT test scenes. Rotations are identical across variants; only bounds differ.
  python eval_bounds.py <npz_dir> <out.jsonl> <seed> [--cv]
--cv: every scene is a test scene; calibration uses clean sub-scenes of the other scenes (run 2 layout).
"""
import os, sys, glob, json, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "..")); sys.path.insert(0, os.path.join(H, "..", "..", "gpu_kit"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
from run_eval import load_scene
from cara.method import Config, calibrate, run
from cara.metrics import rot_errors
DEG = np.pi / 180

VARIANTS = {
    "base": Config(),
    "E1": Config(augment=True),
    "E2": Config(cam_norm="learn", cam_groups=3),
    "E1+E2a": Config(augment=True, cam_norm="learn"),
    "E1+E2": Config(augment=True, cam_norm="learn", cam_groups=3),
    "E2u": Config(cam_groups=3),          # secondary (added after run 1): groups on u, no learned normalizer
}


def stats(err, o):
    reg = o["reg"]; e = err[reg]; B = o["B"][reg]; cov = e <= B
    q = lambda x: x >= np.quantile(x, 0.75)
    hu, hv = q(o["u"][reg]), q(o["v"][reg])
    return dict(cov=float(cov.mean()), cov_hard_u=float(cov[hu].mean()), cov_hard_v=float(cov[hv].mean()),
                width=float(np.median(B) / DEG), flag=float(np.mean(B > 30 * DEG)),
                cov_cat=float(cov[e > 30 * DEG].mean()) if (e > 30 * DEG).any() else None,
                cat_reg=float(np.mean(e > 30 * DEG)), reg=float(reg.mean()),
                cams=dict(e=np.round(e / DEG, 3).tolist(), u=np.round(o["u"][reg] / DEG, 4).tolist(),
                          v=np.round(o["v"][reg] / DEG, 4).tolist(), B=np.round(B / DEG, 3).tolist()))


def main():
    npz, out, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]); cv = "--cv" in sys.argv
    files = sorted(glob.glob(os.path.join(npz, "*.npz")))
    allsc = [load_scene(f, "conf_med") for f in files]
    if cv:
        jobs = []
        for s in sorted({x["scene"] for x in allsc}):
            pool = [x for x in allsc if x["scene"] != s and x["cond"] == "clean"]
            tests = [x for x in allsc if x["scene"] == s and "__" in x["file"] and int(x["file"].split("__")[2]) < 8]
            jobs.append((pool, tests))
    else:
        jobs = [([x for x in allsc if x["file"].startswith("calib__")], [x for x in allsc if x["file"].startswith("test__")])]
    rows = []
    for pool, tests in jobs:
        rng = np.random.default_rng(seed); perm = rng.permutation(len(pool)); C = [pool[i] for i in perm]
        splits = (C[:20], C[20:60], C[60:100], C[100:140])
        cals = {k: calibrate(c, *splits) for k, c in VARIANTS.items()}
        for t in tests:
            rec = dict(seed=seed, file=t["file"], scene=t["scene"], cond=t["cond"], v={})
            for k, c in VARIANTS.items():
                o = run(t, c, cals[k]); rec["v"][k] = stats(rot_errors(t["R"], o["R"]), o)
            rows.append(rec)
        print(seed, "fold done", tests[0]["scene"] if tests else "", flush=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
