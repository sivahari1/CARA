"""Bound extensions on the synthetic conditions of the main study (same generator and seeds)."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *
from synthetic_main import CONDS
from cara.method import Config, calibrate, run
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_bounds import stats
V = {"base": Config(), "E1": Config(augment=True), "E2": Config(cam_norm="learn", cam_groups=3), "E1+E2a": Config(augment=True, cam_norm="learn"), "E1+E2": Config(augment=True, cam_norm="learn", cam_groups=3)}
N_TEST = int(os.environ.get("N_TEST", 10))
seed = int(sys.argv[1]); out = sys.argv[2]
rng = np.random.default_rng(1000 + seed)
Dfit, De, Dr, Dc = calib_sets(rng)
t0 = time.time(); cals = {k: calibrate(c, Dfit, De, Dr, Dc) for k, c in V.items()}; print("calib", time.time() - t0, flush=True)
rows = []
for cname, kw in CONDS.items():
    trng = np.random.default_rng(seed * 7919 + list(CONDS).index(cname) * 101)
    for t in range(N_TEST):
        sc = make_scene(trng, **kw)
        rec = dict(seed=seed, cond=cname, scene=t, v={})
        for k, c in V.items():
            o = run(sc, c, cals[k]); rec["v"][k] = stats(rot_errors(sc["R"], o["R"]), o)
        rows.append(rec)
    print(cname, flush=True)
with open(out, "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
