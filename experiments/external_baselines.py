"""Released-implementation baselines on the same test scenes as the main studies
(deterministic regeneration): GLOMAP rotation averaging (pycolmap 4.2) and
Shonan averaging with Huber loss (GTSAM). Outputs results/external_synth.jsonl,
results/external_real.jsonl, results/external_full.json"""
import os
os.environ.setdefault("GLOG_minloglevel", "2")
from common import *
from multiprocessing import Pool
import synthetic_main as SM
import real_notredame as RN
from cara.external import run_glomap_ra, run_shonan


def ext(sc):
    out = {}
    for name, f in [("GLOMAP-RA", run_glomap_ra), ("Shonan-H", run_shonan)]:
        t0 = time.time(); R = f(sc); dt = time.time() - t0
        s = summarize(rot_errors(sc["R"], R)); s["time"] = dt; out[name] = s
    return out


def synth_seed(seed):
    rows = []
    for ci, (cname, kw) in enumerate(SM.CONDS.items()):
        trng = np.random.default_rng(seed * 7919 + ci * 101)
        for t in range(20):
            sc = make_scene(trng, **kw)
            rows.append(dict(seed=seed, cond=cname, scene=t, **ext(sc)))
    return rows


def real_seed(seed):
    rng = np.random.default_rng(500 + seed)
    perm = rng.permutation(RN.DATA["n"])
    A, Bc = perm[: RN.DATA["n"] // 2], perm[RN.DATA["n"] // 2:]
    _ = [RN.draw(rng, A) for _ in range(140)]      # consume the same random stream as the main study
    rows = []
    for ci, (cname, kw) in enumerate(RN.CONDS.items()):
        trng = np.random.default_rng(seed * 7919 + ci * 101)
        for t in range(20):
            sc = RN.draw(trng, Bc, extra_noise=kw["kappa"] * RN.AXIS, extra_out=kw["out"])
            rows.append(dict(seed=seed, cond=cname, scene=t, **ext(sc)))
    return rows


if __name__ == "__main__":
    t0 = time.time()
    with Pool(2) as p:
        s = p.map(synth_seed, range(10))
        r = p.map(real_seed, range(5))
    dump(os.path.join(RES, "external_synth.jsonl"), [x for o in s for x in o])
    dump(os.path.join(RES, "external_real.jsonl"), [x for o in r for x in o])
    D = RN.DATA
    full = dict(n=D["n"], R=D["R"], edges=D["edges"], Rij=D["Rij"], sig_hat=np.ones(len(D["edges"])))
    json.dump(ext(full), open(os.path.join(RES, "external_full.json"), "w"), indent=1, default=float)
    print("done", time.time() - t0)
