import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from dataclasses import replace, asdict
from cara.method import Config, calibrate, run
from cara.sim import make_scene
from cara.metrics import rot_errors, summarize
from cara import baselines as BL
from cara.solvers import solve_cg, DEG
from cara.graph import components

RES = os.path.join(os.path.dirname(__file__), "..", "results")
CHI3_Q90 = 2.5003       # sqrt of chi2_3 0.9 quantile
LOGCHI3 = 1.4404        # exp(E log chi_3): typical-angle -> per-axis std


def calib_sets(rng, n_fit=20, n_each=40, **kw):
    D = [make_scene(rng, **kw) for _ in range(n_fit + 3 * n_each)]
    a, b, c = n_fit, n_fit + n_each, n_fit + 2 * n_each
    return D[:a], D[a:b], D[b:c], D[c:]


def bound_record(sc, out, err, sig_native, sig_recal_axis):
    """coverage / width of per-camera bounds for CARA's estimate."""
    E = sc["edges"]
    reg, keep = out["reg"], out["keep"]
    kk = keep & reg[E[:, 0]] & reg[E[:, 1]]

    def u_from(sig):
        info = np.zeros(sc["n"])
        np.add.at(info, E[kk, 0], sig[kk] ** -2)
        np.add.at(info, E[kk, 1], sig[kk] ** -2)
        return 1 / np.sqrt(np.maximum(info, 1e-300))
    rec = {}
    e = err[reg]
    for name, B in [("crc", out["B"]),
                    ("gauss_native", CHI3_Q90 * u_from(sig_native)),
                    ("gauss_recal", CHI3_Q90 * u_from(sig_recal_axis))]:
        b = B[reg]
        cov = e <= b
        hard = out["u"][reg] >= np.quantile(out["u"][reg], 0.75)   # least-informed quartile
        rec[name] = dict(cov=float(cov.mean()), cov_hard=float(cov[hard].mean()),
                         width=float(np.median(b) / DEG))
    return rec


def dump(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def load(path):
    return [json.loads(l) for l in open(path)]
