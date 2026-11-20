"""Baselines.

L2       spectral + L2 IRLS (chordal / least squares; Martinec & Pajdla 2007)
L1       spectral + L1-IRLS (Chatterjee & Govindu 2013)
CG       spectral + L1-IRLS + Geman-McClure IRLS (Chatterjee & Govindu, TPAMI 2018)
ConfW    CG with native prior weights 1/sigma_hat^2 (isotropic analogue of
         uncertainty-aware averaging, Zhang et al. CVPR 2023)
CycFilt  cycle-consistency filtering with a tuned fixed threshold (Zach et al. 2010
         style), largest component, then CG
ConfGate drop the least-confident edges (tuned percentile of sigma_hat; a
         confidence-only analogue of GeoCond-style gating), largest component, CG
Tuned thresholds are chosen on the D_fit split by maximizing mean AUC@5.
"""
import numpy as np
from .solvers import spectral, irls, solve_cg, DEG
from .graph import Triangles, per_edge_quantile, components
from .metrics import rot_errors, auc


def _largest(n, edges, keep):
    _, lab = components(n, edges, keep)
    return np.where(lab == np.bincount(lab).argmax())[0]


def _finish(n, R, nodes):
    out = np.full((n, 3, 3), np.nan)
    out[nodes] = R[nodes]
    return out


def run_L2(sc, **k):
    n, E, Rij = sc["n"], sc["edges"], sc["Rij"]
    return irls(n, E, Rij, spectral(n, E, Rij), None, "l2", iters=5)


def run_L1(sc, **k):
    n, E, Rij = sc["n"], sc["edges"], sc["Rij"]
    return irls(n, E, Rij, spectral(n, E, Rij), None, "l1", iters=40)


def run_CG(sc, **k):
    return solve_cg(sc["n"], sc["edges"], sc["Rij"])


def run_ConfW(sc, **k):
    w = 1.0 / sc["sig_hat"] ** 2
    return solve_cg(sc["n"], sc["edges"], sc["Rij"], w / w.mean())


def raw_cycle_score(sc, tri=None):
    tri = tri or Triangles(sc["n"], sc["edges"], 30, np.random.default_rng(1))
    return per_edge_quantile(tri.e, tri.cycle_angles(sc["Rij"]), tri.m, 0.5)


def run_CycFilt(sc, thr, tri=None, **k):
    s = raw_cycle_score(sc, tri)
    keep = np.isnan(s) | (s <= thr)       # edges without triangles cannot be tested: kept
    nodes = _largest(sc["n"], sc["edges"], keep)
    R = solve_cg(sc["n"], sc["edges"][keep], sc["Rij"][keep], nodes=nodes)
    return _finish(sc["n"], R, nodes)


def run_ConfGate(sc, pct, **k):
    keep = sc["sig_hat"] <= np.percentile(sc["sig_hat"], 100 - pct)
    nodes = _largest(sc["n"], sc["edges"], keep)
    R = solve_cg(sc["n"], sc["edges"][keep], sc["Rij"][keep], nodes=nodes)
    return _finish(sc["n"], R, nodes)


def tune(D_fit, have_conf=True):
    grid_c = np.array([1, 2, 3, 5, 8, 12, 20, 30, 45, 90]) * DEG
    best_c, bv = None, -1
    for t in grid_c:
        v = np.mean([auc(rot_errors(sc["R"], run_CycFilt(sc, t)), 5) for sc in D_fit])
        if v > bv:
            best_c, bv = t, v
    out = dict(cyc_thr=best_c)
    if have_conf:
        best_p, bv = None, -1
        for p in [0, 10, 20, 30, 40]:
            v = np.mean([auc(rot_errors(sc["R"], run_ConfGate(sc, p)), 5) for sc in D_fit])
            if v > bv:
                best_p, bv = p, v
        out["conf_pct"] = best_p
    return out


def run_all(sc, tuned, have_conf=True):
    out = dict(L2=run_L2(sc), L1=run_L1(sc), CG=run_CG(sc))
    if have_conf:
        out["ConfW"] = run_ConfW(sc)
        out["ConfGate"] = run_ConfGate(sc, tuned["conf_pct"])
    out["CycFilt"] = run_CycFilt(sc, tuned["cyc_thr"])
    return out
