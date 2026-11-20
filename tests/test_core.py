import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from cara.so3 import expm, logm, angle, random_rotations, T
from cara.graph import Triangles, per_edge_quantile
from cara.solvers import spectral, solve_cg
from cara.metrics import rot_errors
from cara.method import crc_threshold, rho_bracket, quantile
from cara.sim import make_scene

DEG = np.pi / 180


def test_so3_roundtrip():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(50, 3))
    w = w / np.linalg.norm(w, axis=1, keepdims=True) * rng.uniform(0, 3, (50, 1))
    assert np.allclose(logm(expm(w)), w, atol=1e-8)
    assert np.allclose(angle(expm(w)), np.linalg.norm(w, axis=1), atol=1e-7)


def test_noiseless_recovery():
    rng = np.random.default_rng(1)
    sc = make_scene(rng, n=60, p_out=0.0, sig_med=1e-9, spread=0.0)
    R = spectral(sc["n"], sc["edges"], sc["Rij"])
    assert rot_errors(sc["R"], R).max() < 1e-6
    tri = Triangles(sc["n"], sc["edges"])
    assert tri.cycle_angles(sc["Rij"]).max() < 1e-6


def test_robust_solver_with_outliers():
    rng = np.random.default_rng(2)
    sc = make_scene(rng, n=80, p_out=0.2)
    e = rot_errors(sc["R"], solve_cg(sc["n"], sc["edges"], sc["Rij"]))
    assert np.median(e) < 3 * DEG


def test_per_edge_quantile():
    te = np.array([0, 0, 0, 2, 2])
    v = np.array([3.0, 1.0, 2.0, 5.0, 4.0])
    q = per_edge_quantile(te, v, 3, 0.5)
    assert q[0] == 2.0 and np.isnan(q[1]) and q[2] == 4.0


def test_crc_validity_monte_carlo():
    """E[loss on a new scene] <= alpha for exchangeable scenes (Theorem 1)."""
    rng = np.random.default_rng(3)
    alpha, losses = 0.1, []
    for t in range(400):
        scales = rng.lognormal(0, 0.3, 31)          # scene-level heterogeneity
        scenes = [rng.exponential(s, rng.integers(20, 60)) for s in scales]
        lam = crc_threshold(scenes[:30], alpha)
        losses.append(np.mean(scenes[30] > lam))
    assert np.mean(losses) <= alpha + 0.01, np.mean(losses)


def test_rho_bracket_contains_truth():
    rng = np.random.default_rng(4)
    ref = np.sort(rng.gamma(3, 1, 200000))
    for rho, eps in [(1.0, 0.2), (2.0, 0.3), (3.0, 0.35)]:
        inl = rho * rng.gamma(3, 1, int(100000 * (1 - eps)))
        out = rng.uniform(0, 50, int(100000 * eps))     # adversarial-ish outliers
        s = np.concatenate([inl, out])
        lo, hi = rho_bracket(s, ref, 0.5, 0.4)
        assert lo <= rho * 1.02 and hi >= rho * 0.98, (rho, lo, hi)


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
