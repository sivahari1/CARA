"""Synthetic view graphs with learned-confidence-like uncertainty.

Noise model: per-edge per-axis std sigma_true ~ LogNormal(log sig_med, spread).
Native uncertainty sigma_hat = sigma_true / overconf * LogNormal(0, conf_noise):
rank-correlated with the truth but overconfident, matching the 2026 audit of
DUSt3R/VGGT-family confidences (median level off by ~2.4x).
Shift: sigma_true *= noise_mult while sigma_hat is unchanged.
Outliers: fraction p_out of edges; uniform random rotations, or with
probability flip_frac a 180-deg flip of the true relative rotation; outliers receive sigma_hat from the inlier law (confidently wrong).
Layout 'ring' adds *correlated* doppelganger edges: cameras on a circle around a
symmetric object; look-alike pairs (i in front half, j in back half, j's mirror
twin overlapping i) get R_obs = R_j Q R_i^T with one fixed Q = Ry(pi), so the
spurious edges are mutually cycle-consistent (the hard case for cycle tests).
"""
import numpy as np
from .so3 import expm, random_rotations, T

DEG = np.pi / 180


def _noise_and_conf(rng, m, sig_med, spread, overconf, conf_noise, noise_mult):
    sig_true = np.exp(rng.normal(np.log(sig_med), spread, m))
    sig_hat = sig_true / overconf * np.exp(rng.normal(0, conf_noise, m))
    return sig_true * noise_mult, sig_hat


def _ry(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def make_scene(rng, n=100, layout="generic", k_local=4, n_rand=150, p_out=0.2,
               noise_mult=1.0, sig_med=1.5 * DEG, spread=0.5, overconf=2.4,
               conf_noise=0.45, dop_frac=0.0, overlap=25 * DEG, flip_frac=0.0):
    if layout == "generic":
        R = random_rotations(n, rng)
        edges = set()
        for i in range(n):
            for d in range(1, k_local + 1):
                a, b = i, (i + d) % n
                edges.add((min(a, b), max(a, b)))
        target = len(edges) + n_rand
        while len(edges) < target:
            i, j = rng.integers(n, size=2)
            if i != j:
                edges.add((int(min(i, j)), int(max(i, j))))
        edges = np.array(sorted(edges))
        dop = np.zeros(len(edges), bool)
    elif layout == "ring":
        th = np.sort(rng.uniform(0, 2 * np.pi, n))
        jit = expm(rng.normal(0, 3 * DEG, (n, 3)))
        R = np.array([jit[i] @ _ry(th[i]) for i in range(n)])
        d = np.abs((th[:, None] - th[None, :] + np.pi) % (2 * np.pi) - np.pi)
        iu = np.triu_indices(n, 1)
        mask = d[iu] < overlap
        edges = set(zip(iu[0][mask].tolist(), iu[1][mask].tolist()))
        k = 0
        while k < n_rand // 3:  # a few genuine wide-baseline retrieval edges
            i, j = rng.integers(n, size=2)
            if i != j and d[i, j] < 2 * overlap:
                edges.add((int(min(i, j)), int(max(i, j)))); k += 1
        edges = sorted(edges)
        # doppelganger candidates: i front (th<pi), j back, mirror of j near i
        dm = np.abs((th[:, None] - (th[None, :] - np.pi) + np.pi) % (2 * np.pi) - np.pi)
        front = th < np.pi
        cand = [(i, j) for i in range(n) for j in range(n)
                if front[i] and not front[j] and dm[i, j] < overlap]
        rng.shuffle(cand)
        nd = int(round(dop_frac * len(cand)))
        dset = set()
        for (i, j) in cand[:nd]:
            dset.add((min(i, j), max(i, j)))
        edges = sorted(set(edges) | dset)
        dop = np.array([e in dset for e in edges], bool)
        edges = np.array(edges)
    else:
        raise ValueError(layout)
    m = len(edges)
    sig_true, sig_hat = _noise_and_conf(rng, m, sig_med, spread, overconf, conf_noise, noise_mult)
    is_out = (rng.random(m) < p_out) & ~dop
    Rij = np.empty((m, 3, 3))
    Q = _ry(np.pi)
    flip = expm(np.array([0.0, np.pi, 0.0]))
    noise = expm(sig_true[:, None] * rng.normal(size=(m, 3)))
    i, j = edges[:, 0], edges[:, 1]
    Rtrue = R[j] @ T(R[i])
    Rij[:] = Rtrue @ noise
    for e in np.where(is_out)[0]:
        Rij[e] = random_rotations(1, rng)[0] if rng.random() > flip_frac else Rtrue[e] @ flip
    for e in np.where(dop)[0]:
        a, b = edges[e]
        # which endpoint is in the back half gets the mirror twin
        if layout == "ring" and not (R is None):
            Rj_tw = R[b] @ Q
            Ri_ = R[a]
            Rij[e] = Rj_tw @ T(Ri_) @ noise[e]
    is_out = is_out | dop
    return dict(n=n, R=R, edges=edges, Rij=Rij, sig_hat=sig_hat, is_out=is_out,
                dop=dop, sig_true=sig_true)
