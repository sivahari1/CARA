"""Failure-mode augmentation for calibration scenes (extension E1).

Two failure models, applied only to CALIBRATION scenes (labels are known there):
  edge  : a fraction of inlier edges replaced by uniformly random rotations, with confidences drawn
          from the confidences of real outlier edges (confidently wrong edges)
  cam   : a fraction of cameras is 'misread': every edge of such a camera carries the same wrong
          rotation D (angle 20-180 deg) plus small noise. These errors close around every cycle, so
          cycle tests cannot detect them; only the calibration can account for them.
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot

DEFAULT_MODES = (("none", 0.0), ("edge", 0.1), ("edge", 0.2), ("cam", 0.1), ("cam", 0.2))


def inject_edge(sc, frac, rng, sig_pool=None):
    sc = dict(sc); m = len(sc["edges"])
    cand = np.where(~sc["is_out"])[0]
    k = min(int(round(frac * m)), len(cand))
    if k == 0:
        return sc
    idx = rng.choice(cand, k, replace=False)
    Rij = sc["Rij"].copy(); Rij[idx] = Rot.random(k, random_state=int(rng.integers(1 << 31))).as_matrix()
    sig = np.array(sc["sig_hat"], float).copy()
    if sig_pool is not None and len(sig_pool):
        sig[idx] = rng.choice(sig_pool, k)
    iso = sc["is_out"].copy(); iso[idx] = True
    sc.update(Rij=Rij, sig_hat=sig, is_out=iso)
    return sc


def inject_cam(sc, frac, rng, ang=(20.0, 180.0), noise_deg=3.0):
    sc = dict(sc); n, E = sc["n"], sc["edges"]
    k = int(round(frac * n))
    if k == 0:
        return sc
    Rij = sc["Rij"].copy(); iso = sc["is_out"].copy()
    for i in rng.choice(n, k, replace=False):
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
        D = Rot.from_rotvec(np.radians(rng.uniform(*ang)) * ax).as_matrix()
        for e in np.where((E[:, 0] == i) | (E[:, 1] == i))[0]:
            N = Rot.from_rotvec(np.radians(noise_deg) * rng.normal(size=3)).as_matrix()
            # camera i observed as D R_i:  R_ij -> R_ij D^T if i is the first index, D R_ij otherwise
            Rij[e] = (Rij[e] @ D.T if E[e, 0] == i else D @ Rij[e]) @ N
            iso[e] = True
    sc.update(Rij=Rij, is_out=iso)
    return sc


def augment(scenes, modes=DEFAULT_MODES, seed=0, sig_pool=None):
    """every scene under every failure mode (the 'none' mode keeps the original scene)"""
    out = []
    for j, sc in enumerate(scenes):
        for mi, (kind, lev) in enumerate(modes):
            rng = np.random.default_rng([seed, j, mi])
            if kind == "none":
                out.append(sc)
            elif kind == "edge":
                out.append(inject_edge(sc, lev, rng, sig_pool))
            elif kind == "cam":
                out.append(inject_cam(sc, lev, rng))
            else:
                raise ValueError(kind)
    return out
