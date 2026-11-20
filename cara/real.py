"""Real data.

Notre Dame (1DSfM, Wilson & Snavely ECCV 2014), in the Chatterjee & Govindu
format (RR relative rotations, I index pairs, Rgt reference rotations):
715 cameras, 64,678 relative rotations; convention R_ij = R_j R_i^T (verified:
median disagreement with Rgt 1.48 deg under this convention vs 31.8 deg for
the transpose). Reference rotations are the 1DSfM/Bundler reconstruction, so
'ground truth' means 'reference', as in all work on this benchmark.

Madrid Metropolis (1DSfM) with per-edge rotation covariances from
Zhang, Larsson & Barath (CVPR 2023, GlobalSfMpy). No reference rotations are
included, so it is used only for label-free calibration auditing.
"""
import os
import numpy as np
import scipy.io as sio
from .so3 import expm, angle, T
from .graph import components

DEG = np.pi / 180
DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def load_notredame():
    d = sio.loadmat(os.path.join(DATA, "Notredame.mat"))
    RR = d["RR"].transpose(2, 0, 1).astype(float)
    I = d["I"].T.astype(int) - 1
    Rgt = d["Rgt"].transpose(2, 0, 1).astype(float)
    err = angle(T(Rgt[I[:, 1]] @ T(Rgt[I[:, 0]])) @ RR)
    return dict(n=len(Rgt), R=Rgt, edges=I, Rij=RR, err=err)


def subscene(data, cams, rng, tau_out=5 * DEG, extra_noise=0.0, extra_out=0.0):
    """Induced sub-graph on `cams` (largest component), optional shift:
    extra per-axis Gaussian noise (rad) and injected random-rotation outliers.
    Inlier labels come from the *unshifted* edge error (<= tau_out)."""
    cams = np.sort(np.asarray(cams))
    loc = -np.ones(data["n"], int)
    loc[cams] = np.arange(len(cams))
    E = data["edges"]
    sel = (loc[E[:, 0]] >= 0) & (loc[E[:, 1]] >= 0)
    edges = loc[E[sel]]
    Rij = data["Rij"][sel].copy()
    err = data["err"][sel]
    n = len(cams)
    _, lab = components(n, edges, np.ones(len(edges), bool))
    big = np.where(lab == np.bincount(lab).argmax())[0]
    loc2 = -np.ones(n, int)
    loc2[big] = np.arange(len(big))
    s2 = (loc2[edges[:, 0]] >= 0) & (loc2[edges[:, 1]] >= 0)
    edges, Rij, err = loc2[edges[s2]], Rij[s2], err[s2]
    m = len(edges)
    is_out = err > tau_out
    if extra_noise > 0:
        Rij = Rij @ expm(extra_noise * rng.normal(size=(m, 3)))
    if extra_out > 0:
        inj = rng.random(m) < extra_out
        from .so3 import random_rotations
        Rij[inj] = random_rotations(int(inj.sum()), rng)
        is_out = is_out | inj
    return dict(n=len(big), R=data["R"][cams[big]], edges=edges, Rij=Rij,
                sig_hat=np.ones(m), is_out=is_out, cams=cams[big])


def load_madrid_cov():
    p = os.path.join(DATA, "Madrid_Metropolis", "covariance_rot.txt")
    L = [l.split() for l in open(p) if not l.startswith("#")]
    ids = np.array([[int(a[0]), int(a[1])] for a in L])
    v = np.array([[int(x) for x in a[2:11]] for a in L], dtype=np.uint64).view(np.float64)
    C = np.zeros((len(v), 3, 3))
    C[:, 0, 0], C[:, 1, 1], C[:, 2, 2] = v[:, 0], v[:, 1], v[:, 2]
    C[:, 0, 1] = C[:, 1, 0] = v[:, 3]
    C[:, 0, 2] = C[:, 2, 0] = v[:, 4]
    C[:, 1, 2] = C[:, 2, 1] = v[:, 5]
    uniq, inv = np.unique(ids, return_inverse=True)
    edges = inv.reshape(-1, 2)
    Rij = expm(v[:, 6:9])
    sig = np.sqrt(np.trace(C, axis1=1, axis2=2) / 3)
    return dict(n=len(uniq), edges=edges, Rij=Rij, sig_hat=sig, cov=C)
