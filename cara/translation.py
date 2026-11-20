"""Translation averaging for extension E5 (downstream effect of the rotation estimate).

Relative pose of pair (i, j) from the front end, camera-from-world: x_j = R_ij x_i + t_ij.
Since t_ij = R_j (c_i - c_j) for camera centres c_i = -R_i^T t_i, the unit direction
    d_ij = R_j^T t_ij / |t_ij|   is parallel to  c_i - c_j   (R_j: the estimated global rotation of camera j)
Least unsquared deviations (LUD, Ozyesil and Singer 2015):
    min_{c, s}  sum_e w_e || c_i - c_j - s_e d_e ||_2   s.t.  s_e >= 1,  sum c = 0
solved by iteratively reweighted least squares.
"""
import numpy as np
from scipy.sparse import coo_matrix, vstack, identity
from scipy.sparse.linalg import lsqr


def directions(edges, tij, R):
    """unit directions d_e ∥ c_i - c_j in the world frame, given camera rotations R (world->camera)"""
    d = np.einsum("mba,mb->ma", R[edges[:, 1]], tij)          # R_j^T t_ij
    nrm = np.linalg.norm(d, axis=1, keepdims=True)
    return d / np.maximum(nrm, 1e-12)


def lud(n, edges, d, w=None, iters=15, nodes=None):
    """returns centres (n,3) (NaN for cameras outside `nodes`), up to similarity"""
    m = len(edges)
    if w is None:
        w = np.ones(m)
    if nodes is None:
        nodes = np.unique(edges)
    loc = -np.ones(n, int); loc[nodes] = np.arange(len(nodes)); N = len(nodes)
    ok = (loc[edges[:, 0]] >= 0) & (loc[edges[:, 1]] >= 0)
    E, D, W = loc[edges[ok]], d[ok], w[ok]; M = len(E)
    from scipy.optimize import lsq_linear
    rw = np.ones(M)
    # unknown vector z = [c (3N, axis-major per camera), s (M)]; 3 rows per edge
    r3 = np.arange(3 * M).reshape(M, 3)
    ci = np.stack([3 * E[:, 0] + k for k in range(3)], 1); cj = np.stack([3 * E[:, 1] + k for k in range(3)], 1)
    for _ in range(iters):
        sw = np.repeat(np.sqrt(W * rw), 3).reshape(M, 3)
        rows = np.concatenate([r3.ravel(), r3.ravel(), r3.ravel()])
        cols = np.concatenate([ci.ravel(), cj.ravel(), (3 * N + np.repeat(np.arange(M), 3))])
        vals = np.concatenate([sw.ravel(), -sw.ravel(), (-sw * D).ravel()])
        A = coo_matrix((vals, (rows, cols)), shape=(3 * M, 3 * N + M))
        G = coo_matrix((np.ones(3 * N), (np.tile(np.arange(3), N), np.arange(3 * N))), shape=(3, 3 * N + M))
        A = vstack([A, G]).tocsr(); b = np.zeros(3 * M + 3)
        lb = np.r_[np.full(3 * N, -np.inf), np.ones(M)]
        if A.shape[1] <= 2000:     # small problems: dense bounded-variable least squares (exact, fast)
            z = lsq_linear(A.toarray(), b, bounds=(lb, np.full(3 * N + M, np.inf)), method="bvls").x
        else:
            z = lsq_linear(A, b, bounds=(lb, np.full(3 * N + M, np.inf)), method="trf", lsmr_tol="auto", max_iter=200).x
        c = z[:3 * N].reshape(N, 3); s = z[3 * N:]
        res = np.linalg.norm(c[E[:, 0]] - c[E[:, 1]] - s[:, None] * D, axis=1)
        rw = 1.0 / np.maximum(res, 1e-4)
    out = np.full((n, 3), np.nan); out[nodes] = c
    return out


def align_error(C_est, C_gt):
    """similarity (Umeyama) alignment on cameras with finite estimates; returns per-camera error
    divided by the median distance of GT centres to their centroid (scale-free), NaN if unregistered"""
    ok = np.all(np.isfinite(C_est), 1)
    err = np.full(len(C_gt), np.nan)
    if ok.sum() < 3:
        return err
    X, Y = C_est[ok], C_gt[ok]
    mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my
    U, S, Vt = np.linalg.svd(Yc.T @ Xc)
    Dm = np.eye(3); Dm[2, 2] = np.sign(np.linalg.det(U @ Vt))
    Rm = U @ Dm @ Vt; sc = np.trace(np.diag(S) @ Dm) / max(np.sum(Xc ** 2), 1e-12)
    Yh = sc * Xc @ Rm.T + my
    scale = np.median(np.linalg.norm(C_gt - C_gt.mean(0), axis=1))
    err[ok] = np.linalg.norm(Yh - Y, axis=1) / max(scale, 1e-12)
    return err


def centres_gt(R, t):
    return -np.einsum("nji,nj->ni", R, t)
