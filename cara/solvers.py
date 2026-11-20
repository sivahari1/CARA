"""Rotation averaging solvers.

spectral : chordal spectral relaxation (Martinec & Pajdla 2007; Arrigoni et al.)
irls     : Lie-algebra IRLS (Govindu 2004; Chatterjee & Govindu 2013/2018) with
           kernels 'l2', 'l1', 'gm' (Geman-McClure), and optional prior weights.
All solvers work on the sub-graph induced by `nodes` (default: all).
"""
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve, eigsh
from .so3 import expm, logm, proj_so3, T

DEG = np.pi / 180


def _restrict(n, edges, nodes):
    if nodes is None:
        nodes = np.arange(n)
    nodes = np.asarray(nodes)
    loc = -np.ones(n, int)
    loc[nodes] = np.arange(len(nodes))
    sel = (loc[edges[:, 0]] >= 0) & (loc[edges[:, 1]] >= 0)
    return nodes, loc, sel


def spectral(n, edges, Rij, w=None, nodes=None):
    nodes, loc, sel = _restrict(n, edges, nodes)
    E = loc[edges[sel]]
    Rr = Rij[sel]
    w = np.ones(len(E)) if w is None else np.asarray(w)[sel]
    k = len(nodes)
    rows, cols, vals = [], [], []
    for a in range(3):
        for b in range(3):
            # block (j,i) += w R_ij ; block (i,j) += w R_ij^T
            rows.append(3 * E[:, 1] + a); cols.append(3 * E[:, 0] + b); vals.append(w * Rr[:, a, b])
            rows.append(3 * E[:, 0] + b); cols.append(3 * E[:, 1] + a); vals.append(w * Rr[:, a, b])
    M = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(3 * k, 3 * k)).tocsr()
    if 3 * k <= 1500:
        _, V = np.linalg.eigh(M.toarray())
        X = V[:, -3:]
    else:
        _, X = eigsh(M, k=3, which="LA", tol=1e-8)
    Xb = X.reshape(k, 3, 3)
    if np.sum(np.sign(np.linalg.det(Xb))) < 0:
        X[:, 0] *= -1
        Xb = X.reshape(k, 3, 3)
    R = np.full((n, 3, 3), np.nan)
    R[nodes] = proj_so3(Xb)
    return R


def residuals(R, edges, Rij):
    """rotation-vector residual d_e = log(R_j^T R_ij R_i)."""
    return logm(T(R[edges[:, 1]]) @ Rij @ R[edges[:, 0]])


def irls(n, edges, Rij, R0, prior_w=None, kernel="gm", c=5 * DEG, nodes=None,
         iters=30, tol=1e-7):
    nodes, loc, sel = _restrict(n, edges, nodes)
    E = edges[sel]
    Rr = Rij[sel]
    pw = np.ones(len(E)) if prior_w is None else np.asarray(prior_w)[sel]
    R = R0.copy()
    k = len(nodes)
    li, lj = loc[E[:, 0]], loc[E[:, 1]]
    for _ in range(iters):
        d = residuals(R, E, Rr)
        r = np.linalg.norm(d, axis=1)
        if kernel == "l2":
            kw = np.ones_like(r)
        elif kernel == "l1":
            kw = 1.0 / np.maximum(r, 1e-4)
        elif kernel == "gm":
            kw = 1.0 / (1.0 + (r / c) ** 2) ** 2
        else:
            raise ValueError(kernel)
        w = pw * kw
        # weighted graph Laplacian; anchor local node 0
        rows = np.concatenate([li, lj, li, lj])
        cols = np.concatenate([li, lj, lj, li])
        vals = np.concatenate([w, w, -w, -w])
        L = csr_matrix(coo_matrix((vals, (rows, cols)), shape=(k, k)))
        b = np.zeros((k, 3))
        np.add.at(b, lj, w[:, None] * d)
        np.add.at(b, li, -w[:, None] * d)
        L = L[1:, 1:]
        om = np.zeros((k, 3))
        if k > 1:
            sol = spsolve(L.tocsc(), b[1:])
            om[1:] = sol.reshape(k - 1, 3)
        R[nodes] = R[nodes] @ expm(om)
        if np.abs(om).max() < tol:
            break
    return R


def solve_cg(n, edges, Rij, prior_w=None, nodes=None, c=5 * DEG):
    """Chatterjee & Govindu style: spectral init -> L1-IRLS -> Geman-McClure IRLS."""
    R0 = spectral(n, edges, Rij, prior_w, nodes)
    R1 = irls(n, edges, Rij, R0, prior_w, "l1", nodes=nodes, iters=20)
    return irls(n, edges, Rij, R1, prior_w, "gm", c=c, nodes=nodes, iters=20)
