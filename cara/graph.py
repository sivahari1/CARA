"""View-graph helpers: triangle enumeration, cycle scores, components."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .so3 import angle, T


class Triangles:
    """For every edge e=(i,j), up to K third vertices k with edges (j,k),(k,i).
    Stores, per triangle, the edge ids and orientation flags needed to form
    the cycle R_ki R_jk R_ij (which is I for noise-free, consistent data)."""

    def __init__(self, n, edges, K=30, rng=None):
        rng = rng if rng is not None else np.random.default_rng(0)
        m = len(edges)
        lut = {}
        nbr = [set() for _ in range(n)]
        for e, (i, j) in enumerate(edges):
            lut[(i, j)] = (e, 0)
            lut[(j, i)] = (e, 1)
            nbr[i].add(j)
            nbr[j].add(i)
        te, e2, f2, e3, f3 = [], [], [], [], []
        for e, (i, j) in enumerate(edges):
            ks = list(nbr[i] & nbr[j])
            if len(ks) > K:
                ks = list(rng.choice(ks, K, replace=False))
            for k in ks:
                a, fa = lut[(j, k)]
                b, fb = lut[(k, i)]
                te.append(e); e2.append(a); f2.append(fa); e3.append(b); f3.append(fb)
        self.m = m
        self.e = np.array(te, int)
        self.e2 = np.array(e2, int); self.f2 = np.array(f2, bool)
        self.e3 = np.array(e3, int); self.f3 = np.array(f3, bool)

    def __len__(self):
        return len(self.e)

    def cycle_angles(self, Rij):
        if len(self.e) == 0:
            return np.zeros(0)
        A = Rij[self.e2].copy(); A[self.f2] = T(A[self.f2])   # R_jk
        B = Rij[self.e3].copy(); B[self.f3] = T(B[self.f3])   # R_ki
        return angle(B @ A @ Rij[self.e])

    def normalized(self, cyc, sig):
        return cyc / np.sqrt(sig[self.e] ** 2 + sig[self.e2] ** 2 + sig[self.e3] ** 2)


def per_edge_quantile(tri_e, vals, m, q, mask=None):
    """q-quantile (lower order statistic) of vals grouped by edge id; NaN if none."""
    if mask is not None:
        tri_e, vals = tri_e[mask], vals[mask]
    out = np.full(m, np.nan)
    if len(vals) == 0:
        return out
    order = np.lexsort((vals, tri_e))
    se, sv = tri_e[order], vals[order]
    counts = np.bincount(se, minlength=m)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    has = counts > 0
    idx = starts[has] + np.floor(q * (counts[has] - 1)).astype(int)
    out[has] = sv[idx]
    return out


def components(n, edges, keep):
    E = edges[keep]
    if len(E) == 0:
        return n, np.arange(n)
    A = coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n))
    return connected_components(A, directed=False)
