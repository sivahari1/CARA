"""Evaluation: gauge alignment, per-camera rotation errors, AUC."""
import numpy as np
from .so3 import angle, proj_so3, T

DEG = np.pi / 180


def rot_errors(Rgt, Rest):
    """Per-camera geodesic error after robust gauge alignment R_i -> R_i G.
    Cameras with NaN estimates (unregistered) get error = inf."""
    n = len(Rgt)
    ok = ~np.isnan(Rest).any(axis=(1, 2))
    err = np.full(n, np.inf)
    if ok.sum() == 0:
        return err
    A, B = Rest[ok], Rgt[ok]
    M = T(A) @ B  # per-camera implied gauge
    G = proj_so3(M.sum(0))
    for _ in range(5):
        e = angle(T(B) @ A @ G)
        keep = e <= max(np.median(e) * 2.0, 1e-9)
        G = proj_so3(M[keep].sum(0))
    err[ok] = angle(T(B) @ A @ G)
    return err


def auc(err, thr_deg):
    """Area under the recall curve up to thr (deg), unregistered = failure."""
    e = np.sort(np.asarray(err) / DEG)
    n = len(e)
    xs = np.linspace(0, thr_deg, 201)
    rec = np.searchsorted(e, xs, side="right") / n
    return np.trapezoid(rec, xs) / thr_deg


def summarize(err):
    e = err / DEG
    reg = np.isfinite(e)
    ef = e[reg]
    return dict(
        median=float(np.median(ef)) if reg.any() else np.nan,
        mean=float(np.mean(ef)) if reg.any() else np.nan,
        p95=float(np.percentile(ef, 95)) if reg.any() else np.nan,
        reg=float(reg.mean()),
        cat=float(np.mean(~(e <= 30))),          # >30 deg or unregistered
        auc2=auc(err, 2), auc5=auc(err, 5), auc10=auc(err, 10),
    )
