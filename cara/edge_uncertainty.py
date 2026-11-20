"""Extension E3: edge-level uncertainty from several VGGT signals (run 2 files only).

Features per edge (fixed before run 2 was computed):
  f1  log(1 / median depth confidence over both images)         (the pre-registered sigma_hat)
  f2  -min over the two images of the mean log confidence          (worst image)
  f3  -max over the two images of the mean log confidence          (best image)
  f4  -log(min over the two images of the 10% confidence quantile)
  f5  log(angle between the relative rotations of the last two camera-head iterations, deg, + 1e-3)
The learned uncertainty is  log sigma = beta . [1, f1..f5], fitted by least squares to the log error of
inlier edges (error <= 5 deg) of the fitting split of the calibration scenes. It replaces sigma_hat; the
rest of CARA (including its power-law recalibration) is unchanged.
"""
import numpy as np
from .so3 import angle, T

NAMES = ["f1_med", "f2_worst_logmean", "f3_best_logmean", "f4_worst_q10", "f5_iter_delta"]


def rel_rot(ext):
    return np.einsum("mab,mcb->mac", ext[:, 1, :, :3], ext[:, 0, :, :3])


def edge_features(z, keep_idx=None):
    cs = z["conf_stats"]                       # (M,2,4): median, mean log, q10, q90
    f1 = np.log(1.0 / np.maximum(z["conf_med"], 1e-6))
    f2 = -cs[:, :, 1].min(1)
    f3 = -cs[:, :, 1].max(1)
    f4 = -np.log(np.maximum(cs[:, :, 2].min(1), 1e-6))
    dl = np.degrees(angle(T(rel_rot(z["ext_prev"])) @ rel_rot(z["ext"])))
    f5 = np.log(dl + 1e-3)
    F = np.stack([f1, f2, f3, f4, f5], 1)
    return F if keep_idx is None else F[keep_idx]


def fit(F_list, err_list, tau=5 * np.pi / 180):
    X = np.concatenate(F_list); y = np.concatenate(err_list)
    m = y <= tau
    A = np.c_[np.ones(m.sum()), X[m]]
    return np.linalg.lstsq(A, np.log(np.maximum(y[m], 1e-5)), rcond=None)[0]


def predict(beta, F, lo=1e-4, hi=np.pi):
    """clipped to [1e-4, pi] rad: a few edges have extreme confidences, which would otherwise give a
    zero or infinite uncertainty"""
    return np.clip(np.exp(np.clip(np.c_[np.ones(len(F)), F] @ beta, -50, 50)), lo, hi)
