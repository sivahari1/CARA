"""Sanity check after stage 1: relative-rotation errors of the clean calibration edges.
Expect a median of a few degrees. A median near 90-180 deg means a convention error
(e.g. transposed rotations); stop and report it before running stage 2.
  python check_frontend.py --in out_imc"""
import os, sys, glob, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cara.so3 import angle, T
from scipy.stats import spearmanr
ap = argparse.ArgumentParser(); ap.add_argument("--in", dest="inp", required=True); a = ap.parse_args()
errs, errs_t, conf = [], [], []
for f in sorted(glob.glob(os.path.join(a.inp, "*__clean.npz")))[:60]:
    z = np.load(f, allow_pickle=True); E, R, G = z["edges"], z["Rij"], z["R_gt"]
    Rt = G[E[:, 1]] @ T(G[E[:, 0]])
    errs += list(np.degrees(angle(T(Rt) @ R))); errs_t += list(np.degrees(angle(Rt @ R))); conf += list(z["conf_med"])
errs, errs_t, conf = map(np.array, (errs, errs_t, conf))
print(f"edges {len(errs)}  median err {np.median(errs):.2f} deg (transposed: {np.median(errs_t):.2f})")
print(f"fraction > 5 deg: {np.mean(errs > 5):.3f}   > 30 deg: {np.mean(errs > 30):.3f}")
print(f"Spearman(confidence, error) = {spearmanr(conf, errs).correlation:.3f}  (expect negative)")
