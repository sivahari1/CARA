"""E1/E2 development on ETH3D CALIBRATION scenes only (leave-one-scene-out), with injected gross
outliers as the shift. Test scenes are not used here. Writes per-camera records for design decisions."""
import os, sys, glob, pickle, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "..")); sys.path.insert(0, os.path.join(H, "..", "..", "gpu_kit"))
from run_eval import load_scene
from cara.method import Config, calibrate, run
from cara.metrics import rot_errors
from scipy.spatial.transform import Rotation as Rot

NPZ = os.path.join(H, "..", "..", "data", "eth3d_vggt")
calib = [load_scene(f, "conf_med") for f in sorted(glob.glob(os.path.join(NPZ, "calib__*__clean.npz")))]
scenes = sorted({c["scene"] for c in calib})
out_sig = np.concatenate([c["sig_hat"][c["is_out"]] for c in calib])     # confidence of real VGGT outliers


def inject(sc, frac, rng):
    sc = dict(sc); m = len(sc["edges"])
    k = int(round(frac * m))
    if k == 0:
        return sc
    cand = np.where(~sc["is_out"])[0]
    idx = rng.choice(cand, min(k, len(cand)), replace=False)
    Rij = sc["Rij"].copy(); Rij[idx] = Rot.random(len(idx), random_state=int(rng.integers(1e9))).as_matrix()
    sig = sc["sig_hat"].copy(); sig[idx] = rng.choice(out_sig, len(idx))
    iso = sc["is_out"].copy(); iso[idx] = True
    sc.update(Rij=Rij, sig_hat=sig, is_out=iso)
    return sc


def inject_cam(sc, frac, rng, ang=(20, 180)):
    """camera-level failure: a fraction of cameras gets a common wrong rotation on all its edges
    (cycle-consistent, like an image VGGT misreads) plus per-edge noise; confidences unchanged"""
    sc = dict(sc); n = sc["n"]; E = sc["edges"]
    k = int(round(frac * n))
    if k == 0:
        return sc
    cams = rng.choice(n, k, replace=False)
    Rij = sc["Rij"].copy(); iso = sc["is_out"].copy()
    for i in cams:
        th = np.radians(rng.uniform(*ang)); ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
        Dl = Rot.from_rotvec(th * ax).as_matrix()
        for e in np.where((E[:, 0] == i) | (E[:, 1] == i))[0]:
            N = Rot.from_rotvec(np.radians(3) * rng.normal(size=3)).as_matrix()
            # camera i observed as R_i' = D R_i :  R_ij' = R_ij D^T (i first) or D R_ij (i second)
            Rij[e] = (Rij[e] @ Dl.T if E[e, 0] == i else Dl @ Rij[e]) @ N
            iso[e] = True
    sc.update(Rij=Rij, is_out=iso)
    return sc


LEVELS = [("none", 0.0), ("edge", 0.1), ("edge", 0.2), ("cam", 0.1), ("cam", 0.2)]
recs = []
cfgs = {"CARA": Config(), "CARA-static": Config(rho="none")}
for held in scenes:
    rng = np.random.default_rng(zlib_seed := abs(hash(held)) % 2**32)
    pool = [c for c in calib if c["scene"] != held]; dev = [c for c in calib if c["scene"] == held]
    perm = np.random.default_rng(0).permutation(len(pool)); P = [pool[i] for i in perm]
    splits = (P[:20], P[20:50], P[50:80], P[80:120])
    cals = {k: calibrate(c, *splits) for k, c in cfgs.items()}
    for kind, lev in LEVELS:
        for j, sc0 in enumerate(dev):
            r_ = np.random.default_rng([j, int(lev * 100), len(held), len(kind)])
            sc = inject_cam(sc0, lev, r_) if kind == "cam" else inject(sc0, lev, r_)
            for k in cfgs:
                o = run(sc, cfgs[k], cals[k])
                err = rot_errors(sc["R"], o["R"])
                recs.append(dict(held=held, lev=lev, kind=kind, edges=sc["edges"], sig_hat=sc["sig_hat"], is_out=sc["is_out"], sub=j, meth=k, err=err, u=o["u"], reg=o["reg"], lam=cals[k].lam_cam,
                                 rho=o["rho"], s0=o["s0"], s1=o["s1"], feat=o["feat"], frac_out=float(sc["is_out"].mean()),
                                 lam_edge=cals[k].lam_edge))
    print(held, "done", flush=True)
pickle.dump(recs, open(os.path.join(H, "dev_eth3d.pkl"), "wb"))
