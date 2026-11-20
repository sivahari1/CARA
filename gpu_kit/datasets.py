"""Ground-truth readers. Every reader returns a list of scenes:
    dict(name, images=[abs paths], R=(N,3,3) world->camera, t=(N,3), K=(N,3,3) or None,
         pairs=(M,2) int candidate view-graph pairs, depth_paths=[...] or None)

IMC 2022 (Kaggle "image-matching-challenge-2022", train/<scene>/):
    calibration.csv: image_id, camera_intrinsics (9, row-major), rotation_matrix (9, row-major),
    translation_vector (3); x_cam = R x_world + T.  pair_covisibility.csv: pair (id1-id2), covisibility.
ETH3D high-res multi-view training (<scene>/dslr_calibration_undistorted/images.txt, COLMAP text,
    QW QX QY QZ TX TY TZ = world->camera), images in <scene>/images/dslr_images_undistorted/.
    Ground-truth depth (optional, from multi_view_training_dslr_depth): <scene>/ground_truth_depth/
    dslr_images/<name>, raw little-endian float32, image size, invalid = inf or 0.
    CHECK these paths against your download; adjust ETH3D_* constants if they differ.
"""
import os, csv, glob
import numpy as np
from scipy.spatial.transform import Rotation as Rot

ETH3D_IMG = "images/dslr_images_undistorted"
ETH3D_CAL = "dslr_calibration_undistorted/images.txt"
ETH3D_DEPTH = "ground_truth_depth/dslr_images"


def _pairs_knn_from_poses(R, t, k=8):
    """fallback view graph: k nearest cameras by viewing-direction angle and distance"""
    C = -np.einsum("nji,nj->ni", R, t)                 # centres
    d = R[:, 2, :]                                       # viewing directions (third row)
    ang = np.arccos(np.clip(d @ d.T, -1, 1))
    dist = np.linalg.norm(C[:, None] - C[None], axis=-1)
    score = ang / (ang.std() + 1e-9) + dist / (dist.std() + 1e-9)
    np.fill_diagonal(score, np.inf)
    P = set()
    for i in range(len(R)):
        for j in np.argsort(score[i])[:k]:
            P.add((min(i, j), max(i, j)))
    return np.array(sorted(P))


def _check(found, scenes, root):
    if scenes:
        miss = set(scenes) - set(found)
        if miss:
            raise SystemExit(f"scenes not found under {root}: {sorted(miss)}")


def read_imc2022(root, scenes=None, covis_min=0.1, max_images=None, rng=None):
    out = []
    for sdir in sorted(glob.glob(os.path.join(root, "train", "*"))):
        name = os.path.basename(sdir)
        if scenes and name not in scenes:
            continue
        ids, R, t, K = [], [], [], []
        with open(os.path.join(sdir, "calibration.csv")) as f:
            for row in csv.DictReader(f):
                ids.append(row["image_id"])
                K.append(np.array(row["camera_intrinsics"].split(), float).reshape(3, 3))
                R.append(np.array(row["rotation_matrix"].split(), float).reshape(3, 3))
                t.append(np.array(row["translation_vector"].split(), float))
        idx = {s: i for i, s in enumerate(ids)}
        pairs, wts = [], {}
        pc = os.path.join(sdir, "pair_covisibility.csv")
        if os.path.exists(pc):
            with open(pc) as f:
                for row in csv.DictReader(f):
                    if float(row["covisibility"]) >= covis_min:
                        a, b = row["pair"].split("-")
                        if a in idx and b in idx:
                            key = (min(idx[a], idx[b]), max(idx[a], idx[b]))
                            pairs.append(key); wts[key] = float(row["covisibility"])
        R, t, K = np.array(R), np.array(t), np.array(K)
        pairs = np.array(sorted(set(pairs))) if pairs else _pairs_knn_from_poses(R, t)
        pw = np.array([wts.get((int(i), int(j)), 1.0) for i, j in pairs])
        imgs = [os.path.join(sdir, "images", s + ".jpg") for s in ids]
        out.append(dict(name=name, images=imgs, R=R, t=t, K=K, pairs=pairs, pair_w=pw, depth_paths=None))
    _check([o["name"] for o in out], scenes, root)
    return out


def read_eth3d(root, scenes=None, k=8):
    """ETH3D high-res multi-view training (undistorted). Finds images.txt under each scene
    (dslr_calibration_undistorted/images.txt) and resolves image paths relative to <scene>/images/."""
    out = []
    for sdir in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(sdir):
            continue
        cands = glob.glob(os.path.join(sdir, "**", "images.txt"), recursive=True)
        cands = [c for c in cands if "undistorted" in c] or cands
        if not cands:
            continue
        name = os.path.basename(sdir)
        if scenes and name not in scenes:
            continue
        names, R, t = [], [], []
        # COLMAP images.txt alternates a pose line and a (possibly EMPTY) points line, so pose lines
        # are identified by content: 10 tokens, integer id, and an image file name at the end.
        pose_lines = []
        for l in open(cands[0]):
            p = l.split()
            if l.startswith("#") or len(p) != 10:
                continue
            if p[0].isdigit() and os.path.splitext(p[9])[1].lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
                pose_lines.append(l)
        for l in pose_lines:
            p = l.split()
            qw, qx, qy, qz = map(float, p[1:5])
            R.append(Rot.from_quat([qx, qy, qz, qw]).as_matrix())
            t.append(np.array(p[5:8], float))
            names.append(p[9])
        R, t = np.array(R), np.array(t)
        imgs = []
        for n in names:
            p = os.path.join(sdir, "images", n)
            if not os.path.exists(p):
                hit = glob.glob(os.path.join(sdir, "**", os.path.basename(n)), recursive=True)
                p = hit[0] if hit else p
            imgs.append(p)
        missing = sum(not os.path.exists(p) for p in imgs)
        if missing:
            raise SystemExit(f"{name}: {missing} of {len(imgs)} images not found (first: {imgs[0]})")
        out.append(dict(name=name, images=imgs, R=R, t=t, K=None, pairs=_pairs_knn_from_poses(R, t, k),
                        depth_paths=None))
    _check([o["name"] for o in out], scenes, root)
    return out


def cap_pairs(pairs, w, n, k):
    """keep, for every image, its k highest-weight pairs (union over images)"""
    keep = np.zeros(len(pairs), bool)
    for i in range(n):
        idx = np.where((pairs[:, 0] == i) | (pairs[:, 1] == i))[0]
        keep[idx[np.argsort(-w[idx])[:k]]] = True
    return pairs[keep]


def subsample(scene, n, rng, k=8):
    """random n-image sub-scene; keeps GT pairs among the chosen images (or kNN),
    capped to the k strongest (covisibility) pairs per image"""
    N = len(scene["images"])
    if N <= n:
        return scene
    sel = np.sort(rng.choice(N, n, replace=False))
    loc = -np.ones(N, int); loc[sel] = np.arange(n)
    P = scene["pairs"]
    m = (loc[P[:, 0]] >= 0) & (loc[P[:, 1]] >= 0)
    pairs = loc[P[m]]
    if "pair_w" in scene and len(pairs):
        pairs = cap_pairs(pairs, scene["pair_w"][m], n, k)
    if len(pairs) < 2 * n:
        pairs = _pairs_knn_from_poses(scene["R"][sel], scene["t"][sel], k)
    s = dict(scene)
    s.update(images=[scene["images"][i] for i in sel], R=scene["R"][sel], t=scene["t"][sel],
             K=None if scene["K"] is None else scene["K"][sel], pairs=pairs,
             depth_paths=None if scene["depth_paths"] is None else [scene["depth_paths"][i] for i in sel],
             name=f"{scene['name']}_sub{int(rng.integers(1e9))}")
    return s
