"""Stage 1 (GPU): build view graphs with VGGT relative rotations and confidences.

For every sub-scene and condition it writes one .npz with
  edges (M,2), Rij (M,3,3) = R_j R_i^T from VGGT run on the image pair,
  conf_med (M,) median depth confidence over both images, conf_logmean (M,),
  R_gt (N,3,3) world->camera, names, cond, scene.
Usage (GPU machine, inside the VGGT repo environment):
  python frontend_vggt.py --dataset imc --root /data/imc2022 --out out_imc --role calib \
      --scenes-file splits/imc_calib.txt --n-sub 30 --per-scene 20 --conds clean
  python frontend_vggt.py --dataset imc --root /data/imc2022 --out out_imc --role test \
      --scenes-file splits/imc_test.txt --n-sub 30 --per-scene 10 \
      --conds clean,fog0.7,fog0.5,fog0.3,low0.3,low0.15,rain0.5
  python frontend_vggt.py ... --mock      # no torch/weights: synthetic predictions, for testing
"""
import os, sys, json, argparse, hashlib, zlib, time
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from datasets import read_imc2022, read_eth3d, subsample
import degrade


def parse_cond(c):
    if c == "clean":
        return ("clean", None)
    for k in ("fog", "low", "rain"):
        if c.startswith(k):
            return (k, float(c[len(k):]))
    raise ValueError(c)


class VGGTPair:
    def __init__(self, device="cuda", batch=8):
        import torch
        from vggt.models.vggt import VGGT
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        self.torch, self.p2e = torch, pose_encoding_to_extri_intri
        self.device, self.batch = device, batch
        cap = torch.cuda.get_device_capability()[0] if device == "cuda" else 0
        self.dtype = torch.bfloat16 if cap >= 8 else torch.float16
        self.model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()

    def depth_single(self, path):
        """depth of one clean image (only used to render fog when no GT depth exists).
        'pad' mode: the long side is scaled to 518 and the short side padded symmetrically."""
        from vggt.utils.load_fn import load_and_preprocess_images
        torch = self.torch
        im = load_and_preprocess_images([path], mode="pad").to(self.device)[None]
        with torch.no_grad(), torch.autocast("cuda", dtype=self.dtype):
            tok, ps = self.model.aggregator(im)
            depth, _ = self.model.depth_head(tok, im, ps)
        d = depth[0, 0, ..., 0].float().cpu().numpy()            # (518, 518)
        W, H = Image.open(path).size
        S = d.shape[0]; s = S / max(W, H)
        w, h = int(round(W * s)), int(round(H * s))
        x0, y0 = (S - w) // 2, (S - h) // 2
        crop = d[y0:y0 + h, x0:x0 + w]
        return np.asarray(Image.fromarray(crop).resize((W, H), Image.BILINEAR)).astype(np.float32)

    def pairs(self, paths_a, paths_b):
        """returns dict: Rij (M,3,3), ext (M,2,3,4) last camera-head iteration, ext_prev (M,2,3,4)
        previous iteration, conf_stats (M,2,4) per image [median, mean log, q10, q90] of depth confidence"""
        from vggt.utils.load_fn import load_and_preprocess_images
        torch = self.torch
        ext_l, ext_p, cs, cmj = [], [], [], []
        for s in range(0, len(paths_a), self.batch):
            A, B = paths_a[s:s + self.batch], paths_b[s:s + self.batch]
            imgs = torch.stack([load_and_preprocess_images([a, b], mode="pad") for a, b in zip(A, B)]).to(self.device)
            with torch.no_grad(), torch.autocast("cuda", dtype=self.dtype):
                tok, ps = self.model.aggregator(imgs)
                poses = self.model.camera_head(tok)
                _, conf = self.model.depth_head(tok, imgs, ps)
            e1, _ = self.p2e(poses[-1].float(), imgs.shape[-2:])
            e0, _ = self.p2e(poses[-2].float() if len(poses) > 1 else poses[-1].float(), imgs.shape[-2:])
            ext_l.append(e1.cpu().numpy()); ext_p.append(e0.cpu().numpy())
            c = conf.float().flatten(2).cpu().numpy()                     # (B,2,HW)
            cmj.append(np.median(c.reshape(len(A), -1), 1))                # v1 definition: both images jointly
            lc = np.log(np.maximum(c, 1e-6))
            cs.append(np.stack([np.median(c, -1), lc.mean(-1), np.quantile(c, 0.1, -1), np.quantile(c, 0.9, -1)], -1))
        ext, extp = np.concatenate(ext_l), np.concatenate(ext_p)
        Rij = np.einsum("mab,mcb->mac", ext[:, 1, :, :3], ext[:, 0, :, :3])   # R_1 R_0^T
        return dict(Rij=Rij, ext=ext, ext_prev=extp, conf_stats=np.concatenate(cs), conf_med=np.concatenate(cmj))


class MockPair:
    """Synthetic stand-in: GT relative rotation + noise that grows with degradation severity;
    confidence rank-correlated with the noise and overconfident. For pipeline testing only."""
    SEV = {"clean": 1.0, "fog": None, "low": None, "rain": 1.6}

    def __init__(self, rng):
        self.rng = rng

    def set_scene(self, R_gt, idx_of, cond, t_gt=None):
        self.R_gt, self.idx_of = R_gt, idx_of
        if t_gt is not None:
            self.t_gt = t_gt
        kind, lev = cond
        self.mult = {"clean": 1.0, "fog": 1 + 2 * (1 - (lev or 1)), "low": 1 + 3 * (0.3 - (lev or 0.3)) / 0.3 + 1, "rain": 1.6}[kind]

    def pairs(self, pa, pb):
        from scipy.spatial.transform import Rotation as Rot
        R, cs, ext, extp = [], [], [], []
        for a, b in zip(pa, pb):
            i, j = self.idx_of[a], self.idx_of[b]
            s = np.exp(self.rng.normal(np.log(1.5), 0.5)) * self.mult * np.pi / 180
            Rt = self.R_gt[j] @ self.R_gt[i].T
            if self.rng.random() < 0.15:
                Rt = Rot.random(random_state=int(self.rng.integers(1e9))).as_matrix()
            Rm = Rt @ Rot.from_rotvec(s * self.rng.normal(size=3)).as_matrix()
            R.append(Rm)
            c = 2.4 / s * np.exp(self.rng.normal(0, 0.45))          # overconfident, rank-correlated
            cs.append([[c, np.log(c), 0.5 * c, 2 * c]] * 2)
            e = np.zeros((2, 3, 4)); e[0, :, :3] = np.eye(3); e[1, :, :3] = Rm
            e[1, :, 3] = self.t_gt[j] - Rt @ self.t_gt[i] if hasattr(self, "t_gt") else 0
            ext.append(e)
            ep = e.copy(); ep[1, :, :3] = Rm @ Rot.from_rotvec(0.3 * s * self.rng.normal(size=3)).as_matrix(); extp.append(ep)
        cs = np.array(cs)
        return dict(Rij=np.array(R), ext=np.array(ext), ext_prev=np.array(extp), conf_stats=cs, conf_med=cs[:, 0, 0])

    def depth_single(self, path):
        W, H = Image.open(path).size
        return np.full((H, W), 10.0, np.float32)


MAX_SIDE = 1024      # VGGT works at 518 px; degraded copies are stored at <= 1024 px to save time and disk
_DEPTH = {}


def _small(p, cache):
    """downscaled clean copy (<= MAX_SIDE) used as the base for all degradations"""
    q = os.path.join(cache, "clean_" + hashlib.md5(p.encode()).hexdigest()[:12] + ".jpg")
    if not os.path.exists(q):
        im = Image.open(p).convert("RGB")
        im.thumbnail((MAX_SIDE, MAX_SIDE))
        im.save(q + ".tmp.jpg", quality=95); os.replace(q + ".tmp.jpg", q)   # atomic
    return q


def degrade_images(sub, cond, cache, model, rng):
    kind, lev = cond
    if kind == "clean":
        # the same <= 1024 px copy that the degradations start from: identical preprocessing for all
        # conditions, and ~20x faster than decoding 24-megapixel originals for every pair
        return [_small(p, cache) for p in sub["images"]]
    out = []
    for k, p in enumerate(sub["images"]):
        tag = hashlib.md5(f"{p}{kind}{lev}".encode()).hexdigest()[:12]
        q = os.path.join(cache, f"{tag}.jpg")
        if not os.path.exists(q):
            base = _small(p, cache)
            im = Image.open(base).convert("RGB")
            if kind == "fog":
                if sub["depth_paths"] is not None:
                    full = Image.open(p)
                    d = degrade.read_depth_raw(sub["depth_paths"][k], *full.size)
                    d = np.asarray(Image.fromarray(d).resize(im.size, Image.NEAREST))
                else:
                    if base not in _DEPTH:
                        _DEPTH[base] = model.depth_single(base)
                    d = _DEPTH[base]
                im = degrade.fog(im, d, lev, rng)
            elif kind == "low":
                im = degrade.lowlight(im, lev, rng)
            else:
                im = degrade.rain(im, lev, rng)
            im.save(q + ".tmp.jpg", quality=95); os.replace(q + ".tmp.jpg", q)   # atomic
        out.append(q)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["imc", "eth3d"], required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--role", choices=["calib", "test", "cv"], required=True)
    ap.add_argument("--scenes-file", default=None)
    ap.add_argument("--n-sub", type=int, default=30)
    ap.add_argument("--per-scene", type=int, default=10)
    ap.add_argument("--conds", default="clean")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--k", type=int, default=8, help="max pairs per image (covisibility / kNN)")
    ap.add_argument("--cache", default=None, help="folder for degraded images (default <out>/_degraded); reusable across runs")
    ap.add_argument("--n-test-sub", type=int, default=None,
                    help="role cv: sub-scenes 0..n_test_sub-1 get all conditions, the rest clean only (calibration pool)")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    scenes = [l.strip() for l in open(a.scenes_file) if l.strip()] if a.scenes_file else None
    data = read_imc2022(a.root, scenes) if a.dataset == "imc" else read_eth3d(a.root, scenes)
    os.makedirs(a.out, exist_ok=True); cache = a.cache or os.path.join(a.out, "_degraded"); os.makedirs(cache, exist_ok=True)
    model = MockPair(rng) if a.mock else VGGTPair(batch=a.batch)
    conds = [parse_cond(c) for c in a.conds.split(",")]
    manifest = []
    nts = a.per_scene if a.n_test_sub is None else a.n_test_sub
    total = len(data) * (nts * len(conds) + (a.per_scene - nts)); done = 0; t_start = time.time(); n_new = 0
    for sc in data:
        sid = zlib.crc32(sc["name"].encode())
        for s in range(a.per_scene):
            # every sub-scene and condition has its own seed, so an interrupted run resumes identically
            sub = subsample(sc, a.n_sub, np.random.default_rng([a.seed, sid, s]), k=a.k)
            P = sub["pairs"]
            for ci, cond in enumerate(conds):
                if s >= nts and cond[0] != "clean":
                    continue
                done += 1
                cname = cond[0] if cond[1] is None else f"{cond[0]}{cond[1]}"
                fn = os.path.join(a.out, f"{a.role}__{sc['name']}__{s:03d}__{cname}.npz")
                manifest.append(dict(file=os.path.basename(fn), scene=sc["name"], sub=s, cond=cname, n=len(sub["images"]), m=len(P)))
                if os.path.exists(fn):
                    continue
                crng = np.random.default_rng([a.seed, sid, s, ci])
                paths = degrade_images(sub, cond, cache, model, crng)
                if a.mock:
                    model.set_scene(sub["R"], {p: i for i, p in enumerate(paths)}, cond, sub["t"])
                out = model.pairs([paths[i] for i in P[:, 0]], [paths[j] for j in P[:, 1]])
                cst = out["conf_stats"]
                cm = out["conf_med"]
                cl = cst[:, :, 1].mean(1)
                np.savez_compressed(fn + ".tmp.npz", edges=P, Rij=out["Rij"], conf_med=cm, conf_logmean=cl,
                                    conf_stats=cst, ext=out["ext"], ext_prev=out["ext_prev"], t_gt=sub["t"],
                                    R_gt=sub["R"], role_sub=("test" if s < nts else "calib"),
                                    names=np.array([os.path.basename(p) for p in sub["images"]]),
                                    cond=cname, scene=sc["name"], role=a.role)
                os.replace(fn + ".tmp.npz", fn)            # atomic: a disconnect never leaves a half file
                n_new += 1
                el = time.time() - t_start
                eta = el / n_new * (total - done) / 3600
                print(f"[{done}/{total}] {os.path.basename(fn)} n={len(sub['images'])} m={len(P)}  ETA {eta:.1f} h", flush=True)
    json.dump(manifest, open(os.path.join(a.out, f"manifest_{a.role}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
