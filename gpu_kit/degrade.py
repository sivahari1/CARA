"""Physically based image degradations with a severity parameter.

fog   : Koschmieder model  I = J t + A (1 - t),  t = exp(-beta d).  beta is chosen per image so
        that the MEDIAN transmission equals t_med (severity), which makes severities comparable
        across scenes with different depth ranges. Depth d: ground truth if available, otherwise a
        depth map predicted on the CLEAN image (used only to render the fog; the evaluated
        pipeline sees only the fogged image). Airlight A ~ U(0.8, 0.95), slight blue tint.
lowlight : exposure scaling k, gamma, and Poisson-Gaussian sensor noise.
rain  : random slanted streaks + slight blur + contrast loss (appearance only).
Severity levels used in the protocol: fog t_med in {0.7, 0.5, 0.3}; lowlight k in {0.3, 0.15, 0.07}.
"""
import numpy as np
from PIL import Image, ImageFilter


def fog(img, depth, t_med, rng):
    J = np.asarray(img, np.float32) / 255.0
    d = np.asarray(depth, np.float32)
    valid = np.isfinite(d) & (d > 0)
    if valid.sum() < 100:
        d = np.full(J.shape[:2], 1.0, np.float32); valid = np.ones_like(d, bool)
    fill = np.nanpercentile(d[valid], 99)
    d = np.where(valid, d, fill)                       # sky / invalid -> far
    beta = -np.log(t_med) / np.median(d[valid])
    t = np.exp(-beta * d)[..., None]
    A = rng.uniform(0.8, 0.95) * np.array([0.97, 0.99, 1.0], np.float32)
    I = J * t + A * (1 - t)
    return Image.fromarray((np.clip(I, 0, 1) * 255).astype(np.uint8))


def lowlight(img, k, rng, read_noise=0.01, photons=300.0):
    J = (np.asarray(img, np.float32) / 255.0) ** 2.2           # to linear
    L = k * J
    L = rng.poisson(L * photons) / photons + rng.normal(0, read_noise, L.shape)
    I = np.clip(L, 0, 1) ** (1 / 2.2)                           # back to display gamma
    return Image.fromarray((np.clip(I, 0, 1) * 255).astype(np.uint8))


def rain(img, density, rng, length=25, angle_deg=15):
    J = np.asarray(img, np.float32) / 255.0
    H, W = J.shape[:2]
    layer = np.zeros((H, W), np.float32)
    n = int(density * H * W / 400)
    ys, xs = rng.integers(0, H, n), rng.integers(0, W, n)
    dx, dy = np.sin(np.deg2rad(angle_deg)), np.cos(np.deg2rad(angle_deg))
    for s in range(length):
        yy = np.clip((ys + s * dy).astype(int), 0, H - 1); xx = np.clip((xs + s * dx).astype(int), 0, W - 1)
        layer[yy, xx] = 0.8
    I = 0.85 * J + 0.1 + layer[..., None] * 0.5
    out = Image.fromarray((np.clip(I, 0, 1) * 255).astype(np.uint8))
    return out.filter(ImageFilter.GaussianBlur(0.8))


def read_depth_raw(path, W, H):
    """ETH3D ground-truth depth: raw float32, row-major, size W*H."""
    d = np.fromfile(path, dtype=np.float32)
    if d.size != W * H:
        raise ValueError(f"depth size {d.size} != {W}x{H} for {path}")
    d = d.reshape(H, W)
    d[~np.isfinite(d)] = 0
    return d
