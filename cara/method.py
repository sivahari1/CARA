"""CARA: Calibrated, conformally gated, drift-Aware Rotation Averaging.

Pipeline for one scene (see paper, Alg. 1):
  1  power-law recalibration of native uncertainty            sigma = g(sigma_hat)
  2  cycle scores s_e = Q_q over triangles of normalized cycle angle
  3  label-free drift factor rho from low quantiles of s vs. a reference
  4  conformal (CRC) edge gate s_e/rho <= lam_edge, round-2 rescue with clean triangles
  5  consensus bridging of components (abstain if no consensus)
  6  weighted robust averaging (spectral -> L1-IRLS -> GM-IRLS), kernel scale c*rho
  7  conformal (CRC) residual gate r_e <= lam_res, bridging, re-solve
  8  per-camera bounds B_i = lam_cam * u_i,  u_i = (sum_kept sigma_e^-2)^(-1/2)
Calibration uses four disjoint scene splits (fit / edge / res / cam), see calibrate().
"""
from dataclasses import dataclass, field, replace
import time
import numpy as np
from .so3 import angle, proj_so3, T
from .graph import Triangles, per_edge_quantile, components
from .solvers import spectral, irls, solve_cg, DEG
from .metrics import rot_errors


@dataclass
class Config:
    recal: bool = True
    tri_q: float = 0.25          # triangle quantile for edge score
    K: int = 30                  # max triangles per edge
    rounds: int = 2              # 1 = no rescue round
    resid_gate: bool = True
    clip: bool = True
    bridge: str = "consensus"    # consensus | naive | none
    min_support: int = 2
    final: str = "soft"          # hard: solve on gated edges; soft: GM refine on all edges from gated init
    core_k: int = 2              # prune cameras with < core_k kept edges before bridging
    bridge_thr: float = 3.0      # whitened gauge disagreement for consensus (fixed, not calibrated)
    rho: str = "point"           # none | point | safe
    rho_p: float = 0.2           # quantile for point estimate
    safe_p: float = 0.5          # quantile for safe (upper) estimate
    safe_eps: float = 0.4        # assumed max outlier fraction
    adapt_kernel: bool = True
    c: float = 5 * DEG
    alpha_edge: float = 0.10
    alpha_res: float = 0.10
    alpha_cam: float = 0.10
    # extensions E1/E2 (defaults reproduce the original method)
    cam_norm: str = "u"          # u | learn : per-camera normalizer (E2)
    cam_groups: int = 1          # >1: group-conditional bound thresholds by normalizer quantile (E2)
    augment: bool = False        # calibrate bounds on failure-mode augmented scenes (E1)


@dataclass
class Calib:
    a: float = 0.0
    b: float = 1.0
    const_sigma: float = None
    ref: np.ndarray = None        # sorted reference inlier scores (rho = 1)
    lam_edge: float = np.inf
    lam_res: float = np.inf
    lam_cam: float = np.inf
    beta: np.ndarray = None       # learned normalizer coefficients (E2)
    cam_cuts: np.ndarray = None   # group boundaries on log normalizer (E2)
    lam_cam_g: np.ndarray = None  # per-group thresholds (E2)


# ---------------------------------------------------------------- helpers
def quantile(x, p):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(np.quantile(x, np.clip(p, 0, 1), method="lower"))


def crc_threshold(score_lists, alpha, B=1.0):
    """Conformal risk control (Angelopoulos et al., ICLR 2024) for the loss
    L_s(lam) = mean_i 1[score_si > lam] (non-increasing in lam, bounded by B).
    Returns the smallest lam with n/(n+1) Rhat(lam) + B/(n+1) <= alpha."""
    n = len(score_lists)
    if B / (n + 1) > alpha:
        return np.inf
    arrs = [np.where(np.isfinite(s), s, np.inf) for s in score_lists if len(s) > 0]
    n = len(arrs)
    grid = np.unique(np.concatenate([a[np.isfinite(a)] for a in arrs] + [np.array([0.0])]))
    R = np.zeros(len(grid))
    for a in arrs:
        a = np.sort(a)
        # fraction of scores strictly greater than lam
        R += 1.0 - np.searchsorted(a, grid, side="right") / len(a)
    R /= n
    ok = n / (n + 1) * R + B / (n + 1) <= alpha
    return float(grid[np.argmax(ok)]) if ok.any() else np.inf


def recal_sigma(sig_hat, cal: Calib, cfg: Config):
    if cal.const_sigma is not None:
        return np.full_like(sig_hat, cal.const_sigma, dtype=float)
    if not cfg.recal:
        return np.asarray(sig_hat, float)
    return np.exp(cal.a) * np.asarray(sig_hat, float) ** cal.b


def edge_true_errors(sc):
    E, R = sc["edges"], sc["R"]
    return angle(T(R[E[:, 1]]) @ sc["Rij"] @ R[E[:, 0]])


def estimate_rho(s, cal: Calib, cfg: Config):
    if cfg.rho == "none" or cal.ref is None:
        return 1.0
    if cfg.rho == "point":
        num, den = quantile(s, cfg.rho_p), quantile(cal.ref, cfg.rho_p)
    elif cfg.rho == "safe":
        p, e = cfg.safe_p, cfg.safe_eps
        num, den = quantile(s, p), quantile(cal.ref, (p - e) / (1 - e))
    else:
        raise ValueError(cfg.rho)
    if not np.isfinite(num) or not np.isfinite(den) or den <= 0:
        return 1.0
    return max(1.0, num / den)


def rho_bracket(s, ref, p, eps):
    """Lemma 2: population bracket for the scale shift under eps-contamination."""
    lo = quantile(s, p) / quantile(ref, p / (1 - eps)) if p / (1 - eps) < 1 else np.nan
    hi = quantile(s, p) / quantile(ref, (p - eps) / (1 - eps)) if p > eps else np.nan
    return lo, hi


# ---------------------------------------------------------------- bridging
def prune_core(n, edges, keep, k):
    """k-core of the kept graph: repeatedly detach cameras with < k kept edges.
    Detached cameras can only return through consensus bridging, i.e. when at
    least min_support of their edges agree on the same pose."""
    keep = keep.copy()
    if k <= 1:
        return keep
    while True:
        deg = np.bincount(edges[keep].ravel(), minlength=n)
        low = (deg > 0) & (deg < k)
        if not low.any():
            return keep
        keep &= ~(low[edges[:, 0]] | low[edges[:, 1]])


def _solve_nodes(n, edges, Rij, keep, w, nodes, c):
    if len(nodes) == 1:
        R = np.full((n, 3, 3), np.nan)
        R[nodes[0]] = np.eye(3)
        return R
    return solve_cg(n, edges[keep], Rij[keep], w[keep], nodes=nodes, c=c)


def bridge(n, edges, Rij, sig, keep, w, cfg: Config, c, cand_score=None):
    """Attach components to the main component. 'consensus': a component is
    attached only if >= min_support candidate bridge edges agree on the
    relative gauge (whitened disagreement <= cfg.bridge_thr); otherwise its cameras
    abstain (unregistered). 'naive': add lowest-score rejected edges until
    connected (the prototype's connect-back). Returns keep, registered mask."""
    keep = keep.copy()
    if cfg.bridge == "naive":
        parent = np.arange(n)

        def f(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for e in np.where(keep)[0]:
            parent[f(edges[e, 0])] = f(edges[e, 1])
        order = np.argsort(np.nan_to_num(cand_score, nan=np.inf)) if cand_score is not None else np.arange(len(edges))
        for e in order:
            if not keep[e]:
                a, b = f(edges[e, 0]), f(edges[e, 1])
                if a != b:
                    parent[a] = b
                    keep[e] = True
        ncomp, lab = components(n, edges, keep)
        reg = lab == np.bincount(lab).argmax()
        return keep, reg
    ncomp, lab = components(n, edges, keep)
    main = np.bincount(lab).argmax()
    reg = lab == main
    if cfg.bridge == "none" or ncomp == 1:
        return keep, reg
    thr = cfg.bridge_thr
    for _pass in range(3):          # attach all components per pass; passes allow chains
        ncomp, lab = components(n, edges, keep)
        sizes = np.bincount(lab, minlength=ncomp)
        main = sizes.argmax()
        reg = lab == main
        Rm = _solve_nodes(n, edges, Rij, keep, w, np.where(reg)[0], c)
        # candidate bridge edges: rejected edges with exactly one endpoint in main
        e_ok = ~keep & (reg[edges[:, 0]] ^ reg[edges[:, 1]])
        other = np.where(reg[edges[:, 0]], lab[edges[:, 1]], lab[edges[:, 0]])
        attach = []
        for cc in np.unique(other[e_ok]):
            cand = np.where(e_ok & (other == cc))[0]
            if len(cand) < cfg.min_support:
                continue
            nodes = np.where(lab == cc)[0]
            Rc = _solve_nodes(n, edges, Rij, keep, w, nodes, c)
            i, j = edges[cand, 0], edges[cand, 1]
            fwd = reg[i]
            G = np.where(fwd[:, None, None],
                         T(Rc[j]) @ Rij[cand] @ Rm[i],
                         T(Rc[i]) @ T(Rij[cand]) @ Rm[j])
            D = angle(T(G)[:, None] @ G[None, :]) / np.sqrt(sig[cand][:, None] ** 2 + sig[cand][None, :] ** 2)
            agree = D <= thr
            sup = agree.sum(1)
            best = sup.argmax()
            if sup[best] >= cfg.min_support:
                attach.append(cand[agree[best]])
        if not attach:
            break
        keep[np.concatenate(attach)] = True
    ncomp, lab = components(n, edges, keep)
    reg = lab == np.bincount(lab).argmax()
    return keep, reg


# ---------------------------------------------------------------- pipeline
def stage_scores(sc, cfg: Config, cal: Calib, tri=None):
    sig0 = recal_sigma(sc["sig_hat"], cal, cfg)
    if tri is None:
        tri = Triangles(sc["n"], sc["edges"], cfg.K, np.random.default_rng(1))
    cyc = tri.cycle_angles(sc["Rij"])
    s0 = per_edge_quantile(tri.e, tri.normalized(cyc, sig0), tri.m, cfg.tri_q)
    rho = estimate_rho(s0, cal, cfg)
    return dict(sig=sig0 * rho, s1=s0 / rho, rho=rho, tri=tri, cyc=cyc, s0=s0)


def stage_gate(sc, st, cfg: Config, lam_edge):
    tri, sig = st["tri"], st["sig"]
    s1 = st["s1"]
    # edges without any triangle cannot be tested by cycles: deferred to the residual gate
    keep = np.isnan(s1) | (s1 <= lam_edge)
    if cfg.rounds >= 2 and len(tri):
        mask = keep[tri.e2] & keep[tri.e3]
        s2 = per_edge_quantile(tri.e, tri.normalized(st["cyc"], sig), tri.m, cfg.tri_q, mask)
        keep = keep | (np.nan_to_num(s2, nan=np.inf) <= lam_edge)   # rescue only
    return keep


def weights(sig, cfg: Config):
    w = 1.0 / sig ** 2
    if cfg.clip:
        w = np.clip(w, *np.quantile(w, [0.05, 0.95]))
    return w / w.mean()


def whitened_residuals(sc, R, sig):
    E = sc["edges"]
    r = angle(T(R[E[:, 1]]) @ sc["Rij"] @ R[E[:, 0]]) / sig
    return np.nan_to_num(r, nan=np.inf)


def run(sc, cfg: Config, cal: Calib, stop_after=None, tri=None):
    t0 = time.time()
    n, E, Rij = sc["n"], sc["edges"], sc["Rij"]
    st = stage_scores(sc, cfg, cal, tri)
    if stop_after == "scores":
        return st
    sig, rho = st["sig"], st["rho"]
    c = cfg.c * (rho if cfg.adapt_kernel else 1.0)
    keep = stage_gate(sc, st, cfg, cal.lam_edge)
    w = weights(sig, cfg)
    if cfg.bridge == "consensus":
        keep = prune_core(n, E, keep, cfg.core_k)
    keep, reg = bridge(n, E, Rij, sig, keep, w, cfg, c, cand_score=st["s1"])
    R = solve_cg(n, E[keep], Rij[keep], w[keep], nodes=np.where(reg)[0], c=c)
    if stop_after == "first_solve":
        return dict(st=st, R=R, reg=reg, keep=keep, sig=sig)
    if cfg.resid_gate:
        r = whitened_residuals(sc, R, sig)
        keep = r <= cal.lam_res
        if cfg.bridge == "consensus":
            keep = prune_core(n, E, keep, cfg.core_k)
        keep, reg = bridge(n, E, Rij, sig, keep, w, cfg, c, cand_score=r)
        R = solve_cg(n, E[keep], Rij[keep], w[keep], nodes=np.where(reg)[0], c=c)
    if cfg.final == "soft":
        from .solvers import irls
        R = irls(n, E, Rij, R, w, "gm", c=c, nodes=np.where(reg)[0], iters=20)
    kk = keep & reg[E[:, 0]] & reg[E[:, 1]]
    info = np.zeros(n)
    np.add.at(info, E[kk, 0], sig[kk] ** -2)
    np.add.at(info, E[kk, 1], sig[kk] ** -2)
    u = np.where(reg, 1.0 / np.sqrt(np.maximum(info, 1e-300)), np.inf)
    R[~reg] = np.nan
    # per-camera diagnostics (label-free): kept/total degree and local whitened residuals
    r_fin = whitened_residuals(sc, R, sig)
    deg_all = np.bincount(E.ravel(), minlength=n).astype(float)
    deg_keep = np.bincount(E[kk].ravel(), minlength=n).astype(float)
    res_med = np.full(n, np.nan)
    for i in np.where(reg)[0]:
        inc = kk & ((E[:, 0] == i) | (E[:, 1] == i))
        if inc.any():
            res_med[i] = np.median(r_fin[inc])
    ls = np.log(np.asarray(sc["sig_hat"], float))
    cam_sig = np.zeros(n)
    for i in range(n):
        inc = (E[:, 0] == i) | (E[:, 1] == i)
        if inc.any():
            cam_sig[i] = np.median(ls[inc]) - np.median(ls)
    feat = dict(deg_all=deg_all, deg_keep=deg_keep, rej_frac=1 - deg_keep / np.maximum(deg_all, 1), res_med=res_med,
                cam_sig=cam_sig)
    v = cam_normalizer(u, feat, cfg, cal)
    B = cam_bounds(v, cfg, cal)
    return dict(R=R, reg=reg, keep=keep, u=u, v=v, B=B, rho=rho, sig=sig,
                s1=st["s1"], s0=st["s0"], feat=feat, time=time.time() - t0)


def cam_features(feat):
    """label-free per-camera features: log local whitened residual, rejected-edge fraction,
    relative log confidence of the camera's edges, log kept degree"""
    return np.stack([np.log(np.maximum(np.nan_to_num(feat["res_med"], nan=1.0), 0.05)), feat["rej_frac"],
                     feat["cam_sig"], np.log(np.maximum(feat["deg_keep"], 1))], 1)


def cam_normalizer(u, feat, cfg: Config, cal: Calib):
    if cfg.cam_norm == "u" or cal.beta is None:
        return u
    X = cam_features(feat)
    return u * np.exp(np.clip(np.c_[np.ones(len(u)), X] @ cal.beta, -5, 5))


def cam_bounds(v, cfg: Config, cal: Calib):
    if cfg.cam_groups > 1 and cal.lam_cam_g is not None:
        g = np.digitize(np.log(v), cal.cam_cuts)
        return v * cal.lam_cam_g[np.minimum(g, len(cal.lam_cam_g) - 1)]
    return cal.lam_cam * v


# ---------------------------------------------------------------- calibration
def inlier_mask(sc):
    return ~sc["is_out"]


def calibrate(cfg: Config, D_fit, D_edge, D_res, D_cam, verbose=False):
    cal = Calib()
    # 1) power-law recalibration on inlier edges of D_fit
    x, y = [], []
    for sc in D_fit:
        m = inlier_mask(sc)
        x.append(np.log(sc["sig_hat"][m])); y.append(np.log(edge_true_errors(sc)[m] + 1e-9))
    x, y = np.concatenate(x), np.concatenate(y)
    if np.std(x) < 1e-9:
        cal.const_sigma = float(np.exp(np.mean(y)))
    else:
        cal.b, cal.a = np.polyfit(x, y, 1)
    # 2) reference inlier score distribution at rho = 1 (D_fit)
    ref_cfg = replace(cfg, rho="none")
    ref = []
    for sc in D_fit:
        st = stage_scores(sc, ref_cfg, cal)
        ref.append(st["s0"][inlier_mask(sc)])
    cal.ref = np.sort(np.concatenate(ref)[np.isfinite(np.concatenate(ref))])
    # 3) edge gate threshold (CRC on round-1 scores; rescue only lowers the loss)
    lists = []
    for sc in D_edge:
        st = stage_scores(sc, cfg, cal)
        s1 = st["s1"][inlier_mask(sc)]
        lists.append(s1[np.isfinite(s1)])      # loss over cycle-testable inlier edges
    cal.lam_edge = crc_threshold(lists, cfg.alpha_edge)
    # 4) residual gate threshold (CRC on whitened residuals after first solve)
    if cfg.resid_gate:
        lists = []
        for sc in D_res:
            out = run(sc, replace(cfg, resid_gate=False), cal, stop_after="first_solve")
            r = whitened_residuals(sc, out["R"], out["sig"])
            lists.append(r[inlier_mask(sc)])
        cal.lam_res = crc_threshold(lists, cfg.alpha_res)
    # 5) per-camera bounds. E1: failure-mode augmentation of the calibration scenes; E2: learned
    #    normalizer (fitted on D_fit) and group-conditional thresholds (groups fixed on D_fit)
    Dc, Df = D_cam, D_fit
    if cfg.augment:
        from .augment import augment
        pool = np.concatenate([np.asarray(sc["sig_hat"], float)[sc["is_out"]] for sc in D_fit])
        Dc = augment(D_cam, seed=11, sig_pool=pool)
        Df = augment(D_fit, seed=12, sig_pool=pool)
    cal_tmp = replace(cal, lam_cam=1.0, beta=None)
    base = replace(cfg, cam_norm="u", cam_groups=1)
    if cfg.cam_norm == "learn" or cfg.cam_groups > 1:
        Xs, ys, recs = [], [], []
        for sc in Df:
            out = run(sc, base, cal_tmp)
            err = rot_errors(sc["R"], out["R"]); reg = out["reg"]
            Xs.append(cam_features(out["feat"])[reg]); ys.append(np.log(np.maximum(err[reg], 1e-4) / out["u"][reg]))
            recs.append((out["u"], out["feat"], reg))
        if cfg.cam_norm == "learn":
            X, y = np.concatenate(Xs), np.concatenate(ys)
            cal.beta = np.linalg.lstsq(np.c_[np.ones(len(X)), X], y, rcond=None)[0]
        if cfg.cam_groups > 1:
            lv = np.concatenate([np.log(cam_normalizer(u, f, cfg, cal)[reg]) for u, f, reg in recs])
            cal.cam_cuts = np.quantile(lv, np.linspace(0, 1, cfg.cam_groups + 1)[1:-1])
    lists = []
    for sc in Dc:
        out = run(sc, base, cal_tmp)
        err = rot_errors(sc["R"], out["R"]); reg = out["reg"]
        v = cam_normalizer(out["u"], out["feat"], cfg, cal)
        lists.append((err[reg] / v[reg], np.log(v[reg])))
    cal.lam_cam = crc_threshold([a for a, _ in lists], cfg.alpha_cam)
    if cfg.cam_groups > 1:
        lam = []
        for gi in range(cfg.cam_groups):
            L = [a[np.digitize(b, cal.cam_cuts) == gi] for a, b in lists]
            lam.append(crc_threshold([l for l in L if len(l)], cfg.alpha_cam))
        cal.lam_cam_g = np.array(lam)
    if verbose:
        print(f"calib: b={cal.b:.3f} lam_edge={cal.lam_edge:.3f} lam_res={cal.lam_res:.3f} lam_cam={cal.lam_cam:.3f}")
    return cal
