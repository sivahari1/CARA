"""Label-free calibration audit of per-edge rotation uncertainties.

If the per-axis std sigma_e were correct and noise isotropic Gaussian, the
normalized cycle angle t = angle(C_ijk)/sqrt(sum sigma^2) of a clean triangle
would follow a Maxwell (chi_3) law to first order. A scale miscalibration kappa
gives t ~ kappa * chi_3, so kappa(p) = Q_p(t)/Q_p(chi_3) at low p (dominated by
clean triangles) estimates kappa without reference poses.
Validated on synthetic scenes with known kappa, then applied to the real
Madrid Metropolis covariances of Zhang et al. (CVPR 2023).
"""
from common import *
from scipy.stats import chi
from cara.graph import Triangles
from cara.real import load_madrid_cov

PS = [0.05, 0.1, 0.2, 0.3, 0.5]


def kappa_curve(sc, K=30):
    tri = Triangles(sc["n"], sc["edges"], K, np.random.default_rng(0))
    t = tri.normalized(tri.cycle_angles(sc["Rij"]), sc["sig_hat"])
    return {p: float(np.quantile(t, p) / chi(3).ppf(p)) for p in PS}, len(t)


if __name__ == "__main__":
    out = {"synthetic": [], "madrid": None}
    rng = np.random.default_rng(11)
    for kappa in [1.0, 2.4, 5.0, 10.0]:
        for po in [0.0, 0.2]:
            curves = []
            for _ in range(5):
                sc = make_scene(rng, n=150, n_rand=300, p_out=po, overconf=kappa, conf_noise=0.0)
                curves.append(kappa_curve(sc)[0])
            out["synthetic"].append(dict(kappa=kappa, p_out=po, est={p: float(np.mean([c[p] for c in curves])) for p in PS}))
            print(out["synthetic"][-1], flush=True)
    m = load_madrid_cov()
    curve, ntri = kappa_curve(m)
    sig = m["sig_hat"] / DEG
    out["madrid"] = dict(n=m["n"], m=len(m["edges"]), n_tri=ntri, est=curve,
                         sigma_axis_deg_quartiles=[float(np.quantile(sig, q)) for q in [0.25, 0.5, 0.75]])
    print(out["madrid"])
    json.dump(out, open(os.path.join(RES, "audit_madrid.json"), "w"), indent=1)
