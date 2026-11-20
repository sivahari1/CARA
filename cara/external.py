"""Released-implementation baselines.

GLOMAP-RA : rotation averaging of GLOMAP as shipped in COLMAP 4 (pycolmap 4.2,
            `run_rotation_averaging`, default options: Geman-McClure IRLS after L1,
            uniform weights). Pan et al., ECCV 2024.
Shonan    : Shonan rotation averaging (GTSAM, Dellaert et al., ECCV 2020), certifiable
            for the chordal L2 cost; optionally with Huber robust noise ('Shonan-H').
Both return world->camera rotations (n,3,3), NaN for cameras a method does not register.
"""
import numpy as np
from .so3 import T


def run_glomap_ra(sc, reweighting="UNIFORM", seed=0):
    import pycolmap as pc
    n, E, Rij = sc["n"], sc["edges"], sc["Rij"]
    rec = pc.Reconstruction()
    cam = pc.Camera.create(1, pc.CameraModelId.SIMPLE_PINHOLE, 1000.0, 1000, 1000)
    rec.add_camera_with_trivial_rig(cam)
    for i in range(n):
        rec.add_image_with_trivial_frame(pc.Image(name=f"im{i}", camera_id=1, image_id=i + 1))
    g = pc.PoseGraph()
    for (i, j), R in zip(E, Rij):
        M = np.zeros((3, 4)); M[:, :3] = R          # cam_j_from_cam_i rotation; translation unused
        M[:, 3] = np.array([1.0, 0.0, 0.0])
        e = pc.PoseGraphEdge(pc.Rigid3d(M))
        e.num_matches = 100
        g.add_edge(int(i) + 1, int(j) + 1, e)
    opt = pc.RotationEstimatorOptions()
    opt.random_seed = seed
    opt.reweighting = getattr(pc.RotationAveragingReweighting, reweighting)
    ok = pc.run_rotation_averaging(opt, g, rec, [])
    out = np.full((n, 3, 3), np.nan)
    if not ok:
        return out
    for i in range(n):
        im = rec.image(i + 1)
        if im.has_pose:
            out[i] = im.cam_from_world().rotation.matrix() if callable(im.cam_from_world) else im.cam_from_world.rotation.matrix()
    return out


def run_shonan(sc, huber=True, p=4, pmax=8, seed=0):
    import gtsam
    n, E, Rij = sc["n"], sc["edges"], sc["Rij"]
    noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.05)
    meas = [gtsam.BinaryMeasurementRot3(int(i), int(j), gtsam.Rot3(T(R)), noise) for (i, j), R in zip(E, Rij)]
    params = gtsam.ShonanAveragingParameters3(gtsam.LevenbergMarquardtParams.CeresDefaults())
    params.setUseHuber(huber)
    params.setCertifyOptimality(not huber)
    sa = gtsam.ShonanAveraging3(meas, params)
    out = np.full((n, 3, 3), np.nan)
    try:
        init = sa.initializeRandomly()
        vals, _ = sa.run(init, p, p) if huber else sa.run(init, 3, pmax)
    except RuntimeError:
        return out                                  # did not converge: no camera registered
    for i in range(n):
        if vals.exists(i):
            out[i] = T(vals.atRot3(i).matrix())     # camera->world  ->  world->camera
    return out
