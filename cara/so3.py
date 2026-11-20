"""Batched SO(3) utilities. Convention: absolute R_i maps world -> camera i,
relative R_ij = R_j R_i^T (so R_j = R_ij R_i)."""
import numpy as np
from scipy.spatial.transform import Rotation as _Rot


def expm(w):
    w = np.asarray(w, float)
    return _Rot.from_rotvec(w.reshape(-1, 3)).as_matrix().reshape(w.shape[:-1] + (3, 3))


def logm(R):
    R = np.asarray(R, float)
    return _Rot.from_matrix(R.reshape(-1, 3, 3)).as_rotvec().reshape(R.shape[:-2] + (3,))


def angle(R):
    """geodesic angle (rad) of rotation matrices, batched over leading dims."""
    c = (np.trace(R, axis1=-2, axis2=-1) - 1.0) / 2.0
    return np.arccos(np.clip(c, -1.0, 1.0))


def proj_so3(M):
    U, _, Vt = np.linalg.svd(M)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.ones(M.shape[:-2] + (3,))
    D[..., 2] = d
    return (U * D[..., None, :]) @ Vt


def random_rotations(n, rng):
    return _Rot.random(n, random_state=int(rng.integers(2**31))).as_matrix()


def T(R):
    return np.swapaxes(R, -1, -2)
