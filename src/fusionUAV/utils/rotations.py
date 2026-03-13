from __future__ import annotations

import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=float,
    )


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    n = np.linalg.norm(q)
    if n <= 0.0:
        raise ValueError("Quaternion norm must be positive.")
    return q / n


def quat_conj(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Hamilton product for quaternions stored as [w, x, y, z].
    """
    w1, x1, y1, z1 = np.asarray(q1, dtype=float).reshape(4)
    w2, x2, y2, z2 = np.asarray(q2, dtype=float).reshape(4)

    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def rotvec_to_quat(phi: np.ndarray) -> np.ndarray:
    """
    Exponential map Exp(phi): R^3 -> unit quaternion, Hamilton convention.
    """
    phi = np.asarray(phi, dtype=float).reshape(3)
    angle = np.linalg.norm(phi)

    if angle < 1e-12:
        # small-angle approximation: q ≈ [1, 0.5*phi]
        return quat_normalize(np.array([1.0, 0.5 * phi[0], 0.5 * phi[1], 0.5 * phi[2]], dtype=float))

    axis = phi / angle
    half = 0.5 * angle
    s = np.sin(half)
    return np.array([np.cos(half), axis[0] * s, axis[1] * s, axis[2] * s], dtype=float)


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """
    Log map Log(q): unit quaternion -> R^3 rotation vector.
    """
    q = quat_normalize(q)
    qw = q[0]
    qv = q[1:]
    nv = np.linalg.norm(qv)

    if nv < 1e-12:
        return 2.0 * qv

    angle = 2.0 * np.arctan2(nv, qw)
    axis = qv / nv
    return axis * angle


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    w, x, y, z = q

    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def rotate_vector(q: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Rotate 3D vector x using q ⊗ [0,x] ⊗ q*.
    """
    q = quat_normalize(q)
    xq = np.array([0.0, *np.asarray(x, dtype=float).reshape(3)], dtype=float)
    yq = quat_multiply(quat_multiply(q, xq), quat_conj(q))
    return yq[1:]