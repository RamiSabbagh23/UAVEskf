from __future__ import annotations

import numpy as np

from fusionUAV.eskf.state import ESKFState
from fusionUAV.utils.rotations import quat_to_rotmat, skew


def update_mag_unit_vector(
    state: ESKFState,
    z_mag_b: np.ndarray,
    mag_n: np.ndarray,
    R_mag: np.ndarray,
) -> tuple[ESKFState, np.ndarray, np.ndarray, np.ndarray]:
    """
    Magnetometer update using a normalized field-direction measurement.

    Measurement model:
        z = R_nb(q).T @ mag_n + v

    where:
        - z_mag_b is the measured magnetic field direction in body frame (3,)
        - mag_n   is the known Earth magnetic field direction in navigation frame (3,)
        - R_nb(q) maps body -> navigation
        - R_nb(q).T maps navigation -> body

    Important:
        This update uses only field DIRECTION, not magnitude.
    """
    x = state.copy()

    z = np.asarray(z_mag_b, dtype=float).reshape(3)
    m_n = np.asarray(mag_n, dtype=float).reshape(3)
    R = np.asarray(R_mag, dtype=float).reshape(3, 3)

    # Normalize measurement and reference field direction
    z_norm = np.linalg.norm(z)
    m_norm = np.linalg.norm(m_n)
    if z_norm <= 0.0 or m_norm <= 0.0:
        raise ValueError("Magnetometer vectors must have nonzero norm.")

    z = z / z_norm
    m_n = m_n / m_norm

    R_nb = quat_to_rotmat(x.q)

    # Predicted body-frame magnetic direction
    z_hat = R_nb.T @ m_n

    # Innovation
    innovation = z - z_hat

    # Jacobian wrt error state [dp, dv, dtheta, dba, dbw, dg]
    H = np.zeros((3, 18), dtype=float)

    # For local-angle ESKF with q_true = q ⊗ dq:
    # z_hat ≈ R(q)^T m_n - [R(q)^T m_n]_x dtheta
    # so H_theta = skew(z_hat)
    H[:, 6:9] = skew(z_hat)

    S = H @ x.P @ H.T + R
    K = x.P @ H.T @ np.linalg.inv(S)

    delta_x_hat = K @ innovation

    # Joseph-form covariance update
    I = np.eye(18, dtype=float)
    x.P = (I - K @ H) @ x.P @ (I - K @ H).T + K @ R @ K.T
    x.P = 0.5 * (x.P + x.P.T)

    # print('z_hat=', z_hat)
    # print("z = ", z)
    # print("innovation = ", innovation)
    # print('K=', K)
    # print("mn = ", m_n)
    # print("Rot = ", R_nb)
    # print('delta_x_hat=', delta_x_hat)
    # print('x.P=', x.P)
    # print('H=', H)
    # print("dtheta=", delta_x_hat[6:9])
    # print("---")



    return x, delta_x_hat, K, innovation