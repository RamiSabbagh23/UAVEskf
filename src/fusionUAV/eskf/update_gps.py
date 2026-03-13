from __future__ import annotations

import numpy as np

from fusionUAV.eskf.state import ESKFState


def update_gps_position(
    state: ESKFState,
    z_gps_pos: np.ndarray,
    R_gps: np.ndarray,
) -> tuple[ESKFState, np.ndarray, np.ndarray, np.ndarray]:
    """
    GPS position update for the ESKF.

    Measurement model:
        y = p + v
    where p is the nominal position state and v ~ N(0, R_gps).

    Returns:
        state_upd   : state with updated covariance P
        delta_x_hat : estimated error-state mean (to inject later)
        K           : Kalman gain
        innovation  : residual z - h(x)
    """
    x = state.copy()

    z = np.asarray(z_gps_pos, dtype=float).reshape(3)
    R = np.asarray(R_gps, dtype=float).reshape(3, 3)

    # predicted measurement from nominal state
    z_hat = x.p.copy()

    # innovation
    innovation = z - z_hat

    # Jacobian wrt error state [dp, dv, dtheta, dba, dbw, dg]
    H = np.zeros((3, 18), dtype=float)
    H[:, 0:3] = np.eye(3, dtype=float)

    S = H @ x.P @ H.T + R
    K = x.P @ H.T @ np.linalg.inv(S)

    delta_x_hat = K @ innovation

    # Joseph-form covariance update for numerical stability
    I = np.eye(18, dtype=float)
    x.P = (I - K @ H) @ x.P @ (I - K @ H).T + K @ R @ K.T
    x.P = 0.5 * (x.P + x.P.T)

    return x, delta_x_hat, K, innovation