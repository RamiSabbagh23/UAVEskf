from __future__ import annotations

import numpy as np

from fusionUAV.eskf.state import ESKFState


def update_baro_altitude(
    state: ESKFState,
    z_baro_alt_m: float,
    sigma_baro_alt_m: float,
) -> tuple[ESKFState, np.ndarray, np.ndarray, float]:
    """
    Barometer altitude update for the ESKF.

    Assumption used here:
        - navigation frame is NED
        - state.p[2] is 'down' in meters
        - barometer reports altitude above origin in meters (positive up)

    Therefore:
        h(x) = -p_z

    Returns:
        state_upd   : state with updated covariance P
        delta_x_hat : estimated error-state mean (to inject later)
        K           : Kalman gain
        innovation  : scalar residual z - h(x)
    """
    x = state.copy()

    z = float(z_baro_alt_m)
    R = np.array([[float(sigma_baro_alt_m) ** 2]], dtype=float)

    # predicted barometric altitude from nominal state
    z_hat = -x.p[2]

    # innovation
    innovation = z - z_hat

    # Jacobian wrt error state [dp, dv, dtheta, dba, dbw, dg]
    H = np.zeros((1, 18), dtype=float)
    H[0, 2] = -1.0   # h(x) = -p_z

    S = H @ x.P @ H.T + R
    K = x.P @ H.T @ np.linalg.inv(S)

    delta_x_hat = (K * innovation).reshape(18)

    # Joseph-form covariance update
    I = np.eye(18, dtype=float)
    x.P = (I - K @ H) @ x.P @ (I - K @ H).T + K @ R @ K.T
    x.P = 0.5 * (x.P + x.P.T)
    
    return x, delta_x_hat, K, innovation