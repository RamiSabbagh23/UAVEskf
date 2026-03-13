from __future__ import annotations

import numpy as np

from fusionUAV.eskf.state import ESKFState
from fusionUAV.eskf.jacobian import build_Fx_Fi_Qi, propagate_covariance
from fusionUAV.utils.rotations import quat_multiply, quat_normalize, quat_to_rotmat, rotvec_to_quat


def predict_nominal(
    state: ESKFState,
    acc_m: np.ndarray,
    gyro_m: np.ndarray,
    dt: float,
) -> ESKFState:
    """
    Nominal-state IMU propagation for ESKF.

    State convention:
        p  : position in navigation frame
        v  : velocity in navigation frame
        q  : body -> navigation quaternion [w, x, y, z]
        ba : accelerometer bias in body frame
        bw : gyroscope bias in body frame
        g  : gravity in navigation frame
    """
    if dt <= 0.0:
        return state.copy()

    x = state.copy()

    acc_m = np.asarray(acc_m, dtype=float).reshape(3)
    gyro_m = np.asarray(gyro_m, dtype=float).reshape(3)

    # Bias-corrected IMU
    acc_b = acc_m - x.ba
    omega_b = gyro_m - x.bw

    # Rotation body -> navigation
    R_nb = quat_to_rotmat(x.q)

    # Navigation-frame acceleration
    a_n = R_nb @ acc_b + x.g

    # Position / velocity propagation
    x.p = x.p + x.v * dt + 0.5 * a_n * dt * dt 
    x.v = x.v + a_n * dt

    # Quaternion propagation using rotation increment
    dq = rotvec_to_quat(omega_b * dt)
    x.q = quat_normalize(quat_multiply(x.q, dq))

    # Biases and gravity are modeled as constant in the nominal state
    x.ba = x.ba.copy()
    x.bw = x.bw.copy()
    x.g = x.g.copy()

    return x

def predict(
    state: ESKFState,
    acc_m: np.ndarray,
    gyro_m: np.ndarray,
    dt: float,
    sigma_acc_white: float,
    sigma_gyro_white: float,
    sigma_acc_bias_rw: float,
    sigma_gyro_bias_rw: float,
) -> ESKFState:
    """
    Full ESKF prediction:
      1) propagate nominal state with IMU
      2) propagate error covariance with first-order discrete Jacobians
    """
    if dt <= 0.0:
        return state.copy()

    x_pred = predict_nominal(state, acc_m, gyro_m, dt)

    Fx, Fi, Qi = build_Fx_Fi_Qi(
        state=state,
        acc_m=acc_m,
        gyro_m=gyro_m,
        dt=dt,
        sigma_acc_white=sigma_acc_white,
        sigma_gyro_white=sigma_gyro_white,
        sigma_acc_bias_rw=sigma_acc_bias_rw,
        sigma_gyro_bias_rw=sigma_gyro_bias_rw,
    )

    x_pred.P = propagate_covariance(state.P, Fx, Fi, Qi)
    return x_pred