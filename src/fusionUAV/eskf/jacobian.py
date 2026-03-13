from __future__ import annotations

import numpy as np

from fusionUAV.eskf.state import ESKFState
from fusionUAV.utils.rotations import quat_to_rotmat, skew


def build_Fx_Fi_Qi(
    state: ESKFState,
    acc_m: np.ndarray,
    gyro_m: np.ndarray,
    dt: float,
    sigma_acc_white: float,
    sigma_gyro_white: float,
    sigma_acc_bias_rw: float,
    sigma_gyro_bias_rw: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the discrete-time error-state transition Jacobian Fx,
    perturbation Jacobian Fi, and perturbation covariance Qi.

    Error-state ordering:
        [dp, dv, dtheta, dba, dbw, dg]  -> 18 states

    Perturbation ordering:
        [vi, thetai, ai, omegai] -> 12 perturbation components
    """
    acc_m = np.asarray(acc_m, dtype=float).reshape(3)
    gyro_m = np.asarray(gyro_m, dtype=float).reshape(3)

    R = quat_to_rotmat(state.q)
    a_hat = acc_m - state.ba
    w_hat = gyro_m - state.bw

    Fx = np.eye(18, dtype=float)

    # dp <- dp + dv * dt
    Fx[0:3, 3:6] = np.eye(3) * dt

    # dv <- dv + (-R [a_hat]x dtheta - R dba + dg) * dt
    Fx[3:6, 6:9] = -R @ skew(a_hat) * dt
    Fx[3:6, 9:12] = -R * dt
    Fx[3:6, 15:18] = np.eye(3) * dt

    # dtheta <- R^T{(wm - bw) dt} dtheta - dbw * dt
    # Small-angle discrete approximation:
    Fx[6:9, 6:9] = np.eye(3) - skew(w_hat) * dt
    Fx[6:9, 12:15] = -np.eye(3) * dt

    # dba, dbw, dg keep identity blocks already set

    Fi = np.zeros((18, 12), dtype=float)

    # [vi, thetai, ai, omegai]
    Fi[3:6, 0:3] = np.eye(3)      # vi into dv
    Fi[6:9, 3:6] = np.eye(3)      # thetai into dtheta
    Fi[9:12, 6:9] = np.eye(3)     # ai into dba
    Fi[12:15, 9:12] = np.eye(3)   # omegai into dbw

    Vi = (sigma_acc_white ** 2) * (dt ** 2) * np.eye(3, dtype=float)
    Thetai = (sigma_gyro_white ** 2) * (dt ** 2) * np.eye(3, dtype=float)
    Ai = (sigma_acc_bias_rw ** 2) * dt * np.eye(3, dtype=float)
    Omegai = (sigma_gyro_bias_rw ** 2) * dt * np.eye(3, dtype=float)

    Qi = np.zeros((12, 12), dtype=float)
    Qi[0:3, 0:3] = Vi
    Qi[3:6, 3:6] = Thetai
    Qi[6:9, 6:9] = Ai
    Qi[9:12, 9:12] = Omegai

    return Fx, Fi, Qi


def propagate_covariance(
    P: np.ndarray,
    Fx: np.ndarray,
    Fi: np.ndarray,
    Qi: np.ndarray,
) -> np.ndarray:
    """
    Discrete covariance propagation.
    """
    P = np.asarray(P, dtype=float)
    P_next = Fx @ P @ Fx.T + Fi @ Qi @ Fi.T

    # enforce symmetry numerically
    P_next = 0.5 * (P_next + P_next.T)
    return P_next