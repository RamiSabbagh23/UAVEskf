from __future__ import annotations

import numpy as np

from fusionUAV.eskf.state import ESKFState
from fusionUAV.utils.rotations import skew


def reset_error_state_covariance(
    state: ESKFState,
    delta_x_hat: np.ndarray,
    use_exact_reset_jacobian: bool = True,
) -> ESKFState:
    """
    Reset the error-state mean to zero after injection and update the covariance.

    Error-state ordering:
        [dp, dv, dtheta, dba, dbw, dg]

    If use_exact_reset_jacobian is False, use G = I as the common approximation.
    """
    dx = np.asarray(delta_x_hat, dtype=float).reshape(18)

    x = state.copy()

    G = np.eye(18, dtype=float)

    if use_exact_reset_jacobian:
        dtheta_hat = dx[6:9]
        G[6:9, 6:9] = np.eye(3) - 0.5 * skew(dtheta_hat)

    x.P = G @ x.P @ G.T
    x.P = 0.5 * (x.P + x.P.T)  # enforce symmetry numerically

    return x