from __future__ import annotations

import numpy as np

from fusionUAV.eskf.state import ESKFState
from fusionUAV.utils.rotations import quat_multiply, quat_normalize, rotvec_to_quat


def inject_error(state: ESKFState, delta_x_hat: np.ndarray) -> ESKFState:
    """
    Inject the estimated error-state mean into the nominal state.

    Error-state ordering:
        delta_x = [dp, dv, dtheta, dba, dbw, dg]
                = [0:3, 3:6, 6:9, 9:12, 12:15, 15:18]
    """
    dx = np.asarray(delta_x_hat, dtype=float).reshape(18)

    x = state.copy()

    dp = dx[0:3]
    dv = dx[3:6]
    dtheta = dx[6:9]
    dba = dx[9:12]
    dbw = dx[12:15]
    dg = dx[15:18]

    x.p = x.p + dp
    x.v = x.v + dv
    x.q = quat_normalize(quat_multiply(x.q, rotvec_to_quat(dtheta)))
    x.ba = x.ba + dba
    x.bw = x.bw + dbw
    x.g = x.g + dg

    return x