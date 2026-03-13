from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


STATE_SIZE = 18


def _as_vec3(x, default):
    if x is None:
        return np.array(default, dtype=float)
    arr = np.asarray(x, dtype=float).reshape(3)
    return arr.copy()


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    n = np.linalg.norm(q)
    if n <= 0.0:
        raise ValueError("Quaternion norm must be positive.")
    return q / n


@dataclass
class ESKFState:
    # Nominal state
    p: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))          # position
    v: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))          # velocity
    q: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))   # quaternion [w, x, y, z]
    ba: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))         # accel bias
    bw: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))         # gyro bias
    g: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 9.81]))        # NED gravity

    # Error-state covariance
    P: np.ndarray = field(default_factory=lambda: np.eye(STATE_SIZE, dtype=float) * 1e-3)

    def __post_init__(self):
        self.p = _as_vec3(self.p, [0.0, 0.0, 0.0])
        self.v = _as_vec3(self.v, [0.0, 0.0, 0.0])
        self.q = quat_normalize(self.q)
        self.ba = _as_vec3(self.ba, [0.0, 0.0, 0.0])
        self.bw = _as_vec3(self.bw, [0.0, 0.0, 0.0])
        self.g = _as_vec3(self.g, [0.0, 0.0, 9.81])

        self.P = np.asarray(self.P, dtype=float)
        if self.P.shape != (STATE_SIZE, STATE_SIZE):
            raise ValueError(f"P must have shape ({STATE_SIZE}, {STATE_SIZE})")

    @staticmethod
    def nominal_size() -> int:
        # p(3) + v(3) + q(4) + ba(3) + bw(3) + g(3)
        return 19

    @staticmethod
    def error_size() -> int:
        # dp(3) + dv(3) + dtheta(3) + dba(3) + dbw(3) + dg(3)
        return 18

    def copy(self) -> "ESKFState":
        return ESKFState(
            p=self.p.copy(),
            v=self.v.copy(),
            q=self.q.copy(),
            ba=self.ba.copy(),
            bw=self.bw.copy(),
            g=self.g.copy(),
            P=self.P.copy(),
        )

    def nominal_vector(self) -> np.ndarray:
        return np.hstack([self.p, self.v, self.q, self.ba, self.bw, self.g])

    def reset_covariance(self, sigma_p=1.0, sigma_v=1.0, sigma_theta=0.1,
                         sigma_ba=0.1, sigma_bw=0.01, sigma_g=0.1) -> None:
        self.P = np.zeros((STATE_SIZE, STATE_SIZE), dtype=float)
        self.P[0:3, 0:3] = np.eye(3) * sigma_p**2
        self.P[3:6, 3:6] = np.eye(3) * sigma_v**2
        self.P[6:9, 6:9] = np.eye(3) * sigma_theta**2
        self.P[9:12, 9:12] = np.eye(3) * sigma_ba**2
        self.P[12:15, 12:15] = np.eye(3) * sigma_bw**2
        self.P[15:18, 15:18] = np.eye(3) * sigma_g**2