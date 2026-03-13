from __future__ import annotations

import numpy as np

from fusionUAV.eskf.inject import inject_error
from fusionUAV.eskf.predict import predict
from fusionUAV.eskf.reset import reset_error_state_covariance
from fusionUAV.eskf.state import ESKFState
from fusionUAV.eskf.update_baro import update_baro_altitude
from fusionUAV.eskf.update_gps import update_gps_position
from fusionUAV.eskf.update_mag import update_mag_unit_vector


class ESKF:
    def __init__(
        self,
        state: ESKFState | None = None,
        sigma_acc_white: float = 0.05,
        sigma_gyro_white: float = 0.005,
        sigma_acc_bias_rw: float = 0.0001,
        sigma_gyro_bias_rw: float = 0.00001,
        use_exact_reset_jacobian: bool = True,
    ) -> None:
        self.state = ESKFState() if state is None else state

        self.sigma_acc_white = float(sigma_acc_white)
        self.sigma_gyro_white = float(sigma_gyro_white)
        self.sigma_acc_bias_rw = float(sigma_acc_bias_rw)
        self.sigma_gyro_bias_rw = float(sigma_gyro_bias_rw)

        self.use_exact_reset_jacobian = bool(use_exact_reset_jacobian)

    def predict(
        self,
        acc_m: np.ndarray,
        gyro_m: np.ndarray,
        dt: float,
    ) -> ESKFState:
        self.state = predict(
            state=self.state,
            acc_m=acc_m,
            gyro_m=gyro_m,
            dt=dt,
            sigma_acc_white=self.sigma_acc_white,
            sigma_gyro_white=self.sigma_gyro_white,
            sigma_acc_bias_rw=self.sigma_acc_bias_rw,
            sigma_gyro_bias_rw=self.sigma_gyro_bias_rw,
        )
        return self.state

    def update_gps(
        self,
        z_gps_pos: np.ndarray,
        R_gps: np.ndarray,
    ) -> tuple[ESKFState, np.ndarray]:
        self.state, dx, K, innovation = update_gps_position(
            state=self.state,
            z_gps_pos=z_gps_pos,
            R_gps=R_gps,
        )
        self.state = inject_error(self.state, dx)
        self.state = reset_error_state_covariance(
            self.state,
            dx,
            use_exact_reset_jacobian=self.use_exact_reset_jacobian,
        )
        return self.state, innovation

    def update_baro(
        self,
        z_baro_alt_m: float,
        sigma_baro_alt_m: float,
    ) -> tuple[ESKFState, float]:
        self.state, dx, K, innovation = update_baro_altitude(
            state=self.state,
            z_baro_alt_m=z_baro_alt_m,
            sigma_baro_alt_m=sigma_baro_alt_m,
        )
        self.state = inject_error(self.state, dx)
        self.state = reset_error_state_covariance(
            self.state,
            dx,
            use_exact_reset_jacobian=self.use_exact_reset_jacobian,
        )
        return self.state, innovation

    def update_mag(
        self,
        z_mag_b: np.ndarray,
        mag_n: np.ndarray,
        R_mag: np.ndarray,
    ) -> tuple[ESKFState, np.ndarray]:
        self.state, dx, K, innovation = update_mag_unit_vector(
            state=self.state,
            z_mag_b=z_mag_b,
            mag_n=mag_n,
            R_mag=R_mag,
        )
        self.state = inject_error(self.state, dx)
        self.state = reset_error_state_covariance(
            self.state,
            dx,
            use_exact_reset_jacobian=self.use_exact_reset_jacobian,
        )
        return self.state, innovation

    def get_state(self) -> ESKFState:
        return self.state.copy()

    def set_state(self, state: ESKFState) -> None:
        self.state = state.copy()