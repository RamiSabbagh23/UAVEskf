from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

G0 = 9.80665


@dataclass
class IMUNoiseConfig:
    sigma_acc_white: float
    sigma_gyro_white: float
    sigma_acc_bias_rw: float
    sigma_gyro_bias_rw: float


@dataclass
class GPSNoiseConfig:
    R_gps: np.ndarray


@dataclass
class BaroNoiseConfig:
    sigma_baro_alt_m: float


@dataclass
class MagNoiseConfig:
    R_mag: np.ndarray
    mag_n: np.ndarray


@dataclass
class SensorConfigBundle:
    imu: IMUNoiseConfig
    gps: GPSNoiseConfig
    baro: BaroNoiseConfig
    mag: MagNoiseConfig


def _load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_vehicle_settings(data: dict[str, Any], vehicle_name: str) -> dict[str, Any]:
    vehicles = data.get("Vehicles", {})
    if vehicle_name not in vehicles:
        raise KeyError(f"Vehicle '{vehicle_name}' not found in settings.json under Vehicles.")
    return vehicles[vehicle_name]


def _get_sensor_block(vehicle_cfg: dict[str, Any], sensor_name: str) -> dict[str, Any]:
    sensors = vehicle_cfg.get("Sensors", {})
    if sensor_name not in sensors:
        raise KeyError(f"Sensor '{sensor_name}' not found under vehicle Sensors.")
    return sensors[sensor_name]


def _std_from_variance(var_value: Any, default: float) -> float:
    if var_value is None:
        return float(default)
    var_value = float(var_value)
    if var_value < 0.0:
        raise ValueError("Variance must be nonnegative.")
    return float(np.sqrt(var_value))


def _pick_first_float(cfg: dict[str, Any], keys: list[str], default: float | None = None) -> float:
    for key in keys:
        if key in cfg and cfg[key] is not None:
            return float(cfg[key])
    if default is None:
        raise KeyError(f"None of keys found: {keys}")
    return float(default)


from typing import Any


def calculate_baro_sigma(
    baro_cfg: dict[str, Any],
    current_pressure_pa: float = 101325.0,
    current_temperature_k: float = 288.15,
) -> float:

    R_SPECIFIC_AIR = 287.05287  # J/(kg*K)
    G = 9.80665                 # m/s^2

    sigma_pressure_pa = float(baro_cfg.get("UncorrelatedNoiseSigma", 2.75))
    if sigma_pressure_pa < 0.0:
        raise ValueError("UncorrelatedNoiseSigma must be non-negative.")
    if current_pressure_pa <= 0.0:
        raise ValueError("current_pressure_pa must be positive.")
    if current_temperature_k <= 0.0:
        raise ValueError("current_temperature_k must be positive.")
    # Local pressure gradient magnitude |dP/dh| in Pa/m
    pa_per_meter = current_pressure_pa * G / (R_SPECIFIC_AIR * current_temperature_k)

    sigma_alt_m = sigma_pressure_pa / pa_per_meter
    return float(sigma_alt_m)


def _deg_per_sqrt_hr_to_rad_per_sqrt_s(x: float) -> float:
    return float(x) * np.pi / 180.0 / np.sqrt(3600.0)


def _deg_per_hr_to_rad_per_s(x: float) -> float:
    return float(x) * np.pi / 180.0 / 3600.0


def _mg_to_mps2(x: float) -> float:
    return float(x) * 1e-3 * G0


def _ug_to_mps2(x: float) -> float:
    return float(x) * 1e-6 * G0


def build_sensor_configs(
    settings_path: str | Path,
    vehicle_name: str = "Drone1",
    imu_name: str = "Imu",
    gps_name: str = "Gps",
    baro_name: str = "Barometer",
    mag_name: str = "Magnetometer",
) -> SensorConfigBundle:
    """
    Build ESKF sensor-noise configs from AirSim settings.json.

    This version intentionally keeps the mapping conservative and explicit:
    - IMU white-noise std values are taken from linear/angular random-walk style entries if present.
    - Bias random-walk std values are taken from bias-stability / random-walk style entries if present.
    - GPS covariance is diagonal from horizontal/vertical variances if present.
    - Barometer is scalar altitude std from variance if present.
    - Magnetometer covariance is diagonal from a single variance/std if present.
    """

    data = _load_json(settings_path)
    vehicle_cfg = _get_vehicle_settings(data, vehicle_name)

    imu_cfg = _get_sensor_block(vehicle_cfg, imu_name)
    gps_cfg = _get_sensor_block(vehicle_cfg, gps_name)
    baro_cfg = _get_sensor_block(vehicle_cfg, baro_name)
    mag_cfg = _get_sensor_block(vehicle_cfg, mag_name)

    # ---- IMU ----
    if any(k in imu_cfg for k in ("AccelNoiseSigma", "LinearAccelerationNoiseSigma")):
        sigma_acc_white = _pick_first_float(
            imu_cfg,
            ["AccelNoiseSigma", "LinearAccelerationNoiseSigma"],
        )
    else:
        sigma_acc_white = _mg_to_mps2(
            _pick_first_float(imu_cfg, ["VelocityRandomWalk"], default=0.24)
        )

    if any(k in imu_cfg for k in ("GyroNoiseSigma", "AngularVelocityNoiseSigma")):
        sigma_gyro_white = _pick_first_float(
            imu_cfg,
            ["GyroNoiseSigma", "AngularVelocityNoiseSigma"],
        )
    else:
        sigma_gyro_white = _deg_per_sqrt_hr_to_rad_per_sqrt_s(
            _pick_first_float(imu_cfg, ["AngularRandomWalk"], default=0.30)
        )

    if any(k in imu_cfg for k in ("AccelBiasRandomWalkSigma", "AccelerometerBiasSigma")):
        sigma_acc_bias_rw = _pick_first_float(
            imu_cfg,
            ["AccelBiasRandomWalkSigma", "AccelerometerBiasSigma"],
        )
    else:
        accel_bias_stability_si = _ug_to_mps2(
            _pick_first_float(imu_cfg, ["AccelBiasStability"], default=36.0)
        )
        accel_bias_tau = _pick_first_float(imu_cfg, ["AccelBiasStabilityTau"], default=800.0)
        if accel_bias_tau <= 0.0:
            raise ValueError("AccelBiasStabilityTau must be positive.")
        sigma_acc_bias_rw = accel_bias_stability_si * np.sqrt(2.0 / accel_bias_tau)

    if any(k in imu_cfg for k in ("GyroBiasRandomWalkSigma", "GyroscopeBiasSigma")):
        sigma_gyro_bias_rw = _pick_first_float(
            imu_cfg,
            ["GyroBiasRandomWalkSigma", "GyroscopeBiasSigma"],
        )
    else:
        gyro_bias_stability_si = _deg_per_hr_to_rad_per_s(
            _pick_first_float(imu_cfg, ["GyroBiasStability"], default=4.6)
        )
        gyro_bias_tau = _pick_first_float(imu_cfg, ["GyroBiasStabilityTau"], default=500.0)
        if gyro_bias_tau <= 0.0:
            raise ValueError("GyroBiasStabilityTau must be positive.")
        sigma_gyro_bias_rw = gyro_bias_stability_si * np.sqrt(2.0 / gyro_bias_tau)

    imu_noise = IMUNoiseConfig(
        sigma_acc_white=sigma_acc_white,
        sigma_gyro_white=sigma_gyro_white,
        sigma_acc_bias_rw=sigma_acc_bias_rw,
        sigma_gyro_bias_rw=sigma_gyro_bias_rw,
    )

    # ---- GPS ----
    sigma_gps_xy = _std_from_variance(
        gps_cfg.get("HorizontalVariance", gps_cfg.get("PositionXYVariance", 0.25)),
        default=0.5,
    )
    sigma_gps_z = _std_from_variance(
        gps_cfg.get("VerticalVariance", gps_cfg.get("PositionZVariance", 0.50)),
        default=0.70710678,
    )

    R_gps = np.diag([sigma_gps_xy**2, sigma_gps_xy**2, sigma_gps_z**2])
    gps_noise = GPSNoiseConfig(R_gps=R_gps)

    # ---- Barometer ----
    sigma_baro_alt_m = calculate_baro_sigma(baro_cfg)
    baro_noise = BaroNoiseConfig(sigma_baro_alt_m=sigma_baro_alt_m)

    # ---- Magnetometer ----
    sigma_mag = float(mag_cfg.get("NoiseSigma", 0.01))
    if sigma_mag < 0.0:
        raise ValueError("NoiseSigma must be nonnegative.")
    R_mag = np.diag([sigma_mag**2, sigma_mag**2, sigma_mag**2])

    mag_n = np.asarray(
        mag_cfg.get("ReferenceFieldNED", [0.25043193, 0.03408979, 0.36581179]),
        dtype=float,
    ).reshape(3)

    mag_noise = MagNoiseConfig(R_mag=R_mag, mag_n=mag_n)

    return SensorConfigBundle(
        imu=imu_noise,
        gps=gps_noise,
        baro=baro_noise,
        mag=mag_noise,
    )