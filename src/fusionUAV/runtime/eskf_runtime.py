from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import airsim
import numpy as np
import yaml

from fusionUAV.runtime.automatic_control import (
    AutomaticControlConfig,
    AutomaticControlStep,
)
from fusionUAV.eskf.eskf_filter import ESKF
from fusionUAV.eskf.state import ESKFState
from fusionUAV.runtime.manual_control import ManualControlConfig, ManualControlKeysConfig
from fusionUAV.sensors.airsim_io import (
    read_baro_sample,
    read_gps_sample,
    read_imu_sample,
    read_mag_sample,
)
from fusionUAV.sensors.build_from_settings import build_sensor_configs


@dataclass
class PathsConfig:
    airsim_settings: str


@dataclass
class VehicleConfig:
    name: str


@dataclass
class SensorsConfig:
    use_gps: bool
    use_baro: bool
    use_mag: bool


@dataclass
class PeriodsConfig:
    imu: float
    gps: float
    baro: float
    mag: float


@dataclass
class InitialCovarianceConfig:
    sigma_p: float
    sigma_v: float
    sigma_theta: float
    sigma_ba: float
    sigma_bw: float
    sigma_g: float


@dataclass
class ControlConfig:
    mode: str = "manual"


@dataclass
class PrintingConfig:
    state_period: float = 1.0


@dataclass
class PlottingConfig:
    airsim_enable: bool = True
    airsim_period: float = 0.2
    max_points: int = 2000
    duration: float = 0.35
    live_enable: bool = True
    live_max_points: int = 1000
    live_title: str = "Live ESKF vs Ground Truth"


@dataclass
class AppConfig:
    paths: PathsConfig
    vehicle: VehicleConfig
    sensors: SensorsConfig
    periods: PeriodsConfig
    initial_covariance: InitialCovarianceConfig
    control: ControlConfig
    manual_control: ManualControlConfig
    automatic_control: AutomaticControlConfig
    printing: PrintingConfig
    plotting: PlottingConfig


def _resolve_optional_path(path_value: Any, base_dir: Path) -> str:
    if path_value is None or str(path_value).strip() == "":
        return ""

    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def _parse_manual_control(
    raw: dict[str, Any],
    base_dir: Path,
) -> ManualControlConfig:
    manual_raw = raw.get("manual_control")
    if manual_raw is None:
        # Backward-compat fallback for accidentally nested config.
        manual_raw = raw.get("initial_covariance", {}).get("manual_control", {})

    manual_raw = dict(manual_raw or {})
    if "record_path" in manual_raw:
        manual_raw["record_path"] = _resolve_optional_path(
            manual_raw.get("record_path"),
            base_dir=base_dir,
        )
    keys_raw = dict(manual_raw.pop("keys", {}) or {})
    keys = ManualControlKeysConfig(**keys_raw)
    return ManualControlConfig(keys=keys, **manual_raw)


def _parse_control(raw: dict[str, Any]) -> ControlConfig:
    control_raw = dict(raw.get("control", {}) or {})
    if "mode" not in control_raw:
        manual_enabled = bool(raw.get("manual_control", {}).get("enable", False))
        control_raw["mode"] = "manual" if manual_enabled else "off"
    return ControlConfig(**control_raw)


def _parse_automatic_control(
    raw: dict[str, Any],
    base_dir: Path,
) -> AutomaticControlConfig:
    automatic_raw = dict(raw.get("automatic_control", {}) or {})
    if "recorded_commands_path" in automatic_raw:
        automatic_raw["recorded_commands_path"] = _resolve_optional_path(
            automatic_raw.get("recorded_commands_path"),
            base_dir=base_dir,
        )
    steps_raw = automatic_raw.pop("steps", []) or []
    steps = [AutomaticControlStep(**dict(step)) for step in steps_raw]
    return AutomaticControlConfig(steps=steps, **automatic_raw)


def _load_yaml_config(config_path: str | Path) -> AppConfig:
    config_path = Path(config_path).expanduser().resolve()
    base_dir = config_path.parent

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    paths_raw = dict(raw["paths"])
    vehicle_raw = dict(raw["vehicle"])
    sensors_raw = dict(raw["sensors"])
    periods_raw = dict(raw["periods"])
    cov_raw = dict(raw["initial_covariance"])

    return AppConfig(
        paths=PathsConfig(
            airsim_settings=paths_raw["airsim_settings"],
        ),
        vehicle=VehicleConfig(
            name=vehicle_raw["name"],
        ),
        sensors=SensorsConfig(
            use_gps=bool(sensors_raw["use_gps"]),
            use_baro=bool(sensors_raw["use_baro"]),
            use_mag=bool(sensors_raw["use_mag"]),
        ),
        periods=PeriodsConfig(
            imu=float(periods_raw["imu"]),
            gps=float(periods_raw["gps"]),
            baro=float(periods_raw["baro"]),
            mag=float(periods_raw["mag"]),
        ),
        initial_covariance=InitialCovarianceConfig(
            sigma_p=float(cov_raw["sigma_p"]),
            sigma_v=float(cov_raw["sigma_v"]),
            sigma_theta=float(cov_raw["sigma_theta"]),
            sigma_ba=float(cov_raw["sigma_ba"]),
            sigma_bw=float(cov_raw["sigma_bw"]),
            sigma_g=float(cov_raw["sigma_g"]),
        ),
        control=_parse_control(raw),
        manual_control=_parse_manual_control(raw, base_dir=base_dir),
        automatic_control=_parse_automatic_control(raw, base_dir=base_dir),
        printing=PrintingConfig(**dict(raw.get("printing", {}) or {})),
        plotting=PlottingConfig(**dict(raw.get("plotting", {}) or {})),
    )


def load_app_config(config_path: str | Path) -> AppConfig:
    return _load_yaml_config(config_path)


def ned_position_from_gps(
    latitude_deg: float,
    longitude_deg: float,
    altitude_m: float,
    home_geopoint: Any,
) -> np.ndarray:
    lat0 = np.deg2rad(float(home_geopoint.latitude))
    h0 = float(home_geopoint.altitude)

    lat = np.deg2rad(latitude_deg)
    lon = np.deg2rad(longitude_deg)
    lon0 = np.deg2rad(float(home_geopoint.longitude))
    h = float(altitude_m)

    r_earth = 6378137.0
    d_north = (lat - lat0) * r_earth
    d_east = (lon - lon0) * r_earth * np.cos(lat0)
    d_down = -(h - h0)

    return np.array([d_north, d_east, d_down], dtype=float)


class LiveESKFRunner:
    def __init__(self, config_path: str | Path):
        self.app_cfg = _load_yaml_config(config_path)

        self.vehicle_name = self.app_cfg.vehicle.name
        self.sensor_cfg = build_sensor_configs(
            settings_path=str(Path(self.app_cfg.paths.airsim_settings).expanduser()),
            vehicle_name=self.vehicle_name,
        )

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        self.home_origin = self.client.getHomeGeoPoint(vehicle_name=self.vehicle_name)
        self.baro_alt_offset = self.home_origin.altitude

        # Estimated constant translation between:
        # - GPS/home-referenced NED frame (used by filter state p)
        # - AirSim ground-truth position frame (vehicle start frame).
        self.gt_to_gps_ned_offset = np.zeros(3, dtype=float)

        self.state = ESKFState()
        imu0 = read_imu_sample(self.client, vehicle_name=self.vehicle_name)

        gps0 = self._read_startup_gps_sample()
        if gps0.valid:
            self.state.p = gps0.position_ned_m.copy()

            gt0 = self.client.simGetGroundTruthKinematics(vehicle_name=self.vehicle_name)
            gt0_p = np.array(
                [gt0.position.x_val, gt0.position.y_val, gt0.position.z_val],
                dtype=float,
            )
            self.gt_to_gps_ned_offset = gps0.position_ned_m - gt0_p

        self.state.reset_covariance(
            sigma_p=self.app_cfg.initial_covariance.sigma_p,
            sigma_v=self.app_cfg.initial_covariance.sigma_v,
            sigma_theta=self.app_cfg.initial_covariance.sigma_theta,
            sigma_ba=self.app_cfg.initial_covariance.sigma_ba,
            sigma_bw=self.app_cfg.initial_covariance.sigma_bw,
            sigma_g=self.app_cfg.initial_covariance.sigma_g,
        )

        self.eskf = ESKF(
            state=self.state,
            sigma_acc_white=self.sensor_cfg.imu.sigma_acc_white,
            sigma_gyro_white=self.sensor_cfg.imu.sigma_gyro_white,
            sigma_acc_bias_rw=self.sensor_cfg.imu.sigma_acc_bias_rw,
            sigma_gyro_bias_rw=self.sensor_cfg.imu.sigma_gyro_bias_rw,
            use_exact_reset_jacobian=True,
        )

        self.prev_imu_t = imu0.t
        self.prev_gps_t = imu0.t
        self.prev_baro_t = imu0.t
        self.prev_mag_t = imu0.t

        self.last_imu = imu0
        self.last_gps = None
        self.last_baro = None
        self.last_mag = None
        self.last_gt = None

    def _read_startup_gps_sample(self, timeout_s: float = 2.0):
        gps = read_gps_sample(
            client=self.client,
            home_geopoint=self.home_origin,
            ned_position_from_gps_fn=ned_position_from_gps,
            vehicle_name=self.vehicle_name,
        )
        if gps.valid:
            return gps

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            time.sleep(0.05)
            gps = read_gps_sample(
                client=self.client,
                home_geopoint=self.home_origin,
                ned_position_from_gps_fn=ned_position_from_gps,
                vehicle_name=self.vehicle_name,
            )
            if gps.valid:
                return gps

        return gps

    def step(self) -> None:
        imu = read_imu_sample(self.client, vehicle_name=self.vehicle_name)
        self.last_imu = imu

        dt = imu.t - self.prev_imu_t
        if dt > 0.0 and dt >= self.app_cfg.periods.imu:
            self.eskf.predict(
                acc_m=imu.acc_mps2,
                gyro_m=imu.gyro_rps,
                dt=dt,
            )
            self.prev_imu_t = imu.t

        if self.app_cfg.sensors.use_gps:
            gps = read_gps_sample(
                client=self.client,
                home_geopoint=self.home_origin,
                ned_position_from_gps_fn=ned_position_from_gps,
                vehicle_name=self.vehicle_name,
            )
            self.last_gps = gps

            dt_gps = gps.t - self.prev_gps_t
            if gps.valid and dt_gps > 0.0 and dt_gps >= self.app_cfg.periods.gps:
                self.eskf.update_gps(
                    z_gps_pos=gps.position_ned_m,
                    R_gps=self.sensor_cfg.gps.R_gps,
                )
                self.prev_gps_t = gps.t

        if self.app_cfg.sensors.use_baro:
            baro = read_baro_sample(self.client, vehicle_name=self.vehicle_name)
            self.last_baro = baro

            dt_baro = baro.t - self.prev_baro_t
            if dt_baro > 0.0 and dt_baro >= self.app_cfg.periods.baro:
                self.eskf.update_baro(
                    z_baro_alt_m=(baro.altitude_m - self.baro_alt_offset),
                    sigma_baro_alt_m=self.sensor_cfg.baro.sigma_baro_alt_m,
                )
                self.prev_baro_t = baro.t

        if self.app_cfg.sensors.use_mag:
            mag = read_mag_sample(self.client, vehicle_name=self.vehicle_name)
            self.last_mag = mag

            dt_mag = mag.t - self.prev_mag_t
            if dt_mag > 0.0 and dt_mag >= self.app_cfg.periods.mag:
                if np.linalg.norm(mag.field_body) > 1e-9:
                    self.eskf.update_mag(
                        z_mag_b=mag.field_body,
                        mag_n=self.sensor_cfg.mag.mag_n,
                        R_mag=self.sensor_cfg.mag.R_mag,
                    )
                    self.prev_mag_t = mag.t

        self.last_gt = self.client.simGetGroundTruthKinematics(
            vehicle_name=self.vehicle_name
        )

    def get_estimate(self):
        return self.eskf.get_state()

    def get_ground_truth_arrays(self):
        if self.last_gt is None:
            self.last_gt = self.client.simGetGroundTruthKinematics(
                vehicle_name=self.vehicle_name
            )

        p_gt = np.array(
            [
                self.last_gt.position.x_val,
                self.last_gt.position.y_val,
                self.last_gt.position.z_val,
            ],
            dtype=float,
        )
        # Align AirSim GT frame to filter's GPS/home NED frame for fair comparison.
        p_gt = p_gt + self.gt_to_gps_ned_offset
        v_gt = np.array(
            [
                self.last_gt.linear_velocity.x_val,
                self.last_gt.linear_velocity.y_val,
                self.last_gt.linear_velocity.z_val,
            ],
            dtype=float,
        )
        q_gt = np.array(
            [
                self.last_gt.orientation.w_val,
                self.last_gt.orientation.x_val,
                self.last_gt.orientation.y_val,
                self.last_gt.orientation.z_val,
            ],
            dtype=float,
        )
        return p_gt, v_gt, q_gt
