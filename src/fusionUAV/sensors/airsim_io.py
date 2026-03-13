from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class IMUSample:
    t: float
    acc_mps2: np.ndarray
    gyro_rps: np.ndarray


@dataclass
class GPSSample:
    t: float
    position_ned_m: np.ndarray
    valid: bool


@dataclass
class BaroSample:
    t: float
    altitude_m: float


@dataclass
class MagSample:
    t: float
    field_body: np.ndarray


def _vec3_to_numpy(v: Any) -> np.ndarray:
    return np.array([float(v.x_val), float(v.y_val), float(v.z_val)], dtype=float)


def read_imu_sample(client, vehicle_name: str = "") -> IMUSample:
    data = client.getImuData(vehicle_name=vehicle_name)
    return IMUSample(
        t=float(data.time_stamp) * 1e-9,
        acc_mps2=_vec3_to_numpy(data.linear_acceleration),
        gyro_rps=_vec3_to_numpy(data.angular_velocity),
    )


def read_gps_sample(
    client,
    home_geopoint,
    ned_position_from_gps_fn,
    vehicle_name: str = "",
) -> GPSSample:
    gps = client.getGpsData(vehicle_name=vehicle_name)

    gnss = gps.gnss
    geo = gnss.geo_point

    valid = bool(gnss.fix_type >= 2)

    if valid:
        p_ned = np.asarray(
            ned_position_from_gps_fn(
                latitude_deg=float(geo.latitude),
                longitude_deg=float(geo.longitude),
                altitude_m=float(geo.altitude),
                home_geopoint=home_geopoint,
            ),
            dtype=float,
        ).reshape(3)
    else:
        p_ned = np.zeros(3, dtype=float)

    return GPSSample(
        t=float(gps.time_stamp) * 1e-9,
        position_ned_m=p_ned,
        valid=valid,
    )


def read_baro_sample(client, vehicle_name: str = "") -> BaroSample:
    data = client.getBarometerData(vehicle_name=vehicle_name)
    return BaroSample(
        t=float(data.time_stamp) * 1e-9,
        altitude_m=float(data.altitude),
    )


def read_mag_sample(client, vehicle_name: str = "") -> MagSample:
    data = client.getMagnetometerData(vehicle_name=vehicle_name)
    return MagSample(
        t=float(data.time_stamp) * 1e-9,
        field_body=_vec3_to_numpy(data.magnetic_field_body),
    )