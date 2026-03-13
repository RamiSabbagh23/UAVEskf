from __future__ import annotations

import sys
import time
from pathlib import Path

import airsim
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fusionUAV.eskf.eskf_filter import ESKF
from fusionUAV.eskf.state import ESKFState
from fusionUAV.sensors.airsim_io import (
    read_baro_sample,
    read_gps_sample,
    read_imu_sample,
    read_mag_sample,
)
from fusionUAV.sensors.build_from_settings import build_sensor_configs


# =========================
# USER PARAMETERS
# =========================

SETTINGS_PATH = str(Path.home() / "Documents" / "AirSim" / "settings.json")
VEHICLE_NAME = "Drone1"

USE_GPS = 1
USE_BARO = 1
USE_MAG = 1

IMU_PERIOD = 0.001
GPS_PERIOD = 0.001
BARO_PERIOD = 0.001
MAG_PERIOD = 0.001

PRINT_SENSOR_RATES = 0
PRINT_STATES = 1

PRINT_RATE_PERIOD = 1.0
PRINT_STATE_PERIOD = 1.0

# =========================
# AIRSIM PLOT PARAMETERS
# =========================

PLOT_IN_AIRSIM = 1
PLOT_PERIOD = 0.20
PLOT_MAX_POINTS = 2000

GT_COLOR = [0.0, 1.0, 0.0, 1.0]      # green
EST_COLOR = [1.0, 0.0, 0.0, 1.0]     # red
GT_POINT_COLOR = [0.0, 1.0, 0.0, 1.0]
EST_POINT_COLOR = [1.0, 0.0, 0.0, 1.0]

GT_LINE_THICKNESS = 8.0
EST_LINE_THICKNESS = 8.0
GT_POINT_SIZE = 12.0
EST_POINT_SIZE = 12.0

PLOT_DURATION = 0.35


# =========================
# GPS -> NED CONVERSION
# =========================

def ned_position_from_gps(
    latitude_deg: float,
    longitude_deg: float,
    altitude_m: float,
    home_geopoint,
) -> np.ndarray:
    """
    Simple local tangent-plane approximation around home.
    Good enough for short AirSim trajectories.
    """
    lat0 = np.deg2rad(float(home_geopoint.latitude))
    lon0 = np.deg2rad(float(home_geopoint.longitude))
    h0 = float(home_geopoint.altitude)

    lat = np.deg2rad(latitude_deg)
    lon = np.deg2rad(longitude_deg)
    h = float(altitude_m)

    R_earth = 6378137.0
    d_north = (lat - lat0) * R_earth
    d_east = (lon - lon0) * R_earth * np.cos(lat0)
    d_down = -(h - h0)

    return np.array([d_north, d_east, d_down], dtype=float)


# =========================
# AIRSIM PLOT HELPERS
# =========================

def np_to_vector3r(p: np.ndarray) -> airsim.Vector3r:
    return airsim.Vector3r(float(p[0]), float(p[1]), float(p[2]))


def trim_path(path_list, max_points: int):
    if len(path_list) > max_points:
        del path_list[:-max_points]


def plot_gt_vs_est_in_airsim(client, gt_path, est_path):
    if len(gt_path) >= 2:
        client.simPlotLineStrip(
            points=gt_path,
            color_rgba=GT_COLOR,
            thickness=GT_LINE_THICKNESS,
            duration=PLOT_DURATION,
            is_persistent=False,
        )

    if len(est_path) >= 2:
        client.simPlotLineStrip(
            points=est_path,
            color_rgba=EST_COLOR,
            thickness=EST_LINE_THICKNESS,
            duration=PLOT_DURATION,
            is_persistent=False,
        )

    if len(gt_path) >= 1:
        client.simPlotPoints(
            points=[gt_path[-1]],
            color_rgba=GT_POINT_COLOR,
            size=GT_POINT_SIZE,
            duration=PLOT_DURATION,
            is_persistent=False,
        )

    if len(est_path) >= 1:
        client.simPlotPoints(
            points=[est_path[-1]],
            color_rgba=EST_POINT_COLOR,
            size=EST_POINT_SIZE,
            duration=PLOT_DURATION,
            is_persistent=False,
        )


# =========================
# MAIN
# =========================

def main():

    cfg = build_sensor_configs(
        settings_path=SETTINGS_PATH,
        vehicle_name=VEHICLE_NAME,
    )

    client = airsim.MultirotorClient()
    client.confirmConnection()

    client.reset()
    time.sleep(1)

    home_origin = client.getHomeGeoPoint(vehicle_name=VEHICLE_NAME)
    baro_alt_offset = home_origin.altitude

    # =========================
    # INITIALIZE ESKF
    # =========================

    state = ESKFState()

    state.reset_covariance(
        sigma_p=2.0,
        sigma_v=1.0,
        sigma_theta=0.2,
        sigma_ba=0.2,
        sigma_bw=0.02,
        sigma_g=0.2,
    )

    eskf = ESKF(
        state=state,
        sigma_acc_white=cfg.imu.sigma_acc_white,
        sigma_gyro_white=cfg.imu.sigma_gyro_white,
        sigma_acc_bias_rw=cfg.imu.sigma_acc_bias_rw,
        sigma_gyro_bias_rw=cfg.imu.sigma_gyro_bias_rw,
        use_exact_reset_jacobian=True,
    )

    imu0 = read_imu_sample(client, vehicle_name=VEHICLE_NAME)

    prev_imu_t = imu0.t
    prev_gps_t = imu0.t
    prev_baro_t = imu0.t
    prev_mag_t = imu0.t

    last_rate_print = time.time()
    last_state_print = time.time()
    last_plot_time = time.time()

    imu_rate = 0.0
    gps_rate = 0.0
    baro_rate = 0.0
    mag_rate = 0.0

    gt_path = []
    est_path = []

    try:

        while True:
            imu = read_imu_sample(client, vehicle_name=VEHICLE_NAME)
            dt = (imu.t - prev_imu_t)

            if dt >= IMU_PERIOD:
                eskf.predict(
                    acc_m=imu.acc_mps2,
                    gyro_m=imu.gyro_rps,
                    dt=dt,
                )
                imu_rate = 1.0 / dt
                prev_imu_t = imu.t

            if USE_GPS:
                gps = read_gps_sample(
                    client=client,
                    home_geopoint=home_origin,
                    ned_position_from_gps_fn=ned_position_from_gps,
                    vehicle_name=VEHICLE_NAME,
                )

                if (gps.valid) and (gps.t - prev_gps_t) >= GPS_PERIOD:
                    eskf.update_gps(
                        z_gps_pos=gps.position_ned_m,
                        R_gps=cfg.gps.R_gps,
                    )

                    gps_rate = 1.0 / (gps.t - prev_gps_t)
                    prev_gps_t = gps.t

            if USE_BARO:
                baro = read_baro_sample(client, vehicle_name=VEHICLE_NAME)
                if (baro.t - prev_baro_t) >= BARO_PERIOD:
                    eskf.update_baro(
                        z_baro_alt_m=(baro.altitude_m - baro_alt_offset),
                        sigma_baro_alt_m=cfg.baro.sigma_baro_alt_m,
                    )
                    baro_rate = 1.0 / (baro.t - prev_baro_t)
                    prev_baro_t = baro.t

            if USE_MAG:
                mag = read_mag_sample(client, vehicle_name=VEHICLE_NAME)
                if (mag.t - prev_mag_t) >= MAG_PERIOD:
                    if np.linalg.norm(mag.field_body) > 1e-9:
                        eskf.update_mag(
                            z_mag_b=mag.field_body,
                            mag_n=cfg.mag.mag_n,
                            R_mag=cfg.mag.R_mag,
                        )
                        mag_rate = 1.0 / (mag.t - prev_mag_t)
                        prev_mag_t = mag.t

            # =========================
            # AIRSIM GT VS ESTIMATION PLOT
            # =========================

            if PLOT_IN_AIRSIM and (time.time() - last_plot_time) >= PLOT_PERIOD:
                x = eskf.get_state()
                gt = client.simGetGroundTruthKinematics(vehicle_name=VEHICLE_NAME)

                p_est = np.array(x.p, dtype=float)
                p_gt = np.array(
                    [gt.position.x_val, gt.position.y_val, gt.position.z_val],
                    dtype=float,
                )

                est_path.append(np_to_vector3r(p_est))
                gt_path.append(np_to_vector3r(p_gt))

                trim_path(est_path, PLOT_MAX_POINTS)
                trim_path(gt_path, PLOT_MAX_POINTS)

                plot_gt_vs_est_in_airsim(client, gt_path, est_path)

                last_plot_time = time.time()

            # =========================
            # PRINT SENSOR RATES
            # =========================

            if PRINT_SENSOR_RATES and (time.time() - last_rate_print > PRINT_RATE_PERIOD):

                print(
                    f"Rates | IMU: {imu_rate:6.2f} Hz | "
                    f"GPS: {gps_rate:5.2f} Hz | "
                    f"BARO: {baro_rate:5.2f} Hz | "
                    f"MAG: {mag_rate:5.2f} Hz"
                )

                last_rate_print = time.time()

            # =========================
            # PRINT STATES
            # =========================

            if PRINT_STATES and (time.time() - last_state_print > PRINT_STATE_PERIOD):

                x = eskf.get_state()
                gt = client.simGetGroundTruthKinematics(vehicle_name=VEHICLE_NAME)

                print("\nESTIMATED STATE")

                print(f"p = {np.round(x.p,4)}")
                print(f"v = {np.round(x.v,4)}")
                print(f"q = {np.round(x.q,4)}")
                print(f"ba = {np.round(x.ba,4)}")
                print(f"bw = {np.round(x.bw,4)}")
                print(f"g = {np.round(x.g,4)}")

                print("-----")

                p_gt = np.array([gt.position.x_val, gt.position.y_val, gt.position.z_val])
                v_gt = np.array([gt.linear_velocity.x_val, gt.linear_velocity.y_val, gt.linear_velocity.z_val])
                q_gt = np.array([gt.orientation.w_val, gt.orientation.x_val, gt.orientation.y_val, gt.orientation.z_val])

                print("GROUND TRUTH")

                print(f"p_gt = {np.round(p_gt,4)}")
                print(f"v_gt = {np.round(v_gt,4)}")
                print(f"q_gt = {np.round(q_gt,4)}")

                print("-----")

                pos_err = x.p - p_gt
                vel_err = x.v - v_gt
                q_err = x.q - q_gt

                print("ERROR")

                print(f"pos_err = {np.round(pos_err,4)}")
                print(f"vel_err = {np.round(vel_err,4)}")
                print(f"q_err = {np.round(q_err,4)}")

                print("--------------------------------------------")

                last_state_print = time.time()

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()