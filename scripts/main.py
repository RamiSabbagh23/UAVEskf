from __future__ import annotations

import sys
import time
from pathlib import Path

import airsim
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from fusionUAV.runtime.automatic_control import AutomaticControlRunner
from fusionUAV.runtime.eskf_runtime import LiveESKFRunner, load_app_config
from fusionUAV.runtime.manual_control import ManualControlRunner
from fusionUAV.runtime.run_logger import LiveRunLogger
from fusionUAV.visualization.airsim_plot import AirSimPlotConfig, AirSimTrajectoryPlotter
from fusionUAV.visualization.live_plot import LivePlotConfig, LivePositionPlotter


def _reset_airsim(vehicle_name: str, timeout_s: float = 5.0) -> None:
    client = airsim.MultirotorClient()
    client.confirmConnection()
    try:
        pre_reset_gps_ts = float(client.getGpsData(vehicle_name=vehicle_name).time_stamp)
    except Exception:
        pre_reset_gps_ts = -1.0

    try:
        client.armDisarm(False, vehicle_name=vehicle_name)
        client.enableApiControl(False, vehicle_name=vehicle_name)
    except Exception:
        pass

    client.reset()

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        gt = client.simGetGroundTruthKinematics(vehicle_name=vehicle_name)
        p_gt = np.array(
            [gt.position.x_val, gt.position.y_val, gt.position.z_val],
            dtype=float,
        )

        gt_reset = np.linalg.norm(p_gt[:2]) <= 0.25 and abs(p_gt[2]) <= 1.0

        try:
            gps_ts = float(client.getGpsData(vehicle_name=vehicle_name).time_stamp)
            gps_fresh = gps_ts > pre_reset_gps_ts
        except Exception:
            gps_fresh = True

        if gt_reset and gps_fresh:
            return

        time.sleep(0.05)

    print("Warning: AirSim reset finished before GPS refreshed; startup alignment may be stale.")


def main() -> None:
    config_path = ROOT / "configs" / "eskf_config.yaml"
    app_cfg = load_app_config(config_path)
    _reset_airsim(vehicle_name=app_cfg.vehicle.name)
    runner = LiveESKFRunner(config_path)
    logger = LiveRunLogger()

    control_loop_sleep = 0.001
    live_plot_period = 0.001

    airsim_plotter = AirSimTrajectoryPlotter(
        client=runner.client,
        cfg=AirSimPlotConfig(
            enable=runner.app_cfg.plotting.airsim_enable,
            period=runner.app_cfg.plotting.airsim_period,
            max_points=runner.app_cfg.plotting.max_points,
            duration=runner.app_cfg.plotting.duration,
        ),
    )

    live_plotter: LivePositionPlotter | None = None
    if runner.app_cfg.plotting.live_enable:
        live_plotter = LivePositionPlotter(
            cfg=LivePlotConfig(
                enable=runner.app_cfg.plotting.live_enable,
                max_points=runner.app_cfg.plotting.live_max_points,
                figure_title=runner.app_cfg.plotting.live_title,
            )
        )
        live_plotter.show(block=False)

    controller = None
    control_mode = runner.app_cfg.control.mode.strip().lower()
    if control_mode in {"manual", "automatic"}:
        control_client = airsim.MultirotorClient()
        control_client.confirmConnection()
        if control_mode == "manual":
            controller = ManualControlRunner(
                client=control_client,
                vehicle_name=runner.vehicle_name,
                cfg=runner.app_cfg.manual_control,
            )
        else:
            controller = AutomaticControlRunner(
                client=control_client,
                vehicle_name=runner.vehicle_name,
                cfg=runner.app_cfg.automatic_control,
            )
        controller.start()
    elif control_mode != "off":
        raise ValueError(f"Unsupported control mode: {runner.app_cfg.control.mode}")

    start_t = time.time()
    last_airsim_plot_t = 0.0
    last_live_plot_t = 0.0
    last_state_print_t = 0.0

    try:
        while True:
            if controller is not None:
                controller.update()
                if controller.stop_requested:
                    print(f"{control_mode} control requested stop.")
                    break

            runner.step()
            wall_t = time.time()
            sample_t = wall_t - start_t

            x = runner.get_estimate()
            p_gt, v_gt, q_gt = runner.get_ground_truth_arrays()
            p_est = np.array(x.p, dtype=float)
            v_est = np.array(x.v, dtype=float)
            q_est = np.array(x.q, dtype=float)

            logger.append(
                t=sample_t,
                p_est=p_est,
                p_gt=p_gt,
                v_est=v_est,
                v_gt=v_gt,
                q_est=q_est,
                q_gt=q_gt,
                ba_est=np.array(x.ba, dtype=float),
                bw_est=np.array(x.bw, dtype=float),
                g_est=np.array(x.g, dtype=float),
            )

            do_live_plot = (
                live_plotter is not None
                and (wall_t - last_live_plot_t) >= live_plot_period
            )
            do_airsim_plot = (
                airsim_plotter.cfg.enable
                and (wall_t - last_airsim_plot_t) >= airsim_plotter.cfg.period
            )
            do_state_print = (
                (wall_t - last_state_print_t) >= runner.app_cfg.printing.state_period
            )

            if do_live_plot or do_airsim_plot or do_state_print:
                if do_live_plot and live_plotter is not None:
                    live_plotter.update(
                        t=sample_t,
                        p_est=p_est,
                        p_gt=p_gt,
                    )
                    last_live_plot_t = wall_t

                if do_airsim_plot:
                    airsim_plotter.update(
                        p_gt=p_gt,
                        p_est=p_est,
                    )
                    last_airsim_plot_t = wall_t

                if do_state_print:
                    pos_err = p_est - p_gt
                    print(
                        f"p_est={np.round(p_est, 3)} | "
                        f"p_gt={np.round(p_gt, 3)} | "
                        f"err={np.round(pos_err, 3)}"
                    )
                    last_state_print_t = wall_t

            time.sleep(control_loop_sleep)

    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        if controller is not None:
            controller.stop()
        if not logger.is_empty():
            try:
                output_dir = logger.finalize(ROOT, live_plotter=live_plotter)
                if output_dir is not None:
                    print(f"Saved run artifacts to {output_dir}")
            except Exception as exc:
                print(f"Failed to save run history: {exc}")
        if live_plotter is not None:
            live_plotter.close()


if __name__ == "__main__":
    main()
