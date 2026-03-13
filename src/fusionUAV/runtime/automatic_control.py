from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import airsim
import numpy as np
import yaml


@dataclass
class AutomaticControlStep:
    duration: float
    mode: str = "velocity"
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw_rate: float = 0.0
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float | None = None
    z: float | None = None
    name: str = ""


@dataclass
class AutomaticControlConfig:
    loop_sleep: float = 0.02
    takeoff_on_start: bool = True
    finish_hover_duration: float = 1.0
    land_on_finish: bool = True
    use_recorded_commands: bool = False
    recorded_commands_path: str = "outputs/control/last_manual_control.yaml"
    steps: list[AutomaticControlStep] = field(default_factory=list)


class AutomaticControlRunner:
    def __init__(
        self,
        client: airsim.MultirotorClient,
        vehicle_name: str = "",
        cfg: AutomaticControlConfig | None = None,
    ):
        self.client = client
        self.vehicle_name = vehicle_name
        self.cfg = cfg or AutomaticControlConfig()

        self._started = False
        self._stop_requested = False
        self._last_update_t = 0.0
        self._current_step_index = 0
        self._current_step_started_t: float | None = None
        self._finish_hover_started_t: float | None = None
        self._active_steps: list[AutomaticControlStep] = []
        self._recorded_sequence_has_takeoff = False
        self._recorded_sequence_has_land = False

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested

    def start(self) -> None:
        if self._started:
            return

        self.client.enableApiControl(True, vehicle_name=self.vehicle_name)
        self.client.armDisarm(True, vehicle_name=self.vehicle_name)

        self._active_steps = self._resolve_steps()
        self._recorded_sequence_has_takeoff = any(
            step.mode.strip().lower() == "takeoff" for step in self._active_steps
        )
        self._recorded_sequence_has_land = any(
            step.mode.strip().lower() == "land" for step in self._active_steps
        )

        if self.cfg.takeoff_on_start and not self._recorded_sequence_has_takeoff:
            print("Automatic control: takeoff")
            self.client.takeoffAsync(vehicle_name=self.vehicle_name).join()

        self._started = True
        self._stop_requested = False
        self._last_update_t = 0.0
        self._current_step_index = 0
        self._current_step_started_t = None
        self._finish_hover_started_t = None

    def stop(self) -> None:
        if not self._started:
            return

        self.client.armDisarm(False, vehicle_name=self.vehicle_name)
        self.client.enableApiControl(False, vehicle_name=self.vehicle_name)
        self._started = False

    def update(self) -> None:
        if not self._started or self._stop_requested:
            return

        now = time.monotonic()
        if (now - self._last_update_t) < self.cfg.loop_sleep:
            return
        self._last_update_t = now

        if self._current_step_index < len(self._active_steps):
            self._run_active_step(now)
            return

        self._finish(now)

    def _run_active_step(self, now: float) -> None:
        step = self._active_steps[self._current_step_index]

        if self._current_step_started_t is None:
            self._execute_step(step)
            self._current_step_started_t = now
            return

        if (now - self._current_step_started_t) >= step.duration:
            self._current_step_index += 1
            self._current_step_started_t = None
            if self._current_step_index >= len(self._active_steps):
                self.client.hoverAsync(vehicle_name=self.vehicle_name)
                self._finish_hover_started_t = now

    def _finish(self, now: float) -> None:
        if self._finish_hover_started_t is None:
            self.client.hoverAsync(vehicle_name=self.vehicle_name)
            self._finish_hover_started_t = now
            return

        if (now - self._finish_hover_started_t) < self.cfg.finish_hover_duration:
            return

        if self.cfg.land_on_finish and not self._recorded_sequence_has_land:
            print("Automatic control: landing")
            self.client.landAsync(vehicle_name=self.vehicle_name).join()

        self._stop_requested = True

    def _execute_step(self, step: AutomaticControlStep) -> None:
        label = step.name or f"step_{self._current_step_index + 1}"
        mode = step.mode.strip().lower()

        if mode == "hover":
            print(f"Automatic control: {label} | hover duration={step.duration:.2f}")
            self.client.hoverAsync(vehicle_name=self.vehicle_name)
            return

        if mode == "takeoff":
            print(f"Automatic control: {label} | takeoff")
            self.client.takeoffAsync(vehicle_name=self.vehicle_name).join()
            return

        if mode == "land":
            print(f"Automatic control: {label} | land")
            self.client.landAsync(vehicle_name=self.vehicle_name).join()
            return

        if mode == "velocity":
            print(
                f"Automatic control: {label} | "
                f"vx={step.vx:.2f} vy={step.vy:.2f} vz={step.vz:.2f} "
                f"yaw_rate={step.yaw_rate:.2f} duration={step.duration:.2f}"
            )
            self.client.moveByVelocityBodyFrameAsync(
                vx=step.vx,
                vy=step.vy,
                vz=step.vz,
                duration=step.duration,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=step.yaw_rate),
                vehicle_name=self.vehicle_name,
            )
            return

        if mode == "attitude":
            pose = self.client.simGetVehiclePose(vehicle_name=self.vehicle_name)
            _, _, current_yaw = airsim.to_eularian_angles(pose.orientation)
            target_yaw = current_yaw if step.yaw_deg is None else np.deg2rad(step.yaw_deg)
            target_z = float(pose.position.z_val) if step.z is None else float(step.z)

            print(
                f"Automatic control: {label} | "
                f"roll_deg={step.roll_deg:.2f} pitch_deg={step.pitch_deg:.2f} "
                f"yaw_deg={np.rad2deg(target_yaw):.2f} z={target_z:.2f} "
                f"duration={step.duration:.2f}"
            )
            self.client.moveByRollPitchYawZAsync(
                roll=np.deg2rad(step.roll_deg),
                pitch=np.deg2rad(step.pitch_deg),
                yaw=float(target_yaw),
                z=target_z,
                duration=step.duration,
                vehicle_name=self.vehicle_name,
            )
            return

        raise ValueError(f"Unsupported automatic control step mode: {step.mode}")

    def _resolve_steps(self) -> list[AutomaticControlStep]:
        if self.cfg.use_recorded_commands:
            recorded_steps = self._load_recorded_steps()
            if len(recorded_steps) > 0:
                print(
                    f"Automatic control: loaded {len(recorded_steps)} recorded steps "
                    f"from {self.cfg.recorded_commands_path}"
                )
                return recorded_steps
            print(
                "Automatic control: recorded command file is empty or missing, "
                "falling back to configured automatic steps."
            )

        return list(self.cfg.steps)

    def _load_recorded_steps(self) -> list[AutomaticControlStep]:
        path = Path(self.cfg.recorded_commands_path).expanduser()
        if not path.exists():
            return []

        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        steps_raw = raw.get("steps", []) or []
        return [AutomaticControlStep(**dict(step)) for step in steps_raw]
