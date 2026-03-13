from __future__ import annotations

import time
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path

import airsim
import yaml

try:
    import keyboard
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    keyboard = None


@dataclass
class ManualControlKeysConfig:
    emergency: str = "esc"
    takeoff: str = "t"
    land: str = "l"
    forward: str = "w"
    backward: str = "s"
    left: str = "a"
    right: str = "d"
    up: str = "up"
    down: str = "down"
    yaw_left: str = "left"
    yaw_right: str = "right"


@dataclass
class ManualControlConfig:
    enable: bool = False
    speed: float = 5.0
    yaw_speed: float = 30.0
    command_duration: float = 0.1
    loop_sleep: float = 0.02
    record_enable: bool = True
    record_path: str = "outputs/control/last_manual_control.yaml"
    keys: ManualControlKeysConfig = field(default_factory=ManualControlKeysConfig)


class ManualControlRunner:
    def __init__(
        self,
        client: airsim.MultirotorClient,
        vehicle_name: str = "",
        cfg: ManualControlConfig | None = None,
    ):
        self.client = client
        self.vehicle_name = vehicle_name
        self.cfg = cfg or ManualControlConfig()

        self._started = False
        self._stop_requested = False
        self._last_update_t = 0.0
        self._was_moving = False
        self._takeoff_pressed_prev = False
        self._land_pressed_prev = False
        self._flight_started = False
        self._recorded_steps: list[dict[str, float | str]] = []
        self._active_record_step: dict[str, float | str] | None = None
        self._active_record_started_t: float | None = None

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested

    def start(self) -> None:
        if not self.cfg.enable:
            return

        if keyboard is None:
            raise RuntimeError(
                "Manual control requires the 'keyboard' package. "
                "Install dependencies and retry."
            )

        if self._started:
            return

        self.client.enableApiControl(True, vehicle_name=self.vehicle_name)
        self.client.armDisarm(True, vehicle_name=self.vehicle_name)

        print("--- Drone Manual Control Ready ---")
        print("Controls:")
        print("  T: Takeoff | L: Land | Esc: Emergency Land & Exit")
        print("  W/S: Forward/Backward | A/D: Left/Right (Strafe)")
        print("  Up/Down Arrows: Rise/Sink")
        print("  Left/Right Arrows: Yaw (Rotate)")
        print("----------------------------------")

        self._started = True
        self._stop_requested = False
        self._last_update_t = 0.0
        self._was_moving = False
        self._takeoff_pressed_prev = False
        self._land_pressed_prev = False
        self._flight_started = False
        self._recorded_steps = []
        self._active_record_step = None
        self._active_record_started_t = None

    def stop(self) -> None:
        if not self.cfg.enable or not self._started:
            return

        self._flush_active_record_step(time.monotonic())
        self._save_recording()
        self.client.armDisarm(False, vehicle_name=self.vehicle_name)
        self.client.enableApiControl(False, vehicle_name=self.vehicle_name)
        self._started = False

    def update(self) -> None:
        if not self.cfg.enable or not self._started or self._stop_requested:
            return

        now = time.monotonic()
        if (now - self._last_update_t) < self.cfg.loop_sleep:
            return
        self._last_update_t = now

        keys = self.cfg.keys

        try:
            if keyboard.is_pressed(keys.emergency):
                self._append_instant_record_step("land", now)
                self._flight_started = False
                print("Emergency stop triggered. Landing...")
                self.client.landAsync(vehicle_name=self.vehicle_name).join()
                self._stop_requested = True
                return

            takeoff_pressed = keyboard.is_pressed(keys.takeoff)
            if takeoff_pressed and not self._takeoff_pressed_prev:
                self._append_instant_record_step("takeoff", now)
                self._flight_started = True
                self.client.takeoffAsync(vehicle_name=self.vehicle_name)

            land_pressed = keyboard.is_pressed(keys.land)
            if land_pressed and not self._land_pressed_prev:
                self._append_instant_record_step("land", now)
                self._flight_started = False
                self.client.landAsync(vehicle_name=self.vehicle_name)

            vx = 0.0
            vy = 0.0
            vz = 0.0
            yaw_rate = 0.0
            moving = False

            if keyboard.is_pressed(keys.forward):
                vx = self.cfg.speed
                moving = True
            elif keyboard.is_pressed(keys.backward):
                vx = -self.cfg.speed
                moving = True

            if keyboard.is_pressed(keys.left):
                vy = -self.cfg.speed
                moving = True
            elif keyboard.is_pressed(keys.right):
                vy = self.cfg.speed
                moving = True

            if keyboard.is_pressed(keys.up):
                vz = -self.cfg.speed
                moving = True
            elif keyboard.is_pressed(keys.down):
                vz = self.cfg.speed
                moving = True

            if keyboard.is_pressed(keys.yaw_left):
                yaw_rate = -self.cfg.yaw_speed
                moving = True
            elif keyboard.is_pressed(keys.yaw_right):
                yaw_rate = self.cfg.yaw_speed
                moving = True

            if moving:
                self._flight_started = True

            self._update_recording(
                now=now,
                moving=moving,
                vx=vx,
                vy=vy,
                vz=vz,
                yaw_rate=yaw_rate,
            )

            if moving:
                self.client.moveByVelocityBodyFrameAsync(
                    vx=vx,
                    vy=vy,
                    vz=vz,
                    duration=self.cfg.command_duration,
                    yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate),
                    vehicle_name=self.vehicle_name,
                )
            elif self._was_moving:
                self.client.hoverAsync(vehicle_name=self.vehicle_name)

            self._was_moving = moving
            self._takeoff_pressed_prev = takeoff_pressed
            self._land_pressed_prev = land_pressed

        except Exception as exc:  # pragma: no cover - hardware/runtime interaction
            print(f"Manual control loop error: {exc}")
            self._stop_requested = True

    def _update_recording(
        self,
        now: float,
        moving: bool,
        vx: float,
        vy: float,
        vz: float,
        yaw_rate: float,
    ) -> None:
        if not self.cfg.record_enable:
            return

        current_step: dict[str, float | str] | None = None
        if moving:
            current_step = {
                "mode": "velocity",
                "name": "manual_velocity",
                "vx": float(vx),
                "vy": float(vy),
                "vz": float(vz),
                "yaw_rate": float(yaw_rate),
            }
        elif self._flight_started:
            current_step = {
                "mode": "hover",
                "name": "manual_hover",
            }

        if current_step is None:
            self._flush_active_record_step(now)
            return

        if self._active_record_step is None:
            self._active_record_step = current_step
            self._active_record_started_t = now
            return

        if self._record_steps_equal(self._active_record_step, current_step):
            return

        self._flush_active_record_step(now)
        self._active_record_step = current_step
        self._active_record_started_t = now

    def _append_instant_record_step(self, mode: str, now: float) -> None:
        if not self.cfg.record_enable:
            return

        self._flush_active_record_step(now)
        self._recorded_steps.append(
            {
                "mode": mode,
                "name": mode,
                "duration": 0.0,
            }
        )

    def _flush_active_record_step(self, now: float) -> None:
        if not self.cfg.record_enable:
            return

        if self._active_record_step is None or self._active_record_started_t is None:
            self._active_record_step = None
            self._active_record_started_t = None
            return

        duration = max(0.0, float(now - self._active_record_started_t))
        if duration > 1e-3:
            step = dict(self._active_record_step)
            step["duration"] = duration
            self._recorded_steps.append(step)

        self._active_record_step = None
        self._active_record_started_t = None

    def _save_recording(self) -> None:
        if not self.cfg.record_enable or len(self._recorded_steps) == 0:
            return

        output_path = Path(self.cfg.record_path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "format": "fusionuav_manual_record_v1",
            "saved_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "vehicle_name": self.vehicle_name,
            "command_duration": float(self.cfg.command_duration),
            "steps": self._recorded_steps,
        }
        with output_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False)

        print(f"Saved manual command recording to {output_path}")

    @staticmethod
    def _record_steps_equal(
        lhs: dict[str, float | str],
        rhs: dict[str, float | str],
    ) -> bool:
        if lhs.get("mode") != rhs.get("mode"):
            return False

        for key in ("vx", "vy", "vz", "yaw_rate"):
            if float(lhs.get(key, 0.0)) != float(rhs.get(key, 0.0)):
                return False

        return True
