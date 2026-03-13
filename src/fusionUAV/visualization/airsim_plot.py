from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import airsim
import numpy as np


@dataclass
class AirSimPlotConfig:
    enable: bool = True
    period: float = 0.20
    max_points: int = 2000
    duration: float = 0.35

    gt_color: Sequence[float] = (0.0, 1.0, 0.0, 1.0)
    est_color: Sequence[float] = (1.0, 0.0, 0.0, 1.0)
    gt_point_color: Sequence[float] = (0.0, 1.0, 0.0, 1.0)
    est_point_color: Sequence[float] = (1.0, 0.0, 0.0, 1.0)

    gt_line_thickness: float = 8.0
    est_line_thickness: float = 8.0
    gt_point_size: float = 12.0
    est_point_size: float = 12.0


def np_to_vector3r(p: np.ndarray) -> airsim.Vector3r:
    return airsim.Vector3r(float(p[0]), float(p[1]), float(p[2]))


@dataclass
class AirSimTrajectoryPlotter:
    client: airsim.MultirotorClient
    cfg: AirSimPlotConfig = field(default_factory=AirSimPlotConfig)

    gt_path: list[airsim.Vector3r] = field(default_factory=list)
    est_path: list[airsim.Vector3r] = field(default_factory=list)

    def _trim(self) -> None:
        if len(self.gt_path) > self.cfg.max_points:
            del self.gt_path[:-self.cfg.max_points]
        if len(self.est_path) > self.cfg.max_points:
            del self.est_path[:-self.cfg.max_points]

    def add_points(self, p_gt: np.ndarray, p_est: np.ndarray) -> None:
        self.gt_path.append(np_to_vector3r(np.asarray(p_gt, dtype=float)))
        self.est_path.append(np_to_vector3r(np.asarray(p_est, dtype=float)))
        self._trim()

    def draw(self) -> None:
        if not self.cfg.enable:
            return

        if len(self.gt_path) >= 2:
            self.client.simPlotLineStrip(
                points=self.gt_path,
                color_rgba=list(self.cfg.gt_color),
                thickness=float(self.cfg.gt_line_thickness),
                duration=float(self.cfg.duration),
                is_persistent=False,
            )

        if len(self.est_path) >= 2:
            self.client.simPlotLineStrip(
                points=self.est_path,
                color_rgba=list(self.cfg.est_color),
                thickness=float(self.cfg.est_line_thickness),
                duration=float(self.cfg.duration),
                is_persistent=False,
            )

        if len(self.gt_path) >= 1:
            self.client.simPlotPoints(
                points=[self.gt_path[-1]],
                color_rgba=list(self.cfg.gt_point_color),
                size=float(self.cfg.gt_point_size),
                duration=float(self.cfg.duration),
                is_persistent=False,
            )

        if len(self.est_path) >= 1:
            self.client.simPlotPoints(
                points=[self.est_path[-1]],
                color_rgba=list(self.cfg.est_point_color),
                size=float(self.cfg.est_point_size),
                duration=float(self.cfg.duration),
                is_persistent=False,
            )

    def update(self, p_gt: np.ndarray, p_est: np.ndarray) -> None:
        self.add_points(p_gt=p_gt, p_est=p_est)
        self.draw()

    def clear(self) -> None:
        self.gt_path.clear()
        self.est_path.clear()