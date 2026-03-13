from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
    from pyqtgraph.exporters import ImageExporter
except ImportError:  # pragma: no cover - dependency availability is environment-specific
    pg = None
    QtCore = None
    QtGui = None
    QtWidgets = None
    ImageExporter = None


@dataclass
class LiveRunLogger:
    t_data: list[float] = field(default_factory=list)

    est_p: list[np.ndarray] = field(default_factory=list)
    gt_p: list[np.ndarray] = field(default_factory=list)

    est_v: list[np.ndarray] = field(default_factory=list)
    gt_v: list[np.ndarray] = field(default_factory=list)

    est_q: list[np.ndarray] = field(default_factory=list)
    gt_q: list[np.ndarray] = field(default_factory=list)

    est_ba: list[np.ndarray] = field(default_factory=list)
    est_bw: list[np.ndarray] = field(default_factory=list)
    est_g: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.app = None
        if pg is not None and QtWidgets is not None:
            self.app = QtWidgets.QApplication.instance()
            if self.app is None:
                self.app = QtWidgets.QApplication([])

    def append(
        self,
        t: float,
        p_est: np.ndarray,
        p_gt: np.ndarray,
        v_est: np.ndarray,
        v_gt: np.ndarray,
        q_est: np.ndarray,
        q_gt: np.ndarray,
        ba_est: np.ndarray,
        bw_est: np.ndarray,
        g_est: np.ndarray,
    ) -> None:
        self.t_data.append(float(t))

        self.est_p.append(np.asarray(p_est, dtype=float).reshape(3).copy())
        self.gt_p.append(np.asarray(p_gt, dtype=float).reshape(3).copy())

        self.est_v.append(np.asarray(v_est, dtype=float).reshape(3).copy())
        self.gt_v.append(np.asarray(v_gt, dtype=float).reshape(3).copy())

        self.est_q.append(np.asarray(q_est, dtype=float).reshape(4).copy())
        self.gt_q.append(np.asarray(q_gt, dtype=float).reshape(4).copy())

        self.est_ba.append(np.asarray(ba_est, dtype=float).reshape(3).copy())
        self.est_bw.append(np.asarray(bw_est, dtype=float).reshape(3).copy())
        self.est_g.append(np.asarray(g_est, dtype=float).reshape(3).copy())

    def is_empty(self) -> bool:
        return len(self.t_data) == 0

    def finalize(self, root: Path, live_plotter=None) -> Path | None:
        if self.is_empty():
            return None

        output_dir = self._create_output_dir(root)
        self.save_csv(output_dir)
        self.save_plots(output_dir)

        if live_plotter is not None:
            live_plotter.save_png(output_dir / "live_position_plot.png")

        return output_dir

    def _create_output_dir(self, root: Path) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = root / "outputs" / "live_eskf" / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def save_csv(self, output_dir: Path) -> Path:
        import csv

        output_path = output_dir / "state_log.csv"
        header = [
            "t",
            "p_est_x", "p_est_y", "p_est_z",
            "p_gt_x", "p_gt_y", "p_gt_z",
            "v_est_x", "v_est_y", "v_est_z",
            "v_gt_x", "v_gt_y", "v_gt_z",
            "q_est_w", "q_est_x", "q_est_y", "q_est_z",
            "q_gt_w", "q_gt_x", "q_gt_y", "q_gt_z",
            "ba_est_x", "ba_est_y", "ba_est_z",
            "bw_est_x", "bw_est_y", "bw_est_z",
            "g_est_x", "g_est_y", "g_est_z",
        ]
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for idx, t in enumerate(self.t_data):
                writer.writerow([
                    t,
                    *self.est_p[idx],
                    *self.gt_p[idx],
                    *self.est_v[idx],
                    *self.gt_v[idx],
                    *self.est_q[idx],
                    *self.gt_q[idx],
                    *self.est_ba[idx],
                    *self.est_bw[idx],
                    *self.est_g[idx],
                ])
        return output_path

    def save_plots(self, output_dir: Path) -> list[Path]:
        self._require_pyqtgraph()

        return [
            self._save_three_axis_plot(
                output_dir / "position_est_vs_gt.png",
                title="Estimated Position vs Ground Truth",
                ylabels=("x [m]", "y [m]", "z [m]"),
                est_series=self.est_p,
                gt_series=self.gt_p,
            ),
            self._save_three_axis_plot(
                output_dir / "velocity_est_vs_gt.png",
                title="Estimated Velocity vs Ground Truth",
                # ylabels=("vx [m/s]", "vy [m/s]", "vz [m/s]"),
                ylabels=("V<sub>x [m/s]", "V<sub>y [m/s]", "V<sub>z [m/s]"),
                est_series=self.est_v,
                gt_series=self.gt_v,
            ),
            self._save_four_axis_plot(
                output_dir / "quaternion_est_vs_gt.png",
                title="Estimated Quaternion vs Ground Truth",
                # ylabels=("qw", "qx", "qy", "qz"),
                ylabels=("q<sub>w", "q<sub>x", "q<sub>y", "q<sub>z"),
                est_series=self.est_q,
                gt_series=self.gt_q,
            ),
            self._save_three_axis_plot(
                output_dir / "accel_bias_est.png",
                title="Estimated Accelerometer Bias",
                ylabels=("ba_x [m/s^2]", "ba_y [m/s^2]", "ba_z [m/s^2]"),
                est_series=self.est_ba,
                gt_series=None,
            ),
            self._save_three_axis_plot(
                output_dir / "gyro_bias_est.png",
                title="Estimated Gyroscope Bias",
                # ylabels=("bw_x [rad/s]", "bw_y [rad/s]", "bw_z [rad/s]"),
                ylabels=("b<sub>a,x</sub> [m/s²]", "b<sub>a,y</sub> [m/s²]", "b<sub>a,z</sub> [m/s²]"),
                est_series=self.est_bw,
                gt_series=None,
            ),
            self._save_three_axis_plot(
                output_dir / "gravity_vector_est.png",
                title="Estimated Gravity Vector",
                # ylabels=("g_x [m/s^2]", "g_y [m/s^2]", "g_z [m/s^2]"),
                ylabels=("g<sub>x [m/s²]", "g<sub>y [m/s²]", "g<sub>z [m/s²]"),
                est_series=self.est_g,
                gt_series=None,
            ),
        ]

    def _require_pyqtgraph(self) -> None:
        if (
            pg is None
            or QtWidgets is None
            or QtCore is None
            or QtGui is None
            or ImageExporter is None
        ):
            raise RuntimeError(
                "Saving plots requires 'pyqtgraph' and a Qt binding such as 'PyQt5'."
            )
        if self.app is None:
            self.app = QtWidgets.QApplication.instance()
            if self.app is None:
                self.app = QtWidgets.QApplication([])
        pg.setConfigOptions(antialias=True, background="w", foreground="k")

    def _save_three_axis_plot(
        self,
        output_path: Path,
        title: str,
        ylabels: tuple[str, str, str],
        est_series: list[np.ndarray],
        gt_series: list[np.ndarray] | None,
    ) -> Path:
        return self._save_multi_axis_plot(
            output_path=output_path,
            title=title,
            ylabels=list(ylabels),
            est=np.asarray(est_series, dtype=float),
            gt=None if gt_series is None else np.asarray(gt_series, dtype=float),
        )

    def _save_four_axis_plot(
        self,
        output_path: Path,
        title: str,
        ylabels: tuple[str, str, str, str],
        est_series: list[np.ndarray],
        gt_series: list[np.ndarray],
    ) -> Path:
        return self._save_multi_axis_plot(
            output_path=output_path,
            title=title,
            ylabels=list(ylabels),
            est=np.asarray(est_series, dtype=float),
            gt=np.asarray(gt_series, dtype=float),
        )

    def _save_multi_axis_plot(
        self,
        output_path: Path,
        title: str,
        ylabels: list[str],
        est: np.ndarray,
        gt: np.ndarray | None,
    ) -> Path:
        widget = pg.GraphicsLayoutWidget(show=False, title=title)
        widget.resize(1500, 360 * len(ylabels))
        widget.setWindowTitle(title)

        est_pen = pg.mkPen(color=(40, 120, 220), width=2)
        gt_pen = pg.mkPen(
            color=(30, 150, 70),
            width=2,
            style=QtCore.Qt.PenStyle.DashLine,
        )
        tick_font = QtGui.QFont()
        tick_font.setPointSize(11)
        axis_label_style = {"font-size": "13pt"}
        t = np.asarray(self.t_data, dtype=float)

        for idx, ylabel in enumerate(ylabels):
            plot = widget.addPlot(row=idx, col=0)
            plot.showGrid(x=True, y=True, alpha=0.25)
            plot.addLegend(labelTextSize="11pt")
            plot.setLabel("left", ylabel, **axis_label_style)
            plot.getAxis("left").enableAutoSIPrefix(False)
            plot.getAxis("left").setTickFont(tick_font)
            plot.getAxis("bottom").setTickFont(tick_font)
            if idx == len(ylabels) - 1:
                plot.setLabel("bottom", "time [s]", **axis_label_style)
            if idx == 0:
                plot.setTitle(title, size="15pt")
            plot.plot(t, est[:, idx], name="Estimate", pen=est_pen)
            if gt is not None:
                plot.plot(t, gt[:, idx], name="Ground Truth", pen=gt_pen)

        widget.show()
        self.app.processEvents()
        exporter = ImageExporter(widget.scene())
        params = exporter.parameters()
        if "width" in params:
            params["width"] = 2400
        exporter.export(str(output_path))
        widget.close()
        self.app.processEvents()
        return output_path
