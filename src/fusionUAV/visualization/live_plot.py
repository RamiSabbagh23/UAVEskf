from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
except ImportError:  # pragma: no cover - dependency availability is environment-specific
    pg = None
    QtCore = None
    QtGui = None
    QtWidgets = None


@dataclass
class LivePlotConfig:
    enable: bool = True
    max_points: int = 1000
    figure_title: str = "ESKF Position: Estimate vs Ground Truth"


@dataclass
class LivePositionPlotter:
    cfg: LivePlotConfig = field(default_factory=LivePlotConfig)

    t_data: list[float] = field(default_factory=list)

    est_x: list[float] = field(default_factory=list)
    est_y: list[float] = field(default_factory=list)
    est_z: list[float] = field(default_factory=list)

    gt_x: list[float] = field(default_factory=list)
    gt_y: list[float] = field(default_factory=list)
    gt_z: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.app = None
        self.window = None

        if not self.cfg.enable:
            return

        if pg is None or QtWidgets is None or QtCore is None or QtGui is None:
            raise RuntimeError(
                "Live plotting requires 'pyqtgraph' and a Qt binding such as 'PyQt5'."
            )

        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])

        pg.setConfigOptions(antialias=True, background="w", foreground="k")

        self.window = pg.GraphicsLayoutWidget(show=False, title=self.cfg.figure_title)
        self.window.resize(1000, 800)
        self.window.setWindowTitle(self.cfg.figure_title)
        self._tick_font = QtGui.QFont()
        self._tick_font.setPointSize(11)
        self._axis_label_style = {"font-size": "13pt"}

        est_pen = pg.mkPen(color=(40, 120, 220), width=2)
        gt_pen = pg.mkPen(
            color=(30, 150, 70),
            width=2,
            style=QtCore.Qt.PenStyle.DashLine,
        )

        self.plot_x = self._create_plot(
            row=0,
            label="x [m]",
            est_name="Est",
            gt_name="Gt",
            est_pen=est_pen,
            gt_pen=gt_pen,
        )
        self.plot_y = self._create_plot(
            row=1,
            label="y [m]",
            est_name="Est",
            gt_name="Gt",
            est_pen=est_pen,
            gt_pen=gt_pen,
        )
        self.plot_z = self._create_plot(
            row=2,
            label="z [m]",
            est_name="Est",
            gt_name="Gt",
            est_pen=est_pen,
            gt_pen=gt_pen,
            add_bottom_label=True,
        )

        self.curve_est_x, self.curve_gt_x = self.plot_x
        self.curve_est_y, self.curve_gt_y = self.plot_y
        self.curve_est_z, self.curve_gt_z = self.plot_z

    def _create_plot(
        self,
        row: int,
        label: str,
        est_name: str,
        gt_name: str,
        est_pen,
        gt_pen,
        add_bottom_label: bool = False,
    ):
        plot = self.window.addPlot(row=row, col=0)
        plot.showGrid(x=True, y=True, alpha=0.25)
        plot.addLegend(labelTextSize="11pt")
        plot.setLabel("left", label, **self._axis_label_style)
        if add_bottom_label:
            plot.setLabel("bottom", "time [s]", **self._axis_label_style)
        plot.getAxis("left").setTickFont(self._tick_font)
        plot.getAxis("bottom").setTickFont(self._tick_font)
        plot.enableAutoRange(x=True, y=True)
        curve_est = plot.plot(name=est_name, pen=est_pen)
        curve_gt = plot.plot(name=gt_name, pen=gt_pen)
        return curve_est, curve_gt

    def _trim(self) -> None:
        n = self.cfg.max_points
        if len(self.t_data) > n:
            self.t_data = self.t_data[-n:]

            self.est_x = self.est_x[-n:]
            self.est_y = self.est_y[-n:]
            self.est_z = self.est_z[-n:]

            self.gt_x = self.gt_x[-n:]
            self.gt_y = self.gt_y[-n:]
            self.gt_z = self.gt_z[-n:]

    def update(self, t: float, p_est: np.ndarray, p_gt: np.ndarray) -> None:
        if not self.cfg.enable:
            return

        p_est = np.asarray(p_est, dtype=float).reshape(3)
        p_gt = np.asarray(p_gt, dtype=float).reshape(3)

        self.t_data.append(float(t))

        self.est_x.append(float(p_est[0]))
        self.est_y.append(float(p_est[1]))
        self.est_z.append(float(p_est[2]))

        self.gt_x.append(float(p_gt[0]))
        self.gt_y.append(float(p_gt[1]))
        self.gt_z.append(float(p_gt[2]))

        self._trim()
        self.redraw()

    def redraw(self) -> None:
        if not self.cfg.enable or len(self.t_data) == 0:
            return

        t_data = np.asarray(self.t_data, dtype=float)

        self.curve_est_x.setData(t_data, np.asarray(self.est_x, dtype=float))
        self.curve_gt_x.setData(t_data, np.asarray(self.gt_x, dtype=float))

        self.curve_est_y.setData(t_data, np.asarray(self.est_y, dtype=float))
        self.curve_gt_y.setData(t_data, np.asarray(self.gt_y, dtype=float))

        self.curve_est_z.setData(t_data, np.asarray(self.est_z, dtype=float))
        self.curve_gt_z.setData(t_data, np.asarray(self.gt_z, dtype=float))

        self.app.processEvents()

    def show(self, block: bool = False) -> None:
        if not self.cfg.enable or self.window is None:
            return

        self.window.show()
        self.app.processEvents()

        if block:
            exec_fn = getattr(self.app, "exec", None)
            if exec_fn is None:
                exec_fn = self.app.exec_
            exec_fn()

    def close(self) -> None:
        if self.window is None:
            return

        self.window.close()
        if self.app is not None:
            self.app.processEvents()

    def save_png(self, path) -> None:
        if self.window is None:
            return

        import pyqtgraph.exporters

        exporter = pyqtgraph.exporters.ImageExporter(self.window.scene())
        params = exporter.parameters()
        if "width" in params:
            params["width"] = 2200
        exporter.export(str(path))
