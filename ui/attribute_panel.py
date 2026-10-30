from __future__ import annotations

import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class AttributePanel(QGroupBox):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._images: list[np.ndarray | None] = [None, None]

        self.figure_1 = Figure()
        self.axes_1 = self.figure_1.add_subplot(111)
        self.figure_1.subplots_adjust(bottom=0.1)
        self.plot_1 = FigureCanvasQTAgg(self.figure_1)
        self.plot_1.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        # self.plot_1.setMinimumSize(240, 180)

        self.figure_2 = Figure()
        self.axes_2 = self.figure_2.add_subplot(111)
        self.figure_2.subplots_adjust(bottom=0.1)
        self.plot_2 = FigureCanvasQTAgg(self.figure_2)
        self.plot_2.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        # self.plot_2.setMinimumSize(240, 180)

        panel_1 = QWidget()
        layout_1 = QVBoxLayout(panel_1)
        layout_1.setContentsMargins(0, 0, 0, 0)
        layout_1.addWidget(self.plot_1)

        panel_2 = QWidget()
        layout_2 = QVBoxLayout(panel_2)
        layout_2.setContentsMargins(0, 0, 0, 0)
        layout_2.addWidget(self.plot_2)

        layout = QVBoxLayout(self)
        charts_layout = QHBoxLayout()
        charts_layout.setContentsMargins(0, 0, 0, 0)
        charts_layout.setSpacing(5)
        charts_layout.addWidget(panel_1)
        charts_layout.addWidget(panel_2)
        layout.addLayout(charts_layout)
        self.retranslateUi()

    def retranslateUi(self) -> None:
        self.setTitle(self.tr("直方图"))

    def set_image(self, image: np.ndarray | None, *, source: bool = False) -> None:
        self._images[1 if source else 0] = image
        self._refresh_charts()

    def _refresh_charts(self) -> None:
        for index, image in enumerate(self._images):
            axes = self.axes_2 if index else self.axes_1
            plot = self.plot_2 if index else self.plot_1
            axes.clear()

            axes.set_ylim(0, 0.05)
            axes.set_xlim(0, 256)
            axes.spines["top"].set_visible(False)
            axes.spines["right"].set_visible(False)
            axes.spines["left"].set_visible(False)
            axes.set_yticks([])
            axes.set_xticks([0, 50, 100, 150, 200, 255])
            axes.tick_params(axis="x", colors="black", labelsize=5)
            axes.spines["bottom"].set_color("gray")
            axes.spines["bottom"].set_linewidth(1)

            if image is None:
                plot.draw_idle()
                continue

            hist, _, _ = axes.hist(
                image.ravel(),
                bins=256,
                range=(0, 256),
                density=True,
                color="steelblue",
            )
        for axes, plot in (
            (self.axes_1, self.plot_1),
            (self.axes_2, self.plot_2),
        ):
            plot.draw_idle()

    def set_original(self, image: np.ndarray | None) -> None:
        self.set_image(image, source=False)

    def set_result(self, image: np.ndarray | None) -> None:
        self.set_image(image, source=True)
