from __future__ import annotations

import numpy as np
import cv2

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QGroupBox, QLabel, QTabWidget, QVBoxLayout, QWidget


class AttributePanel(QGroupBox):
    def __init__(self, parent=None) -> None:
        super().__init__("属性", parent)
        self.tabs = QTabWidget(self)
        self.tab_1 = QWidget()
        self.tab_2 = QWidget()

        self.info_1 = QLabel("图像1：未打开图片")
        self.figure_1 = Figure()
        self.axes_1 = self.figure_1.add_subplot(111)
        self.plot_1 = FigureCanvasQTAgg(self.figure_1)
        self.plot_1.setMinimumHeight(180)

        self.info_2 = QLabel("图像2：未打开图片")
        self.figure_2 = Figure()
        self.axes_2 = self.figure_2.add_subplot(111)
        self.plot_2 = FigureCanvasQTAgg(self.figure_2)
        self.plot_2.setMinimumHeight(180)

        layout_1 = QVBoxLayout(self.tab_1)
        layout_1.addWidget(self.info_1)
        layout_1.addWidget(self.plot_1)

        layout_2 = QVBoxLayout(self.tab_2)
        layout_2.addWidget(self.info_2)
        layout_2.addWidget(self.plot_2)

        self.tabs.addTab(self.tab_1, "图像1")
        self.tabs.addTab(self.tab_2, "图像2")

        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs)

    def set_image(self, image: np.ndarray | None, *, source: bool = False) -> None:
        target_info = self.info_1 if not source else self.info_2
        target_axes = self.axes_1 if not source else self.axes_2
        target_plot = self.plot_1 if not source else self.plot_2
        target_axes.clear()

        if image is None:
            target_info.setText("图像1：未打开图片" if not source else "图像2：未打开图片")
            target_plot.draw_idle()
            return

        hist = cv2.calcHist([image], [0], None, [256], [0, 256]).ravel()
        target_info.setText(
            f"{('图像1' if not source else '图像2')}：尺寸 {image.shape[1]} × {image.shape[0]} | "
            f"均值 {image.mean():.2f}"
        )
        target_axes.bar(range(256), hist, width=1, color="steelblue")
        target_axes.set_xlim(0, 256)
        target_axes.set_xticks([])
        target_axes.set_yticks([])
        target_plot.figure.tight_layout()
        target_plot.draw_idle()

    def set_original(self, image: np.ndarray | None) -> None:
        self.set_image(image, source=False)

    def set_result(self, image: np.ndarray | None) -> None:
        self.set_image(image, source=True)
