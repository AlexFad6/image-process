from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QGroupBox, QLabel, QTabWidget, QVBoxLayout, QWidget


class AttributePanel(QGroupBox):
    def __init__(self, parent=None) -> None:
        super().__init__("属性与直方图", parent)
        self.tabs = QTabWidget(self)
        self.tab_1 = QWidget()
        self.tab_2 = QWidget()

        self.info_1 = QLabel("图像1：未打开图片")
        self.plot_1 = pg.PlotWidget()
        self.plot_1.setMinimumHeight(180)
        self.plot_1.setLabel("bottom", "灰度")
        self.plot_1.setLabel("left", "像素")

        self.info_2 = QLabel("图像2：未打开图片")
        self.plot_2 = pg.PlotWidget()
        self.plot_2.setMinimumHeight(180)
        self.plot_2.setLabel("bottom", "灰度")
        self.plot_2.setLabel("left", "像素")

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
        target_plot = self.plot_1 if not source else self.plot_2
        target_plot.clear()

        if image is None:
            target_info.setText("图像1：未打开图片" if not source else "图像2：未打开图片")
            return

        hist = np.bincount(image.ravel(), minlength=256)
        target_info.setText(
            f"{('图像1' if not source else '图像2')}：尺寸 {image.shape[1]} × {image.shape[0]} | "
            f"均值 {image.mean():.2f} | 最小/最大 {image.min()} / {image.max()} | 通道 {image.ndim}"
        )
        target_plot.plot(hist, pen="y")

    def set_original(self, image: np.ndarray | None) -> None:
        self.set_image(image, source=False)

    def set_result(self, image: np.ndarray | None) -> None:
        self.set_image(image, source=True)
