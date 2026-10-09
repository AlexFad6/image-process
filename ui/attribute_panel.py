from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QGroupBox, QLabel, QVBoxLayout


class AttributePanel(QGroupBox):
    def __init__(self, parent=None) -> None:
        super().__init__("属性与直方图", parent)
        self.info = QLabel("未打开图片")
        self.plot = pg.PlotWidget(); self.plot.setMinimumHeight(150)
        self.plot.setLabel("bottom", "灰度"); self.plot.setLabel("left", "像素")
        layout = QVBoxLayout(self); layout.addWidget(self.info); layout.addWidget(self.plot)

    def set_image(self, image: np.ndarray | None) -> None:
        self.plot.clear()
        if image is None:
            self.info.setText("未打开图片"); return
        hist = np.bincount(image.ravel(), minlength=256)
        self.info.setText(f"尺寸: {image.shape[1]} × {image.shape[0]}    "
                          f"均值: {image.mean():.2f}    最小/最大: {image.min()}/{image.max()}")
        self.plot.plot(hist, pen="y")
