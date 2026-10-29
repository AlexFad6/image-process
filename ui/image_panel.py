from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .image_view import ImageView


class ImagePanel(QGroupBox):
    convertToGrayRequested = Signal()
    displayModeRequested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__("图像对比", parent)
        self.sync_zoom = True

        outer = QVBoxLayout(self)
        top_bar = QWidget()
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.addStretch()
        self.convert_gray = QPushButton("灰度图")
        self.convert_gray.setToolTip("切换原图、灰度图、频谱图和相位谱图")
        self.convert_gray.clicked.connect(self.displayModeRequested.emit)
        top_layout.addWidget(self.convert_gray)
        outer.addWidget(top_bar)

        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        self.original = ImageView()
        self.result = ImageView()
        for view in (self.original, self.result):
            view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            content_layout.addWidget(view, 1)
        outer.addWidget(content)

        self.original.set_partner(self.result)
        self.result.set_partner(self.original)

    def set_sync_zoom(self, enabled: bool) -> None:
        self.sync_zoom = enabled
        self.original.set_sync(enabled)
        self.result.set_sync(enabled)

    def reset_views(self) -> None:
        self.original.reset_view()
        self.result.reset_view()
        self.sync_views()

    def sync_views(self) -> None:
        state = self.original.get_view_state()
        if state is None:
            return
        self.result.restore_view_state(state)
