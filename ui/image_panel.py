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
    previousRequested = Signal()
    nextRequested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.sync_zoom = True

        outer = QVBoxLayout(self)
        top_bar = QWidget()
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.addStretch()
        self.previous_button = QPushButton()
        self.previous_button.clicked.connect(self.previousRequested.emit)
        top_layout.addWidget(self.previous_button)
        self.next_button = QPushButton()
        self.next_button.clicked.connect(self.nextRequested.emit)
        top_layout.addWidget(self.next_button)
        top_layout.addStretch()
        self.convert_gray = QPushButton()
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
        self.retranslateUi()

    def retranslateUi(self) -> None:
        self.setTitle(self.tr("图像对比"))
        self.convert_gray.setText(self.tr("灰度图"))
        self.convert_gray.setToolTip(self.tr("切换灰度图、频谱图和相位谱图"))
        self.previous_button.setText("←")
        self.previous_button.setToolTip(self.tr("上一张图像"))
        self.next_button.setText("→")
        self.next_button.setToolTip(self.tr("下一张图像"))

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
