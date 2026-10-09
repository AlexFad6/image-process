from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget


class WidgetPanel(QWidget):
    openRequested = Signal()
    saveRequested = Signal()
    sequenceRequested = Signal()
    undoRequested = Signal()
    redoRequested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        for text, signal in [("打开图片", self.openRequested), ("保存当前", self.saveRequested),
                             ("序列导出", self.sequenceRequested), ("◀", self.undoRequested),
                             ("▶", self.redoRequested)]:
            button = QPushButton(text); button.clicked.connect(signal); layout.addWidget(button)
