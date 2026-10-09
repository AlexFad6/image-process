from __future__ import annotations

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox,
                               QPushButton)
from core.app_state import TransformParams


class ParamPanel(QGroupBox):
    changed = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__("变换参数", parent)
        form = QFormLayout(self)
        self.mode = QComboBox(); self.mode.addItems(["linear", "log", "gamma"])
        self.backend = QComboBox(); self.backend.addItems(["numpy", "opencv"])
        self.low = self._spin(0, 255, 0)
        self.high = self._spin(0, 255, 255)
        self.log_c = self._spin(0.01, 20, 1, 0.1)
        self.gamma = self._spin(0.01, 10, 1, 0.1)
        for label, widget in [("模式", self.mode), ("后端", self.backend), ("低值", self.low),
                              ("高值", self.high), ("log c", self.log_c), ("gamma", self.gamma)]:
            form.addRow(label, widget)
        self.reset = QPushButton("重置参数")
        form.addRow(self.reset)
        self._timer = QTimer(self); self._timer.setSingleShot(True); self._timer.setInterval(180)
        for widget in (self.mode, self.backend, self.low, self.high, self.log_c, self.gamma):
            (widget.currentIndexChanged if isinstance(widget, QComboBox) else widget.valueChanged).connect(self._schedule)
        self.reset.clicked.connect(self._reset)
        self._timer.timeout.connect(self._emit)

    @staticmethod
    def _spin(minimum: float, maximum: float, value: float, step: float = 1) -> QDoubleSpinBox:
        box = QDoubleSpinBox(); box.setRange(minimum, maximum); box.setValue(value); box.setSingleStep(step)
        return box

    def _schedule(self, *_args) -> None: self._timer.start()
    def _emit(self) -> None: self.changed.emit(self.params())
    def _reset(self) -> None:
        self.mode.setCurrentText("linear"); self.backend.setCurrentText("numpy")
        self.low.setValue(0); self.high.setValue(255); self.log_c.setValue(1); self.gamma.setValue(1)
    def params(self) -> TransformParams:
        return TransformParams(self.mode.currentText(), self.low.value(), self.high.value(),
                               self.log_c.value(), self.gamma.value(), self.backend.currentText())
