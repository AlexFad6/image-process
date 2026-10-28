from __future__ import annotations

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QLabel, QPushButton

from core.app_state import TransformParams


class ParamPanel(QGroupBox):
    changed = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__("变换参数", parent)
        form = QFormLayout(self)

        self.mode = QComboBox()
        self.mode.addItems(["linear", "log", "gamma"])

        self.low = self._spin(0, 255, 0)
        self.high = self._spin(0, 255, 255)
        self.log_c = self._spin(0.01, 20, 1, 0.1)
        self.gamma = self._spin(0.01, 10, 1, 0.1)

        self.low_label = QLabel("低值")
        self.high_label = QLabel("高值")
        self.log_label = QLabel("log c")
        self.gamma_label = QLabel("gamma")

        form.addRow("模式", self.mode)
        form.addRow(self.low_label, self.low)
        form.addRow(self.high_label, self.high)
        form.addRow(self.log_label, self.log_c)
        form.addRow(self.gamma_label, self.gamma)

        self.reset = QPushButton("重置参数")
        form.addRow(self.reset)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(200)

        for widget in (self.mode, self.low, self.high, self.log_c, self.gamma):
            signal = widget.currentIndexChanged if isinstance(widget, QComboBox) else widget.valueChanged
            signal.connect(self._schedule)

        self.mode.currentIndexChanged.connect(self._sync_visibility)
        self.reset.clicked.connect(self._reset)
        self._timer.timeout.connect(self._emit)

        self._sync_visibility(self.mode.currentIndex())

    @staticmethod
    def _spin(minimum: float, maximum: float, value: float, step: float = 1) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(minimum, maximum)
        box.setValue(value)
        box.setSingleStep(step)
        return box

    def _schedule(self, *_args) -> None:
        self._timer.start()

    def _emit(self) -> None:
        self.changed.emit(self.params())

    def _sync_visibility(self, _index: int) -> None:
        mode = self.mode.currentText()
        self.low_label.setVisible(mode == "linear")
        self.high_label.setVisible(mode == "linear")
        self.low.setVisible(mode == "linear")
        self.high.setVisible(mode == "linear")

        self.log_label.setVisible(mode == "log")
        self.log_c.setVisible(mode == "log")

        self.gamma_label.setVisible(mode == "gamma")
        self.gamma.setVisible(mode == "gamma")

    def _reset(self) -> None:
        self.mode.setCurrentText("linear")
        self.low.setValue(0)
        self.high.setValue(255)
        self.log_c.setValue(1)
        self.gamma.setValue(1)

    def params(self) -> TransformParams:
        return TransformParams(
            mode=self.mode.currentText(),
            low=self.low.value(),
            high=self.high.value(),
            log_c=self.log_c.value(),
            gamma=self.gamma.value(),
        )
