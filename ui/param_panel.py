from __future__ import annotations

from PySide6.QtCore import QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QToolBar,
    QVBoxLayout,
)

from core.app_state import TransformParams


class ParamPanel(QGroupBox):
    changed = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__("变换参数", parent)
        layout = QVBoxLayout(self)
        self.mode_toolbar = QToolBar("模式", self)
        self.mode_toolbar.setMovable(False)
        self.mode_group = QActionGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_actions = {}
        for value, text in (
            ("linear", "线性变换"),
            ("log", "对数变换"),
            ("gamma", "伽马变换"),
        ):
            action = QAction(text, self)
            action.setCheckable(True)
            action.triggered.connect(lambda checked, mode=value: self.set_mode(mode))
            self.mode_group.addAction(action)
            self.mode_toolbar.addAction(action)
            self.mode_actions[value] = action
        self.mode_actions["linear"].setChecked(True)
        layout.addWidget(self.mode_toolbar)

        form = QFormLayout()
        layout.addLayout(form)

        self.a = self._spin(0, 255, 0)
        self.b = self._spin(0, 255, 255)
        self.c = self._spin(0, 255, 0)
        self.d = self._spin(0, 255, 255)
        self.gamma = self._spin(0.01, 10, 1, 0.1)

        self.low_label = QLabel("a")
        self.high_label = QLabel("b")
        self.c_label = QLabel("c")
        self.d_label = QLabel("d")
        self.gamma_label = QLabel("gamma")

        form.addRow(self.low_label, self.a)
        form.addRow(self.high_label, self.b)
        form.addRow(self.c_label, self.c)
        form.addRow(self.d_label, self.d)
        form.addRow(self.gamma_label, self.gamma)

        self.reset = QPushButton("重置参数")
        layout.addWidget(self.reset)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(200)

        for widget in (self.a, self.b, self.c, self.d, self.gamma):
            widget.valueChanged.connect(self._schedule)

        self.reset.clicked.connect(self._reset)
        self._timer.timeout.connect(self._emit)

        self._sync_visibility()

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

    def _sync_visibility(self, *_args) -> None:
        mode = self.mode_name()
        self.low_label.setVisible(mode == "linear")
        self.high_label.setVisible(mode == "linear")
        self.a.setVisible(mode == "linear")
        self.b.setVisible(mode == "linear")

        self.c_label.setVisible(mode in ("log", "gamma"))
        self.c.setVisible(mode in ("log", "gamma"))
        self.d_label.setVisible(mode == "linear")
        self.d.setVisible(mode == "linear")

        self.gamma_label.setVisible(mode == "gamma")
        self.gamma.setVisible(mode == "gamma")

    def _reset(self) -> None:
        self.mode_actions["linear"].setChecked(True)
        self.a.setValue(0)
        self.b.setValue(255)
        self.c.setValue(0)
        self.d.setValue(255)
        self.gamma.setValue(1)
        self._sync_visibility()

    def params(self) -> TransformParams:
        return TransformParams(
            mode=self.mode_name(),
            a=self.a.value(),
            b=self.b.value(),
            c=self.c.value(),
            d=self.d.value(),
            gamma=self.gamma.value(),
        )

    def mode_name(self) -> str:
        for value, action in self.mode_actions.items():
            if action.isChecked():
                return value
        raise RuntimeError("No transform mode is selected")

    def set_mode(self, mode: str) -> None:
        if mode not in self.mode_actions:
            raise ValueError(f"Unsupported transform mode: {mode}")
        self.mode_actions[mode].setChecked(True)
        if mode in ("log", "gamma") and self.c.value() == 0:
            self.c.setValue(1)
        self._sync_visibility()
        self._schedule()
