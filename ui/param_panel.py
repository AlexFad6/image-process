from __future__ import annotations

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSizePolicy,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from core.app_state import TransformParams


class ParamPanel(QGroupBox):
    changed = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__("变换参数", parent)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
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

        self.a, self.a_row, self.a_slider = self._parameter_row("a", 0, 255, 0)
        self.b, self.b_row, self.b_slider = self._parameter_row("b", 0, 255, 255)
        self.c, self.c_row, self.c_slider = self._parameter_row("c", 0, 255, 0)
        self.d, self.d_row, self.d_slider = self._parameter_row("d", 0, 255, 255)
        self.gamma, self.gamma_row, self.gamma_slider = self._parameter_row(
            "gamma", 0.01, 10, 1, 2, 0.01
        )

        form.addRow(self.a_row)
        form.addRow(self.b_row)
        form.addRow(self.c_row)
        form.addRow(self.d_row)
        form.addRow(self.gamma_row)

        self.reset = QPushButton("重置参数")
        layout.addWidget(self.reset)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(200)

        for widget in (self.a, self.b, self.c, self.d, self.gamma):
            widget.valueChanged.connect(self._schedule)
        for slider in (
            self.a_slider,
            self.b_slider,
            self.c_slider,
            self.d_slider,
            self.gamma_slider,
        ):
            slider.valueChanged.connect(self._schedule)

        self.reset.clicked.connect(self._reset)
        self._timer.timeout.connect(self._emit)

        self._sync_visibility()

    @staticmethod
    def _parameter_row(
        label_text: str,
        minimum: float,
        maximum: float,
        value: float,
        decimals: int = 2,
        step: float = 1,
    ) -> tuple[QDoubleSpinBox, QWidget, QSlider]:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel(label_text)
        label.setFixedWidth(45)
        row_layout.addWidget(label)

        scale_factor = 10 ** decimals
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(int(minimum * scale_factor), int(maximum * scale_factor))
        slider.setValue(int(value * scale_factor))
        row_layout.addWidget(slider)

        spinbox = QDoubleSpinBox()
        spinbox.setRange(minimum, maximum)
        spinbox.setDecimals(decimals)
        spinbox.setSingleStep(step)
        spinbox.setValue(value)
        spinbox.setFixedWidth(80)
        spinbox.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        spinbox.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row_layout.addWidget(spinbox)

        def on_slider_changed(slider_value: int) -> None:
            spinbox.blockSignals(True)
            spinbox.setValue(slider_value / scale_factor)
            spinbox.blockSignals(False)

        def on_spinbox_changed(spin_value: float) -> None:
            slider.blockSignals(True)
            slider.setValue(int(round(spin_value * scale_factor)))
            slider.blockSignals(False)

        slider.valueChanged.connect(on_slider_changed)
        spinbox.valueChanged.connect(on_spinbox_changed)
        return spinbox, row, slider

    def _schedule(self, *_args) -> None:
        self._timer.start()

    def _emit(self) -> None:
        self.changed.emit(self.params())

    def _sync_visibility(self, *_args) -> None:
        mode = self.mode_name()
        self.a_row.setVisible(mode == "linear")
        self.b_row.setVisible(mode == "linear")
        self.c_row.setVisible(mode in ("linear", "log", "gamma"))
        self.d_row.setVisible(mode == "linear")
        self.gamma_row.setVisible(mode == "gamma")

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
