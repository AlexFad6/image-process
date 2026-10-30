from __future__ import annotations

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QButtonGroup,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from core.app_state import FrequencyParams, SpatialParams, TransformParams


class ParamPanel(QGroupBox):
    changed = Signal(object)
    spatialChanged = Signal(object)
    frequencyChanged = Signal(object)
    fftRequested = Signal()
    ifftRequested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        self._transform_mode = "linear"

        self.pages = QStackedWidget(self)
        self.transform_page = self._create_transform_page()
        self.spatial_page = self._create_spatial_page()
        self.frequency_page = self._create_frequency_page()
        for page in (self.transform_page, self.spatial_page, self.frequency_page):
            self.pages.addWidget(page)
        layout.addWidget(self.pages)

        self.reset = QPushButton()
        layout.addWidget(self.reset)
        self.reset.clicked.connect(self._reset)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(200)
        self._timer.timeout.connect(self._emit_transform)
        self.retranslateUi()

    def _create_transform_page(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        self.transform_mode_group = QButtonGroup(page)
        transform_modes = QHBoxLayout()
        for text, value in (
            ("线性变换", "linear"),
            ("对数变换", "log"),
            ("伽马变换", "gamma"),
        ):
            button = QRadioButton(text, page)
            button.setProperty("mode", value)
            self.transform_mode_group.addButton(button)
            transform_modes.addWidget(button)
        self.transform_mode_group.buttons()[0].setChecked(True)
        self.transform_mode_group.buttonClicked.connect(
            lambda button: self.set_mode(button.property("mode"))
        )
        layout.addLayout(transform_modes)
        form = QFormLayout()
        layout.addLayout(form)
        self.a, self.a_row, self.a_slider = self._parameter_row("a", 0, 255, 0)
        self.b, self.b_row, self.b_slider = self._parameter_row("b", 0, 255, 255)
        self.c, self.c_row, self.c_slider = self._parameter_row("c", 0, 255, 0)
        self.d, self.d_row, self.d_slider = self._parameter_row("d", 0, 255, 255)
        self.gamma, self.gamma_row, self.gamma_slider = self._parameter_row(
            "gamma", 0.01, 10, 1, 2, 0.01
        )
        for row in (self.a_row, self.b_row, self.c_row, self.d_row, self.gamma_row):
            form.addRow(row)
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
        self._sync_transform_visibility()
        return page

    def _create_spatial_page(self) -> QWidget:
        page = QWidget(self)
        form = QFormLayout(page)
        self.spatial_filter_group = QButtonGroup(page)
        spatial_filters = QVBoxLayout()
        for text, value in (
            ("均值滤波", "mean"),
            ("高斯滤波", "gaussian"),
            ("拉普拉斯", "laplacian"),
            ("Prewitt X", "differenceG_x"),
            ("Prewitt Y", "differenceG_y"),
            ("Roberts", "roberts"),
            ("Sobel X", "sobel_x"),
            ("Sobel Y", "sobel_y"),
            ("中值滤波", "median"),
        ):
            button = QRadioButton(text, page)
            button.setProperty("filter_type", value)
            self.spatial_filter_group.addButton(button)
            spatial_filters.addWidget(button)
        self.spatial_filter_group.buttons()[0].setChecked(True)
        self.spatial_filter_group.buttonClicked.connect(self._emit_spatial)

        self.kernel_size = QSpinBox(page)
        self.kernel_size.setRange(3, 99)
        self.kernel_size.setSingleStep(2)
        self.kernel_size.setValue(3)
        self.kernel_size.valueChanged.connect(self._emit_spatial)

        self.median_ratio = QDoubleSpinBox(page)
        self.median_ratio.setRange(0, 100)
        self.median_ratio.setValue(50)
        self.median_ratio.setSuffix("%")
        self.median_ratio.valueChanged.connect(self._emit_spatial)
        form.addRow(self.tr("滤波器"), spatial_filters)
        form.addRow(self.tr("核大小"), self.kernel_size)
        form.addRow(self.tr("中值比例"), self.median_ratio)
        return page

    def _create_frequency_page(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        form = QFormLayout()
        self.frequency_filter_group = QButtonGroup(page)
        frequency_filters = QHBoxLayout()
        for text, value in (
            ("理想滤波", "ideal"),
            ("高斯滤波", "gaussian"),
            ("巴特沃斯滤波", "butterworth"),
        ):
            button = QRadioButton(text, page)
            button.setProperty("filter_type", value)
            self.frequency_filter_group.addButton(button)
            frequency_filters.addWidget(button)
        self.frequency_filter_group.buttons()[0].setChecked(True)
        self.frequency_cutoff = QDoubleSpinBox(page)
        self.frequency_cutoff.setRange(1, 10000)
        self.frequency_cutoff.setValue(30)
        self.frequency_cutoff.setSingleStep(1)
        form.addRow(self.tr("滤波器"), frequency_filters)
        form.addRow(self.tr("截止频率"), self.frequency_cutoff)
        layout.addLayout(form)
        self.fft_button = QPushButton(page)
        self.ifft_button = QPushButton(page)
        layout.addWidget(self.fft_button)
        layout.addWidget(self.ifft_button)
        layout.addStretch()
        self.frequency_filter_group.buttonClicked.connect(self._emit_frequency)
        self.frequency_cutoff.valueChanged.connect(self._emit_frequency)
        self.fft_button.clicked.connect(self.fftRequested)
        self.ifft_button.clicked.connect(self.ifftRequested)
        self._set_frequency_controls_enabled(False)
        return page

    def retranslateUi(self) -> None:
        self.setTitle(self.tr("处理参数"))
        for label, text in zip(
            (self.a_row, self.b_row, self.c_row, self.d_row, self.gamma_row),
            ("a", "b", "c", "d", "gamma"),
        ):
            label.findChild(QLabel).setText(self.tr(text))
        self.reset.setText(self.tr("重置参数"))
        self.fft_button.setText(self.tr("傅里叶变换"))
        self.ifft_button.setText(self.tr("应用逆变换"))

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

    def _emit_transform(self) -> None:
        self.changed.emit(self.transform_params())

    def _emit_spatial(self, *_args) -> None:
        self.spatialChanged.emit(self.spatial_params())

    def _emit_frequency(self, *_args) -> None:
        self.frequencyChanged.emit(self.frequency_params())

    def _sync_transform_visibility(self) -> None:
        mode = self.transform_mode()
        self.a_row.setVisible(mode == "linear")
        self.b_row.setVisible(mode == "linear")
        self.c_row.setVisible(mode in ("linear", "log", "gamma"))
        self.d_row.setVisible(mode == "linear")
        self.gamma_row.setVisible(mode == "gamma")

    def _set_frequency_controls_enabled(self, enabled: bool) -> None:
        for button in self.frequency_filter_group.buttons():
            button.setEnabled(enabled)
        self.frequency_cutoff.setEnabled(enabled)
        self.ifft_button.setEnabled(enabled)

    def set_frequency_controls_enabled(self, enabled: bool) -> None:
        self._set_frequency_controls_enabled(enabled)

    def _reset(self) -> None:
        operation = self.operation_name()
        if operation == "gray":
            self.set_mode("linear")
            self.a.setValue(0)
            self.b.setValue(255)
            self.c.setValue(0)
            self.d.setValue(255)
            self.gamma.setValue(1)
            self._schedule()
        elif operation == "spatial":
            self.spatial_filter_group.buttons()[0].setChecked(True)
            self.kernel_size.setValue(3)
            self.median_ratio.setValue(50)
            self._emit_spatial()
        else:
            self.frequency_filter_group.buttons()[0].setChecked(True)
            self.frequency_cutoff.setValue(30)
            self._emit_frequency()

    def transform_params(self) -> TransformParams:
        return TransformParams(
            mode=self._transform_mode,
            a=self.a.value(),
            b=self.b.value(),
            c=self.c.value(),
            d=self.d.value(),
            gamma=self.gamma.value(),
        )

    def spatial_params(self) -> SpatialParams:
        return SpatialParams(
            filter_type=self.spatial_filter_group.checkedButton().property("filter_type"),
            kernel_size=self.kernel_size.value(),
            median_ratio=self.median_ratio.value(),
        )

    def frequency_params(self) -> FrequencyParams:
        return FrequencyParams(
            filter_type=self.frequency_filter_group.checkedButton().property("filter_type"),
            cutoff=self.frequency_cutoff.value(),
        )

    def operation_name(self) -> str:
        return ("gray", "spatial", "frequency")[self.pages.currentIndex()]

    def set_operation(self, operation: str) -> None:
        if operation not in ("gray", "spatial", "frequency"):
            raise ValueError(f"Unsupported operation: {operation}")
        self.pages.setCurrentIndex(("gray", "spatial", "frequency").index(operation))
        self._set_frequency_controls_enabled(
            operation == "frequency"
            and self.frequency_filter_group.buttons()[0].isEnabled()
        )

    def transform_mode(self) -> str:
        return self._transform_mode

    def set_mode(self, mode: str) -> None:
        if mode not in ("linear", "log", "gamma"):
            raise ValueError(f"Unsupported transform mode: {mode}")
        self.set_operation("gray")
        self._transform_mode = mode
        for button in self.transform_mode_group.buttons():
            if button.property("mode") == mode:
                button.setChecked(True)
                break
        self._sync_transform_visibility()
        if mode in ("log", "gamma") and self.c.value() == 0:
            self.c.setValue(1)
        if mode == "linear":
            self.a_row.setVisible(True)
            self.b_row.setVisible(True)
            self.d_row.setVisible(True)
            self.gamma_row.setVisible(False)
        elif mode == "log":
            self.a_row.setVisible(False)
            self.b_row.setVisible(False)
            self.d_row.setVisible(False)
            self.gamma_row.setVisible(False)
        else:
            self.a_row.setVisible(False)
            self.b_row.setVisible(False)
            self.d_row.setVisible(False)
            self.gamma_row.setVisible(True)
        self._schedule()

    def params(self) -> TransformParams:
        return self.transform_params()
