from __future__ import annotations

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QButtonGroup,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
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

from core.app_state import FrequencyParams, GeometryParams, SpatialParams, TransformParams


class ParamPanel(QGroupBox):
    changed = Signal(object)
    spatialChanged = Signal(object)
    frequencyChanged = Signal(object)
    geometryChanged = Signal(object)
    fftRequested = Signal()
    ifftRequested = Signal()
    magnitudeIfftRequested = Signal()
    phaseIfftRequested = Signal()

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
        self.geometry_page = self._create_geometry_page()
        for page in (
            self.transform_page,
            self.spatial_page,
            self.frequency_page,
            self.geometry_page,
        ):
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
        transform_modes_container = QWidget(page)
        transform_modes_container.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        transform_modes = QGridLayout(transform_modes_container)
        transform_modes.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        for index, (text, value) in enumerate(
            (
                ("线性变换", "linear"),
                ("对数变换", "log"),
                ("伽马变换", "gamma"),
                ("直方图均衡化", "histogram_equalization"),
            )
        ):
            button = QRadioButton(text, page)
            button.setProperty("mode", value)
            self.transform_mode_group.addButton(button)
            row, column = divmod(index, 3)
            transform_modes.addWidget(button, row, column, Qt.AlignmentFlag.AlignLeft)
        self.transform_mode_group.buttons()[0].setChecked(True)
        self.transform_mode_group.buttonClicked.connect(
            lambda button: self.set_mode(button.property("mode"))
        )
        layout.addWidget(
            transform_modes_container,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
        )
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
            ("双边滤波", "bilateral"),
            ("拉普拉斯", "laplacian"),
            ("Prewitt X", "differenceG_x"),
            ("Prewitt Y", "differenceG_y"),
            ("Roberts", "roberts"),
            ("Sobel X", "sobel_x"),
            ("Sobel Y", "sobel_y"),
            ("中值滤波", "median"),
            ("USM 锐化", "usm"),
            ("高斯白噪声", "gaussian_noise"),
            ("椒盐噪声", "salt_pepper_noise"),
        ):
            button = QRadioButton(text, page)
            button.setProperty("filter_type", value)
            self.spatial_filter_group.addButton(button)
            spatial_filters.addWidget(button)
        self.spatial_filter_group.buttons()[0].setChecked(True)
        self.spatial_filter_group.buttonClicked.connect(self._emit_spatial)
        self.spatial_filter_group.buttonClicked.connect(
            lambda button: self._sync_spatial_visibility()
        )

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
        self.bilateral_sigma_color = QDoubleSpinBox(page)
        self.bilateral_sigma_color.setRange(0.1, 255)
        self.bilateral_sigma_color.setValue(75)
        self.bilateral_sigma_color.valueChanged.connect(self._emit_spatial)
        self.bilateral_sigma_space = QDoubleSpinBox(page)
        self.bilateral_sigma_space.setRange(0.1, 255)
        self.bilateral_sigma_space.setValue(75)
        self.bilateral_sigma_space.valueChanged.connect(self._emit_spatial)
        self.usm_amount = QDoubleSpinBox(page)
        self.usm_amount.setRange(0, 10)
        self.usm_amount.setValue(1.5)
        self.usm_amount.setSingleStep(0.1)
        self.usm_amount.valueChanged.connect(self._emit_spatial)
        self.gaussian_mean = QDoubleSpinBox(page)
        self.gaussian_mean.setRange(-255, 255)
        self.gaussian_mean.setValue(0)
        self.gaussian_mean.valueChanged.connect(self._emit_spatial)
        self.gaussian_std = QDoubleSpinBox(page)
        self.gaussian_std.setRange(0, 255)
        self.gaussian_std.setValue(20)
        self.gaussian_std.valueChanged.connect(self._emit_spatial)
        self.salt_pepper_amount = QDoubleSpinBox(page)
        self.salt_pepper_amount.setRange(0, 100)
        self.salt_pepper_amount.setValue(5)
        self.salt_pepper_amount.setSuffix("%")
        self.salt_pepper_amount.valueChanged.connect(self._emit_spatial)
        form.addRow(spatial_filters)
        form.addRow(self.tr("核大小"), self.kernel_size)
        self.kernel_size_label = form.labelForField(self.kernel_size)
        self.median_ratio_row = QWidget(page)
        median_ratio_layout = QHBoxLayout(self.median_ratio_row)
        median_ratio_layout.setContentsMargins(0, 0, 0, 0)
        median_ratio_layout.addWidget(self.median_ratio)
        form.addRow(self.tr("中值比例"), self.median_ratio_row)
        self.median_ratio_label = form.labelForField(self.median_ratio_row)
        self.bilateral_sigma_color_row = QWidget(page)
        bilateral_color_layout = QHBoxLayout(self.bilateral_sigma_color_row)
        bilateral_color_layout.setContentsMargins(0, 0, 0, 0)
        bilateral_color_layout.addWidget(self.bilateral_sigma_color)
        form.addRow(self.tr("颜色 sigma"), self.bilateral_sigma_color_row)
        self.bilateral_sigma_color_label = form.labelForField(self.bilateral_sigma_color_row)
        self.bilateral_sigma_space_row = QWidget(page)
        bilateral_space_layout = QHBoxLayout(self.bilateral_sigma_space_row)
        bilateral_space_layout.setContentsMargins(0, 0, 0, 0)
        bilateral_space_layout.addWidget(self.bilateral_sigma_space)
        form.addRow(self.tr("空间 sigma"), self.bilateral_sigma_space_row)
        self.bilateral_sigma_space_label = form.labelForField(self.bilateral_sigma_space_row)
        self.usm_amount_row = QWidget(page)
        usm_amount_layout = QHBoxLayout(self.usm_amount_row)
        usm_amount_layout.setContentsMargins(0, 0, 0, 0)
        usm_amount_layout.addWidget(self.usm_amount)
        form.addRow(self.tr("锐化强度"), self.usm_amount_row)
        self.usm_amount_label = form.labelForField(self.usm_amount_row)
        self.gaussian_mean_row = QWidget(page)
        gaussian_mean_layout = QHBoxLayout(self.gaussian_mean_row)
        gaussian_mean_layout.setContentsMargins(0, 0, 0, 0)
        gaussian_mean_layout.addWidget(self.gaussian_mean)
        form.addRow(self.tr("噪声均值"), self.gaussian_mean_row)
        self.gaussian_mean_label = form.labelForField(self.gaussian_mean_row)
        self.gaussian_std_row = QWidget(page)
        gaussian_std_layout = QHBoxLayout(self.gaussian_std_row)
        gaussian_std_layout.setContentsMargins(0, 0, 0, 0)
        gaussian_std_layout.addWidget(self.gaussian_std)
        form.addRow(self.tr("噪声标准差"), self.gaussian_std_row)
        self.gaussian_std_label = form.labelForField(self.gaussian_std_row)
        self.salt_pepper_amount_row = QWidget(page)
        salt_pepper_layout = QHBoxLayout(self.salt_pepper_amount_row)
        salt_pepper_layout.setContentsMargins(0, 0, 0, 0)
        salt_pepper_layout.addWidget(self.salt_pepper_amount)
        form.addRow(self.tr("噪声比例"), self.salt_pepper_amount_row)
        self.salt_pepper_amount_label = form.labelForField(self.salt_pepper_amount_row)
        self._sync_spatial_visibility()
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
        form.addRow(frequency_filters)
        form.addRow(self.tr("截止频率"), self.frequency_cutoff)
        layout.addLayout(form)
        self.fft_button = QPushButton(page)
        self.ifft_button = QPushButton(page)
        self.magnitude_ifft_button = QPushButton(page)
        self.phase_ifft_button = QPushButton(page)
        layout.addWidget(self.fft_button)
        layout.addWidget(self.ifft_button)
        layout.addWidget(self.magnitude_ifft_button)
        layout.addWidget(self.phase_ifft_button)
        layout.addStretch()
        self.frequency_filter_group.buttonClicked.connect(self._emit_frequency)
        self.frequency_cutoff.valueChanged.connect(self._emit_frequency)
        self.fft_button.clicked.connect(self.fftRequested)
        self.ifft_button.clicked.connect(self.ifftRequested)
        self.magnitude_ifft_button.clicked.connect(self.magnitudeIfftRequested)
        self.phase_ifft_button.clicked.connect(self.phaseIfftRequested)
        self._set_frequency_controls_enabled(False)
        return page

    def _create_geometry_page(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        self.geometry_group = QButtonGroup(page)
        for text, value in (
            ("顺时针旋转 90°", "rotate_cw_90"),
            ("逆时针旋转 90°", "rotate_ccw_90"),
            ("旋转 180°", "rotate_180"),
            ("水平翻转", "flip_horizontal"),
            ("垂直翻转", "flip_vertical"),
        ):
            button = QRadioButton(text, page)
            button.setProperty("transform_type", value)
            self.geometry_group.addButton(button)
            layout.addWidget(button)
        self.geometry_group.buttons()[0].setChecked(True)
        self.geometry_group.buttonClicked.connect(self._emit_geometry)
        layout.addStretch()
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
        self.magnitude_ifft_button.setText(self.tr("仅幅度谱恢复"))
        self.phase_ifft_button.setText(self.tr("仅相位谱恢复"))

    def _emit_geometry(self, *_args) -> None:
        self.geometryChanged.emit(self.geometry_params())

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

    def _sync_spatial_visibility(self) -> None:
        selected = self.spatial_filter_group.checkedButton()
        filter_type = selected.property("filter_type") if selected is not None else None
        show_kernel_size = filter_type in ("mean", "gaussian", "bilateral", "median", "usm")
        self.kernel_size.setVisible(show_kernel_size)
        self.kernel_size_label.setVisible(show_kernel_size)
        self.median_ratio_row.setVisible(filter_type == "median")
        self.median_ratio_label.setVisible(filter_type == "median")
        self.bilateral_sigma_color_row.setVisible(filter_type == "bilateral")
        self.bilateral_sigma_color_label.setVisible(filter_type == "bilateral")
        self.bilateral_sigma_space_row.setVisible(filter_type == "bilateral")
        self.bilateral_sigma_space_label.setVisible(filter_type == "bilateral")
        self.usm_amount_row.setVisible(filter_type == "usm")
        self.usm_amount_label.setVisible(filter_type == "usm")
        self.gaussian_mean_label.setVisible(filter_type == "gaussian_noise")
        self.gaussian_std_label.setVisible(filter_type == "gaussian_noise")
        self.salt_pepper_amount_label.setVisible(filter_type == "salt_pepper_noise")
        self.gaussian_mean_row.setVisible(filter_type == "gaussian_noise")
        self.gaussian_std_row.setVisible(filter_type == "gaussian_noise")
        self.salt_pepper_amount_row.setVisible(filter_type == "salt_pepper_noise")

    def _set_frequency_controls_enabled(self, enabled: bool) -> None:
        for button in self.frequency_filter_group.buttons():
            button.setEnabled(enabled)
        self.frequency_cutoff.setEnabled(enabled)
        self.ifft_button.setEnabled(enabled)
        self.magnitude_ifft_button.setEnabled(enabled)
        self.phase_ifft_button.setEnabled(enabled)

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
            self.bilateral_sigma_color.setValue(75)
            self.bilateral_sigma_space.setValue(75)
            self.usm_amount.setValue(1.5)
            self.gaussian_mean.setValue(0)
            self.gaussian_std.setValue(20)
            self.salt_pepper_amount.setValue(5)
            self._sync_spatial_visibility()
            self._emit_spatial()
        elif operation == "frequency":
            self.frequency_filter_group.buttons()[0].setChecked(True)
            self.frequency_cutoff.setValue(30)
            self._emit_frequency()
        else:
            self.geometry_group.buttons()[0].setChecked(True)
            self._emit_geometry()

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
            bilateral_sigma_color=self.bilateral_sigma_color.value(),
            bilateral_sigma_space=self.bilateral_sigma_space.value(),
            usm_amount=self.usm_amount.value(),
            gaussian_mean=self.gaussian_mean.value(),
            gaussian_std=self.gaussian_std.value(),
            salt_pepper_amount=self.salt_pepper_amount.value(),
        )

    def frequency_params(self) -> FrequencyParams:
        return FrequencyParams(
            filter_type=self.frequency_filter_group.checkedButton().property("filter_type"),
            cutoff=self.frequency_cutoff.value(),
        )

    def geometry_params(self) -> GeometryParams:
        return GeometryParams(
            transform_type=self.geometry_group.checkedButton().property("transform_type")
        )

    def operation_name(self) -> str:
        return ("gray", "spatial", "frequency", "geometry")[self.pages.currentIndex()]

    def set_operation(self, operation: str) -> None:
        if operation not in ("gray", "spatial", "frequency", "geometry"):
            raise ValueError(f"Unsupported operation: {operation}")
        self.pages.setCurrentIndex(
            ("gray", "spatial", "frequency", "geometry").index(operation)
        )
        self._set_frequency_controls_enabled(
            operation == "frequency"
            and self.frequency_filter_group.buttons()[0].isEnabled()
        )

    def transform_mode(self) -> str:
        return self._transform_mode

    def set_mode(self, mode: str) -> None:
        if mode not in ("linear", "log", "gamma", "histogram_equalization"):
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
        elif mode == "gamma":
            self.a_row.setVisible(False)
            self.b_row.setVisible(False)
            self.d_row.setVisible(False)
            self.gamma_row.setVisible(True)
        else:
            self.a_row.setVisible(False)
            self.b_row.setVisible(False)
            self.c_row.setVisible(False)
            self.d_row.setVisible(False)
            self.gamma_row.setVisible(False)
        self._schedule()

    def params(self) -> TransformParams:
        return self.transform_params()
