from __future__ import annotations

import numpy as np
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from core.app_state import (
    BinarizationParams,
    EdgeParams,
    FrequencyParams,
    HoughParams,
    GeometryParams,
    RestorationParams,
    SpatialParams,
    TransformParams,
)


class ParamPanel(QGroupBox):
    changed = Signal(object)
    noiseChanged = Signal(object)
    spatialChanged = Signal(object)
    frequencyChanged = Signal(object)
    geometryChanged = Signal(object)
    binarizationChanged = Signal(object)
    edgeChanged = Signal(object)
    houghChanged = Signal(object)
    restorationChanged = Signal(object)
    blindDeconvolutionRequested = Signal()
    wienerFilterRequested = Signal()
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
        self.noise_page = self._create_noise_page()
        self.spatial_page = self._create_spatial_page()
        self.frequency_page = self._create_frequency_page()
        self.geometry_page = self._create_geometry_page()
        self.restoration_page = self._create_restoration_page()
        self.binarization_page = self._create_binarization_page()
        self.edge_page = self._create_edge_page()
        self.hough_page = self._create_hough_page()
        for page in (
            self.transform_page,
            self.noise_page,
            self.spatial_page,
            self.frequency_page,
            self.restoration_page,
            self.geometry_page,
            self.binarization_page,
            self.edge_page,
            self.hough_page,
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
        self._spatial_kernel_sizes = {"mean": 3, "adaptive_median": 7}
        self._active_spatial_filter_type = "mean"
        spatial_filters = QVBoxLayout()
        for text, value in (
            ("算数均值滤波", "mean"),
            ("几何均值滤波", "geometric_mean"),
            ("高斯滤波", "gaussian"),
            ("双边滤波", "bilateral"),
            ("拉普拉斯", "laplacian"),
            ("Prewitt X", "differenceG_x"),
            ("Prewitt Y", "differenceG_y"),
            ("Roberts", "roberts"),
            ("Sobel X", "sobel_x"),
            ("Sobel Y", "sobel_y"),
            ("中值滤波", "median"),
            ("自适应中值滤波", "adaptive_median"),
            ("USM 锐化", "usm"),
        ):
            button = QRadioButton(text, page)
            button.setProperty("filter_type", value)
            self.spatial_filter_group.addButton(button)
            spatial_filters.addWidget(button)
        self.spatial_filter_group.buttons()[0].setChecked(True)
        self.spatial_filter_group.buttonClicked.connect(self._emit_spatial)
        self.spatial_filter_group.buttonClicked.connect(
            self._on_spatial_filter_changed
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
        self._sync_spatial_visibility()
        return page

    def _create_noise_page(self) -> QWidget:
        page = QWidget(self)
        form = QFormLayout(page)
        self.noise_group = QButtonGroup(page)
        noise_options = QVBoxLayout()
        for text, value in (
            ("高斯噪声", "gaussian_noise"),
            ("椒盐噪声", "salt_pepper_noise"),
            ("泊松噪声", "poisson_noise"),
            ("斑点噪声", "speckle_noise"),
            ("均匀噪声", "uniform_noise"),
            ("周期噪声", "periodic_noise"),
        ):
            button = QRadioButton(text, page)
            button.setProperty("filter_type", value)
            self.noise_group.addButton(button)
            noise_options.addWidget(button)
        self.noise_group.buttons()[0].setChecked(True)
        self.noise_group.buttonClicked.connect(self._emit_noise)
        self.noise_group.buttonClicked.connect(
            lambda button: self._sync_noise_visibility()
        )
        self.gaussian_mean = QDoubleSpinBox(page)
        self.gaussian_mean.setRange(-255, 255)
        self.gaussian_mean.setValue(0)
        self.gaussian_mean.valueChanged.connect(self._emit_noise)
        self.gaussian_std = QDoubleSpinBox(page)
        self.gaussian_std.setRange(0, 255)
        self.gaussian_std.setValue(20)
        self.gaussian_std.valueChanged.connect(self._emit_noise)
        self.salt_pepper_amount = QDoubleSpinBox(page)
        self.salt_pepper_amount.setRange(0, 100)
        self.salt_pepper_amount.setValue(5)
        self.salt_pepper_amount.setSuffix("%")
        self.salt_pepper_amount.valueChanged.connect(self._emit_noise)
        self.speckle_mean = QDoubleSpinBox(page)
        self.speckle_mean.setRange(-1, 1)
        self.speckle_mean.setDecimals(3)
        self.speckle_mean.setValue(0)
        self.speckle_mean.valueChanged.connect(self._emit_noise)
        self.speckle_std = QDoubleSpinBox(page)
        self.speckle_std.setRange(0, 1)
        self.speckle_std.setDecimals(3)
        self.speckle_std.setValue(0.1)
        self.speckle_std.valueChanged.connect(self._emit_noise)
        self.uniform_low = QDoubleSpinBox(page)
        self.uniform_low.setRange(-255, 255)
        self.uniform_low.setValue(-20)
        self.uniform_low.valueChanged.connect(self._emit_noise)
        self.uniform_high = QDoubleSpinBox(page)
        self.uniform_high.setRange(-255, 255)
        self.uniform_high.setValue(20)
        self.uniform_high.valueChanged.connect(self._emit_noise)
        self.periodic_angle = QDoubleSpinBox(page)
        self.periodic_angle.setRange(-6.283185, 6.283185)
        self.periodic_angle.setDecimals(4)
        self.periodic_angle.setValue(0)
        self.periodic_angle.valueChanged.connect(self._emit_noise)
        self.periodic_freq = QDoubleSpinBox(page)
        self.periodic_freq.setRange(0, 10)
        self.periodic_freq.setDecimals(4)
        self.periodic_freq.setValue(0.05)
        self.periodic_freq.valueChanged.connect(self._emit_noise)
        self.periodic_amplitude = QDoubleSpinBox(page)
        self.periodic_amplitude.setRange(0, 255)
        self.periodic_amplitude.setValue(20)
        self.periodic_amplitude.valueChanged.connect(self._emit_noise)
        self.periodic_phase = QDoubleSpinBox(page)
        self.periodic_phase.setRange(-6.283185, 6.283185)
        self.periodic_phase.setDecimals(4)
        self.periodic_phase.setValue(0)
        self.periodic_phase.valueChanged.connect(self._emit_noise)
        self.seed = QSpinBox(page)
        self.seed.setRange(0, 2147483647)
        self.seed.setValue(0)
        self.seed.valueChanged.connect(self._emit_noise)
        self.seed_row = QWidget(page)
        seed_layout = QHBoxLayout(self.seed_row)
        seed_layout.setContentsMargins(0, 0, 0, 0)
        seed_label = QLabel(self.tr("随机种子"), self.seed_row)
        seed_label.setFixedWidth(80)
        seed_layout.addWidget(seed_label)
        seed_layout.addWidget(self.seed)
        form.addRow(self.seed_row)
        form.addRow(noise_options)
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
        self.speckle_mean_row = QWidget(page)
        speckle_mean_layout = QHBoxLayout(self.speckle_mean_row)
        speckle_mean_layout.setContentsMargins(0, 0, 0, 0)
        speckle_mean_layout.addWidget(self.speckle_mean)
        form.addRow(self.tr("斑点均值"), self.speckle_mean_row)
        self.speckle_mean_label = form.labelForField(self.speckle_mean_row)
        self.speckle_std_row = QWidget(page)
        speckle_std_layout = QHBoxLayout(self.speckle_std_row)
        speckle_std_layout.setContentsMargins(0, 0, 0, 0)
        speckle_std_layout.addWidget(self.speckle_std)
        form.addRow(self.tr("斑点标准差"), self.speckle_std_row)
        self.speckle_std_label = form.labelForField(self.speckle_std_row)
        self.uniform_low_row = QWidget(page)
        uniform_low_layout = QHBoxLayout(self.uniform_low_row)
        uniform_low_layout.setContentsMargins(0, 0, 0, 0)
        uniform_low_layout.addWidget(self.uniform_low)
        form.addRow(self.tr("均匀噪声下限"), self.uniform_low_row)
        self.uniform_low_label = form.labelForField(self.uniform_low_row)
        self.uniform_high_row = QWidget(page)
        uniform_high_layout = QHBoxLayout(self.uniform_high_row)
        uniform_high_layout.setContentsMargins(0, 0, 0, 0)
        uniform_high_layout.addWidget(self.uniform_high)
        form.addRow(self.tr("均匀噪声上限"), self.uniform_high_row)
        self.uniform_high_label = form.labelForField(self.uniform_high_row)
        self.periodic_angle_row = QWidget(page)
        periodic_angle_layout = QHBoxLayout(self.periodic_angle_row)
        periodic_angle_layout.setContentsMargins(0, 0, 0, 0)
        periodic_angle_layout.addWidget(self.periodic_angle)
        form.addRow(self.tr("周期角度"), self.periodic_angle_row)
        self.periodic_angle_label = form.labelForField(self.periodic_angle_row)
        self.periodic_freq_row = QWidget(page)
        periodic_freq_layout = QHBoxLayout(self.periodic_freq_row)
        periodic_freq_layout.setContentsMargins(0, 0, 0, 0)
        periodic_freq_layout.addWidget(self.periodic_freq)
        form.addRow(self.tr("周期频率"), self.periodic_freq_row)
        self.periodic_freq_label = form.labelForField(self.periodic_freq_row)
        self.periodic_amplitude_row = QWidget(page)
        periodic_amplitude_layout = QHBoxLayout(self.periodic_amplitude_row)
        periodic_amplitude_layout.setContentsMargins(0, 0, 0, 0)
        periodic_amplitude_layout.addWidget(self.periodic_amplitude)
        form.addRow(self.tr("周期强度"), self.periodic_amplitude_row)
        self.periodic_amplitude_label = form.labelForField(self.periodic_amplitude_row)
        self.periodic_phase_row = QWidget(page)
        periodic_phase_layout = QHBoxLayout(self.periodic_phase_row)
        periodic_phase_layout.setContentsMargins(0, 0, 0, 0)
        periodic_phase_layout.addWidget(self.periodic_phase)
        form.addRow(self.tr("周期相位"), self.periodic_phase_row)
        self.periodic_phase_label = form.labelForField(self.periodic_phase_row)
        self._sync_noise_visibility()
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
        self.frequency_mode_group = QButtonGroup(page)
        frequency_modes = QHBoxLayout()
        for text, value in (
            ("低通", "lowpass"),
            ("高通", "highpass"),
            ("带通", "bandpass"),
            ("带阻", "bandreject"),
            ("陷波阻", "notchreject"),
            ("陷波通", "notchpass"),
        ):
            button = QRadioButton(text, page)
            button.setProperty("frequency_mode", value)
            self.frequency_mode_group.addButton(button)
            frequency_modes.addWidget(button)
        self.frequency_mode_group.buttons()[0].setChecked(True)
        form.addRow(frequency_modes)
        self.frequency_cutoff = QDoubleSpinBox(page)
        self.frequency_cutoff.setRange(1, 10000)
        self.frequency_cutoff.setValue(30)
        self.frequency_cutoff.setSingleStep(1)
        form.addRow(frequency_filters)
        self.frequency_pass_group = QButtonGroup(page)
        pass_filters = QHBoxLayout()
        for text, value in (("低通", "lowpass"), ("高通", "highpass")):
            button = QRadioButton(text, page)
            button.setProperty("pass_type", value)
            self.frequency_pass_group.addButton(button)
            pass_filters.addWidget(button)
        self.frequency_pass_group.buttons()[0].setChecked(True)
        form.addRow(pass_filters)
        form.addRow(self.tr("截止频率"), self.frequency_cutoff)
        self.frequency_band_width = QDoubleSpinBox(page)
        self.frequency_band_width.setRange(1, 10000)
        self.frequency_band_width.setValue(20)
        form.addRow(self.tr("带宽"), self.frequency_band_width)
        self.frequency_order = QSpinBox(page)
        self.frequency_order.setRange(1, 10)
        self.frequency_order.setValue(2)
        form.addRow(self.tr("巴特沃斯阶数"), self.frequency_order)
        self.frequency_order_label = form.labelForField(self.frequency_order)
        self.frequency_band_width_label = form.labelForField(self.frequency_band_width)
        self.frequency_notch_radius = QDoubleSpinBox(page)
        self.frequency_notch_radius.setRange(0, 1000)
        self.frequency_notch_radius.setValue(8)
        form.addRow(self.tr("陷波半径"), self.frequency_notch_radius)
        self.frequency_notch_radius_label = form.labelForField(self.frequency_notch_radius)
        notch_row = QWidget(page)
        notch_layout = QHBoxLayout(notch_row)
        notch_layout.setContentsMargins(0, 0, 0, 0)
        self.frequency_notch_x = QSpinBox(page)
        self.frequency_notch_x.setRange(-10000, 10000)
        self.frequency_notch_y = QSpinBox(page)
        self.frequency_notch_y.setRange(-10000, 10000)
        self.frequency_add_notch = QPushButton("添加", page)
        self.frequency_remove_notch = QPushButton("删除选中", page)
        self.frequency_clear_notches = QPushButton("清空", page)
        self.frequency_notch_list = QListWidget(page)
        notch_layout.addWidget(QLabel("x", page))
        notch_layout.addWidget(self.frequency_notch_x)
        notch_layout.addWidget(QLabel("y", page))
        notch_layout.addWidget(self.frequency_notch_y)
        notch_layout.addWidget(self.frequency_add_notch)
        notch_layout.addWidget(self.frequency_remove_notch)
        notch_layout.addWidget(self.frequency_clear_notches)
        form.addRow(self.tr("陷波点"), notch_row)
        form.addRow(self.frequency_notch_list)
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
        self.frequency_pass_group.buttonClicked.connect(self._emit_frequency)
        self.frequency_cutoff.valueChanged.connect(self._emit_frequency)
        self.frequency_band_width.valueChanged.connect(self._emit_frequency)
        self.frequency_order.valueChanged.connect(self._emit_frequency)
        self.frequency_notch_radius.valueChanged.connect(self._emit_frequency)
        self.frequency_mode_group.buttonClicked.connect(self._emit_frequency)
        self.frequency_mode_group.buttonClicked.connect(
            lambda button: self._sync_frequency_visibility()
        )
        self.frequency_add_notch.clicked.connect(self._add_notch_point)
        self.frequency_remove_notch.clicked.connect(self._remove_notch_point)
        self.frequency_clear_notches.clicked.connect(self._clear_notch_points)
        self.frequency_filter_group.buttonClicked.connect(
            lambda button: self._sync_frequency_visibility()
        )
        self._sync_frequency_visibility()
        self.fft_button.clicked.connect(self.fftRequested)
        self.ifft_button.clicked.connect(self.ifftRequested)
        self.magnitude_ifft_button.clicked.connect(self.magnitudeIfftRequested)
        self.phase_ifft_button.clicked.connect(self.phaseIfftRequested)
        self._set_frequency_controls_enabled(False)
        return page

    def _create_restoration_page(self) -> QWidget:
        page = QWidget(self)
        form = QFormLayout(page)
        self.restoration_mode_group = QButtonGroup(page)
        mode_layout = QHBoxLayout()
        for text, value in (
            ("模糊退化", "degradation"),
            ("维纳滤波", "wiener"),
            ("盲去卷积", "blind_deconvolution"),
        ):
            button = QRadioButton(text, page)
            button.setProperty("restoration_mode", value)
            self.restoration_mode_group.addButton(button)
            mode_layout.addWidget(button)
        self.restoration_mode_group.buttons()[0].setChecked(True)
        form.addRow(mode_layout)
        self.restoration_mode_group.buttonClicked.connect(self._sync_restoration_mode)
        self.restoration_mode_group.buttonClicked.connect(self._emit_restoration)
        self.restoration_blind_notice = QLabel(self.tr("盲去卷积参数将在此处设置"), page)
        self.restoration_blind_notice.setWordWrap(True)
        form.addRow(self.restoration_blind_notice)
        self.blind_iterations = QSpinBox(page)
        self.blind_iterations.setRange(1, 200)
        self.blind_iterations.setValue(50)
        form.addRow(self.tr("盲去卷积迭代次数"), self.blind_iterations)
        self.blind_iterations_label = form.labelForField(self.blind_iterations)
        self.blind_update_interval = QSpinBox(page)
        self.blind_update_interval.setRange(1, 50)
        self.blind_update_interval.setValue(2)
        form.addRow(self.tr("PSF 更新间隔"), self.blind_update_interval)
        self.blind_update_interval_label = form.labelForField(
            self.blind_update_interval
        )
        self.blind_epsilon = QDoubleSpinBox(page)
        self.blind_epsilon.setRange(1e-6, 1)
        self.blind_epsilon.setDecimals(12)
        self.blind_epsilon.setValue(1e-6)
        form.addRow(self.tr("稳定项"), self.blind_epsilon)
        self.blind_epsilon_label = form.labelForField(self.blind_epsilon)
        self.blind_start_button = QPushButton(self.tr("开始处理"), page)
        self.blind_start_button.clicked.connect(self.blindDeconvolutionRequested)
        form.addRow(self.blind_start_button)
        self.wiener_noise_ratio = QDoubleSpinBox(page)
        self.wiener_noise_ratio.setRange(1e-6, 1.0)
        self.wiener_noise_ratio.setDecimals(6)
        self.wiener_noise_ratio.setSingleStep(0.001)
        self.wiener_noise_ratio.setValue(0.01)
        form.addRow(self.tr("维纳噪声比"), self.wiener_noise_ratio)
        self.wiener_noise_ratio_label = form.labelForField(self.wiener_noise_ratio)
        self.wiener_noise_ratio.valueChanged.connect(self._emit_restoration)
        self.wiener_start_button = QPushButton(self.tr("开始处理"), page)
        self.wiener_start_button.clicked.connect(self.wienerFilterRequested)
        form.addRow(self.wiener_start_button)
        self.blind_psf_title = QLabel(self.tr("当前 PSF 预览"), page)
        form.addRow(self.blind_psf_title)
        self.blind_psf_preview = QLabel(page)
        self.blind_psf_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.blind_psf_preview.setMinimumSize(180, 180)
        self.blind_psf_preview.setText(self.tr("等待处理"))
        form.addRow(self.blind_psf_preview)
        for widget in (
            self.blind_iterations,
            self.blind_update_interval,
            self.blind_epsilon,
        ):
            widget.valueChanged.connect(self._emit_restoration)
        self.restoration_type = QComboBox(page)
        for text, value in (
            ("运动模糊", "motion"),
            ("离焦模糊", "defocus"),
            ("高斯模糊", "gaussian"),
            ("大气湍流", "turbulence"),
            ("衍射极限", "diffraction"),
        ):
            self.restoration_type.addItem(text, value)
        form.addRow(self.tr("点扩散函数类型"), self.restoration_type)
        self.restoration_type_label = form.labelForField(self.restoration_type)

        def decimal(name, label, minimum, maximum, value, step=0.1):
            widget = QDoubleSpinBox(page)
            widget.setRange(minimum, maximum)
            widget.setValue(value)
            widget.setSingleStep(step)
            setattr(self, name, widget)
            form.addRow(self.tr(label), widget)
            setattr(self, name + "_label", form.labelForField(widget))
            widget.valueChanged.connect(self._emit_restoration)
            return widget

        self.restoration_size = QSpinBox(page)
        self.restoration_size.setRange(3, 255)
        self.restoration_size.setSingleStep(2)
        self.restoration_size.setValue(21)
        form.addRow(self.tr("PSF 大小"), self.restoration_size)
        self.restoration_size_label = form.labelForField(self.restoration_size)
        self.restoration_size.valueChanged.connect(self._emit_restoration)
        decimal("motion_length", "运动长度", 0.1, 255, 15)
        decimal("motion_angle", "运动角度", -180, 180, 0, 1)
        decimal("defocus_radius", "离焦半径", 0.1, 255, 5)
        decimal("gaussian_sigma", "高斯 sigma", 0.1, 255, 3)
        decimal("turbulence_strength", "湍流强度", 0.0001, 10, 0.0025, 0.0001)
        decimal("diffraction_wavelength", "波长 (nm)", 1, 10000, 550)
        decimal("diffraction_aperture", "孔径", 0.01, 1000, 2)
        self.restoration_preview = QLabel(page)
        self.restoration_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.restoration_preview.setMinimumSize(180, 180)
        self.restoration_preview.setText(self.tr("PSF 预览"))
        form.addRow(self.restoration_preview)
        self.restoration_type.currentIndexChanged.connect(self._sync_restoration_visibility)
        self.restoration_type.currentIndexChanged.connect(self._emit_restoration)
        self._sync_restoration_visibility()
        self._sync_restoration_mode()
        return page

    def _sync_restoration_mode(self, *_args) -> None:
        mode = self.restoration_mode_group.checkedButton().property("restoration_mode")
        degradation = mode == "degradation"
        wiener = mode == "wiener"
        blind = mode == "blind_deconvolution"
        self.restoration_blind_notice.setVisible(blind)
        for widget in (
            self.blind_iterations, self.blind_update_interval, self.blind_epsilon
        ):
            widget.setVisible(blind)
        for label in (
            self.blind_iterations_label,
            self.blind_update_interval_label,
            self.blind_epsilon_label,
        ):
            label.setVisible(blind)
        self.blind_start_button.setVisible(blind)
        self.blind_psf_title.setVisible(blind)
        self.blind_psf_preview.setVisible(blind)
        self.wiener_noise_ratio.setVisible(wiener)
        self.wiener_noise_ratio_label.setVisible(wiener)
        self.wiener_start_button.setVisible(wiener)
        for widget in (
            self.restoration_type, self.restoration_size, self.restoration_preview,
            self.motion_length, self.motion_angle, self.defocus_radius,
            self.gaussian_sigma, self.turbulence_strength,
            self.diffraction_wavelength, self.diffraction_aperture,
        ):
            widget.setVisible(degradation or wiener)
        self.restoration_size_label.setVisible(degradation or wiener)
        self.restoration_type_label.setVisible(degradation or wiener)
        for name in (
            "motion_length", "motion_angle", "defocus_radius", "gaussian_sigma",
            "turbulence_strength", "diffraction_wavelength", "diffraction_aperture",
        ):
            getattr(self, name + "_label").setVisible(
                (degradation or wiener) and getattr(self, name).isVisible()
            )
        if degradation or wiener:
            self._sync_restoration_visibility()

    def set_blind_processing(self, processing: bool) -> None:
        self.blind_start_button.setEnabled(not processing)
        self.blind_start_button.setText(
            self.tr("处理中...") if processing else self.tr("开始处理")
        )

    def set_restoration_preview(self, image) -> None:
        self._set_psf_preview(self.restoration_preview, image)

    def set_blind_psf_preview(self, image, step: int | None = None) -> None:
        if image is None:
            self.blind_psf_preview.clear()
            self.blind_psf_preview.setText(self.tr("等待处理"))
            return
        self._set_psf_preview(self.blind_psf_preview, image)
        if step is not None:
            self.blind_psf_title.setText(self.tr("第 {} 步 PSF 预览").format(step))

    @staticmethod
    def _set_psf_preview(target: QLabel, image) -> None:
        values = np.asarray(image, dtype=np.uint8)
        if values.ndim != 2 or values.size == 0:
            target.clear()
            return
        height, width = values.shape
        qimage = QImage(
            values.data,
            width,
            height,
            width,
            QImage.Format.Format_Grayscale8,
        ).copy()
        pixmap = QPixmap.fromImage(qimage)
        target.setPixmap(
            pixmap.scaled(
                target.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
        )

    def _create_geometry_page(self) -> QWidget:
        page = QWidget(self)
        page.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        page.setMinimumSize(0, 0)
        form = QFormLayout(page)
        self.geometry_group = QButtonGroup(page)
        for text, value in (
            ("顺时针旋转 90°", "rotate_cw_90"),
            ("逆时针旋转 90°", "rotate_ccw_90"),
            ("旋转 180°", "rotate_180"),
            ("水平翻转", "flip_horizontal"),
            ("垂直翻转", "flip_vertical"),
            ("缩放", "resize"),
            ("裁剪", "crop"),
            ("任意角度旋转", "rotate"),
            ("仿射变换", "affine"),
            ("透视变换", "perspective"),
        ):
            button = QRadioButton(text, page)
            button.setProperty("transform_type", value)
            self.geometry_group.addButton(button)
            form.addRow(button)
        self.geometry_group.buttons()[0].setChecked(True)
        self.geometry_group.buttonClicked.connect(self._emit_geometry)
        self.geometry_group.buttonClicked.connect(lambda button: self._sync_geometry_visibility())

        def combo(name, items):
            widget = QComboBox(page)
            for text, value in items:
                widget.addItem(text, value)
            widget.currentIndexChanged.connect(self._emit_geometry)
            setattr(self, name, widget)
            form.addRow(name, widget)
            setattr(self, name + "_label", form.labelForField(widget))
            return widget
        def integer(name, label, value, maximum=16384):
            widget = QSpinBox(page)
            widget.setRange(0, maximum)
            widget.setValue(value)
            widget.valueChanged.connect(self._emit_geometry)
            setattr(self, name, widget)
            form.addRow(label, widget)
            setattr(self, name + "_label", form.labelForField(widget))
            return widget
        def decimal(name, label, value, minimum=-10000, maximum=10000):
            widget = QDoubleSpinBox(page)
            widget.setRange(minimum, maximum)
            widget.setValue(value)
            widget.valueChanged.connect(self._emit_geometry)
            setattr(self, name, widget)
            form.addRow(label, widget)
            setattr(self, name + "_label", form.labelForField(widget))
            return widget

        integer("geometry_width", "宽度", 640)
        integer("geometry_height", "高度", 480)
        self.geometry_keep_aspect = QCheckBox(self.tr("保持宽高比"), page)
        self.geometry_keep_aspect.toggled.connect(self._emit_geometry)
        form.addRow(self.geometry_keep_aspect)
        combo("geometry_interpolation", (("最近邻", "nearest"), ("双线性", "linear"), ("双三次", "cubic"), ("面积", "area")))
        integer("geometry_x", "x", 0)
        integer("geometry_y", "y", 0)
        integer("geometry_crop_width", "裁剪宽度", 100)
        integer("geometry_crop_height", "裁剪高度", 100)
        decimal("geometry_angle", "角度", 0, -36000, 36000)
        decimal("geometry_center_x", "中心 x", 0)
        decimal("geometry_center_y", "中心 y", 0)
        decimal("geometry_scale", "缩放比例", 1, 0.01, 100)
        combo("geometry_border_mode", (("常量", "constant"), ("复制", "replicate"), ("反射", "reflect"), ("反射101", "reflect101")))
        combo("geometry_transform_interpolation", (("最近邻", "nearest"), ("双线性", "linear"), ("双三次", "cubic"), ("面积", "area")))
        self.geometry_points = []
        for prefix, title, count in (("src", "源点", 4), ("dst", "目标点", 4)):
            for index in range(count):
                x = decimal(f"geometry_{prefix}_{index}_x", f"{title}{index + 1} x", 0)
                y = decimal(f"geometry_{prefix}_{index}_y", f"{title}{index + 1} y", 0)
                x.setProperty("geometry_label", getattr(self, f"geometry_{prefix}_{index}_x_label"))
                y.setProperty("geometry_label", getattr(self, f"geometry_{prefix}_{index}_y_label"))
                self.geometry_points.extend((x, y))
        integer("geometry_output_width", "输出宽度", 0)
        integer("geometry_output_height", "输出高度", 0)
        self._sync_geometry_visibility()
        return page

    def _create_binarization_page(self) -> QWidget:
        page = QWidget(self)
        form = QFormLayout(page)
        self.binarization_group = QButtonGroup(page)
        methods = QVBoxLayout()
        for text, value in (
            ("手动全局阈值", "manual"),
            ("Otsu 自动阈值", "otsu"),
            ("均值自适应", "adaptive_mean"),
            ("高斯自适应", "adaptive_gaussian"),
        ):
            button = QRadioButton(text, page)
            button.setProperty("method", value)
            self.binarization_group.addButton(button)
            methods.addWidget(button)
        self.binarization_group.buttons()[0].setChecked(True)
        self.binarization_group.buttonClicked.connect(self._emit_binarization)
        self.binarization_group.buttonClicked.connect(
            lambda button: self._sync_binarization_visibility()
        )
        form.addRow(methods)

        self.threshold = QDoubleSpinBox(page)
        self.threshold.setRange(0, 255)
        self.threshold.setValue(128)
        self.threshold.valueChanged.connect(self._emit_binarization)
        form.addRow(self.tr("阈值"), self.threshold)
        self.threshold_label = form.labelForField(self.threshold)

        self.binarization_block_size = QSpinBox(page)
        self.binarization_block_size.setRange(3, 99)
        self.binarization_block_size.setSingleStep(2)
        self.binarization_block_size.setValue(11)
        self.binarization_block_size.valueChanged.connect(self._emit_binarization)
        form.addRow(self.tr("邻域大小"), self.binarization_block_size)
        self.binarization_block_label = form.labelForField(self.binarization_block_size)

        self.binarization_c = QDoubleSpinBox(page)
        self.binarization_c.setRange(-255, 255)
        self.binarization_c.setValue(2)
        self.binarization_c.valueChanged.connect(self._emit_binarization)
        form.addRow(self.tr("常数 C"), self.binarization_c)
        self.binarization_c_label = form.labelForField(self.binarization_c)
        self._sync_binarization_visibility()
        return page

    def _create_edge_page(self) -> QWidget:
        page = QWidget(self)
        form = QFormLayout(page)
        self.edge_method_group = QButtonGroup(page)
        methods = QVBoxLayout()
        for text, value in (
            ("Sobel", "sobel"),
            ("Laplacian", "laplacian"),
            ("Canny", "canny"),
        ):
            button = QRadioButton(text, page)
            button.setProperty("method", value)
            self.edge_method_group.addButton(button)
            methods.addWidget(button)
        self.edge_method_group.buttons()[0].setChecked(True)
        self.edge_method_group.buttonClicked.connect(self._emit_edge)
        self.edge_method_group.buttonClicked.connect(
            lambda button: self._sync_edge_visibility()
        )
        form.addRow(methods)

        self.edge_direction_group = QButtonGroup(page)
        direction_row = QWidget(page)
        directions = QHBoxLayout(direction_row)
        directions.setContentsMargins(0, 0, 0, 0)
        for text, value in (("X 方向", "x"), ("Y 方向", "y"), ("合并幅值", "magnitude")):
            button = QRadioButton(text, page)
            button.setProperty("direction", value)
            self.edge_direction_group.addButton(button)
            directions.addWidget(button)
        self.edge_direction_group.buttons()[-1].setChecked(True)
        self.edge_direction_group.buttonClicked.connect(self._emit_edge)
        form.addRow(direction_row)
        self.edge_direction_row = direction_row

        self.edge_low_threshold = QDoubleSpinBox(page)
        self.edge_low_threshold.setRange(0, 255)
        self.edge_low_threshold.setValue(50)
        self.edge_low_threshold.valueChanged.connect(self._emit_edge)
        form.addRow(self.tr("低阈值"), self.edge_low_threshold)
        self.edge_low_label = form.labelForField(self.edge_low_threshold)

        self.edge_high_threshold = QDoubleSpinBox(page)
        self.edge_high_threshold.setRange(1, 255)
        self.edge_high_threshold.setValue(150)
        self.edge_high_threshold.valueChanged.connect(self._emit_edge)
        form.addRow(self.tr("高阈值"), self.edge_high_threshold)
        self.edge_high_label = form.labelForField(self.edge_high_threshold)
        self._sync_edge_visibility()
        return page

    def _create_hough_page(self) -> QWidget:
        page = QWidget(self)
        form = QFormLayout(page)
        self.hough_mode_group = QButtonGroup(page)
        modes = QHBoxLayout()
        for text, value in (("直线检测", "lines"), ("圆检测", "circles")):
            button = QRadioButton(text, page)
            button.setProperty("hough_mode", value)
            self.hough_mode_group.addButton(button)
            modes.addWidget(button)
        self.hough_mode_group.buttons()[0].setChecked(True)
        self.hough_mode_group.buttonClicked.connect(self._emit_hough)
        self.hough_mode_group.buttonClicked.connect(lambda button: self._sync_hough_visibility())
        form.addRow(modes)

        self.hough_method_group = QButtonGroup(page)
        methods = QHBoxLayout()
        for text, value in (("标准", "HOUGH_STANDARD"), ("概率", "HOUGH_PROBABILISTIC")):
            button = QRadioButton(text, page)
            button.setProperty("hough_method", value)
            self.hough_method_group.addButton(button)
            methods.addWidget(button)
        self.hough_method_group.buttons()[0].setChecked(True)
        self.hough_method_group.buttonClicked.connect(self._emit_hough)
        form.addRow(self.tr("直线方法"), methods)

        def add_double(name, label, minimum, maximum, value, step=1.0):
            widget = QDoubleSpinBox(page)
            widget.setRange(minimum, maximum)
            widget.setSingleStep(step)
            widget.setValue(value)
            widget.valueChanged.connect(self._emit_hough)
            setattr(self, name, widget)
            form.addRow(self.tr(label), widget)
            setattr(self, name + "_label", form.labelForField(widget))

        def add_int(name, label, minimum, maximum, value):
            widget = QSpinBox(page)
            widget.setRange(minimum, maximum)
            widget.setValue(value)
            widget.valueChanged.connect(self._emit_hough)
            setattr(self, name, widget)
            form.addRow(self.tr(label), widget)
            setattr(self, name + "_label", form.labelForField(widget))

        add_double("hough_rho", "rho", 0.01, 100, 1)
        add_double("hough_theta", "theta (度)", 0.01, 180, 1)
        add_int("hough_threshold", "阈值", 1, 10000, 100)
        add_double("hough_min_line_length", "最小线长", 0, 10000, 50)
        add_double("hough_max_line_gap", "最大间隙", 0, 10000, 10)
        add_double("hough_dp", "dp", 1, 20, 1, 0.1)
        add_double("hough_min_dist", "最小圆距", 0.1, 10000, 20)
        add_double("hough_param1", "param1", 0.1, 10000, 100)
        add_double("hough_param2", "param2", 0.1, 10000, 30)
        add_int("hough_min_radius", "最小半径", 0, 10000, 0)
        add_int("hough_max_radius", "最大半径", 0, 10000, 0)
        add_int("hough_thickness", "线宽", 1, 50, 2)
        self.hough_color = []
        color_row = QHBoxLayout()
        for value in (0, 0, 255):
            spin = QSpinBox(page)
            spin.setRange(0, 255)
            spin.setValue(value)
            spin.valueChanged.connect(self._emit_hough)
            self.hough_color.append(spin)
            color_row.addWidget(spin)
        form.addRow(self.tr("颜色 BGR"), color_row)
        self._sync_hough_visibility()
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
        self.magnitude_ifft_button.setText(self.tr("仅幅值谱恢复"))
        self.phase_ifft_button.setText(self.tr("仅相位谱恢复"))

    def _emit_geometry(self, *_args) -> None:
        self.geometryChanged.emit(self.geometry_params())

    def _emit_binarization(self, *_args) -> None:
        self.binarizationChanged.emit(self.binarization_params())

    def _emit_edge(self, *_args) -> None:
        self.edgeChanged.emit(self.edge_params())

    def _emit_hough(self, *_args) -> None:
        self.houghChanged.emit(self.hough_params())

    def _emit_restoration(self, *_args) -> None:
        self.restorationChanged.emit(self.restoration_params())

    def _sync_restoration_visibility(self, *_args) -> None:
        psf_type = self.restoration_type.currentData()
        groups = {
            "motion": ("motion_length", "motion_angle"),
            "defocus": ("defocus_radius",),
            "gaussian": ("gaussian_sigma",),
            "turbulence": ("turbulence_strength",),
            "diffraction": ("diffraction_wavelength", "diffraction_aperture"),
        }
        visible = set(groups[psf_type])
        for name in (
            "motion_length", "motion_angle", "defocus_radius", "gaussian_sigma",
            "turbulence_strength", "diffraction_wavelength", "diffraction_aperture",
        ):
            getattr(self, name).setVisible(name in visible)
            getattr(self, name + "_label").setVisible(name in visible)

    def _sync_hough_visibility(self) -> None:
        is_lines = self.hough_mode_group.checkedButton().property("hough_mode") == "lines"
        is_probability = (
            self.hough_method_group.checkedButton().property("hough_method")
            == "HOUGH_PROBABILISTIC"
        )
        for button in self.hough_method_group.buttons():
            button.setVisible(is_lines)
        for name in ("hough_rho", "hough_theta", "hough_threshold"):
            getattr(self, name).setVisible(True)
            getattr(self, name + "_label").setVisible(True)
        for name in ("hough_min_line_length", "hough_max_line_gap"):
            getattr(self, name).setVisible(is_lines and is_probability)
            getattr(self, name + "_label").setVisible(is_lines and is_probability)
        for name in (
            "hough_dp", "hough_min_dist", "hough_param1", "hough_param2",
            "hough_min_radius", "hough_max_radius",
        ):
            getattr(self, name).setVisible(not is_lines)
            getattr(self, name + "_label").setVisible(not is_lines)

    def _sync_binarization_visibility(self) -> None:
        method = self.binarization_group.checkedButton().property("method")
        is_manual = method == "manual"
        is_adaptive = method in ("adaptive_mean", "adaptive_gaussian")
        self.threshold.setVisible(is_manual)
        self.threshold_label.setVisible(is_manual)
        self.binarization_block_size.setVisible(is_adaptive)
        self.binarization_block_label.setVisible(is_adaptive)
        self.binarization_c.setVisible(is_adaptive)
        self.binarization_c_label.setVisible(is_adaptive)

    def _sync_edge_visibility(self) -> None:
        method = self.edge_method_group.checkedButton().property("method")
        is_sobel = method == "sobel"
        is_canny = method == "canny"
        self.edge_direction_row.setVisible(is_sobel)
        for widget in (
            self.edge_low_threshold,
            self.edge_low_label,
            self.edge_high_threshold,
            self.edge_high_label,
        ):
            widget.setVisible(is_canny)

    def _sync_geometry_visibility(self) -> None:
        mode = self.geometry_group.checkedButton().property("transform_type")
        for name in ("geometry_width", "geometry_height", "geometry_keep_aspect", "geometry_interpolation"):
            getattr(self, name).setVisible(mode == "resize")
            if hasattr(self, name + "_label"):
                getattr(self, name + "_label").setVisible(mode == "resize")
        for name in ("geometry_x", "geometry_y", "geometry_crop_width", "geometry_crop_height"):
            getattr(self, name).setVisible(mode == "crop")
            getattr(self, name + "_label").setVisible(mode == "crop")
        for name in ("geometry_angle", "geometry_center_x", "geometry_center_y", "geometry_scale", "geometry_border_mode", "geometry_transform_interpolation"):
            getattr(self, name).setVisible(mode == "rotate")
            getattr(self, name + "_label").setVisible(mode == "rotate")
        for index, widget in enumerate(self.geometry_points):
            widget.setVisible(mode in ("affine", "perspective") and (mode == "perspective" or index < 6))
            widget_label = self.geometry_points[index].property("geometry_label")
            if widget_label is not None:
                widget_label.setVisible(widget.isVisible())
        self.geometry_output_width.setVisible(mode in ("affine", "perspective"))
        self.geometry_output_height.setVisible(mode in ("affine", "perspective"))
        self.geometry_output_width_label.setVisible(mode in ("affine", "perspective"))
        self.geometry_output_height_label.setVisible(mode in ("affine", "perspective"))

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

    def _emit_noise(self, *_args) -> None:
        self.noiseChanged.emit(self.noise_params())

    def _emit_frequency(self, *_args) -> None:
        self.frequencyChanged.emit(self.frequency_params())

    def _sync_frequency_visibility(self) -> None:
        is_butterworth = (
            self.frequency_filter_group.checkedButton().property("filter_type")
            == "butterworth"
        )
        self.frequency_order.setVisible(is_butterworth)
        self.frequency_order_label.setVisible(is_butterworth)
        mode = self.frequency_mode_group.checkedButton().property("frequency_mode")
        is_band = mode in ("bandpass", "bandstop", "bandreject")
        is_notch = mode in ("notch", "notchreject", "notchpass")
        self.frequency_band_width.setVisible(is_band)
        self.frequency_band_width_label.setVisible(is_band)
        self.frequency_notch_radius.setVisible(is_notch)
        self.frequency_notch_radius_label.setVisible(is_notch)
        self.frequency_notch_x.setVisible(is_notch)
        self.frequency_notch_y.setVisible(is_notch)
        self.frequency_add_notch.setVisible(is_notch)
        self.frequency_remove_notch.setVisible(is_notch)
        self.frequency_clear_notches.setVisible(is_notch)
        self.frequency_notch_list.setVisible(is_notch)

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
        show_kernel_size = filter_type in (
            "mean",
            "geometric_mean",
            "gaussian",
            "bilateral",
            "median",
            "adaptive_median",
            "usm",
        )
        self.kernel_size.setVisible(show_kernel_size)
        self.kernel_size_label.setVisible(show_kernel_size)
        self.kernel_size_label.setText(
            self.tr("最大核大小")
            if filter_type == "adaptive_median"
            else self.tr("核大小")
        )
        self.median_ratio_row.setVisible(filter_type == "median")
        self.median_ratio_label.setVisible(filter_type == "median")
        self.bilateral_sigma_color_row.setVisible(filter_type == "bilateral")
        self.bilateral_sigma_color_label.setVisible(filter_type == "bilateral")
        self.bilateral_sigma_space_row.setVisible(filter_type == "bilateral")
        self.bilateral_sigma_space_label.setVisible(filter_type == "bilateral")
        self.usm_amount_row.setVisible(filter_type == "usm")
        self.usm_amount_label.setVisible(filter_type == "usm")

    def _on_spatial_filter_changed(self, button) -> None:
        previous = self._active_spatial_filter_type
        self._spatial_kernel_sizes[previous] = self.kernel_size.value()
        filter_type = button.property("filter_type")
        default_size = 7 if filter_type == "adaptive_median" else 3
        self.kernel_size.setValue(
            self._spatial_kernel_sizes.get(filter_type, default_size)
        )
        self._active_spatial_filter_type = filter_type
        self._sync_spatial_visibility()

    def _sync_noise_visibility(self) -> None:
        filter_type = self.noise_group.checkedButton().property("filter_type")
        self.gaussian_mean_label.setVisible(filter_type == "gaussian_noise")
        self.gaussian_std_label.setVisible(filter_type == "gaussian_noise")
        self.salt_pepper_amount_label.setVisible(filter_type == "salt_pepper_noise")
        self.gaussian_mean_row.setVisible(filter_type == "gaussian_noise")
        self.gaussian_std_row.setVisible(filter_type == "gaussian_noise")
        self.salt_pepper_amount_row.setVisible(filter_type == "salt_pepper_noise")
        self.speckle_mean_row.setVisible(filter_type == "speckle_noise")
        self.speckle_std_row.setVisible(filter_type == "speckle_noise")
        self.uniform_low_row.setVisible(filter_type == "uniform_noise")
        self.uniform_high_row.setVisible(filter_type == "uniform_noise")
        self.speckle_mean_label.setVisible(filter_type == "speckle_noise")
        self.speckle_std_label.setVisible(filter_type == "speckle_noise")
        self.uniform_low_label.setVisible(filter_type == "uniform_noise")
        self.uniform_high_label.setVisible(filter_type == "uniform_noise")
        periodic_visible = filter_type == "periodic_noise"
        for name in (
            "periodic_angle",
            "periodic_freq",
            "periodic_amplitude",
            "periodic_phase",
        ):
            getattr(self, name + "_row").setVisible(periodic_visible)
            getattr(self, name + "_label").setVisible(periodic_visible)

    def _set_frequency_controls_enabled(self, enabled: bool) -> None:
        for button in self.frequency_filter_group.buttons():
            button.setEnabled(enabled)
        self.frequency_cutoff.setEnabled(enabled)
        self.frequency_mode_group.setExclusive(True)
        for button in self.frequency_mode_group.buttons():
            button.setEnabled(enabled)
        self.frequency_band_width.setEnabled(enabled)
        self.frequency_notch_radius.setEnabled(enabled)
        self.frequency_notch_x.setEnabled(enabled)
        self.frequency_notch_y.setEnabled(enabled)
        self.frequency_add_notch.setEnabled(enabled)
        self.frequency_remove_notch.setEnabled(enabled)
        self.frequency_clear_notches.setEnabled(enabled)
        self.ifft_button.setEnabled(enabled)
        self.magnitude_ifft_button.setEnabled(enabled)
        self.phase_ifft_button.setEnabled(enabled)

    def set_hough_controls_enabled(self, enabled: bool) -> None:
        for widget in self.hough_page.findChildren(QWidget):
            widget.setEnabled(enabled)

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
            self._spatial_kernel_sizes = {"mean": 3, "adaptive_median": 7}
            self._active_spatial_filter_type = "mean"
            self.kernel_size.setValue(3)
            self.median_ratio.setValue(50)
            self.bilateral_sigma_color.setValue(75)
            self.bilateral_sigma_space.setValue(75)
            self.usm_amount.setValue(1.5)
            self._sync_spatial_visibility()
            self._emit_spatial()
        elif operation == "noise":
            self.noise_group.buttons()[0].setChecked(True)
            self.gaussian_mean.setValue(0)
            self.gaussian_std.setValue(20)
            self.salt_pepper_amount.setValue(5)
            self.speckle_mean.setValue(0)
            self.speckle_std.setValue(0.1)
            self.uniform_low.setValue(-20)
            self.uniform_high.setValue(20)
            self.periodic_angle.setValue(0)
            self.periodic_freq.setValue(0.05)
            self.periodic_amplitude.setValue(20)
            self.periodic_phase.setValue(0)
            self.seed.setValue(0)
            self._sync_noise_visibility()
            self._emit_noise()
        elif operation == "frequency":
            self.frequency_filter_group.buttons()[0].setChecked(True)
            self.frequency_pass_group.buttons()[0].setChecked(True)
            self.frequency_cutoff.setValue(30)
            self.frequency_order.setValue(2)
            self.frequency_band_width.setValue(20)
            self.frequency_notch_radius.setValue(8)
            self.frequency_notch_list.clear()
            self._sync_frequency_visibility()
            self._emit_frequency()
        elif operation == "geometry":
            self.geometry_group.buttons()[0].setChecked(True)
            self._emit_geometry()
        elif operation == "restoration":
            self.restoration_mode_group.buttons()[0].setChecked(True)
            self.restoration_type.setCurrentIndex(0)
            self.restoration_size.setValue(31)
            self.motion_length.setValue(15)
            self.motion_angle.setValue(0)
            self.defocus_radius.setValue(5)
            self.gaussian_sigma.setValue(3)
            self.turbulence_strength.setValue(0.0025)
            self.diffraction_wavelength.setValue(550)
            self.diffraction_aperture.setValue(2)
            self.blind_iterations.setValue(50)
            self.blind_update_interval.setValue(2)
            self.blind_epsilon.setValue(1e-6)
            self._sync_restoration_visibility()
            self._sync_restoration_mode()
            self._emit_restoration()
        else:
            operation = self.operation_name()
            if operation == "binarization":
                self.binarization_group.buttons()[0].setChecked(True)
                self.threshold.setValue(128)
                self.binarization_block_size.setValue(11)
                self.binarization_c.setValue(2)
                self._sync_binarization_visibility()
                self._emit_binarization()
            else:
                if operation == "edge":
                    self.edge_method_group.buttons()[0].setChecked(True)
                    self.edge_direction_group.buttons()[-1].setChecked(True)
                    self.edge_low_threshold.setValue(50)
                    self.edge_high_threshold.setValue(150)
                    self._sync_edge_visibility()
                    self._emit_edge()
                else:
                    self.hough_mode_group.buttons()[0].setChecked(True)
                    self.hough_method_group.buttons()[0].setChecked(True)
                    self._sync_hough_visibility()
                    self._emit_hough()

    def hough_params(self) -> HoughParams:
        return HoughParams(
            mode=self.hough_mode_group.checkedButton().property("hough_mode"),
            method=self.hough_method_group.checkedButton().property("hough_method"),
            rho=self.hough_rho.value(),
            theta=self.hough_theta.value(),
            threshold=self.hough_threshold.value(),
            min_line_length=self.hough_min_line_length.value(),
            max_line_gap=self.hough_max_line_gap.value(),
            dp=self.hough_dp.value(),
            min_dist=self.hough_min_dist.value(),
            param1=self.hough_param1.value(),
            param2=self.hough_param2.value(),
            min_radius=self.hough_min_radius.value(),
            max_radius=self.hough_max_radius.value(),
            color=tuple(spin.value() for spin in self.hough_color),
            thickness=self.hough_thickness.value(),
        )

    def restoration_params(self) -> RestorationParams:
        return RestorationParams(
            mode=self.restoration_mode_group.checkedButton().property("restoration_mode"),
            psf_type=self.restoration_type.currentData(),
            size=self.restoration_size.value(),
            motion_length=self.motion_length.value(),
            motion_angle=self.motion_angle.value(),
            defocus_radius=self.defocus_radius.value(),
            gaussian_sigma=self.gaussian_sigma.value(),
            turbulence_strength=self.turbulence_strength.value(),
            diffraction_wavelength=self.diffraction_wavelength.value(),
            diffraction_aperture=self.diffraction_aperture.value(),
            blind_iterations=self.blind_iterations.value(),
            blind_update_interval=self.blind_update_interval.value(),
            blind_epsilon=self.blind_epsilon.value(),
            wiener_noise_ratio=self.wiener_noise_ratio.value(),
        )

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
            gaussian_mean=0,
            gaussian_std=20,
            salt_pepper_amount=5,
            seed=0,
            speckle_mean=0,
            speckle_std=0.1,
            uniform_low=-20,
            uniform_high=20,
            periodic_angle=0,
            periodic_freq=0.05,
            periodic_amplitude=20,
            periodic_phase=0,
        )

    def noise_params(self) -> SpatialParams:
        return SpatialParams(
            filter_type=self.noise_group.checkedButton().property("filter_type"),
            kernel_size=self.kernel_size.value(),
            median_ratio=self.median_ratio.value(),
            bilateral_sigma_color=self.bilateral_sigma_color.value(),
            bilateral_sigma_space=self.bilateral_sigma_space.value(),
            usm_amount=self.usm_amount.value(),
            gaussian_mean=self.gaussian_mean.value(),
            gaussian_std=self.gaussian_std.value(),
            salt_pepper_amount=self.salt_pepper_amount.value(),
            seed=self.seed.value(),
            speckle_mean=self.speckle_mean.value(),
            speckle_std=self.speckle_std.value(),
            uniform_low=self.uniform_low.value(),
            uniform_high=self.uniform_high.value(),
            periodic_angle=self.periodic_angle.value(),
            periodic_freq=self.periodic_freq.value(),
            periodic_amplitude=self.periodic_amplitude.value(),
            periodic_phase=self.periodic_phase.value(),
        )

    def frequency_params(self) -> FrequencyParams:
        return FrequencyParams(
            filter_type=self.frequency_filter_group.checkedButton().property("filter_type"),
            cutoff=self.frequency_cutoff.value(),
            pass_type=self.frequency_pass_group.checkedButton().property("pass_type"),
            order=self.frequency_order.value(),
            mode=self.frequency_mode_group.checkedButton().property("frequency_mode"),
            band_width=self.frequency_band_width.value(),
            notch_radius=self.frequency_notch_radius.value(),
            notch_points=[
                tuple(item.data(Qt.ItemDataRole.UserRole))
                for item in (
                    self.frequency_notch_list.item(index)
                    for index in range(self.frequency_notch_list.count())
                )
            ],
        )

    def _add_notch_point(self) -> None:
        point = (self.frequency_notch_x.value(), self.frequency_notch_y.value())
        self.add_notch_point(point)

    def add_notch_point(self, point: tuple[float, float]) -> None:
        point = (
            float(point[0]),
            float(point[1]),
            float(self.frequency_notch_radius.value()),
        )
        existing_points = [
            item.data(Qt.ItemDataRole.UserRole)
            for item in (
                self.frequency_notch_list.item(i)
                for i in range(self.frequency_notch_list.count())
            )
        ]
        if not any(existing[:2] == point[:2] for existing in existing_points):
            if self.frequency_notch_list.count() >= 128:
                return
            item = QListWidgetItem(
                f"({point[0]}, {point[1]}), r={point[2]}"
            )
            item.setData(Qt.ItemDataRole.UserRole, point)
            self.frequency_notch_list.addItem(item)
            self._emit_frequency()

    def _remove_notch_point(self) -> None:
        row = self.frequency_notch_list.currentRow()
        if row >= 0:
            self.frequency_notch_list.takeItem(row)
            self._emit_frequency()

    def _clear_notch_points(self) -> None:
        self.frequency_notch_list.clear()
        self._emit_frequency()

    def geometry_params(self) -> GeometryParams:
        mode = self.geometry_group.checkedButton().property("transform_type")
        points = [
            (self.geometry_points[index].value(), self.geometry_points[index + 1].value())
            for index in range(0, 16, 2)
        ]
        return GeometryParams(
            transform_type=mode,
            width=self.geometry_width.value(),
            height=self.geometry_height.value(),
            keep_aspect=self.geometry_keep_aspect.isChecked(),
            interpolation=self.geometry_interpolation.currentData(),
            x=self.geometry_x.value(),
            y=self.geometry_y.value(),
            crop_width=self.geometry_crop_width.value(),
            crop_height=self.geometry_crop_height.value(),
            angle=self.geometry_angle.value(),
            center_x=self.geometry_center_x.value(),
            center_y=self.geometry_center_y.value(),
            scale=self.geometry_scale.value(),
            border_mode=self.geometry_border_mode.currentData(),
            rotation_interpolation=self.geometry_transform_interpolation.currentData(),
            src_points=tuple(points[:4]),
            dst_points=tuple(points[4:]),
            output_width=self.geometry_output_width.value(),
            output_height=self.geometry_output_height.value(),
        )

    def binarization_params(self) -> BinarizationParams:
        return BinarizationParams(
            method=self.binarization_group.checkedButton().property("method"),
            threshold=self.threshold.value(),
            block_size=self.binarization_block_size.value(),
            c=self.binarization_c.value(),
        )

    def edge_params(self) -> EdgeParams:
        return EdgeParams(
            method=self.edge_method_group.checkedButton().property("method"),
            direction=self.edge_direction_group.checkedButton().property("direction"),
            low_threshold=self.edge_low_threshold.value(),
            high_threshold=self.edge_high_threshold.value(),
        )

    def operation_name(self) -> str:
        return (
            "gray",
            "noise",
            "spatial",
            "frequency",
            "restoration",
            "geometry",
            "binarization",
            "edge",
            "hough",
        )[self.pages.currentIndex()]

    def set_operation(self, operation: str) -> None:
        if operation not in (
            "gray",
            "noise",
            "spatial",
            "frequency",
            "restoration",
            "geometry",
            "binarization",
            "edge",
            "hough",
        ):
            raise ValueError(f"Unsupported operation: {operation}")
        self.pages.setCurrentIndex(
            (
                "gray",
                "noise",
                "spatial",
                "frequency",
                "restoration",
                "geometry",
                "binarization",
                "edge",
                "hough",
            ).index(operation)
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
