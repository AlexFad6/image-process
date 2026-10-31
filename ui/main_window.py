from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QIcon
from PySide6.QtCore import QEvent, QObject, QSettings, QThread, QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QApplication,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QSizePolicy,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from core.app_state import AppState
from core.exporter import Exporter
from core.image_processor_library import LibraryImageProcessor
from core.image_processor_manual import ManualImageProcessor
from .attribute_panel import AttributePanel
from .image_panel import ImagePanel
from .param_panel import ParamPanel
from utils import install_language, resource_path


class ProcessingWorker(QObject):
    finished = Signal(object, int)
    failed = Signal(str, int)

    def __init__(self, operation: str, payload: tuple, generation: int) -> None:
        super().__init__()
        self.operation = operation
        self.payload = payload
        self.generation = generation

    def run(self) -> None:
        try:
            if self.operation == "spatial":
                processor, image, params = self.payload
                if params.filter_type == "gaussian_noise":
                    result = processor.gaussian_noise(
                        image, params.gaussian_mean, params.gaussian_std
                    )
                elif params.filter_type == "salt_pepper_noise":
                    result = processor.salt_pepper_noise(
                        image, params.salt_pepper_amount
                    )
                elif params.filter_type == "median":
                    result = processor.median_filter(
                        image, params.kernel_size, params.median_ratio
                    )
                elif params.filter_type == "bilateral":
                    result = processor.bilateral_filter(
                        image,
                        params.kernel_size,
                        params.bilateral_sigma_color,
                        params.bilateral_sigma_space,
                    )
                elif params.filter_type == "usm":
                    result = processor.unsharp_mask(
                        image, params.kernel_size, params.usm_amount
                    )
                else:
                    kernel_size = params.kernel_size
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    kernel = processor.gen_filter_kernel(
                        kernel_size, params.filter_type
                    )
                    result = processor.spatial_filter(image, kernel)
                self.finished.emit(result, self.generation)
            elif self.operation == "geometry":
                processor, image, params = self.payload
                result = processor.geometric_transform(image, params.transform_type)
                self.finished.emit(result, self.generation)
            elif self.operation == "binarization":
                processor, image, params = self.payload
                result = processor.binarize(
                    image, params.method, params.threshold, params.block_size, params.c
                )
                self.finished.emit(result, self.generation)
            elif self.operation == "edge":
                processor, image, params = self.payload
                result = processor.edge_detect(
                    image,
                    params.method,
                    params.direction,
                    params.low_threshold,
                    params.high_threshold,
                )
                self.finished.emit(result, self.generation)
            elif self.operation == "transform":
                processor, image, params = self.payload
                if params.mode == "linear":
                    result = processor.linear_transform(
                        image, params.a, params.b, params.c, params.d
                    )
                elif params.mode == "log":
                    result = processor.log_transform(image, params.c)
                elif params.mode == "gamma":
                    result = processor.gamma_transform(image, params.c, params.gamma)
                elif params.mode == "histogram_equalization":
                    result = processor.histogram_equalization(image)
                else:
                    raise ValueError(f"Unsupported transform mode: {params.mode}")
                self.finished.emit(result, self.generation)
            elif self.operation == "fft":
                image, processor = self.payload
                frequency_data = processor.compute_fft(image)
                result = (
                    frequency_data,
                    processor.compute_fft_display_spectrum(frequency_data),
                    processor.compute_fft_phase_spectrum(frequency_data),
                )
                self.finished.emit(result, self.generation)
            elif self.operation == "frequency":
                frequency_data, params, processor = self.payload
                filtered = processor.apply_frequency_filter(
                    frequency_data, params.filter_type, params.cutoff
                )
                result = (
                    filtered,
                    processor.compute_fft_display_spectrum(filtered),
                    processor.compute_fft_phase_spectrum(filtered),
                )
                self.finished.emit(result, self.generation)
            elif self.operation == "ifft":
                frequency_data, processor, component = self.payload
                self.finished.emit(
                    processor.compute_ifft(frequency_data, component),
                    self.generation,
                )
        except Exception as error:
            self.failed.emit(str(error), self.generation)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("image-process")
        self.resize(1200, 760)
        self.setWindowIcon(QIcon(str(resource_path("app_icon.png"))))

        self.state = AppState()
        self.display_mode = "original"
        self.show_phase_spectrum = False
        self.image_queue: list[tuple[np.ndarray, str]] = []
        self.queue_index = -1
        self._processing_generation = 0
        self._processing_jobs: dict[int, tuple[QThread, ProcessingWorker]] = {}

        self.images = ImagePanel()
        self.display_mode_timer = QTimer(self)
        self.display_mode_timer.setSingleShot(True)
        self.display_mode_timer.setInterval(200)
        self.display_mode_timer.timeout.connect(self.cycle_display_mode)
        self.params = ParamPanel()
        self._pending_spatial = self.params.spatial_params()
        self._pending_frequency = self.params.frequency_params()
        self.processor = LibraryImageProcessor
        self.attributes = AttributePanel()

        self._create_menu_bar()
        self._create_operation_toolbar()

        side = QWidget()
        side.setMinimumWidth(0)
        side.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(5, 5, 5, 5)
        self.right_splitter = QSplitter(Qt.Orientation.Vertical)
        self.right_splitter.setMinimumWidth(0)
        self.right_splitter.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding
        )
        self.params.setMinimumWidth(0)
        self.params.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding
        )
        self.attributes.setMinimumWidth(0)
        self.attributes.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding
        )
        self.right_splitter.addWidget(self.params)
        self.right_splitter.addWidget(self.attributes)
        self.right_splitter.setSizes([350, 150])
        self.right_splitter.setChildrenCollapsible(False)
        self.right_splitter.setHandleWidth(8)
        self.right_splitter.setStyleSheet(
            """
            QSplitter::handle {
                background-color: #d4d4d4;
            }
            QSplitter::handle:pressed {
                background-color: #a0a0a0;
            }
            """
        )
        side_layout.addWidget(self.right_splitter)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        left_layout.addWidget(self.images, 1)

        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        content_splitter.addWidget(left)
        content_splitter.addWidget(side)
        content_splitter.setSizes([700, 300])
        content_splitter.setStretchFactor(0, 1)
        content_splitter.setStretchFactor(1, 0)
        content_splitter.setChildrenCollapsible(False)
        content_splitter.setHandleWidth(8)
        content_splitter.setStyleSheet(
            """
            QSplitter::handle {
                background-color: #d4d4d4;
            }
            QSplitter::handle:pressed {
                background-color: #a0a0a0;
            }
            """
        )
        self.setCentralWidget(content_splitter)
        self.setStatusBar(QStatusBar())

        self.images.convertToGrayRequested.connect(self.convert_to_gray)
        self.images.displayModeRequested.connect(self.schedule_display_mode_switch)
        self.params.changed.connect(self.apply_params)
        self.params.spatialChanged.connect(self.apply_spatial_params)
        self.params.geometryChanged.connect(self.apply_geometry_params)
        self.params.binarizationChanged.connect(self.apply_binarization_params)
        self.params.edgeChanged.connect(self.apply_edge_params)
        self.params.fftRequested.connect(self.compute_fft)
        self.params.ifftRequested.connect(self.apply_ifft)
        self.params.magnitudeIfftRequested.connect(
            lambda: self.apply_ifft("magnitude")
        )
        self.params.phaseIfftRequested.connect(lambda: self.apply_ifft("phase"))
        self.params.frequencyChanged.connect(self.apply_frequency_filter)
        self.spatial_timer = QTimer(self)
        self.spatial_timer.setSingleShot(True)
        self.spatial_timer.setInterval(250)
        self.spatial_timer.timeout.connect(self._run_pending_spatial)
        self.frequency_timer = QTimer(self)
        self.frequency_timer.setSingleShot(True)
        self.frequency_timer.setInterval(250)
        self.frequency_timer.timeout.connect(self._run_pending_frequency)
        # 加载默认 cameraman.tif 图像
        self._load_default_image()
        QTimer.singleShot(0, self.images.reset_views)

    def _load_default_image(self) -> None:
        default_path = Path(__file__).resolve().parent.parent / "standard_test_images" / "cameraman.tif"
        image = Exporter.read_image(str(default_path))
        if image is None:
            self._update_ui_state()
            return

        self.image_queue.append((image, str(default_path)))
        self.queue_index = 0
        self._show_queue_image()
        # 两侧图像视图同步
        self._update_ui_state()
        if self.state.source is not None and self.state.source.ndim == 2:
            self.apply_params(self.params.params())
        
    def _create_menu_bar(self) -> None:
        file_menu = self.menuBar().addMenu("文件")
        self.file_menu = file_menu
        self.open_action = self._add_action(file_menu, "打开图像", self.open_image, "Ctrl+O")
        self.save_action = self._add_action(file_menu, "保存当前图像", self.save_image, "Ctrl+S")
        self.export_action = self._add_action(file_menu, "保存图像序列", self.export_sequence)
        file_menu.addSeparator()
        self.exit_action = self._add_action(file_menu, "退出", self.close, "Alt+F4")

        view_menu = self.menuBar().addMenu("视图")
        self.view_menu = view_menu
        sync_action = self._add_action(view_menu, "同步缩放", self.toggle_sync_zoom)
        self.sync_action = sync_action
        sync_action.setCheckable(True)
        sync_action.setChecked(True)
        self.reset_view_action = self._add_action(view_menu, "重置视图", self.images.reset_views)
 
        options_menu = self.menuBar().addMenu("选项")
        self.options_menu = options_menu
        self.phase_action = QAction("显示相位谱", options_menu)
        self.phase_action.toggled.connect(self.toggle_phase_spectrum)
        options_menu.addAction(self.phase_action)
        self.phase_action.setCheckable(True)
        self.phase_action.setChecked(False)
        language_menu = options_menu.addMenu("语言")
        self.language_menu = language_menu
        language_group = QActionGroup(language_menu)
        language_group.setExclusive(True)
        self.chinese_action = QAction("中文", language_menu)
        self.english_action = QAction("English", language_menu)
        for action in (self.chinese_action, self.english_action):
            action.setCheckable(True)
            language_group.addAction(action)
            language_menu.addAction(action)
        current_language = QSettings(
            "AlexFad", "image-process"
        ).value("language", "zh_CN")
        self.chinese_action.setChecked(current_language.startswith("zh"))
        self.english_action.setChecked(current_language.startswith("en"))
        self.chinese_action.triggered.connect(lambda: self.set_language("zh_CN"))
        self.english_action.triggered.connect(lambda: self.set_language("en_US"))

        implementation_menu = self.menuBar().addMenu("实现方式")
        self.implementation_menu = implementation_menu
        implementation_group = QActionGroup(self)
        implementation_group.setExclusive(True)
        self.library_action = self._add_action(implementation_menu, "库函数", self.use_library_backend)
        self.custom_action = self._add_action(implementation_menu, "自编函数", self.use_custom_backend)
        for action in (self.library_action, self.custom_action):
            action.setCheckable(True)
            implementation_group.addAction(action)
        self.library_action.setChecked(True)

        help_menu = self.menuBar().addMenu("帮助")
        self.help_menu = help_menu
        self.about_action = self._add_action(help_menu, "关于", self.show_about)

    @staticmethod
    def _add_action(menu, text: str, slot, shortcut: str | None = None) -> QAction:
        action = QAction(text, menu)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(slot)
        menu.addAction(action)
        return action

    def _create_operation_toolbar(self) -> None:
        toolbar = QToolBar("图像处理", self)
        self.processing_toolbar = toolbar
        toolbar.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

        operation_group = QActionGroup(self)
        operation_group.setExclusive(True)
        operations = (
            ("灰度变换", "gray"),
            ("空间滤波", "spatial"),
            ("频域滤波", "frequency"),
            ("几何变换", "geometry"),
            ("二值化", "binarization"),
            ("边缘检测", "edge"),
        )
        self.operation_actions: dict[str, QAction] = {}
        for text, operation in operations:
            action = QAction(self.tr(text), self)
            action.setCheckable(True)
            action.triggered.connect(
                lambda checked, value=operation: self.switch_operation(value)
            )
            operation_group.addAction(action)
            toolbar.addAction(action)
            self.operation_actions[operation] = action
        self.operation_actions["gray"].setChecked(True)
        self.retranslateUi()

    def switch_operation(self, operation: str) -> None:
        self._processing_generation += 1
        if operation != "frequency" and self.state.frequency_data is not None:
            self.state.clear_frequency_data()
            self.images.result.set_image(None, preserve_view=True)
            self.attributes.set_result(None)
            self._set_frequency_controls_enabled(False)
        self.params.set_operation(operation)
        if operation == "spatial":
            self.apply_spatial_params(self.params.spatial_params())
        elif operation == "geometry":
            self.apply_geometry_params(self.params.geometry_params())
        elif operation == "binarization":
            self.apply_binarization_params(self.params.binarization_params())
        elif operation == "edge":
            self.apply_edge_params(self.params.edge_params())

    def retranslateUi(self) -> None:
        self.setWindowTitle(self.tr("image-process"))
        self.file_menu.setTitle(self.tr("文件"))
        self.open_action.setText(self.tr("打开图像"))
        self.save_action.setText(self.tr("保存当前图像"))
        self.export_action.setText(self.tr("保存图像序列"))
        self.exit_action.setText(self.tr("退出"))
        self.view_menu.setTitle(self.tr("视图"))
        self.sync_action.setText(self.tr("同步缩放"))
        self.reset_view_action.setText(self.tr("重置视图"))
        self.options_menu.setTitle(self.tr("选项"))
        self.phase_action.setText(self.tr("显示相位谱"))
        self.language_menu.setTitle(self.tr("语言"))
        self.chinese_action.setText("中文")
        self.english_action.setText("English")
        self.implementation_menu.setTitle(self.tr("实现方式"))
        self.library_action.setText(self.tr("库函数"))
        self.custom_action.setText(self.tr("自编函数"))
        self.help_menu.setTitle(self.tr("帮助"))
        self.about_action.setText(self.tr("关于"))
        self.processing_toolbar.setWindowTitle(self.tr("图像处理"))
        self.operation_actions["gray"].setText(self.tr("灰度变换"))
        self.operation_actions["spatial"].setText(self.tr("空间滤波"))
        self.operation_actions["frequency"].setText(self.tr("频域滤波"))
        self.operation_actions["geometry"].setText(self.tr("几何变换"))
        self.operation_actions["binarization"].setText(self.tr("二值化"))
        self.operation_actions["edge"].setText(self.tr("边缘检测"))
        self.images.retranslateUi()
        self.params.retranslateUi()
        self.attributes.retranslateUi()

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslateUi()

    def toggle_sync_zoom(self, enabled: bool) -> None:
        self.images.set_sync_zoom(enabled)

    def toggle_phase_spectrum(self, enabled: bool) -> None:
        self.show_phase_spectrum = enabled
        if not enabled and self.display_mode == "phase":
            self.display_mode = "gray"
        self.update_display_mode()

    def set_language(self, language: str) -> None:
        QSettings("AlexFad", "image-process").setValue("language", language)
        app = QApplication.instance()
        if app is None:
            return
        install_language(app, language)
        self.retranslateUi()

    def use_library_backend(self) -> None:
        self.processor = LibraryImageProcessor
        self.apply_params(self.params.params())

    def use_custom_backend(self) -> None:
        self.processor = ManualImageProcessor
        self.apply_params(self.params.params())

    def set_transform_mode(self, mode: str) -> None:
        self.params.set_mode(mode)
        self.apply_params(self.params.params())

    def show_about(self):
        QMessageBox.about(
            self,
            self.tr("关于 image-process"),
            self.tr("图像处理工具\n版本：1.0.0\n\n"
            "GitHub：https://github.com/AlexFad6/image-process\n"
            "作者：AlexFad"),
        )

    def _update_ui_state(self) -> None:
        if self.state.source is None:
            self.params.setEnabled(False)
            self.images.convert_gray.setEnabled(False)
            return
        grayscale = self.state.source.ndim == 2
        self.params.setEnabled(grayscale)
        self.images.convert_gray.setEnabled(True)
        if not grayscale:
            self.statusBar().showMessage(self.tr("请先点击“转为灰度”按钮以进行灰度处理"))

    def convert_to_gray(self) -> None:
        if self.state.source is None:
            return
        gray = self.processor.to_gray(self.state.source)
        if gray is None:
            return
        original_state = self.images.original.get_view_state()
        result_state = self.images.result.get_view_state()
        self.state.source = gray
        self.state.result = None
        self.state.clear_frequency_data()
        self._set_frequency_controls_enabled(False)
        self.display_mode = "gray"
        self.images.original.set_image(gray, preserve_view=True)
        self.images.result.set_image(None, preserve_view=True)
        self.images.original.restore_view_state(original_state)
        self.images.result.restore_view_state(result_state)
        self.attributes.set_original(gray)
        self.attributes.set_result(None)
        self.images.convert_gray.setText(self.tr("频谱图"))
        self._update_ui_state()
        self.apply_params(self.params.params())

    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("打开图片"),
            "",
            self.tr("Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"),
        )
        if not path:
            return
        image = Exporter.read_image(path)
        if image is None:
            QMessageBox.warning(self, self.tr("错误"), self.tr("无法读取图片"))
            return

        self.image_queue.append((image, path))
        self.queue_index = len(self.image_queue) - 1
        self._show_queue_image()

    def _show_queue_image(self) -> None:
        if not self.image_queue or self.queue_index < 0:
            return
        self._processing_generation += 1
        image, path = self.image_queue[self.queue_index]
        self.state.source = image
        self.state.gray_source = self.processor.to_gray(image)
        self.state.source_path = path
        self.state.result = None
        self.state.clear_frequency_data()
        if self.state.gray_source is not None and self.state.gray_source.ndim == 2:
            self.state.frequency_data = LibraryImageProcessor.compute_fft(self.state.gray_source)
            self.state.fft_spectrum_uint8 = (
                LibraryImageProcessor.compute_fft_display_spectrum(
                    self.state.frequency_data
                )
            )
            self.state.phase_spectrum_uint8 = (
                LibraryImageProcessor.compute_fft_phase_spectrum(
                    self.state.frequency_data
                )
            )
        self._set_frequency_controls_enabled(False)
        self.images.original.set_image(image)
        self.images.result.set_image(None)
        self.attributes.set_original(image)
        self.attributes.set_result(None)
        self._update_ui_state()
        self.statusBar().showMessage(f"{self.queue_index + 1}/{len(self.image_queue)}  {path}")
        self.display_mode = "original"
        self.images.convert_gray.setText(self.tr("灰度图"))

    def cycle_display_mode(self) -> None:
        if self.state.source is None:
            return
        if self.display_mode == "original":
            gray = self.state.gray_source
            if gray is None:
                gray = self.processor.to_gray(self.state.source)
            if gray is None:
                return
            self.state.source = gray
            self.state.gray_source = gray
            self.state.result = None
            self._update_ui_state()
            self.display_mode = "gray"
        elif self.display_mode == "gray":
            if self.state.frequency_data is None:
                self.compute_fft()
                return
            self.ensure_frequency_cache()
            self.display_mode = "spectrum"
        elif self.display_mode == "spectrum":
            self.display_mode = "phase" if self.show_phase_spectrum else "gray"
        else:
            self.display_mode = "gray"
        self.update_display_mode()

    def schedule_display_mode_switch(self) -> None:
        # Restart the debounce window on every click.
        self.display_mode_timer.stop()
        self.display_mode_timer.start()

    def ensure_frequency_cache(self) -> None:
        if self.state.source is None or self.state.source.ndim != 2:
            return
        if self.state.frequency_data is None:
            self.state.frequency_data = LibraryImageProcessor.compute_fft(self.state.source)
            self.state.fft_spectrum_uint8 = LibraryImageProcessor.compute_fft_display_spectrum(
                self.state.frequency_data
            )
            self.state.phase_spectrum_uint8 = LibraryImageProcessor.compute_fft_phase_spectrum(
                self.state.frequency_data
            )
        if self.state.result is not None and self.state.result.ndim == 2:
            if self.state.result_frequency_data is None:
                self.state.result_frequency_data = LibraryImageProcessor.compute_fft(self.state.result)
                self.state.result_fft_spectrum_uint8 = (
                    LibraryImageProcessor.compute_fft_display_spectrum(
                        self.state.result_frequency_data
                    )
                )
                self.state.result_phase_spectrum_uint8 = (
                    LibraryImageProcessor.compute_fft_phase_spectrum(
                        self.state.result_frequency_data
                    )
                )

    def update_display_mode(self) -> None:
        if self.state.source is None:
            return
        if self.display_mode == "original":
            source_display = self.state.source
            result_display = self.state.result
            next_text = "灰度图"
        elif self.display_mode == "gray":
            source_display = self.state.source
            result_display = self.state.result
            next_text = "频谱图"
        else:
            self.ensure_frequency_cache()
            if self.display_mode == "phase":
                source_display = self.state.phase_spectrum_uint8
                result_display = self.state.result_phase_spectrum_uint8
                next_text = "灰度图"
            else:
                source_display = self.state.fft_spectrum_uint8
                result_display = self.state.result_fft_spectrum_uint8
                next_text = "相位谱图" if self.show_phase_spectrum else "灰度图"
        self.images.original.set_image(source_display, preserve_view=True)
        self.images.result.set_image(result_display, preserve_view=True)
        self.images.convert_gray.setText(self.tr(next_text))
        self.attributes.set_original(source_display)
        self.attributes.set_result(result_display)

    def compute_fft(self) -> None:
        if self.state.source is None:
            return
        if self.state.source.ndim != 2:
            self.statusBar().showMessage("请先将图像转换为灰度图")
            return
        if self.state.frequency_data is not None:
            self.display_mode = "spectrum"
            self.update_display_mode()
            self.params.set_frequency_controls_enabled(True)
            return
        self._start_processing("fft", (self.state.source.copy(), LibraryImageProcessor))

    def apply_frequency_filter(self, params=None) -> None:
        if self.state.frequency_data is None:
            return
        if params is None:
            params = self.params.frequency_params()
        self._processing_generation += 1
        self.state.frequency_params = params
        self._pending_frequency = params
        self.frequency_timer.start()

    def apply_spatial_params(self, params) -> None:
        self._processing_generation += 1
        self.state.spatial_params = params
        self._pending_spatial = params
        self.spatial_timer.start()

    def apply_geometry_params(self, params) -> None:
        if self.state.source is None:
            return
        self._processing_generation += 1
        self.state.geometry_params = params
        self._start_processing(
            "geometry",
            (self.processor, self.state.source.copy(), params),
        )

    def apply_binarization_params(self, params) -> None:
        if self.state.source is None or self.state.source.ndim != 2:
            return
        self._processing_generation += 1
        self.state.binarization_params = params
        self._start_processing(
            "binarization",
            (self.processor, self.state.source.copy(), params),
        )

    def apply_edge_params(self, params) -> None:
        if self.state.source is None or self.state.source.ndim != 2:
            return
        self._processing_generation += 1
        self.state.edge_params = params
        self._start_processing(
            "edge",
            (self.processor, self.state.source.copy(), params),
        )

    def _run_pending_spatial(self) -> None:
        if self.state.source is None or self.state.source.ndim != 2:
            return
        self._start_processing(
            "spatial",
            (self.processor, self.state.source.copy(), self._pending_spatial),
        )

    def _run_pending_frequency(self) -> None:
        if self.state.frequency_data is None:
            return
        self._start_processing(
            "frequency",
            (
                self.state.frequency_data.copy(),
                self._pending_frequency,
                LibraryImageProcessor,
            ),
        )

    def _start_processing(self, operation: str, payload: tuple) -> None:
        self._processing_generation += 1
        generation = self._processing_generation
        thread = QThread(self)
        worker = ProcessingWorker(operation, payload, generation)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._processing_finished)
        worker.failed.connect(self._processing_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._processing_jobs.pop(generation, None))
        self._processing_jobs[generation] = (thread, worker)
        thread.start()

    def _processing_finished(self, result, generation: int) -> None:
        if generation != self._processing_generation:
            return
        operation = self._processing_jobs[generation][1].operation
        if operation == "spatial":
            self.state.result = result
            self.state.result_frequency_data = None
            self.state.result_fft_spectrum_uint8 = None
            self.state.result_phase_spectrum_uint8 = None
        elif operation == "geometry":
            self.state.result = result
            self.state.clear_frequency_data()
            self.state.result_frequency_data = None
            self.state.result_fft_spectrum_uint8 = None
            self.state.result_phase_spectrum_uint8 = None
            self.display_mode = "original"
        elif operation == "binarization":
            self.state.result = result
            self.state.result_frequency_data = None
            self.state.result_fft_spectrum_uint8 = None
            self.state.result_phase_spectrum_uint8 = None
            self.display_mode = "original"
        elif operation == "edge":
            self.state.result = result
            self.state.result_frequency_data = None
            self.state.result_fft_spectrum_uint8 = None
            self.state.result_phase_spectrum_uint8 = None
            self.display_mode = "original"
        elif operation == "transform":
            self._apply_transform_result(result)
            return
        elif operation in ("fft", "frequency"):
            frequency_data, spectrum, phase = result
            self.state.frequency_data = frequency_data
            self.state.fft_spectrum_uint8 = spectrum
            self.state.phase_spectrum_uint8 = phase
            self.state.result_frequency_data = None
            self.state.result_fft_spectrum_uint8 = None
            self.state.result_phase_spectrum_uint8 = None
            self.display_mode = "spectrum"
            self.params.set_frequency_controls_enabled(True)
        elif operation == "ifft":
            self.state.source = result
            self.state.result = None
            self.state.clear_frequency_data()
            self.display_mode = "gray"
            self.params.set_frequency_controls_enabled(False)
            self._update_ui_state()
        original_state = self.images.original.get_view_state()
        result_state = self.images.result.get_view_state()
        self.update_display_mode()
        self.images.original.restore_view_state(original_state)
        self.images.result.restore_view_state(result_state)
        self.images.sync_views()

    def _processing_failed(self, message: str, generation: int) -> None:
        if generation == self._processing_generation:
            self.statusBar().showMessage(self.tr("处理失败：") + message)

    def apply_ifft(self, component: str = "full") -> None:
        if self.state.frequency_data is None:
            return
        self._start_processing(
            "ifft",
            (self.state.frequency_data.copy(), LibraryImageProcessor, component),
        )

    def _set_frequency_controls_enabled(self, enabled: bool) -> None:
        self.params.set_frequency_controls_enabled(enabled)

    def previous_image(self) -> None:
        if self.queue_index > 0:
            self.queue_index -= 1
            self._show_queue_image()

    def next_image(self) -> None:
        if self.queue_index + 1 < len(self.image_queue):
            self.queue_index += 1
            self._show_queue_image()

    def apply_params(self, params) -> None:
        if self.state.source is None:
            return
        if self.state.source.ndim == 3:
            self.images.result.set_image(None, preserve_view=True)
            self.attributes.set_result(None)
            return
        self.state.params = params
        self._processing_generation += 1
        self._start_processing(
            "transform",
            (self.processor, self.state.source.copy(), params),
        )

    def _apply_transform_result(self, result) -> None:
        self.state.result = result
        self.state.result_frequency_data = None
        self.state.result_fft_spectrum_uint8 = None
        self.state.result_phase_spectrum_uint8 = None
        original_state = self.images.original.get_view_state()
        result_state = self.images.result.get_view_state()
        self.update_display_mode()
        self.images.original.restore_view_state(original_state)
        self.images.result.restore_view_state(result_state)
        self.images.sync_views()

    def save_image(self) -> None:
        if self.state.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存图片", "", "PNG (*.png);;JPEG (*.jpg)")
        if path and not Exporter.save_image(path, self.state.result):
            QMessageBox.warning(self, self.tr("错误"), self.tr("保存失败"))

    def export_sequence(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "选择图片序列", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not files:
            return
        directory = QFileDialog.getExistingDirectory(self, "输出目录")
        if directory:
            log = Exporter.export_sequence(files, directory, self.params.params(), self.processor)
            self.statusBar().showMessage(f"导出完成：{log}")
