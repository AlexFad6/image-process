from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QIcon
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QMainWindow, QMessageBox, QMenuBar, QStatusBar, QVBoxLayout, QWidget

from core.app_state import AppState
from core.exporter import Exporter
from core.image_processor_library import LibraryImageProcessor
from core.image_processor_manual import ManualImageProcessor
from .attribute_panel import AttributePanel
from .image_panel import ImagePanel
from .param_panel import ParamPanel


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("image-process")
        self.resize(1200, 760)
        self.setWindowIcon(QIcon("app_icon.png"))

        self.state = AppState()
        self.image_queue: list[tuple[np.ndarray, str]] = []
        self.queue_index = -1

        self.images = ImagePanel()
        self.params = ParamPanel()
        self.processor = LibraryImageProcessor
        self.attributes = AttributePanel()

        self._create_menu_bar()
        self._create_second_row_menu()

        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.addWidget(self.params)
        side_layout.addWidget(self.attributes, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        left_layout.addWidget(self.images, 1)

        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(left, 2)
        content_layout.addWidget(side, 1)
        self.setCentralWidget(content)
        self.setStatusBar(QStatusBar())

        self.images.convertToGrayRequested.connect(self.convert_to_gray)
        self.params.changed.connect(self.apply_params)
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
        
    def _create_menu_bar(self) -> None:
        file_menu = self.menuBar().addMenu("文件")
        self._add_action(file_menu, "打开图像", self.open_image, "Ctrl+O")
        self._add_action(file_menu, "保存当前图像", self.save_image, "Ctrl+S")
        self._add_action(file_menu, "保存图像序列", self.export_sequence)
        file_menu.addSeparator()
        self._add_action(file_menu, "退出", self.close, "Alt+F4")

        view_menu = self.menuBar().addMenu("视图")
        sync_action = self._add_action(view_menu, "同步缩放", self.toggle_sync_zoom)
        sync_action.setCheckable(True)
        sync_action.setChecked(True)
        self._add_action(view_menu, "重置视图", self.images.reset_views)
        view_menu.addSeparator()
        self._add_action(view_menu, "◀ 上一张", self.previous_image, "Left")
        self._add_action(view_menu, "下一张 ▶", self.next_image, "Right")

        implementation_menu = self.menuBar().addMenu("实现方式")
        implementation_group = QActionGroup(self)
        implementation_group.setExclusive(True)
        self.library_action = self._add_action(implementation_menu, "库函数", self.use_library_backend)
        self.custom_action = self._add_action(implementation_menu, "自编函数", self.use_custom_backend)
        for action in (self.library_action, self.custom_action):
            action.setCheckable(True)
            implementation_group.addAction(action)
        self.library_action.setChecked(True)

        help_menu = self.menuBar().addMenu("帮助")
        self._add_action(help_menu, "关于", self.show_about)

    def _create_second_row_menu(self) -> None:
        second_menu = QMenuBar(self)
        second_menu.setNativeMenuBar(False)
        gray_menu = second_menu.addMenu("灰度变换")
        group = QActionGroup(self)
        group.setExclusive(True)

        for name, mode in [("线性变换", "linear"), ("对数变换", "log"), ("幂指数变换", "gamma")]:
            action = QAction(name, self)
            action.setCheckable(True)
            action.triggered.connect(lambda checked, value=mode: self.set_transform_mode(value))
            group.addAction(action)
            gray_menu.addAction(action)
        group.actions()[0].setChecked(True)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.menuBar())
        layout.addWidget(second_menu)
        self.setMenuWidget(container)

    @staticmethod
    def _add_action(menu, text: str, slot, shortcut: str | None = None) -> QAction:
        action = QAction(text, menu)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(slot)
        menu.addAction(action)
        return action

    def toggle_sync_zoom(self, enabled: bool) -> None:
        self.images.set_sync_zoom(enabled)

    def use_library_backend(self) -> None:
        self.processor = LibraryImageProcessor
        self.apply_params(self.params.params())

    def use_custom_backend(self) -> None:
        self.processor = ManualImageProcessor
        self.apply_params(self.params.params())

    def set_transform_mode(self, mode: str) -> None:
        self.params.mode.setCurrentText(mode)
        self.apply_params(self.params.params())

    def show_about(self):
        QMessageBox.about(
            self,
            "关于 image-process",
            "图像处理工具\n版本：1.0.0\n\n"
            "GitHub：https://github.com/AlexFad6/image-process\n"
            "作者：AlexFad",
        )

    def _update_ui_state(self) -> None:
        if self.state.source is None:
            self.params.setEnabled(False)
            self.images.convert_gray.setEnabled(False)
            return
        grayscale = self.state.source.ndim == 2
        self.params.setEnabled(grayscale)
        self.images.convert_gray.setEnabled(not grayscale)
        if not grayscale:
            self.statusBar().showMessage("请先点击“转为灰度”按钮以进行灰度处理")

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
        self.images.original.set_image(gray, preserve_view=True)
        self.images.result.set_image(None, preserve_view=True)
        self.images.original.restore_view_state(original_state)
        self.images.result.restore_view_state(result_state)
        self.attributes.set_original(gray)
        self.attributes.set_result(None)
        self._update_ui_state()
        self.apply_params(self.params.params())

    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "打开图片",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not path:
            return
        image = Exporter.read_image(path)
        if image is None:
            QMessageBox.warning(self, "错误", "无法读取图片")
            return

        self.image_queue.append((image, path))
        self.queue_index = len(self.image_queue) - 1
        self._show_queue_image()

    def _show_queue_image(self) -> None:
        if not self.image_queue or self.queue_index < 0:
            return
        image, path = self.image_queue[self.queue_index]
        self.state.source = image
        self.state.source_path = path
        self.state.result = None
        self.images.original.set_image(image)
        self.images.result.set_image(None)
        self.attributes.set_original(image)
        self.attributes.set_result(None)
        self._update_ui_state()
        self.statusBar().showMessage(f"{self.queue_index + 1}/{len(self.image_queue)}  {path}")

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
        self.state.result = self.processor.apply(self.state.source, params)
        original_state = self.images.original.get_view_state()
        result_state = self.images.result.get_view_state()
        self.images.original.set_image(self.state.source, preserve_view=True)
        self.images.result.set_image(self.state.result, preserve_view=True)
        self.images.original.restore_view_state(original_state)
        self.images.result.restore_view_state(result_state)
        self.images.sync_views()
        self.attributes.set_original(self.state.source)
        self.attributes.set_result(self.state.result)

    def save_image(self) -> None:
        if self.state.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存图片", "", "PNG (*.png);;JPEG (*.jpg)")
        if path and not Exporter.save_image(path, self.state.result):
            QMessageBox.warning(self, "错误", "保存失败")

    def export_sequence(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "选择图片序列", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not files:
            return
        directory = QFileDialog.getExistingDirectory(self, "输出目录")
        if directory:
            log = Exporter.export_sequence(files, directory, self.params.params(), self.processor)
            self.statusBar().showMessage(f"导出完成：{log}")
