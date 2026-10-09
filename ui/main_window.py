from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (QFileDialog, QMainWindow, QMessageBox,
                               QStatusBar, QWidget, QHBoxLayout, QVBoxLayout)
from core.app_state import AppState
from core.exporter import Exporter
from core.image_processor import ImageProcessor
from .attribute_panel import AttributePanel
from .image_panel import ImagePanel
from .param_panel import ParamPanel


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("image-process")
        self.resize(1200, 760)
        self.state = AppState()
        self.image_queue: list[tuple[np.ndarray, str]] = []
        self.queue_index = -1
        self.images = ImagePanel(); self.params = ParamPanel()
        self.params.backend.setCurrentText("opencv")
        self.attributes = AttributePanel()
        self._create_menu_bar()
        side = QWidget(); side_layout = QVBoxLayout(side)
        side_layout.addWidget(self.params); side_layout.addWidget(self.attributes, 1)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        left_layout.addWidget(self.images, 1)
        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(left, 1)
        content_layout.addWidget(side)
        self.setCentralWidget(content); self.setStatusBar(QStatusBar())
        self.params.changed.connect(self.apply_params)

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
        self.params.backend.setCurrentText("opencv")

    def use_custom_backend(self) -> None:
        self.params.backend.setCurrentText("numpy")

    def show_about(self) -> None:
        QMessageBox.about(
            self,
            "关于 image-process",
            "图像处理工具\n版本：1.0.0\n\n"
            "GitHub：https://github.com/AlexFad6/image-process\n"
            "作者：AlexFad",
        )

    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "打开图片", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if not path: return
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            QMessageBox.warning(self, "错误", "无法读取图片"); return
        self.image_queue.append((image, path))
        self.queue_index = len(self.image_queue) - 1
        self._show_queue_image()

    def _show_queue_image(self) -> None:
        image, path = self.image_queue[self.queue_index]
        self.state.source, self.state.source_path = image, path
        self.state.result = None
        self.apply_params(self.params.params())
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
        if self.state.source is None: return
        self.state.params = params
        self.state.result = ImageProcessor.apply(self.state.source, params)
        self.images.original.set_image(ImageProcessor.to_gray(self.state.source))
        self.images.result.set_image(self.state.result); self.attributes.set_image(self.state.result)

    def save_image(self) -> None:
        if self.state.result is None: return
        path, _ = QFileDialog.getSaveFileName(self, "保存图片", "", "PNG (*.png);;JPEG (*.jpg)")
        if path and not Exporter.save_image(path, self.state.result): QMessageBox.warning(self, "错误", "保存失败")

    def export_sequence(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "选择图片序列", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not files: return
        directory = QFileDialog.getExistingDirectory(self, "输出目录")
        if directory:
            log = Exporter.export_sequence(files, directory, self.params.params())
            self.statusBar().showMessage(f"导出完成：{log}")
