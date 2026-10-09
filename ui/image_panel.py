from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QGroupBox
from .image_view import ImageView


class ImagePanel(QGroupBox):
    def __init__(self, parent=None) -> None:
        super().__init__(None, parent)
        self.sync_zoom = True
        layout = QHBoxLayout(self)
        self.original = ImageView()
        self.result = ImageView()
        layout.addWidget(self.original)
        layout.addWidget(self.result)
        self.original.zoomChanged.connect(self._sync_zoom_from_original)
        self.result.zoomChanged.connect(self._sync_zoom_from_result)
        self.original.horizontalScrollBar().valueChanged.connect(
            lambda value: self._sync_scroll(self.result, "horizontal", value))
        self.original.verticalScrollBar().valueChanged.connect(
            lambda value: self._sync_scroll(self.result, "vertical", value))
        self.result.horizontalScrollBar().valueChanged.connect(
            lambda value: self._sync_scroll(self.original, "horizontal", value))
        self.result.verticalScrollBar().valueChanged.connect(
            lambda value: self._sync_scroll(self.original, "vertical", value))

    @staticmethod
    def _sync_scroll(view: ImageView, axis: str, value: int) -> None:
        bar = getattr(view, f"{axis}ScrollBar")()
        if bar.value() != value:
            bar.blockSignals(True)
            bar.setValue(value)
            bar.blockSignals(False)

    def set_sync_zoom(self, enabled: bool) -> None:
        self.sync_zoom = enabled

    def _sync_zoom_from_original(self, value: float) -> None:
        if self.sync_zoom:
            self.result.set_zoom(value)

    def _sync_zoom_from_result(self, value: float) -> None:
        if self.sync_zoom:
            self.original.set_zoom(value)

    def reset_views(self) -> None:
        self.original.reset_view()
        self.result.reset_view()
