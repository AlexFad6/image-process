from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView


class ImageView(QGraphicsView):
    zoomChanged = Signal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.item = QGraphicsPixmapItem()
        self.scene().addItem(self.item)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self._zoom = 1.0

    def set_image(self, image: np.ndarray | None) -> None:
        if image is None:
            self.item.setPixmap(QPixmap())
            return
        h, w = image.shape[:2]
        qimage = QImage(image.data, w, h, image.strides[0], QImage.Format_Grayscale8).copy()
        self.item.setPixmap(QPixmap.fromImage(qimage))
        self.scene().setSceneRect(0, 0, w, h)
        self.fitInView(self.item, Qt.KeepAspectRatio)
        self._zoom = 1.0

    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)
        self._zoom *= factor
        self.zoomChanged.emit(self._zoom)

    def set_zoom(self, value: float) -> None:
        if value <= 0:
            return
        self.resetTransform()
        self.scale(value, value)
        self._zoom = value

    def reset_view(self) -> None:
        self.resetTransform()
        if not self.item.pixmap().isNull():
            self.fitInView(self.item, Qt.KeepAspectRatio)
        self._zoom = 1.0
