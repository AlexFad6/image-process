from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPoint, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QWidget


class SyncGraphicsView(QGraphicsView):
    viewportChanged = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.partner = None
        self._syncing = False
        self._sync_enabled = True

        self.horizontalScrollBar().valueChanged.connect(self._on_scroll)
        self.verticalScrollBar().valueChanged.connect(self._on_scroll)

    def set_partner(self, partner) -> None:
        self.partner = partner

    def set_sync(self, enabled: bool) -> None:
        self._sync_enabled = bool(enabled)

    def _on_scroll(self) -> None:
        if self._syncing or not self._sync_enabled or self.partner is None:
            self.viewportChanged.emit()
            return

        partner = self.partner
        partner._syncing = True
        partner.horizontalScrollBar().setValue(self.horizontalScrollBar().value())
        partner.verticalScrollBar().setValue(self.verticalScrollBar().value())
        partner._syncing = False
        self.viewportChanged.emit()

    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self._syncing = True
        self.scale(factor, factor)
        self._syncing = False
        self._sync_zoom()
        event.accept()

    def _sync_zoom(self) -> None:
        if self._sync_enabled and self.partner is not None:
            partner = self.partner
            partner._syncing = True
            partner.setTransform(self.transform())
            partner.horizontalScrollBar().setValue(self.horizontalScrollBar().value())
            partner.verticalScrollBar().setValue(self.verticalScrollBar().value())
            partner._syncing = False
        self.viewportChanged.emit()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.viewportChanged.emit()

    def reset_view(self) -> None:
        scene = self.scene()
        if scene is None or scene.sceneRect().isEmpty():
            self.viewportChanged.emit()
            return
        self.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)
        self.viewportChanged.emit()


class MiniMap(QWidget):
    def __init__(self, source_view: QGraphicsView, parent=None) -> None:
        super().__init__(parent)
        self.source_view = source_view
        self.setFixedSize(180, 120)
        self.setCursor(Qt.PointingHandCursor)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._pixmap = None
        self._image_rect = QRect()

    def set_pixmap(self, pixmap: QPixmap | None) -> None:
        self._pixmap = pixmap
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        painter.setPen(QPen(QColor(90, 90, 90), 1))
        painter.setBrush(QColor(0, 0, 0, 200))
        painter.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 6, 6)

        if self._pixmap is None:
            return

        pad = 6
        rect = self.rect().adjusted(pad, pad, -pad, -pad)
        scaled = self._pixmap.scaled(rect.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x = rect.x() + (rect.width() - scaled.width()) // 2
        y = rect.y() + (rect.height() - scaled.height()) // 2
        self._image_rect = QRect(x, y, scaled.width(), scaled.height())
        painter.drawPixmap(x, y, scaled)

        view = self.source_view
        scene = view.scene()
        if scene is None or scene.sceneRect().isEmpty():
            return

        visible = view.mapToScene(view.viewport().rect()).boundingRect()
        scene_rect = scene.sceneRect()
        sx = self._image_rect.width() / scene_rect.width()
        sy = self._image_rect.height() / scene_rect.height()
        rx = self._image_rect.x() + (visible.x() - scene_rect.x()) * sx
        ry = self._image_rect.y() + (visible.y() - scene_rect.y()) * sy
        rw = visible.width() * sx
        rh = visible.height() * sy

        painter.setPen(QPen(QColor(255, 60, 60), 1.5))
        painter.setBrush(QColor(255, 60, 60, 50))
        painter.drawRect(QRectF(rx, ry, rw, rh))

    def mousePressEvent(self, event) -> None:
        self._jump_to(event.position().toPoint())

    def mouseMoveEvent(self, event) -> None:
        if event.buttons() & Qt.LeftButton:
            self._jump_to(event.position().toPoint())

    def _jump_to(self, pos: QPoint) -> None:
        if self._pixmap is None or self._image_rect.isEmpty():
            return
        if not self._image_rect.contains(pos):
            return

        view = self.source_view
        scene = view.scene()
        if scene is None:
            return
        scene_rect = scene.sceneRect()
        sx = scene_rect.width() / self._image_rect.width()
        sy = scene_rect.height() / self._image_rect.height()
        scene_x = scene_rect.x() + (pos.x() - self._image_rect.x()) * sx
        scene_y = scene_rect.y() + (pos.y() - self._image_rect.y()) * sy
        view.centerOn(scene_x, scene_y)


class ImageView(SyncGraphicsView):
    imageClicked = Signal(float, float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pixmap_item = None
        self._image = None
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setBackgroundBrush(QColor(30, 30, 30))
        self.setAlignment(Qt.AlignCenter)

        self.minimap = MiniMap(self, self)
        self.minimap.hide()
        self.viewportChanged.connect(self._update_minimap)

    def get_view_state(self):
        scene = self.scene()
        if scene is None or scene.sceneRect().isEmpty():
            return None
        return {
            "transform": self.transform(),
            "h": self.horizontalScrollBar().value(),
            "v": self.verticalScrollBar().value(),
        }

    def restore_view_state(self, state) -> None:
        if state is None:
            return
        self.setTransform(state["transform"])
        self.horizontalScrollBar().setValue(state["h"])
        self.verticalScrollBar().setValue(state["v"])
        self.viewportChanged.emit()

    def _update_minimap(self) -> None:
        if self._pixmap_item is None:
            self.minimap.hide()
            return

        pixmap = self._pixmap_item.pixmap()
        if pixmap.isNull():
            self.minimap.hide()
            return

        self.minimap.set_pixmap(pixmap)
        scene_rect = self.sceneRect()
        if scene_rect.isEmpty():
            self.minimap.hide()
            return

        viewport_size = self.viewport().size()
        fit_scale = min(viewport_size.width() / scene_rect.width(), viewport_size.height() / scene_rect.height())
        cur_scale = self.transform().m11()
        show = cur_scale > fit_scale * 1.01
        self.minimap.setVisible(show)
        if show:
            self._position_minimap()
            self.minimap.raise_()

    def _position_minimap(self) -> None:
        margin = 16
        width = self.minimap.width()
        height = self.minimap.height()
        x = self.width() - width - margin
        y = self.height() - height - margin
        self.minimap.move(x, y)

    def set_image(self, image, preserve_view: bool = False) -> None:
        view_state = self.get_view_state() if preserve_view else None
        if image is None:
            self.clear_image()
            return

        arr = np.asarray(image)
        if arr.size == 0:
            self.clear_image()
            return

        arr = arr.astype(np.uint8, copy=False)
        if arr.ndim == 2:
            h, w = arr.shape
            qimage = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
            qimage = qimage.copy()
            pixmap = QPixmap.fromImage(qimage)
        elif arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
            h, w, ch = arr.shape
            if ch == 1:
                arr = arr[:, :, 0]
                qimage = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
                qimage = qimage.copy()
            elif ch == 3:
                rgb = np.ascontiguousarray(arr[:, :, ::-1])
                qimage = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
                qimage = qimage.copy()
            else:
                rgba = np.ascontiguousarray(arr[:, :, [2, 1, 0, 3]])
                qimage = QImage(rgba.data, w, h, w * 4, QImage.Format_RGBA8888)
                qimage = qimage.copy()
            pixmap = QPixmap.fromImage(qimage)
        else:
            self.clear_image()
            return

        if self._pixmap_item is None:
            self._pixmap_item = QGraphicsPixmapItem(pixmap)
            self._scene.addItem(self._pixmap_item)
        else:
            self._pixmap_item.setPixmap(pixmap)
        self._pixmap_item.setPos(0, 0)
        self._scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())
        self._image = image
        self.minimap.set_pixmap(pixmap)
        if preserve_view and view_state is not None:
            self.restore_view_state(view_state)
        else:
            self.reset_view()
        self.viewportChanged.emit()

    def clear_image(self) -> None:
        self._image = None
        if self._pixmap_item is not None:
            self._scene.removeItem(self._pixmap_item)
            self._pixmap_item = None
        self._scene.clear()
        self.minimap.set_pixmap(None)
        self.minimap.hide()
        self.viewportChanged.emit()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._pixmap_item is not None:
            point = self.mapToScene(event.position().toPoint())
            rect = self._scene.sceneRect()
            if rect.contains(point):
                self.imageClicked.emit(point.x(), point.y())
        super().mousePressEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_minimap()

    def reset_view(self) -> None:
        scene = self.scene()
        if scene is None or scene.sceneRect().isEmpty():
            self.viewportChanged.emit()
            return
        self.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)
        self.viewportChanged.emit()
