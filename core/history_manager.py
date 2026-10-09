from __future__ import annotations

from dataclasses import replace
from typing import List, Optional
import numpy as np
from .app_state import TransformParams


class HistoryManager:
    def __init__(self, limit: int = 20) -> None:
        self.limit = limit
        self._undo: List[tuple[np.ndarray, TransformParams]] = []
        self._redo: List[tuple[np.ndarray, TransformParams]] = []

    def push(self, image: np.ndarray, params: TransformParams) -> None:
        self._undo.append((image.copy(), replace(params)))
        del self._undo[:-self.limit]
        self._redo.clear()

    def undo(self, image: np.ndarray, params: TransformParams) -> Optional[tuple[np.ndarray, TransformParams]]:
        if not self._undo:
            return None
        self._redo.append((image.copy(), replace(params)))
        return self._undo.pop()

    def redo(self, image: np.ndarray, params: TransformParams) -> Optional[tuple[np.ndarray, TransformParams]]:
        if not self._redo:
            return None
        self._undo.append((image.copy(), replace(params)))
        return self._redo.pop()

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()

    def __len__(self) -> int:
        return len(self._undo)
