from __future__ import annotations

import cv2
import numpy as np
from .app_state import TransformParams


class ImageProcessor:
    """Color-to-gray conversion and point gray-level transforms."""

    @staticmethod
    def to_gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image.astype(np.uint8, copy=False)
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def apply(image: np.ndarray, params: TransformParams) -> np.ndarray:
        gray = ImageProcessor.to_gray(image)
        if params.backend == "opencv":
            return ImageProcessor._opencv(gray, params)
        return ImageProcessor._numpy(gray, params)

    @staticmethod
    def _numpy(gray: np.ndarray, p: TransformParams) -> np.ndarray:
        x = gray.astype(np.float32)
        if p.mode == "linear":
            if p.high <= p.low:
                return np.zeros_like(gray)
            y = (x - p.low) * 255.0 / (p.high - p.low)
        elif p.mode == "log":
            y = p.log_c * np.log1p(x) * 255.0 / np.log(256.0)
        else:
            y = 255.0 * np.power(np.clip(x / 255.0, 0, 1), max(0.01, p.gamma))
        return np.clip(y, 0, 255).astype(np.uint8)

    @staticmethod
    def _opencv(gray: np.ndarray, p: TransformParams) -> np.ndarray:
        if p.mode == "linear":
            if p.high <= p.low:
                return np.zeros_like(gray)
            clipped = np.clip(gray, p.low, p.high).astype(np.float32)
            scaled = cv2.add(
                cv2.multiply(clipped, 255.0 / (p.high - p.low)),
                -p.low * 255.0 / (p.high - p.low),
            )
            return np.clip(scaled, 0, 255).astype(np.uint8)
        if p.mode == "log":
            f = gray.astype(np.float32)
            return np.clip(p.log_c * cv2.log(f + 1) * 255 / np.log(256), 0, 255).astype(np.uint8)
        lut = np.array([255 * (i / 255) ** max(0.01, p.gamma) for i in range(256)],
                       dtype=np.uint8)
        return cv2.LUT(gray, lut)
