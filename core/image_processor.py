from __future__ import annotations

import cv2
import numpy as np

from .app_state import TransformParams


class ImageProcessor:
    """Gray image point transforms and conversion utilities.

    The class keeps algorithm logic decoupled from Qt so it can be tested and reused
    in scripts or in non-GUI pipelines.
    """

    @staticmethod
    def to_gray_if_needed(image: np.ndarray | None) -> np.ndarray | None:
        if image is None:
            return None
        if image.ndim == 2:
            return image.astype(np.uint8, copy=False)
        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if image.shape[-1] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.astype(np.uint8, copy=False)

    @staticmethod
    def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
        image = np.asarray(image, dtype=np.float32)
        image = np.clip(image, 0, 255)
        return image.astype(np.uint8)

    @staticmethod
    def clip_image(image: np.ndarray) -> np.ndarray:
        return np.clip(image, 0, 255).astype(np.float32)

    @staticmethod
    def linear_transform(img: np.ndarray, a: float, b: float, c: float, d: float, mode: str) -> np.ndarray:
        """Implement grayscale linear transform y = c*x + d or piecewise mapping."""
        src = np.asarray(img, dtype=np.float32)
        if mode == "numpy":
            output = np.empty_like(src, dtype=np.float32)
            if b <= a:
                return np.zeros_like(src, dtype=np.uint8)
            output = (src - a) * (c - d) / (b - a) + d
            return ImageProcessor.normalize_to_uint8(output)
        if mode == "opencv":
            if b <= a:
                return np.zeros_like(src, dtype=np.uint8)
            clip = np.clip(src, a, b).astype(np.float32)
            output = (clip - a) * (c - d) / (b - a) + d
            return ImageProcessor.normalize_to_uint8(output)
        raise ValueError(f"Unsupported mode: {mode}")

    @staticmethod
    def log_transform(img: np.ndarray, c: float, mode: str) -> np.ndarray:
        """Implement logarithmic transform y = c * log(1 + x)."""
        src = np.asarray(img, dtype=np.float32)
        if mode == "numpy":
            output = c * np.log1p(src) * 255.0 / np.log(256.0)
            return ImageProcessor.normalize_to_uint8(output)
        if mode == "opencv":
            output = c * cv2.log(src + 1.0) * 255.0 / np.log(256.0)
            return ImageProcessor.normalize_to_uint8(output)
        raise ValueError(f"Unsupported mode: {mode}")

    @staticmethod
    def gamma_transform(img: np.ndarray, c: float, gamma: float, mode: str) -> np.ndarray:
        """Implement gamma transform y = c * x ^ gamma."""
        src = np.asarray(img, dtype=np.float32)
        gamma = max(0.01, float(gamma))
        if mode == "numpy":
            normalized = np.clip(src / 255.0, 0.0, 1.0)
            output = c * np.power(normalized, gamma) * 255.0
            return ImageProcessor.normalize_to_uint8(output)
        if mode == "opencv":
            normalized = np.clip(src / 255.0, 0.0, 1.0)
            output = c * np.power(normalized, gamma) * 255.0
            return ImageProcessor.normalize_to_uint8(output)
        raise ValueError(f"Unsupported mode: {mode}")

    @staticmethod
    def apply(image: np.ndarray, params: TransformParams) -> np.ndarray:
        gray = ImageProcessor.to_gray_if_needed(image)
        if gray is None:
            return np.zeros((1, 1), dtype=np.uint8)
        if params.mode == "linear":
            a = float(params.low)
            b = float(params.high)
            c = 0.0
            d = 255.0
            if params.backend == "numpy":
                return ImageProcessor.linear_transform(gray, a, b, c, d, "numpy")
            return ImageProcessor.linear_transform(gray, a, b, c, d, "opencv")
        if params.mode == "log":
            if params.backend == "numpy":
                return ImageProcessor.log_transform(gray, max(0.01, float(params.log_c)), "numpy")
            return ImageProcessor.log_transform(gray, max(0.01, float(params.log_c)), "opencv")
        if params.backend == "numpy":
            return ImageProcessor.gamma_transform(gray, 1.0, float(params.gamma), "numpy")
        return ImageProcessor.gamma_transform(gray, 1.0, float(params.gamma), "opencv")

    @staticmethod
    def to_gray(image: np.ndarray) -> np.ndarray:
        return ImageProcessor.to_gray_if_needed(image) or np.zeros((1, 1), dtype=np.uint8)
