from __future__ import annotations

import cv2
import numpy as np

from .app_state import TransformParams


class LibraryImageProcessor:
    """Image processing algorithms implemented with OpenCV."""

    @staticmethod
    def to_gray(image: np.ndarray | None) -> np.ndarray | None:
        if image is None or image.ndim == 2:
            return image
        if image.shape[-1] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        if image.shape[-1] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.astype(np.uint8, copy=False)

    @staticmethod
    def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(image, dtype=np.float32), 0, 255).astype(np.uint8)

    @staticmethod
    def clip_image(image: np.ndarray) -> np.ndarray:
        return np.clip(image, 0, 255).astype(np.float32)

    @staticmethod
    def linear_transform(img: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
        if b <= a:
            return np.zeros_like(img, dtype=np.uint8)
        clipped = np.clip(np.asarray(img, dtype=np.float32), a, b)
        return LibraryImageProcessor.normalize_to_uint8((clipped - a) * (c - d) / (b - a) + d)

    @staticmethod
    def log_transform(img: np.ndarray, c: float) -> np.ndarray:
        source = np.asarray(img, dtype=np.float32)
        output = c * cv2.log(source + 1.0) * 255.0 / np.log(256.0)
        return LibraryImageProcessor.normalize_to_uint8(output)

    @staticmethod
    def gamma_transform(img: np.ndarray, c: float, gamma: float) -> np.ndarray:
        source = np.clip(np.asarray(img, dtype=np.float32) / 255.0, 0.0, 1.0)
        return LibraryImageProcessor.normalize_to_uint8(c * cv2.pow(source, max(0.01, float(gamma))) * 255.0)

    @staticmethod
    def apply(image: np.ndarray, params: TransformParams) -> np.ndarray:
        if params.mode == "linear":
            return LibraryImageProcessor.linear_transform(image, params.low, params.high, 0.0, 255.0)
        if params.mode == "log":
            return LibraryImageProcessor.log_transform(image, max(0.01, params.log_c))
        if params.mode == "gamma":
            return LibraryImageProcessor.gamma_transform(image, 1.0, params.gamma)
        raise ValueError(f"Unsupported transform mode: {params.mode}")

    @staticmethod
    def histogram_equalization(image: np.ndarray) -> np.ndarray:
        return cv2.equalizeHist(image)

    @staticmethod
    def gen_filter_kernel(size: int, filter_type: str) -> np.ndarray:
        if filter_type == "mean":
            return np.ones((size, size), np.float32) / (size * size)
        if filter_type == "gaussian":
            vector = cv2.getGaussianKernel(size, 0)
            return vector @ vector.T
        kernels = {
            "laplacian": [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
            "differenceG_x": [[-1, 1], [0, 0]],
            "differenceG_y": [[-1, 0], [1, 0]],
            "roberts": [[-1, 1], [1, -1]],
            "sobel_x": [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            "sobel_y": [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        }
        if filter_type not in kernels:
            raise ValueError(f"Unsupported filter type: {filter_type}")
        return np.asarray(kernels[filter_type], dtype=np.float32)

    @staticmethod
    def spatial_filter(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def median_filter(image: np.ndarray, size: int, ratio: float) -> np.ndarray:
        if ratio != 50:
            raise ValueError("The library median filter supports ratio=50 only")
        return cv2.medianBlur(image, size)
