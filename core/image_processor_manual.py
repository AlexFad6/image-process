from __future__ import annotations

import numpy as np

from .app_state import TransformParams


class ManualImageProcessor:
    """Image processing algorithms implemented with NumPy operations."""

    @staticmethod
    def to_gray(image: np.ndarray | None) -> np.ndarray | None:
        if image is None or image.ndim == 2:
            return image
        if image.shape[-1] == 4:
            image = image[..., :3]
        if image.shape[-1] != 3:
            return image.astype(np.uint8, copy=False)
        gray = (
            0.114 * image[..., 0]
            + 0.587 * image[..., 1]
            + 0.299 * image[..., 2]
        )
        return np.clip(gray, 0, 255).astype(np.uint8)

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
        output = (np.asarray(img, dtype=np.float32) - a) * (c - d) / (b - a) + d
        return ManualImageProcessor.normalize_to_uint8(output)

    @staticmethod
    def log_transform(img: np.ndarray, c: float) -> np.ndarray:
        output = c * np.log1p(np.asarray(img, dtype=np.float32)) * 255.0 / np.log(256.0)
        return ManualImageProcessor.normalize_to_uint8(output)

    @staticmethod
    def gamma_transform(img: np.ndarray, c: float, gamma: float) -> np.ndarray:
        normalized = np.clip(np.asarray(img, dtype=np.float32) / 255.0, 0.0, 1.0)
        output = c * np.power(normalized, max(0.01, float(gamma))) * 255.0
        return ManualImageProcessor.normalize_to_uint8(output)

    @staticmethod
    def apply(image: np.ndarray, params: TransformParams) -> np.ndarray:
        if params.mode == "linear":
            return ManualImageProcessor.linear_transform(image, params.low, params.high, 0.0, 255.0)
        if params.mode == "log":
            return ManualImageProcessor.log_transform(image, max(0.01, params.log_c))
        if params.mode == "gamma":
            return ManualImageProcessor.gamma_transform(image, 1.0, params.gamma)
        raise ValueError(f"Unsupported transform mode: {params.mode}")

    @staticmethod
    def histogram_equalization(image: np.ndarray) -> np.ndarray:
        values = np.asarray(image, dtype=np.uint8)
        hist = np.bincount(values.ravel(), minlength=256)
        cdf = hist.cumsum()
        nonzero = cdf[cdf > 0]
        if nonzero.size == 0 or nonzero[-1] == nonzero[0]:
            return values.copy()
        lut = np.zeros(256, dtype=np.uint8)
        lut[cdf > 0] = np.round((cdf[cdf > 0] - nonzero[0]) * 255 / (nonzero[-1] - nonzero[0])).astype(np.uint8)
        return lut[values]

    @staticmethod
    def gen_filter_kernel(size: int, filter_type: str) -> np.ndarray:
        if filter_type == "mean":
            return np.ones((size, size), dtype=np.float32) / (size * size)
        if filter_type == "gaussian":
            axis = np.arange(size, dtype=np.float32) - (size - 1) / 2
            kernel = np.exp(-(axis ** 2) / 2)
            kernel /= kernel.sum()
            return np.outer(kernel, kernel)
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
        values = np.asarray(image, dtype=np.float32)
        height, width = values.shape[:2]
        kh, kw = kernel.shape
        padded = np.pad(values, ((kh // 2, kh // 2), (kw // 2, kw // 2)), mode="edge")
        result = np.zeros((height, width), dtype=np.float32)
        for y in range(kh):
            for x in range(kw):
                result += kernel[y, x] * padded[y:y + height, x:x + width]
        return ManualImageProcessor.normalize_to_uint8(result)

    @staticmethod
    def median_filter(image: np.ndarray, size: int, ratio: float) -> np.ndarray:
        values = np.asarray(image, dtype=np.uint8)
        if size % 2 == 0 or size < 1:
            raise ValueError("Filter size must be a positive odd number")
        radius = size // 2
        padded = np.pad(values, radius, mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, (size, size))
        index = int(np.clip(ratio, 0, 100) / 100 * (size * size - 1))
        flattened = windows.reshape(windows.shape[:2] + (size * size,))
        return np.partition(flattened, index, axis=-1)[..., index]
