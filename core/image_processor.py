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
    def to_gray(image: np.ndarray | None) -> np.ndarray | None:
        if image is None:
            return None
        if image.ndim == 2:
            return image.astype(np.uint8, copy=False)
        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif image.shape[-1] == 3:
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
        elif mode == "opencv":
            if b <= a:
                return np.zeros_like(src, dtype=np.uint8)
            clip = np.clip(src, a, b).astype(np.float32)
            output = (clip - a) * (c - d) / (b - a) + d
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        return ImageProcessor.normalize_to_uint8(output)

    @staticmethod
    def log_transform(img: np.ndarray, c: float, mode: str) -> np.ndarray:
        """Implement logarithmic transform y = c * log(1 + x)."""
        src = np.asarray(img, dtype=np.float32)
        if mode == "numpy":
            output = c * np.log1p(src) * 255.0 / np.log(256.0)
        elif mode == "opencv":
            output = c * cv2.log(src + 1.0) * 255.0 / np.log(256.0)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        return ImageProcessor.normalize_to_uint8(output)

    @staticmethod
    def gamma_transform(img: np.ndarray, c: float, gamma: float, mode: str) -> np.ndarray:
        """Implement gamma transform y = c * x ^ gamma."""
        src = np.asarray(img, dtype=np.float32)
        gamma = max(0.01, float(gamma))
        if mode == "numpy":
            normalized = np.clip(src / 255.0, 0.0, 1.0)
            output = c * np.power(normalized, gamma) * 255.0
        elif mode == "opencv":
            normalized = np.clip(src / 255.0, 0.0, 1.0)
            output = c * np.power(normalized, gamma) * 255.0
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        return ImageProcessor.normalize_to_uint8(output)

    @staticmethod
    def apply(image: np.ndarray, params: TransformParams) -> np.ndarray:
        if params.mode == "linear":
            a = float(params.low)
            b = float(params.high)
            c = 0.0
            d = 255.0
            if params.backend == "numpy":
                return ImageProcessor.linear_transform(image, a, b, c, d, "numpy")
            return ImageProcessor.linear_transform(image, a, b, c, d, "opencv")
        if params.mode == "log":
            if params.backend == "numpy":
                return ImageProcessor.log_transform(image, max(0.01, float(params.log_c)), "numpy")
            return ImageProcessor.log_transform(image, max(0.01, float(params.log_c)), "opencv")
        if params.backend == "numpy":
            return ImageProcessor.gamma_transform(image, 1.0, float(params.gamma), "numpy")
        return ImageProcessor.gamma_transform(image, 1.0, float(params.gamma), "opencv")
    
    @staticmethod
    def histogram_equalization(image: np.ndarray, mode: str) -> np.ndarray:
        #直方图均衡化
        if mode == "numpy":
            hist, bins = np.histogram(image.flatten(), 256, [0, 256])
            cdf = hist.cumsum()
            cdf_normalized = cdf * hist.max() / cdf.max()
            cdf_m = np.ma.masked_equal(cdf, 0)
            cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
            cdf_final = np.ma.filled(cdf_m, 0).astype(np.uint8)
            equalized = cdf_final[image]
            return equalized
        elif mode == "opencv":
            equalized = cv2.equalizeHist(image)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        return equalized
    
    @staticmethod
    def gen_filter_kernel(size: int, filter_type: str) -> np.ndarray:
        #生成滤波器核
        if filter_type == "mean":
            # 均值滤波器
            kernel = np.ones((size, size), np.float32) / (size * size)
        elif filter_type == "gaussian":
            # 高斯滤波器
            kernel = cv2.getGaussianKernel(size, 0)
            kernel = kernel @ kernel.T
        elif filter_type == "laplacian":
            # 拉普拉斯滤波器
            kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], np.float32)
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")
        return kernel
    
    @staticmethod
    def spatial_filter(image: np.ndarray, kernel: np.ndarray, mode: str) -> np.ndarray:
        #用给出的核对图像进行空间滤波
        if mode == "numpy":
            filtered = cv2.filter2D(image, -1, kernel)
        elif mode == "opencv":
            filtered = cv2.filter2D(image, -1, kernel)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        return filtered

    @staticmethod
    def median_filter(image: np.ndarray, size: int, ratio: float, mode: str) -> np.ndarray:
        #百分比滤波器，基于对模板所覆盖像素的灰度值的排序。ratio参数为填充灰度值排序的百分比，50%为中值滤波器，100%为最大值滤波器，0%为最小值滤波器
        if mode == "numpy":
            # 这里可以使用numpy的排序功能来实现统计排序滤波
            pass
        elif mode == "opencv":
            filtered = cv2.medianBlur(image, size)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        return filtered

