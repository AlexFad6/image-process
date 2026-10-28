from __future__ import annotations

import cv2
import numpy as np

class LibraryImageProcessor:
    """使用OpenCV实现的图像处理算法。"""
    """Image processing algorithms implemented with OpenCV."""

    @staticmethod
    def to_gray(image: np.ndarray | None) -> np.ndarray | None:
        """将图像转换为灰度图像。"""
        """Convert the image to grayscale."""
        
        if image is None or image.ndim == 2:
            return image
        if image.shape[-1] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        if image.shape[-1] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.astype(np.uint8, copy=False)

    @staticmethod
    def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
        """将值域线性映射到[0,255]"""
        """Normalize the value range to [0, 255]"""

        return cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    @staticmethod
    def linear_transform(img: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
        """线性变换，把输入区间 [a, b] 映射到输出区间 [c, d]"""
        """Linear transformation, maps input range [a, b] to output range [c, d]"""
        """y = (x - a) * (d - c) / (b - a) + c"""

        if b <= a:
            return np.zeros_like(img, dtype=np.uint8)

        alpha = (d - c) / (b - a)
        beta  = c - a * alpha

        return cv2.addWeighted(img.astype(np.float32), alpha,
                            np.zeros_like(img, dtype=np.float32), 0.0,
                            beta).astype(np.uint8)

    @staticmethod
    def log_transform(img: np.ndarray, c: float) -> np.ndarray:
        """对数变换"""
        """Logarithmic transformation"""
        """y = c * log(1 + x)"""

        output = c * np.log1p(np.asarray(img, dtype=np.float32)) * 255.0 / np.log(256.0)
        return np.clip(output, 0, 255).astype(np.uint8)

    @staticmethod
    def gamma_transform(img: np.ndarray, c: float, gamma: float) -> np.ndarray:
        """伽马变换"""
        """Gamma transformation"""
        """y = c * x^gamma"""

        i = np.arange(256, dtype=np.float32)
        table = np.clip(c * np.power(i / 255.0, gamma) * 255.0, 0, 255).astype(np.uint8)
        return cv2.LUT(img, table)

    @staticmethod
    def histogram_equalization(image: np.ndarray) -> np.ndarray:
        """直方图均衡化"""
        """Histogram equalization"""

        return cv2.equalizeHist(image)

    @staticmethod
    def gen_filter_kernel(size: int, filter_type: str) -> np.ndarray:
        """生成滤波器核"""
        """Generate filter kernel"""
        
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
        """空间滤波"""
        """Spatial filtering"""

        # 注意 OpenCV 的 filter2D 实际计算的是相关而不是卷积
        # Note that OpenCV's filter2D actually computes correlation rather than convolution
        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def median_filter(image: np.ndarray, size: int, ratio: float) -> np.ndarray:
        if ratio != 50:
            raise ValueError("The library median filter supports ratio=50 only")
        return cv2.medianBlur(image, size)
