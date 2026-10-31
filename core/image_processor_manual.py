from __future__ import annotations

import numpy as np

class ManualImageProcessor:
    """使用NumPy操作实现的图像处理算法。"""
    """Image processing algorithms implemented with NumPy operations."""

    @staticmethod
    def to_gray(image: np.ndarray | None) -> np.ndarray | None:
        """将图像转换为灰度图像。"""
        """Convert the image to grayscale."""

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
        """将值域线性映射到[0,255]"""
        """Normalize the value range to [0, 255]"""

        values = np.asarray(image, dtype=np.float32)
        low, high = values.min(), values.max()
        if low == high:
            return np.zeros(values.shape, dtype=np.uint8)
        values = (values - low) / (high - low) * 255
        return np.clip(values, 0, 255).astype(np.uint8)

    @staticmethod
    def linear_transform(img: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
        """线性变换，把输入区间 [a, b] 映射到输出区间 [c, d]"""
        """Linear transformation, maps input range [a, b] to output range [c, d]"""
        """y = (x - a) * (d - c) / (b - a) + c"""

        if b <= a:
            return np.zeros_like(img, dtype=np.uint8)

        output = (np.asarray(img, dtype=np.float32) - a) * (d - c) / (b - a) + c
        return np.clip(output, 0, 255).astype(np.uint8)

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
        return table[np.asarray(img)]

    @staticmethod
    def histogram_equalization(image: np.ndarray) -> np.ndarray:
        """直方图均衡化"""
        """Histogram equalization"""

        values = np.asarray(image, dtype=np.uint8)
        hist = np.bincount(values.ravel(), minlength=256)
        cdf = hist.cumsum()
        cdf_min = cdf[cdf > 0].min()
        N = cdf[-1]

        if N == 0 or N == cdf_min:
            return values.copy()

        lut = np.zeros(256, dtype=np.uint8)
        mask = cdf > 0
        vals = (cdf[mask] - cdf_min) * 255.0 / (N - cdf_min)
        lut[mask] = np.clip(np.round(vals), 0, 255).astype(np.uint8)

        return lut[values]

    @staticmethod
    def gen_filter_kernel(size: int, filter_type: str) -> np.ndarray:
        """生成滤波器核"""
        """Generate filter kernel"""
        
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
        """空间滤波"""
        """Spatial filtering"""

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
        result = np.partition(flattened, index, axis=-1)[..., index]
        return np.ascontiguousarray(result)

    @staticmethod
    def bilateral_filter(
        image: np.ndarray, size: int, sigma_color: float, sigma_space: float
    ) -> np.ndarray:
        if size % 2 == 0 or size < 1:
            raise ValueError("Filter size must be a positive odd number")
        if sigma_color <= 0 or sigma_space <= 0:
            raise ValueError("Bilateral filter sigmas must be greater than zero")
        values = np.asarray(image, dtype=np.float32)
        radius = size // 2
        padded = np.pad(values, radius, mode="edge")
        axis = np.arange(-radius, radius + 1, dtype=np.float32)
        spatial = np.exp(-(axis[:, None] ** 2 + axis[None, :] ** 2) / (2 * sigma_space ** 2))
        result = np.zeros_like(values)
        weights = np.zeros_like(values)
        for y in range(size):
            for x in range(size):
                neighbor = padded[y:y + values.shape[0], x:x + values.shape[1]]
                range_weight = np.exp(-((neighbor - values) ** 2) / (2 * sigma_color ** 2))
                weight = spatial[y, x] * range_weight
                result += weight * neighbor
                weights += weight
        return np.clip(result / weights, 0, 255).astype(np.uint8)

    @staticmethod
    def unsharp_mask(image: np.ndarray, size: int, amount: float) -> np.ndarray:
        if size % 2 == 0 or size < 1:
            raise ValueError("Filter size must be a positive odd number")
        if amount < 0:
            raise ValueError("USM amount must not be negative")
        values = np.asarray(image, dtype=np.float32)
        kernel = ManualImageProcessor.gen_filter_kernel(size, "gaussian")
        radius = size // 2
        padded = np.pad(values, radius, mode="edge")
        blurred = np.zeros_like(values)
        for y in range(size):
            for x in range(size):
                blurred += kernel[y, x] * padded[
                    y:y + values.shape[0], x:x + values.shape[1]
                ]
        return np.clip(
            values + amount * (values - blurred),
            0,
            255,
        ).astype(np.uint8)

    @staticmethod
    def gaussian_noise(image: np.ndarray, mean: float, std: float) -> np.ndarray:
        values = np.asarray(image, dtype=np.float32)
        noise = np.random.default_rng(0).normal(mean, std, values.shape)
        return np.ascontiguousarray(np.clip(values + noise, 0, 255).astype(np.uint8))

    @staticmethod
    def salt_pepper_noise(image: np.ndarray, amount: float) -> np.ndarray:
        values = np.asarray(image, dtype=np.uint8).copy()
        probability = np.clip(amount, 0, 100) / 100.0
        rng = np.random.default_rng(0)
        mask = rng.random(values.shape) < probability
        salt = rng.random(values.shape) < 0.5
        values[mask & salt] = 255
        values[mask & ~salt] = 0
        return np.ascontiguousarray(values)

    @staticmethod
    def geometric_transform(image: np.ndarray, transform_type: str) -> np.ndarray:
        """Rotate or flip an image using NumPy operations."""
        if transform_type == "rotate_cw_90":
            return np.rot90(image, k=-1).copy()
        if transform_type == "rotate_ccw_90":
            return np.rot90(image, k=1).copy()
        if transform_type == "rotate_180":
            return np.rot90(image, k=2).copy()
        if transform_type == "flip_horizontal":
            return np.flip(image, axis=1).copy()
        if transform_type == "flip_vertical":
            return np.flip(image, axis=0).copy()
        raise ValueError(f"Unsupported geometric transform: {transform_type}")

    @staticmethod
    def binarize(
        image: np.ndarray, method: str, threshold: float, block_size: int, c: float
    ) -> np.ndarray:
        values = np.asarray(image, dtype=np.uint8)
        if method == "manual":
            return np.where(values > threshold, 255, 0).astype(np.uint8)
        if method == "otsu":
            hist = np.bincount(values.ravel(), minlength=256).astype(np.float64)
            total = values.size
            cumulative_weight = np.cumsum(hist)
            cumulative_mean = np.cumsum(hist * np.arange(256))
            denominator = cumulative_weight * (total - cumulative_weight)
            score = np.zeros(256, dtype=np.float64)
            valid = denominator > 0
            score[valid] = (
                (total * cumulative_mean[valid]
                 - cumulative_weight[valid] * cumulative_mean[-1]) ** 2
                / denominator[valid]
            )
            return np.where(values > int(np.argmax(score)), 255, 0).astype(np.uint8)
        if block_size < 3 or block_size % 2 == 0:
            raise ValueError("Adaptive threshold block size must be a positive odd number")
        radius = block_size // 2
        padded = np.pad(values.astype(np.float32), radius, mode="reflect")
        windows = np.lib.stride_tricks.sliding_window_view(
            padded, (block_size, block_size)
        )
        if method == "adaptive_mean":
            local = windows.mean(axis=(-2, -1))
        elif method == "adaptive_gaussian":
            axis = np.arange(-radius, radius + 1, dtype=np.float32)
            kernel = np.exp(-(axis ** 2) / (2 * (block_size / 6) ** 2))
            kernel /= kernel.sum()
            local = np.einsum("i,j,hwij->hw", kernel, kernel, windows)
        else:
            raise ValueError(f"Unsupported binarization method: {method}")
        return np.where(values > local - c, 255, 0).astype(np.uint8)
