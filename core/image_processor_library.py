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

        if b == a:
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
        if size % 2 == 0 or size < 1:
            raise ValueError("Filter size must be a positive odd number")
        if ratio == 50:
            return cv2.medianBlur(image, size)
        values = np.asarray(image, dtype=np.uint8)
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
        return cv2.bilateralFilter(image, size, sigma_color, sigma_space)

    @staticmethod
    def unsharp_mask(image: np.ndarray, size: int, amount: float) -> np.ndarray:
        if size % 2 == 0 or size < 1:
            raise ValueError("Filter size must be a positive odd number")
        if amount < 0:
            raise ValueError("USM amount must not be negative")
        blurred = cv2.GaussianBlur(image, (size, size), 0)
        return np.clip(
            image.astype(np.float32) + amount * (image.astype(np.float32) - blurred),
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
        """Rotate or flip an image using the requested geometric transform."""
        transforms = {
            "rotate_cw_90": cv2.ROTATE_90_CLOCKWISE,
            "rotate_ccw_90": cv2.ROTATE_90_COUNTERCLOCKWISE,
            "rotate_180": cv2.ROTATE_180,
        }
        if transform_type in transforms:
            return cv2.rotate(image, transforms[transform_type])
        if transform_type == "flip_horizontal":
            return cv2.flip(image, 1)
        if transform_type == "flip_vertical":
            return cv2.flip(image, 0)
        raise ValueError(f"Unsupported geometric transform: {transform_type}")

    @staticmethod
    def binarize(
        image: np.ndarray, method: str, threshold: float, block_size: int, c: float
    ) -> np.ndarray:
        values = np.asarray(image, dtype=np.uint8)
        if method == "manual":
            _, result = cv2.threshold(values, threshold, 255, cv2.THRESH_BINARY)
            return result
        if method == "otsu":
            _, result = cv2.threshold(values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return result
        if block_size < 3 or block_size % 2 == 0:
            raise ValueError("Adaptive threshold block size must be a positive odd number")
        adaptive_method = (
            cv2.ADAPTIVE_THRESH_MEAN_C
            if method == "adaptive_mean"
            else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            if method == "adaptive_gaussian"
            else None
        )
        if adaptive_method is None:
            raise ValueError(f"Unsupported binarization method: {method}")
        return cv2.adaptiveThreshold(
            values, 255, adaptive_method, cv2.THRESH_BINARY, block_size, c
        )

    @staticmethod
    def compute_fft(gray_uint8: np.ndarray) -> np.ndarray:
        """Return the centered OpenCV complex spectrum as float32 HxWx2 data."""
        gray_float32 = np.asarray(gray_uint8, dtype=np.float32)
        dft = cv2.dft(gray_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
        return np.fft.fftshift(dft, axes=(0, 1)).astype(np.float32, copy=False)

    @staticmethod
    def compute_fft_display_spectrum(dft_shift: np.ndarray) -> np.ndarray:
        """Return a uint8 magnitude spectrum intended only for display."""
        magnitude = cv2.magnitude(dft_shift[..., 0], dft_shift[..., 1])
        spectrum = 20.0 * np.log1p(magnitude)
        return LibraryImageProcessor.normalize_to_uint8(spectrum)

    @staticmethod
    def compute_fft_phase_spectrum(dft_shift: np.ndarray) -> np.ndarray:
        """Return a uint8 phase spectrum intended only for display."""
        phase = np.arctan2(dft_shift[..., 1], dft_shift[..., 0])
        phase = (phase + np.pi) * (255.0 / (2.0 * np.pi))
        return np.clip(phase, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_frequency_filter(
        dft_shift: np.ndarray, filter_type: str, cutoff: float
    ) -> np.ndarray:
        """Apply an ideal, Gaussian, or Butterworth low-pass transfer function."""
        if cutoff <= 0:
            raise ValueError("Frequency cutoff must be greater than zero")

        height, width = dft_shift.shape[:2]
        y, x = np.ogrid[:height, :width]
        distance = np.sqrt((x - width / 2) ** 2 + (y - height / 2) ** 2)
        if filter_type == "ideal":
            transfer = (distance <= cutoff).astype(np.float32)
        elif filter_type == "gaussian":
            transfer = np.exp(-(distance ** 2) / (2 * cutoff ** 2)).astype(np.float32)
        elif filter_type == "butterworth":
            order = 2
            transfer = (1.0 / (1.0 + (distance / cutoff) ** (2 * order))).astype(np.float32)
        else:
            raise ValueError(f"Unsupported frequency filter type: {filter_type}")

        transfer_complex = np.repeat(transfer[..., np.newaxis], 2, axis=2)
        transfer_complex[..., 1] = 0.0
        return cv2.mulSpectrums(dft_shift, transfer_complex, flags=0)

    @staticmethod
    def compute_ifft(dft_shift: np.ndarray, component: str = "full") -> np.ndarray:
        """Return a normalized image reconstructed from a selected spectrum component."""
        if component not in ("full", "magnitude", "phase"):
            raise ValueError(f"Unsupported spectrum component: {component}")
        if component != "full":
            real = dft_shift[..., 0]
            imaginary = dft_shift[..., 1]
            magnitude = cv2.magnitude(real, imaginary)
            if component == "magnitude":
                dft_shift = np.stack((magnitude, np.zeros_like(magnitude)), axis=-1)
            else:
                phase = np.arctan2(imaginary, real)
                dft_shift = np.stack((np.cos(phase), np.sin(phase)), axis=-1)
        unshifted = np.fft.ifftshift(dft_shift, axes=(0, 1))
        spatial = cv2.idft(unshifted, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        return LibraryImageProcessor.normalize_to_uint8(np.abs(spatial))

    