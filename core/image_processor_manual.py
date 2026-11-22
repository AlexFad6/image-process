from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np
from scipy.signal import fftconvolve

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
    def generate_psf(
        psf_type: str,
        size: int,
        motion_length: float = 15.0,
        motion_angle: float = 0.0,
        defocus_radius: float = 5.0,
        gaussian_sigma: float = 3.0,
        turbulence_strength: float = 0.0025,
        diffraction_wavelength: float = 550.0,
        diffraction_aperture: float = 2.0,
    ) -> np.ndarray:
        """Generate a normalized point spread function for restoration."""
        normalized = ManualImageProcessor.generate_psf_kernel(
            psf_type, size, motion_length, motion_angle, defocus_radius,
            gaussian_sigma, turbulence_strength, diffraction_wavelength,
            diffraction_aperture,
        )
        peak = float(normalized.max())
        if not np.isfinite(peak) or peak <= 0:
            raise ValueError("Generated PSF has no positive peak")
        return np.ascontiguousarray(
            np.clip(normalized / peak * 255, 0, 255).astype(np.uint8)
        )

    @staticmethod
    def generate_psf_kernel(
        psf_type: str,
        size: int,
        motion_length: float = 15.0,
        motion_angle: float = 0.0,
        defocus_radius: float = 5.0,
        gaussian_sigma: float = 3.0,
        turbulence_strength: float = 0.0025,
        diffraction_wavelength: float = 550.0,
        diffraction_aperture: float = 2.0,
    ) -> np.ndarray:
        """Generate a normalized float PSF kernel for image convolution."""
        if size < 3 or size % 2 == 0:
            raise ValueError("PSF size must be an odd number of at least 3")
        if psf_type not in ("motion", "defocus", "gaussian", "turbulence", "diffraction"):
            raise ValueError(f"Unsupported PSF type: {psf_type}")
        axis = np.arange(size, dtype=np.float32) - (size - 1) / 2
        x, y = np.meshgrid(axis, axis)
        if psf_type == "motion":
            if motion_length <= 0:
                raise ValueError("Motion length must be greater than zero")
            angle = np.deg2rad(motion_angle)
            along = x * np.cos(angle) + y * np.sin(angle)
            across = -x * np.sin(angle) + y * np.cos(angle)
            psf = (np.abs(along) <= motion_length / 2) & (np.abs(across) <= 0.5)
        elif psf_type == "defocus":
            if defocus_radius <= 0:
                raise ValueError("Defocus radius must be greater than zero")
            psf = x * x + y * y <= defocus_radius * defocus_radius
        elif psf_type == "gaussian":
            if gaussian_sigma <= 0:
                raise ValueError("Gaussian sigma must be greater than zero")
            psf = np.exp(-(x * x + y * y) / (2 * gaussian_sigma * gaussian_sigma))
        elif psf_type == "turbulence":
            if turbulence_strength <= 0:
                raise ValueError("Turbulence strength must be greater than zero")
            frequency_radius = np.hypot(x, y) / size
            otf = np.exp(-turbulence_strength * frequency_radius ** (5 / 6))
            psf = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(otf))))
        else:
            if diffraction_wavelength <= 0 or diffraction_aperture <= 0:
                raise ValueError("Diffraction parameters must be greater than zero")
            numerical_radius = diffraction_aperture / diffraction_wavelength * 1000
            radial = np.hypot(x, y) / max(numerical_radius, 1e-6)
            psf = np.sinc(radial) ** 2
        psf = np.maximum(np.asarray(psf, dtype=np.float32), 0)
        total = float(psf.sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError("Generated PSF has no energy")
        normalized = psf / total
        return np.ascontiguousarray(normalized)

    @staticmethod
    def apply_psf(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Convolve an image with a normalized float PSF kernel."""
        values = np.asarray(image, dtype=np.float32)
        if values.ndim not in (2, 3):
            raise ValueError("Image must be grayscale or multi-channel")
        kernel = np.asarray(kernel, dtype=np.float32)
        if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
            raise ValueError("PSF kernel must be a square 2D array")
        radius = kernel.shape[0] // 2
        if radius < 1:
            raise ValueError("PSF kernel must have a positive radius")
        pad_width = ((radius, radius), (radius, radius))
        if values.ndim == 3:
            pad_width += ((0, 0),)
        padded = np.pad(values, pad_width, mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(
            padded, kernel.shape, axis=(0, 1)
        )
        result = np.sum(windows * kernel, axis=(-2, -1))
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def estimate_wiener_noise_ratio(
        image: np.ndarray,
        noise_std: float | None = None,
    ) -> float:
        """Estimate N/S from an input noise standard deviation or local noise variance."""
        values = np.asarray(image, dtype=np.float32)
        if values.ndim == 3:
            values = ManualImageProcessor.to_gray(values).astype(np.float32)
        if values.ndim != 2:
            raise ValueError("Wiener noise estimation requires a grayscale image")

        normalized = np.clip(values / 255.0, 0, 1)
        signal_power = max(float(np.mean(normalized * normalized)), 1e-12)

        if noise_std is not None:
            if noise_std < 0:
                raise ValueError("Noise standard deviation must be non-negative")
            noise_std_normalized = float(noise_std) / 255.0
            noise_power = noise_std_normalized * noise_std_normalized
        else:
            laplacian = cv2.Laplacian(normalized, cv2.CV_32F, ksize=3)
            edge_laplacian = laplacian[np.abs(laplacian) < np.quantile(np.abs(laplacian), 0.50)]
            if edge_laplacian.size:
                noise_std_normalized = float(np.std(edge_laplacian))
            else:
                noise_std_normalized = float(np.std(normalized))
            noise_power = noise_std_normalized * noise_std_normalized

        return float(np.clip(noise_power / signal_power, 1e-12, 1.0))

    @staticmethod
    def wiener_filter(
        image: np.ndarray,
        kernel: np.ndarray,
        noise_ratio: float = 0.01,
    ) -> np.ndarray:
        """Use a frequency-domain Wiener filter to restore a grayscale image."""
        values = np.asarray(image, dtype=np.float32)
        if values.ndim == 3:
            values = ManualImageProcessor.to_gray(values).astype(np.float32)
        if values.ndim != 2:
            raise ValueError("Wiener filtering requires a grayscale image")

        psf = np.asarray(kernel, dtype=np.float32)
        if psf.ndim != 2 or psf.shape[0] != psf.shape[1]:
            raise ValueError("PSF kernel must be a square 2D array")
        psf = np.maximum(psf, 0)
        psf /= np.maximum(float(psf.sum()), 1e-12)
        if noise_ratio <= 0:
            raise ValueError("Wiener noise ratio must be positive")

        height, width = values.shape
        kh, kw = psf.shape
        kh = min(kh, height)
        kw = min(kw, width)
        if kh % 2 == 0:
            kh -= 1
        if kw % 2 == 0:
            kw -= 1
        kh = max(kh, 1)
        kw = max(kw, 1)
        source_y = (psf.shape[0] - kh) // 2
        source_x = (psf.shape[1] - kw) // 2
        psf = psf[source_y:source_y + kh, source_x:source_x + kw]
        psf /= np.maximum(float(psf.sum()), 1e-12)

        psf_padded = np.zeros((height, width), dtype=np.float32)
        start_y = (height - kh) // 2
        start_x = (width - kw) // 2
        psf_padded[start_y:start_y + kh, start_x:start_x + kw] = psf

        psf_shifted = np.fft.fftshift(psf_padded)
        image_fft = np.fft.fft2(values)
        psf_fft = np.fft.fft2(psf_shifted)
        psf_conj = np.conj(psf_fft)
        restored_fft = psf_conj / (np.abs(psf_fft) ** 2 + noise_ratio) * image_fft
        restored = np.real(np.fft.ifft2(restored_fft))
        return np.clip(restored, 0, 255).astype(np.uint8)

    @staticmethod
    def estimate_psf_by_normalized_sparsity(
        image: np.ndarray,
        initial_psf: np.ndarray,
        sparsity_weight: float = 0.02,
        iterations: int = 8,
    ) -> np.ndarray:
        """Refine a PSF by minimizing normalized gradient sparsity."""
        values = np.asarray(image, dtype=np.float32)
        if values.ndim == 3:
            values = ManualImageProcessor.to_gray(values).astype(np.float32)
        psf = np.maximum(np.asarray(initial_psf, dtype=np.float32), 0)
        psf /= max(float(psf.sum()), 1e-12)

        observed = np.clip(values / 255.0, 0, 1)
        gx = np.hypot(
            np.diff(observed, axis=1, prepend=observed[:, :1]),
            np.diff(observed, axis=0, prepend=observed[:1, :]),
        )
        edge_mask = gx > np.quantile(gx, 0.55)
        estimate = observed.copy()

        for _ in range(max(int(iterations), 1)):
            blurred = cv2.filter2D(
                estimate, cv2.CV_32F, psf, borderType=cv2.BORDER_REFLECT
            )
            ratio = observed / np.maximum(blurred, 1e-12)
            updated = estimate * cv2.filter2D(
                ratio,
                cv2.CV_32F,
                psf[::-1, ::-1],
                borderType=cv2.BORDER_REFLECT,
            )
            estimate = np.clip(updated, 0, 1)

            estimate_blurred = cv2.filter2D(
                estimate, cv2.CV_32F, psf, borderType=cv2.BORDER_REFLECT
            )
            ratio = observed / np.maximum(estimate_blurred, 1e-12)
            correlation = fftconvolve(ratio * edge_mask, estimate[::-1, ::-1], mode="full")
            sensitivity = fftconvolve(
                edge_mask.astype(np.float32), estimate[::-1, ::-1], mode="full"
            )
            radius = psf.shape[0] // 2
            center_y = estimate.shape[0] - 1
            center_x = estimate.shape[1] - 1
            updated_psf = correlation[
                center_y - radius:center_y + radius + 1,
                center_x - radius:center_x + radius + 1,
            ]
            sensitivity = sensitivity[
                center_y - radius:center_y + radius + 1,
                center_x - radius:center_x + radius + 1,
            ]
            updated_psf = np.maximum(updated_psf / np.maximum(sensitivity, 1e-12), 1e-8)
            updated_psf *= psf
            weak = updated_psf < np.quantile(updated_psf, 0.25)
            updated_psf[weak] *= max(1.0 - sparsity_weight, 0.0)
            updated_psf /= max(float(updated_psf.sum()), 1e-12)
            psf = (1.0 - sparsity_weight) * psf + sparsity_weight * updated_psf
            psf = np.maximum(psf, 1e-8)
            psf /= psf.sum()
        return np.ascontiguousarray(psf)

    @staticmethod
    def estimate_motion_psf(
        image: np.ndarray,
        size: int = 21,
    ) -> tuple[np.ndarray, float, float]:
        """Estimate a motion PSF by searching length/angle candidates."""
        values = np.asarray(image, dtype=np.float32)
        if values.ndim == 3:
            values = ManualImageProcessor.to_gray(values).astype(np.float32)
        if values.ndim != 2:
            raise ValueError("Motion PSF estimation requires a grayscale image")

        if size < 3:
            size = 3
        if size % 2 == 0:
            size -= 1

        normalized = np.clip(values / 255.0, 0, 1)
        sobel_x = cv2.Sobel(normalized, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(normalized, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.hypot(sobel_x, sobel_y)
        gradient_threshold = np.quantile(gradient, 0.75)
        edge_mask = gradient >= gradient_threshold

        best_score = -np.inf
        best_psf = None
        best_length = 0.0
        best_angle = 0.0

        gradient_orientation = np.rad2deg(np.arctan2(sobel_y, sobel_x)) % 180.0
        orientation_hist, orientation_bins = np.histogram(
            gradient_orientation,
            bins=36,
            range=(0, 180),
            weights=gradient,
        )
        dominant_gradient_angle = float(
            (orientation_bins[np.argmax(orientation_hist)]
             + orientation_bins[np.argmax(orientation_hist) + 1]) / 2.0
        )
        candidate_motion_angle = (dominant_gradient_angle + 90.0) % 180.0
        if candidate_motion_angle > 90.0:
            candidate_motion_angle -= 180.0

        candidate_angles = np.unique(np.clip(
            candidate_motion_angle + np.arange(-15.0, 16.0, 5.0),
            -90.0,
            90.0,
        ))

        for length in np.arange(3.0, size + 1.0, 1.0):
            for angle in candidate_angles:
                psf = ManualImageProcessor.generate_psf_kernel(
                    "motion",
                    size,
                    motion_length=float(length),
                    motion_angle=float(angle),
                )
                blurred = cv2.filter2D(
                    normalized,
                    cv2.CV_32F,
                    psf,
                    borderType=cv2.BORDER_REFLECT,
                )

                residual = normalized - blurred
                residual_score = -float(np.mean(residual[edge_mask] ** 2))

                kernel_angle = float(angle)
                kernel_direction = np.deg2rad(kernel_angle)
                axis_x = np.cos(kernel_direction)
                axis_y = np.sin(kernel_direction)
                kernel_alignment = abs(
                    float(np.cos(kernel_direction - np.deg2rad(candidate_motion_angle)))
                )

                kernel_mass = float(np.sum(psf > np.max(psf) * 0.1))
                length_prior = 0.002 * float(length)
                score = residual_score + length_prior + 0.001 * kernel_alignment

                if score > best_score:
                    best_score = score
                    best_psf = psf
                    best_length = float(length)
                    best_angle = float(angle)

        return np.ascontiguousarray(best_psf), best_length, best_angle

    @staticmethod
    def threshold_psf(psf: np.ndarray, threshold_percent: float = 7.0) -> np.ndarray:
        """MATLAB-style dynamic threshold followed by normalization."""
        psf = np.maximum(np.asarray(psf, dtype=np.float32), 0)
        if psf.ndim != 2 or psf.size == 0:
            raise ValueError("PSF must be a non-empty 2D array")
        threshold_percent = float(np.clip(threshold_percent, 1.0, 100.0))
        threshold = float(psf.max()) / threshold_percent
        psf[psf < threshold] = 0
        total = float(psf.sum())
        if not np.isfinite(total) or total <= 1e-12:
            psf[:] = 0
            psf[psf.shape[0] // 2, psf.shape[1] // 2] = 1
        else:
            psf /= total
        return np.ascontiguousarray(psf)

    @staticmethod
    def edgetaper(image: np.ndarray, psf: np.ndarray) -> np.ndarray:
        """Reduce FFT/RL boundary artifacts by tapering the image edges."""
        values = np.asarray(image, dtype=np.float32)
        psf = np.asarray(psf, dtype=np.float32)
        if psf.ndim != 2 or psf.shape[0] != psf.shape[1]:
            raise ValueError("PSF must be a square 2D array")
        taper = max(1, psf.shape[0] // 2)
        tapered = values.copy()
        padded = cv2.copyMakeBorder(
            values,
            taper,
            taper,
            taper,
            taper,
            borderType=cv2.BORDER_REFLECT,
        )
        center = padded[taper:taper + values.shape[0], taper:taper + values.shape[1]]
        return np.ascontiguousarray(center)

    @staticmethod
    def match_histogram(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Match grayscale output histogram to the observed reference."""
        source = np.asarray(image, dtype=np.uint8)
        target = np.asarray(reference, dtype=np.uint8)
        if source.ndim == 3:
            source = ManualImageProcessor.to_gray(source)
        if target.ndim == 3:
            target = ManualImageProcessor.to_gray(target)
        source_hist = np.bincount(source.ravel(), minlength=256).astype(np.float64)
        target_hist = np.bincount(target.ravel(), minlength=256).astype(np.float64)
        source_cdf = np.cumsum(source_hist)
        target_cdf = np.cumsum(target_hist)
        source_cdf /= max(source_cdf[-1], 1)
        target_cdf /= max(target_cdf[-1], 1)
        mapping = np.interp(source_cdf, target_cdf, np.arange(256), left=0, right=255)
        matched = mapping[source]
        return np.clip(matched, 0, 255).astype(np.uint8)

    @staticmethod
    def blind_deconvolution(
        image: np.ndarray,
        initial_psf: np.ndarray,
        iterations: int = 30,
        update_interval: int = 5,
        epsilon: float = 1e-8,
        progress_callback: Callable[[np.ndarray, int], None] | None = None,
        image_damping: float = 1.0,
        psf_damping: float = 0.05,
        tv_weight: float = 0.02,
        l0_weight: float = 0.02,
        psf_support: int = 0,
        multiscale_levels: int = 1,
        sparsity_weight: float = 0.02,
        prior_weight: float = 0.35,
        update_psf: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Stable fixed-PSF Richardson-Lucy restoration after blind kernel search."""
        values = np.asarray(image, dtype=np.float32)
        if values.ndim == 3:
            values = ManualImageProcessor.to_gray(values)
        if values.ndim != 2:
            raise ValueError("Blind deconvolution requires a grayscale image")
        if iterations < 1 or update_interval < 1 or epsilon <= 0:
            raise ValueError("Blind deconvolution parameters are invalid")
        psf = np.asarray(initial_psf, dtype=np.float32).copy()
        if psf.ndim != 2 or psf.shape[0] != psf.shape[1] or psf.shape[0] % 2 == 0:
            raise ValueError("Initial PSF must be an odd square array")
        psf = np.maximum(psf, 0)
        total = float(psf.sum())
        if not np.isfinite(total) or total <= epsilon:
            raise ValueError("Initial PSF has no energy")
        psf /= total
        psf = ManualImageProcessor.threshold_psf(psf, threshold_percent=7.0)
        observed = np.clip(values / 255.0, 0, 1)

        def preview_psf(current_psf: np.ndarray) -> np.ndarray:
            peak = float(current_psf.max())
            if not np.isfinite(peak) or peak <= epsilon:
                raise ValueError("Estimated PSF has no positive peak")
            return np.ascontiguousarray(
                np.clip(current_psf / peak * 255, 0, 255).astype(np.uint8)
            )

        if progress_callback is not None:
            progress_callback(preview_psf(psf), 0)

        observed_full = np.clip(values / 255.0, 0, 1)
        tapered = ManualImageProcessor.edgetaper(values, psf)
        observed = np.clip(tapered / 255.0, 0, 1)
        estimate = observed.copy()
        psf_flip = psf[::-1, ::-1]

        for step in range(1, iterations + 1):
            estimate_blurred = np.maximum(
                cv2.filter2D(
                    estimate,
                    cv2.CV_32F,
                    psf,
                    borderType=cv2.BORDER_REFLECT,
                ),
                epsilon,
            )
            ratio = observed / estimate_blurred
            estimate *= cv2.filter2D(
                ratio,
                cv2.CV_32F,
                psf_flip,
                borderType=cv2.BORDER_REFLECT,
            )
            estimate = np.clip(estimate, 0, 1)

            gy = np.abs(np.diff(estimate, axis=0, prepend=estimate[:1, :]))
            gx = np.abs(np.diff(estimate, axis=1, prepend=estimate[:, :1]))
            gradient = np.maximum(gx, gy)
            threshold = np.quantile(gradient, 0.75)
            strong_edges = gradient >= threshold
            weak_edges = ~strong_edges
            estimate = np.where(weak_edges, estimate * 0.999, estimate)
            estimate = np.clip(estimate, 0, 1)

            if step % update_interval == 0:
                if progress_callback is not None:
                    progress_callback(preview_psf(psf), step)

        restored_full = np.clip(estimate * 255, 0, 255).astype(np.uint8)
        restored_full = ManualImageProcessor.match_histogram(
            restored_full,
            np.clip(values, 0, 255).astype(np.uint8),
        )
        preview = preview_psf(psf)
        return np.ascontiguousarray(restored_full), np.ascontiguousarray(preview)
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
    def geometric_mean_filter(image: np.ndarray, size: int) -> np.ndarray:
        """使用几何平均值执行几何均值滤波。"""
        """Apply geometric mean filtering."""

        values = np.asarray(image, dtype=np.uint8)
        if size % 2 == 0 or size < 1:
            raise ValueError("Filter size must be a positive odd number")

        def filter_channel(channel: np.ndarray) -> np.ndarray:
            radius = size // 2
            padded = np.pad(channel, radius, mode="edge")
            windows = np.lib.stride_tricks.sliding_window_view(
                padded, (size, size)
            )
            flattened = windows.reshape(channel.shape[0], channel.shape[1], size * size)
            # Clip zeros to one before taking logs so dark pixels remain valid.
            result = np.exp(
                np.mean(np.log(np.maximum(flattened, 1)), axis=-1)
            )
            return np.clip(result, 0, 255).astype(np.uint8)

        if values.ndim == 2:
            return np.ascontiguousarray(filter_channel(values))
        if values.ndim == 3:
            channels = [
                filter_channel(values[..., channel])
                for channel in range(values.shape[-1])
            ]
            return np.ascontiguousarray(np.stack(channels, axis=-1))
        raise ValueError("Image must be a 2D grayscale or 3D multi-channel array")

    @staticmethod
    def adaptive_median_filter(image: np.ndarray, max_size: int) -> np.ndarray:
        """使用逐级扩大的窗口执行自适应中值滤波。"""
        """Apply adaptive median filtering with progressively larger windows."""

        values = np.asarray(image, dtype=np.uint8)
        if max_size % 2 == 0 or max_size < 3:
            raise ValueError("Maximum filter size must be an odd number of at least 3")

        def filter_channel(channel: np.ndarray) -> np.ndarray:
            height, width = channel.shape
            max_radius = max_size // 2
            padded = np.pad(channel, max_radius, mode="edge")
            result = np.empty_like(channel)
            unresolved = np.ones((height, width), dtype=bool)

            for size in range(3, max_size + 1, 2):
                radius = size // 2
                y_start = max_radius - radius
                x_start = max_radius - radius
                window_area = padded[
                    y_start : y_start + height + size - 1,
                    x_start : x_start + width + size - 1,
                ]
                windows = np.lib.stride_tricks.sliding_window_view(
                    window_area, (size, size)
                )
                flattened = windows.reshape(height, width, size * size)
                z_min = flattened.min(axis=-1)
                z_max = flattened.max(axis=-1)
                z_median = np.median(flattened, axis=-1).astype(np.uint8)
                stage_a = (z_median > z_min) & (z_median < z_max)

                if size == max_size:
                    result[unresolved] = np.where(
                        (channel > z_min) & (channel < z_max),
                        channel,
                        z_median,
                    )[unresolved]
                    unresolved &= ~stage_a
                    break

                resolved = unresolved & stage_a
                result[resolved] = np.where(
                    (channel > z_min) & (channel < z_max),
                    channel,
                    z_median,
                )[resolved]
                unresolved &= ~stage_a

            return result

        if values.ndim == 2:
            return np.ascontiguousarray(filter_channel(values))
        if values.ndim == 3:
            channels = [
                filter_channel(values[..., channel])
                for channel in range(values.shape[-1])
            ]
            return np.ascontiguousarray(np.stack(channels, axis=-1))
        raise ValueError("Image must be a 2D grayscale or 3D multi-channel array")

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
    def gaussian_noise(
        image: np.ndarray, mean: float, std: float, seed: int = 0
    ) -> np.ndarray:
        values = np.asarray(image, dtype=np.float32)
        noise = np.random.default_rng(seed).normal(mean, std, values.shape)
        return np.ascontiguousarray(np.clip(values + noise, 0, 255).astype(np.uint8))

    @staticmethod
    def salt_pepper_noise(
        image: np.ndarray, amount: float, seed: int = 0
    ) -> np.ndarray:
        values = np.asarray(image, dtype=np.uint8).copy()
        probability = np.clip(amount, 0, 100) / 100.0
        rng = np.random.default_rng(seed)
        mask = rng.random(values.shape) < probability
        salt = rng.random(values.shape) < 0.5
        values[mask & salt] = 255
        values[mask & ~salt] = 0
        return np.ascontiguousarray(values)

    @staticmethod
    def uniform_noise(
        image: np.ndarray, low: float = -20.0, high: float = 20.0, seed: int = 0
    ) -> np.ndarray:
        if low > high:
            raise ValueError("Uniform noise low bound must not exceed high bound")
        values = np.asarray(image, dtype=np.float32)
        noise = np.random.default_rng(seed).uniform(low, high, values.shape)
        return np.ascontiguousarray(np.clip(values + noise, 0, 255).astype(np.uint8))

    @staticmethod
    def periodic_noise(
        image: np.ndarray,
        angle: float = 0.0,
        freq: float = 0.05,
        amplitude: float = 20.0,
        phase: float = 0.0,
    ) -> np.ndarray:
        """Add a two-dimensional sinusoidal stripe pattern to an image.

        ``angle`` and ``phase`` are in radians, while ``freq`` is measured in
        cycles per pixel. The angle controls the stripe direction.
        """
        if freq < 0:
            raise ValueError("Periodic noise frequency must not be negative")
        if amplitude < 0:
            raise ValueError("Periodic noise amplitude must not be negative")

        values = np.asarray(image, dtype=np.float32)
        height, width = values.shape[:2]
        y, x = np.indices((height, width), dtype=np.float32)
        projected = x * np.cos(angle) + y * np.sin(angle)
        pattern = amplitude * np.sin(2.0 * np.pi * freq * projected + phase)
        if values.ndim == 3:
            pattern = pattern[..., np.newaxis]
        return np.ascontiguousarray(
            np.clip(values + pattern, 0, 255).astype(np.uint8)
        )

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

    @staticmethod
    def edge_detect(
        image: np.ndarray,
        method: str,
        direction: str,
        low_threshold: float,
        high_threshold: float,
    ) -> np.ndarray:
        values = np.asarray(image, dtype=np.uint8)
        if method == "sobel":
            padded = np.pad(values.astype(np.float32), 1, mode="edge")
            sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
            sobel_y = sobel_x.T
            x = sum(
                sobel_x[row, col] * padded[row:row + values.shape[0], col:col + values.shape[1]]
                for row in range(3)
                for col in range(3)
            )
            y = sum(
                sobel_y[row, col] * padded[row:row + values.shape[0], col:col + values.shape[1]]
                for row in range(3)
                for col in range(3)
            )
            if direction == "x":
                result = np.abs(x)
            elif direction == "y":
                result = np.abs(y)
            elif direction == "magnitude":
                result = np.hypot(x, y)
            else:
                raise ValueError(f"Unsupported Sobel direction: {direction}")
            return ManualImageProcessor.normalize_to_uint8(result)
        if method == "laplacian":
            kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
            padded = np.pad(values.astype(np.float32), 1, mode="edge")
            result = sum(
                kernel[row, col] * padded[row:row + values.shape[0], col:col + values.shape[1]]
                for row in range(3)
                for col in range(3)
            )
            return ManualImageProcessor.normalize_to_uint8(np.abs(result))
        if method == "canny":
            if low_threshold < 0 or high_threshold <= low_threshold:
                raise ValueError("Canny thresholds must satisfy 0 <= low < high")
            blurred = ManualImageProcessor.spatial_filter(
                values, ManualImageProcessor.gen_filter_kernel(3, "gaussian")
            )
            return ManualImageProcessor.binarize(
                ManualImageProcessor.edge_detect(
                    blurred, "sobel", "magnitude", low_threshold, high_threshold
                ),
                "manual",
                high_threshold,
                11,
                0,
            )
        raise ValueError(f"Unsupported edge detection method: {method}")
