from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class TransformParams:
    mode: str = "linear"
    a: float = 0.0
    b: float = 255.0
    c: float = 0.0
    d: float = 255.0
    gamma: float = 1.0


@dataclass
class SpatialParams:
    filter_type: str = "mean"
    kernel_size: int = 3
    median_ratio: float = 50.0
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0
    usm_amount: float = 1.5
    gaussian_mean: float = 0.0
    gaussian_std: float = 20.0
    salt_pepper_amount: float = 5.0


@dataclass
class FrequencyParams:
    filter_type: str = "ideal"
    cutoff: float = 30.0
    pass_type: str = "lowpass"
    order: int = 2


@dataclass
class GeometryParams:
    transform_type: str = "rotate_cw_90"


@dataclass
class BinarizationParams:
    method: str = "manual"
    threshold: float = 128.0
    block_size: int = 11
    c: float = 2.0


@dataclass
class EdgeParams:
    method: str = "sobel"
    direction: str = "magnitude"
    low_threshold: float = 50.0
    high_threshold: float = 150.0


@dataclass
class AppState:
    source: Optional[np.ndarray] = None
    gray_source: Optional[np.ndarray] = None
    result: Optional[np.ndarray] = None
    source_path: str = ""
    params: TransformParams = field(default_factory=TransformParams)
    spatial_params: SpatialParams = field(default_factory=SpatialParams)
    frequency_params: FrequencyParams = field(default_factory=FrequencyParams)
    geometry_params: GeometryParams = field(default_factory=GeometryParams)
    binarization_params: BinarizationParams = field(default_factory=BinarizationParams)
    edge_params: EdgeParams = field(default_factory=EdgeParams)
    operation_log: list[dict] = field(default_factory=list)
    frequency_data: Optional[np.ndarray] = None
    fft_spectrum_uint8: Optional[np.ndarray] = None
    phase_spectrum_uint8: Optional[np.ndarray] = None
    result_frequency_data: Optional[np.ndarray] = None
    result_fft_spectrum_uint8: Optional[np.ndarray] = None
    result_phase_spectrum_uint8: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.source = None
        self.gray_source = None
        self.result = None
        self.source_path = ""
        self.params = TransformParams()
        self.spatial_params = SpatialParams()
        self.frequency_params = FrequencyParams()
        self.geometry_params = GeometryParams()
        self.binarization_params = BinarizationParams()
        self.edge_params = EdgeParams()
        self.operation_log.clear()
        self.clear_frequency_data()

    def clear_frequency_data(self) -> None:
        self.frequency_data = None
        self.fft_spectrum_uint8 = None
        self.phase_spectrum_uint8 = None
        self.result_frequency_data = None
        self.result_fft_spectrum_uint8 = None
        self.result_phase_spectrum_uint8 = None
