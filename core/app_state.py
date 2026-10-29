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
class AppState:
    source: Optional[np.ndarray] = None
    gray_source: Optional[np.ndarray] = None
    result: Optional[np.ndarray] = None
    source_path: str = ""
    params: TransformParams = field(default_factory=TransformParams)
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
        self.operation_log.clear()
        self.clear_frequency_data()

    def clear_frequency_data(self) -> None:
        self.frequency_data = None
        self.fft_spectrum_uint8 = None
        self.phase_spectrum_uint8 = None
        self.result_frequency_data = None
        self.result_fft_spectrum_uint8 = None
        self.result_phase_spectrum_uint8 = None
