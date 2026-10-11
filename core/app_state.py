from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class TransformParams:
    mode: str = "linear"
    low: float = 0.0
    high: float = 255.0
    log_c: float = 1.0
    gamma: float = 1.0
    backend: str = "opencv"


@dataclass
class AppState:
    source: Optional[np.ndarray] = None
    result: Optional[np.ndarray] = None
    source_path: str = ""
    params: TransformParams = field(default_factory=TransformParams)
    operation_log: list[dict] = field(default_factory=list)

    def reset(self) -> None:
        self.source = None
        self.result = None
        self.source_path = ""
        self.params = TransformParams()
        self.operation_log.clear()
