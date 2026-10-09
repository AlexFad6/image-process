from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional
import cv2
import numpy as np
from .app_state import TransformParams
from .image_processor import ImageProcessor


class Exporter:
    @staticmethod
    def save_image(path: str, image: np.ndarray) -> bool:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        return bool(cv2.imwrite(str(target), image))

    @staticmethod
    def export_sequence(input_paths: list[str], output_dir: str,
                        params: TransformParams,
                        progress: Optional[Callable[[int, int], None]] = None) -> str:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        records = []
        total = len(input_paths)
        for index, name in enumerate(input_paths, 1):
            image = cv2.imread(name, cv2.IMREAD_UNCHANGED)
            if image is None:
                records.append({"input": name, "ok": False, "error": "unreadable"})
                if progress:
                    progress(index, total)
                continue
            target = out / (Path(name).stem + "_gray.png")
            ok = Exporter.save_image(str(target), ImageProcessor.apply(image, params))
            records.append({"input": name, "output": str(target), "ok": ok})
            if progress:
                progress(index, total)
        log = out / "export_log.json"
        log.write_text(json.dumps({"created": datetime.now().isoformat(timespec="seconds"),
                                   "parameters": vars(params), "records": records},
                                  indent=2, ensure_ascii=False), encoding="utf-8")
        return str(log)
