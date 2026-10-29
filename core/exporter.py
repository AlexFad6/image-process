from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from .app_state import TransformParams


class Exporter:
    @staticmethod
    def read_image(path: str) -> Optional[np.ndarray]:
        candidate = str(Path(path))
        try:
            data = np.fromfile(candidate, dtype=np.uint8)
        except OSError:
            return None
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

    @staticmethod
    def save_image(path: str, image: np.ndarray) -> bool:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        suffix = target.suffix.lower() or ".png"
        ok, encoded = cv2.imencode(suffix, image)
        if not ok:
            return False

        try:
            encoded.tofile(str(target))
            return True
        except OSError:
            return False

    @staticmethod
    def export_sequence(
        input_paths: list[str],
        output_dir: str,
        params: TransformParams,
        processor,
        include_suffix: bool = True,
        include_log: bool = True,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> str:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        records: list[dict] = []
        total = len(input_paths)

        for index, src_name in enumerate(input_paths, start=0):
            image = Exporter.read_image(src_name)
            if image is None:
                records.append({
                    "index": index,
                    "input": src_name,
                    "ok": False,
                    "error": "unreadable",
                })
                if progress is not None:
                    progress(index, total)
                continue

            if params.mode == "linear":
                processed = processor.linear_transform(image, params.a, params.b, params.c, params.d)
            elif params.mode == "log":
                processed = processor.log_transform(image, params.c)
            elif params.mode == "gamma":
                processed = processor.gamma_transform(image, params.c, params.gamma)
            else:
                raise ValueError(f"Unsupported transform mode: {params.mode}")
            stem = Path(src_name).stem
            suffix = f"_{params.mode}" if include_suffix else ""
            target_name = f"{index:03d}{suffix}.png"
            target = out / target_name
            ok = Exporter.save_image(str(target), processed)
            records.append({
                "index": index,
                "input": src_name,
                "output": str(target),
                "transform": params.mode,
                "ok": ok,
                "source_stem": stem,
            })

            if progress is not None:
                progress(index + 1, total)

        if include_log:
            log_path = out / "operations.json"
            operations = {
                "created": datetime.now().isoformat(timespec="seconds"),
                "parameters": vars(params),
                "records": records,
            }
            log_path.write_text(json.dumps(operations, ensure_ascii=False, indent=2), encoding="utf-8")
            return str(log_path)

        return ""
