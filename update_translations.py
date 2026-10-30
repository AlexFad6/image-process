from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRANSLATIONS = ROOT / "translations"
SOURCES = [str(path) for path in [ROOT / "main.py", ROOT / "utils.py", *sorted((ROOT / "ui").glob("*.py"))]]


def find_tool(*names: str) -> str:
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    raise RuntimeError(f"Unable to find any of: {', '.join(names)}")


def main() -> None:
    TRANSLATIONS.mkdir(exist_ok=True)
    lupdate = find_tool("pyside6-lupdate", "pylupdate6", "pylupdate")
    lrelease = find_tool("pyside6-lrelease", "lrelease")
    ts_files = [str(TRANSLATIONS / "zh_CN.ts"), str(TRANSLATIONS / "en_US.ts")]
    subprocess.run([lupdate, *SOURCES, "-ts", *ts_files], check=True, cwd=ROOT)
    for ts_file in ts_files:
        subprocess.run([lrelease, ts_file], check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
