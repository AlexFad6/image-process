from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QLocale, QTranslator

def resource_path(*parts: str) -> Path:
    """Return a resource path for both source runs and PyInstaller bundles."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base.joinpath(*parts)


def install_language(app: QCoreApplication, locale_name: str) -> bool:
    old_translator = getattr(app, "_image_process_translator", None)
    if old_translator is not None:
        app.removeTranslator(old_translator)

    translator = QTranslator(app)
    candidates = [locale_name, locale_name.split("_", 1)[0], "zh_CN"]
    for candidate in candidates:
        path = resource_path("translations", f"{candidate}.qm")
        if path.is_file() and translator.load(str(path)):
            app.installTranslator(translator)
            app._image_process_translator = translator
            return True

    app._image_process_translator = None
    return False


def system_language() -> str:
    return QLocale.system().name()
