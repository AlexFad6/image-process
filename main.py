"""Entry point for the application."""
from __future__ import annotations

import sys
import numpy as np
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow
from utils import install_language, system_language


def main() -> int:
    np.random.seed(0)
    app = QApplication(sys.argv)
    app.setApplicationName("image-process")
    language = QSettings("AlexFad", "image-process").value(
        "language", system_language()
    )
    install_language(app, language)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
