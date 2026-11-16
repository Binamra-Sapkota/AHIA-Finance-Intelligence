"""AHIA-R2 Book Intelligence: Desktop Application Entry Point.

Launches the PySide6 research demonstration shell.
"""

import ctypes
import os
import sys
from pathlib import Path


def _preload_linux_xcb_cursor() -> None:
    """Preload libxcb-cursor.so.0 on Linux platforms.

    Qt 6.5+ on Ubuntu/Debian requires libxcb-cursor.so.0 to load the xcb platform
    plugin. If not present system-wide, we search the local project lib/ and
    virtual environment paths to ensure seamless, zero-config startup.
    """
    if sys.platform != "linux":
        return

    project_root = Path(__file__).resolve().parent
    candidates = [
        "libxcb-cursor.so.0",
        str(project_root / "lib" / "libxcb-cursor.so.0"),
        str(project_root / ".venv" / "lib" / "libxcb-cursor.so.0"),
        "/usr/lib/x86_64-linux-gnu/libxcb-cursor.so.0",
        "/usr/lib64/libxcb-cursor.so.0",
        "/usr/lib/libxcb-cursor.so.0",
    ]

    for candidate in candidates:
        try:
            ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
            return
        except OSError:
            continue


# Preload xcb cursor library before importing PySide6
_preload_linux_xcb_cursor()

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow


def create_application() -> QApplication:
    """Create and configure the PySide6 QApplication instance."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    app.setApplicationName("AHIA-R2 Book Intelligence")
    app.setApplicationDisplayName("AHIA-R2 Book Intelligence")
    app.setOrganizationName("AHIA-R2 Research Team")
    return app


def main() -> int:
    """Launch the desktop application window and enter the event loop."""
    app = create_application()
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
