"""AHIA-R2 Book Intelligence: Desktop Application Entry Point.

Launches the PySide6 research demonstration shell.
"""

import os
import sys
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow


def create_application() -> QApplication:
    """Create and configure the PySide6 QApplication instance."""
    # Enable high-DPI scaling attributes if supported
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
