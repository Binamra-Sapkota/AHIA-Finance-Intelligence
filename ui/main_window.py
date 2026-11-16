"""Primary PySide6 Desktop Application Window for AHIA-R2 Book Intelligence.

Integrates the navigation sidebar, stacked view container, and header status bar.
CRITICAL RULE: This module never imports PyTorch or neural weights directly.
"""

from typing import Optional
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from ui.components.sidebar import SidebarNav
from ui.styles.theme import DARK_THEME_QSS
from ui.views.chat_view import ChatView
from ui.views.library_view import LibraryView
from ui.views.monitor_view import MonitorView
from ui.views.settings_view import SettingsView
from ui.views.training_view import TrainingView


class MainWindow(QMainWindow):
    """Main application shell coordinating navigation between the five subsystems."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("AHIA-R2 Book Intelligence — Research Demonstration")
        self.resize(1180, 760)
        self.setMinimumSize(960, 640)
        self._setup_ui()

    def _setup_ui(self) -> None:
        # Apply dark theme styling
        self.setStyleSheet(DARK_THEME_QSS)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        root_layout = QHBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # 1. Left Navigation Sidebar
        self.sidebar = SidebarNav(self)
        self.sidebar.nav_changed.connect(self._on_navigation_changed)
        root_layout.addWidget(self.sidebar)

        # Subtle vertical separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color: #253043; background-color: #253043; width: 1px;")
        root_layout.addWidget(sep)

        # 2. Main Content Stack
        content_container = QWidget()
        content_layout = QVBoxLayout(content_container)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # Top Header Bar
        top_header = QFrame()
        top_header.setStyleSheet(
            "background-color: #0E131E; border-bottom: 1px solid #253043; padding: 8px 16px;"
        )
        top_header_layout = QHBoxLayout(top_header)
        top_header_layout.setContentsMargins(12, 6, 12, 6)

        header_title = QLabel("AHIA-R2 Research Engine")
        header_title.setStyleSheet("font-size: 13px; font-weight: 700; color: #E2E8F0; letter-spacing: 0.5px;")

        top_header_layout.addWidget(header_title)
        top_header_layout.addStretch()

        badge_offline = QLabel("● 100% OFFLINE")
        badge_offline.setProperty("class", "badge-offline")

        badge_model = QLabel("15.2M TRANSFORMER (RANDOM INIT)")
        badge_model.setProperty("class", "badge-depth")

        badge_demo = QLabel("DEMONSTRATION SHELL")
        badge_demo.setProperty("class", "badge-demo")

        top_header_layout.addWidget(badge_offline)
        top_header_layout.addWidget(badge_model)
        top_header_layout.addWidget(badge_demo)

        content_layout.addWidget(top_header)

        # Stacked Views Widget
        self.stacked_views = QStackedWidget()

        # Instantiate all 5 views
        self.chat_view = ChatView()
        self.library_view = LibraryView()
        self.training_view = TrainingView()
        self.monitor_view = MonitorView()
        self.settings_view = SettingsView()

        self.stacked_views.addWidget(self.chat_view)       # Index 0: AI Chat
        self.stacked_views.addWidget(self.library_view)    # Index 1: Book Library
        self.stacked_views.addWidget(self.training_view)   # Index 2: Model Training
        self.stacked_views.addWidget(self.monitor_view)    # Index 3: Architecture Monitor
        self.stacked_views.addWidget(self.settings_view)   # Index 4: Settings

        content_layout.addWidget(self.stacked_views, stretch=1)
        root_layout.addWidget(content_container, stretch=1)

        # Status Bar
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet(
            "QStatusBar { background-color: #0B0F17; border-top: 1px solid #1E2837; color: #94A3B8; font-size: 11px; padding: 4px 12px; }"
        )
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage(
            "Ready • Memory: Nominal • Governor Limit: 10.0s / 3 Iterations • Grounding: Verified"
        )

    def _on_navigation_changed(self, index: int) -> None:
        """Switch active stacked view based on sidebar selection."""
        if 0 <= index < self.stacked_views.count():
            self.stacked_views.setCurrentIndex(index)
            view_names = ["AI Chat", "Book Library", "Model Training", "Architecture Monitor", "Settings"]
            self.status_bar.showMessage(f"Active View: {view_names[index]} • Demonstration Mode")
