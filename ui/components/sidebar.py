"""Left navigation sidebar component for AHIA-R2."""

from typing import Callable, List
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class SidebarNav(QWidget):
    """Main navigation sidebar supporting view switching and system health overview."""

    nav_changed = Signal(int)

    NAV_ITEMS = [
        ("AI Chat", "💬", "Ask questions grounded in books"),
        ("Book Library", "📚", "Manage indexed finance books"),
        ("Model Training", "⚡", "Dataset, tokenizer & Transformer"),
        ("Architecture Monitor", "📊", "Adaptive depth, experts & telemetry"),
        ("Settings", "⚙️", "Device, checkpoints & governor limits"),
    ]

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(240)
        self.buttons: List[QPushButton] = []
        self._current_index = 0
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 16, 12, 16)
        layout.setSpacing(6)

        # Brand / App Title
        title_box = QFrame()
        title_box_layout = QVBoxLayout(title_box)
        title_box_layout.setContentsMargins(4, 4, 4, 12)
        title_box_layout.setSpacing(4)

        app_title = QLabel("AHIA-R2")
        app_title.setStyleSheet("font-size: 18px; font-weight: 700; color: #FFFFFF; letter-spacing: 0.5px;")

        app_subtitle = QLabel("Book Intelligence Research")
        app_subtitle.setStyleSheet("font-size: 11px; color: #60A5FA; font-weight: 600; text-transform: uppercase;")

        offline_badge = QLabel("● 100% OFFLINE")
        offline_badge.setStyleSheet(
            "font-size: 10px; color: #34D399; font-weight: 700; background: rgba(16, 185, 129, 0.15); "
            "padding: 2px 6px; border-radius: 4px; border: 1px solid rgba(16, 185, 129, 0.3);"
        )

        title_box_layout.addWidget(app_title)
        title_box_layout.addWidget(app_subtitle)
        title_box_layout.addWidget(offline_badge)
        layout.addWidget(title_box)

        # Divider
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet("color: #253043; margin-bottom: 8px;")
        layout.addWidget(divider)

        # Nav Buttons
        for idx, (label, icon, desc) in enumerate(self.NAV_ITEMS):
            btn = QPushButton(f"{icon}  {label}")
            btn.setProperty("class", "nav-btn")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setToolTip(desc)
            btn.clicked.connect(lambda checked=False, i=idx: self.set_active_index(i))
            self.buttons.append(btn)
            layout.addWidget(btn)

        layout.addStretch()

        # Footer Research Note
        footer_frame = QFrame()
        footer_frame.setProperty("class", "card")
        footer_layout = QVBoxLayout(footer_frame)
        footer_layout.setContentsMargins(8, 8, 8, 8)
        footer_layout.setSpacing(4)

        footer_label = QLabel("15.2M Param Transformer")
        footer_label.setStyleSheet("font-size: 11px; font-weight: 600; color: #94A3B8;")

        footer_detail = QLabel("Random Init • No External APIs")
        footer_detail.setStyleSheet("font-size: 10px; color: #64748B;")

        footer_layout.addWidget(footer_label)
        footer_layout.addWidget(footer_detail)
        layout.addWidget(footer_frame)

        self._update_button_states()

    def set_active_index(self, index: int) -> None:
        """Switch active navigation index and emit signal."""
        if 0 <= index < len(self.buttons) and index != self._current_index:
            self._current_index = index
            self._update_button_states()
            self.nav_changed.emit(index)

    def _update_button_states(self) -> None:
        for idx, btn in enumerate(self.buttons):
            is_active = (idx == self._current_index)
            btn.setProperty("active", "true" if is_active else "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)
