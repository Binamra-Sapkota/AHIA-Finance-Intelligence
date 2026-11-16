"""Modern dark theme stylesheet and color constants for AHIA-R2 Book Intelligence.

Technical research aesthetic with obsidian backgrounds, subtle borders,
electric blue accents, and emerald grounding indicators.
"""


class ThemeColors:
    """Design token palette for the AHIA-R2 UI."""

    BG_MAIN = "#0B0F17"
    BG_SIDEBAR = "#0F141F"
    BG_CARD = "#151B26"
    BG_CARD_HOVER = "#1C2433"
    BG_INPUT = "#0D121D"

    BORDER = "#253043"
    BORDER_LIGHT = "#334155"
    BORDER_ACCENT = "#3B82F6"

    TEXT_PRIMARY = "#F3F4F6"
    TEXT_SECONDARY = "#94A3B8"
    TEXT_MUTED = "#64748B"
    TEXT_ACCENT = "#60A5FA"

    ACCENT_BLUE = "#3B82F6"
    ACCENT_BLUE_HOVER = "#2563EB"
    ACCENT_EMERALD = "#10B981"
    ACCENT_AMBER = "#F59E0B"
    ACCENT_ROSE = "#EF4444"
    ACCENT_PURPLE = "#8B5CF6"


DARK_THEME_QSS = """
/* Global Window & Fonts */
QWidget {
    background-color: #0B0F17;
    color: #F3F4F6;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 13px;
    selection-background-color: #3B82F6;
    selection-color: #FFFFFF;
}

/* Scrollbars */
QScrollBar:vertical {
    border: none;
    background: #0B0F17;
    width: 8px;
    margin: 0px 0px 0px 0px;
}
QScrollBar::handle:vertical {
    background: #273244;
    min-height: 24px;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover {
    background: #3B82F6;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    border: none;
    background: #0B0F17;
    height: 8px;
}
QScrollBar::handle:horizontal {
    background: #273244;
    min-width: 24px;
    border-radius: 4px;
}
QScrollBar::handle:horizontal:hover {
    background: #3B82F6;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

/* Card Panels */
QFrame[class="card"] {
    background-color: #151B26;
    border: 1px solid #253043;
    border-radius: 8px;
    padding: 12px;
}

QFrame[class="card-elevated"] {
    background-color: #1A2230;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 14px;
}

/* Buttons */
QPushButton {
    background-color: #1E2738;
    color: #F3F4F6;
    border: 1px solid #2B374E;
    border-radius: 6px;
    padding: 7px 14px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #27344B;
    border-color: #3B82F6;
}
QPushButton:pressed {
    background-color: #151C28;
}
QPushButton:disabled {
    background-color: #131822;
    color: #475569;
    border-color: #1C2433;
}

QPushButton[class="primary"] {
    background-color: #2563EB;
    color: #FFFFFF;
    border: 1px solid #3B82F6;
    font-weight: 600;
}
QPushButton[class="primary"]:hover {
    background-color: #1D4ED8;
    border-color: #60A5FA;
}
QPushButton[class="primary"]:pressed {
    background-color: #1E40AF;
}

QPushButton[class="danger"] {
    background-color: #991B1B;
    color: #FEE2E2;
    border: 1px solid #DC2626;
}
QPushButton[class="danger"]:hover {
    background-color: #B91C1C;
}

/* Navigation Sidebar Buttons */
QPushButton[class="nav-btn"] {
    background-color: transparent;
    color: #94A3B8;
    border: none;
    border-radius: 6px;
    padding: 10px 14px;
    text-align: left;
    font-size: 13px;
    font-weight: 500;
}
QPushButton[class="nav-btn"]:hover {
    background-color: #161F2E;
    color: #F3F4F6;
}
QPushButton[class="nav-btn"][active="true"] {
    background-color: #1E293B;
    color: #60A5FA;
    font-weight: 600;
    border-left: 3px solid #3B82F6;
}

/* Inputs & Text Edits */
QLineEdit, QTextEdit, QPlainTextEdit {
    background-color: #0D121D;
    color: #F3F4F6;
    border: 1px solid #253043;
    border-radius: 6px;
    padding: 8px 10px;
}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
    border: 1px solid #3B82F6;
    background-color: #0F1624;
}

/* ComboBox & SpinBox */
QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #0D121D;
    color: #F3F4F6;
    border: 1px solid #253043;
    border-radius: 6px;
    padding: 6px 10px;
    min-height: 20px;
}
QComboBox:focus, QSpinBox:focus {
    border: 1px solid #3B82F6;
}
QComboBox::drop-down {
    border: none;
    width: 24px;
}
QComboBox QAbstractItemView {
    background-color: #151B26;
    color: #F3F4F6;
    border: 1px solid #253043;
    selection-background-color: #2563EB;
}

/* Tables & Lists */
QTableWidget, QTableView, QListWidget {
    background-color: #111622;
    border: 1px solid #253043;
    border-radius: 6px;
    gridline-color: #1F2837;
    color: #F3F4F6;
}
QHeaderView::section {
    background-color: #151B26;
    color: #94A3B8;
    padding: 6px 10px;
    border: none;
    border-bottom: 1px solid #253043;
    font-weight: 600;
    font-size: 12px;
}
QTableWidget::item:selected, QListWidget::item:selected {
    background-color: #1E2A3F;
    color: #FFFFFF;
}

/* Status Badges */
QLabel[class="badge-demo"] {
    background-color: rgba(245, 158, 11, 0.15);
    color: #FBBF24;
    border: 1px solid rgba(245, 158, 11, 0.35);
    border-radius: 4px;
    padding: 3px 8px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
}

QLabel[class="badge-offline"] {
    background-color: rgba(16, 185, 129, 0.15);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.35);
    border-radius: 4px;
    padding: 3px 8px;
    font-size: 11px;
    font-weight: 600;
}

QLabel[class="badge-depth"] {
    background-color: rgba(59, 130, 246, 0.15);
    color: #60A5FA;
    border: 1px solid rgba(59, 130, 246, 0.35);
    border-radius: 4px;
    padding: 3px 8px;
    font-size: 11px;
    font-weight: 600;
}

/* Progress Bars */
QProgressBar {
    background-color: #0D121D;
    border: 1px solid #253043;
    border-radius: 4px;
    height: 10px;
    text-align: center;
    color: transparent;
}
QProgressBar::chunk {
    background-color: #3B82F6;
    border-radius: 3px;
}
"""
