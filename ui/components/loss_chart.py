"""Pure PySide6 QPainter-based vector loss chart widget.

Renders multi-depth training loss curves without external charting dependencies.
"""

from typing import Dict, List, Tuple
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget


class LossChartWidget(QWidget):
    """Custom vector chart rendering training loss across adaptive execution depths."""

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(240)
        self.setStyleSheet("background-color: #0E131E; border-radius: 8px;")

        # Demonstration loss curve points (Step, Loss Value)
        self.series: Dict[str, Tuple[QColor, List[Tuple[int, float]]]] = {
            "Total Loss": (QColor("#10B981"), [
                (0, 4.2), (100, 3.4), (200, 2.7), (300, 2.2), (400, 1.85),
                (500, 1.55), (600, 1.35), (700, 1.20), (800, 1.10), (900, 1.02), (1000, 0.95)
            ]),
            "Depth 8 Loss": (QColor("#8B5CF6"), [
                (0, 4.5), (100, 3.6), (200, 2.9), (300, 2.3), (400, 1.95),
                (500, 1.62), (600, 1.40), (700, 1.25), (800, 1.14), (900, 1.05), (1000, 0.98)
            ]),
            "Depth 4 Loss": (QColor("#3B82F6"), [
                (0, 4.0), (100, 3.2), (200, 2.5), (300, 2.05), (400, 1.72),
                (500, 1.48), (600, 1.29), (700, 1.15), (800, 1.06), (900, 0.98), (1000, 0.92)
            ]),
            "Depth 2 Loss": (QColor("#F59E0B"), [
                (0, 3.8), (100, 3.0), (200, 2.3), (300, 1.90), (400, 1.60),
                (500, 1.38), (600, 1.22), (700, 1.08), (800, 0.99), (900, 0.92), (1000, 0.86)
            ]),
        }

    def paintEvent(self, event) -> None:
        """Render axes, grid, loss curves, and legend."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        margin_left = 50
        margin_right = 20
        margin_top = 35
        margin_bottom = 35

        plot_w = w - margin_left - margin_right
        plot_h = h - margin_top - margin_bottom

        if plot_w <= 0 or plot_h <= 0:
            return

        # Background
        painter.fillRect(0, 0, w, h, QBrush(QColor("#0E131E")))

        # Grid lines and Y ticks (Loss 0.0 to 5.0)
        grid_pen = QPen(QColor("#1F2837"), 1, Qt.PenStyle.DashLine)
        painter.setPen(grid_pen)
        font = QFont("monospace", 9)
        painter.setFont(font)

        y_steps = [1.0, 2.0, 3.0, 4.0, 5.0]
        for y_val in y_steps:
            norm_y = 1.0 - (y_val / 5.0)
            py = margin_top + norm_y * plot_h
            painter.drawLine(int(margin_left), int(py), int(w - margin_right), int(py))

            # Y-label
            painter.setPen(QColor("#64748B"))
            painter.drawText(QRectF(5, py - 8, margin_left - 10, 16), Qt.AlignmentFlag.AlignRight, f"{y_val:.1f}")
            painter.setPen(grid_pen)

        # X ticks (Steps 0 to 1000)
        x_steps = [0, 250, 500, 750, 1000]
        for x_val in x_steps:
            norm_x = x_val / 1000.0
            px = margin_left + norm_x * plot_w
            painter.drawLine(int(px), int(margin_top), int(px), int(margin_top + plot_h))

            # X-label
            painter.setPen(QColor("#64748B"))
            painter.drawText(QRectF(px - 25, h - margin_bottom + 6, 50, 18), Qt.AlignmentFlag.AlignCenter, str(x_val))
            painter.setPen(grid_pen)

        # Draw Loss Curves
        for name, (color, data) in self.series.items():
            if not data:
                continue

            path = QPainterPath()
            curve_pen = QPen(color, 2)
            painter.setPen(curve_pen)

            first = True
            for step, loss in data:
                norm_x = min(1.0, max(0.0, step / 1000.0))
                norm_y = min(1.0, max(0.0, 1.0 - (loss / 5.0)))
                px = margin_left + norm_x * plot_w
                py = margin_top + norm_y * plot_h

                if first:
                    path.moveTo(px, py)
                    first = False
                else:
                    path.lineTo(px, py)

            painter.drawPath(path)

            # Draw small point markers
            painter.setBrush(QBrush(color))
            for step, loss in data:
                norm_x = step / 1000.0
                norm_y = 1.0 - (loss / 5.0)
                px = margin_left + norm_x * plot_w
                py = margin_top + norm_y * plot_h
                painter.drawEllipse(QPointF(px, py), 2.5, 2.5)

        # Legend at top
        legend_x = margin_left
        legend_y = 10
        painter.setFont(QFont("sans-serif", 9))
        for name, (color, _) in self.series.items():
            painter.fillRect(int(legend_x), int(legend_y + 4), 10, 10, QBrush(color))
            painter.setPen(QColor("#94A3B8"))
            painter.drawText(QRectF(legend_x + 14, legend_y, 90, 18), Qt.AlignmentFlag.AlignVCenter, name)
            legend_x += 105

        # Watermark label
        painter.setPen(QColor("#475569"))
        painter.drawText(
            QRectF(w - 180, 10, 160, 18),
            Qt.AlignmentFlag.AlignRight,
            "DEMO SYNTHETIC LOSS",
        )
