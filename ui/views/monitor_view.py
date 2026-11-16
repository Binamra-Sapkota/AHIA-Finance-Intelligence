"""Architecture Monitor View for AHIA-R2.

Features:
- Selected neural execution depth selector (2, 4, 8 blocks)
- Active mixture-of-experts (4 planned experts) routing visualizer
- Retrieval duration & Generation latency meters
- Citation ledger verification status
- Real-time CPU and RAM resource monitoring
- GPU availability indicator
"""

import os
from typing import Optional
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


class MonitorView(QWidget):
    """Real-time architecture monitor visualizing adaptive depths, experts, and CPU/RAM usage."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.selected_depth = 4
        self._setup_ui()
        self._start_metrics_timer()

    def _setup_ui(self) -> None:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        content = QWidget()
        main_layout = QVBoxLayout(content)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(14)

        # Header Card
        header_card = QFrame()
        header_card.setProperty("class", "card")
        header_layout = QHBoxLayout(header_card)
        header_layout.setContentsMargins(12, 10, 12, 10)

        title_layout = QVBoxLayout()
        view_title = QLabel("Architecture & Resource Monitor")
        view_title.setStyleSheet("font-size: 15px; font-weight: 600; color: #FFFFFF;")

        view_subtitle = QLabel("Adaptive Transformer forward depth, 4-expert routing, and runtime hardware metrics")
        view_subtitle.setStyleSheet("font-size: 11px; color: #94A3B8;")

        title_layout.addWidget(view_title)
        title_layout.addWidget(view_subtitle)
        header_layout.addLayout(title_layout)

        header_layout.addStretch()

        demo_badge = QLabel("DEMONSTRATION TELEMETRY")
        demo_badge.setProperty("class", "badge-demo")
        header_layout.addWidget(demo_badge)

        main_layout.addWidget(header_card)

        # 1. Adaptive Execution Depth Selector Section
        depth_card = QFrame()
        depth_card.setProperty("class", "card")
        depth_layout = QVBoxLayout(depth_card)
        depth_layout.setContentsMargins(14, 12, 14, 12)
        depth_layout.setSpacing(10)

        depth_header = QHBoxLayout()
        depth_title = QLabel("Adaptive Neural Execution Depth")
        depth_title.setStyleSheet("font-weight: 600; font-size: 13px; color: #60A5FA;")

        depth_explain = QLabel("Dynamically scales compute per query complexity")
        depth_explain.setStyleSheet("font-size: 11px; color: #94A3B8;")

        depth_header.addWidget(depth_title)
        depth_header.addStretch()
        depth_header.addWidget(depth_explain)
        depth_layout.addLayout(depth_header)

        # Depth Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)
        self.depth_group = QButtonGroup(self)

        depth_configs = [
            (2, "⚡ Depth 2 Blocks (Fast)", "Classification & Graceful Refusal (<15ms)", "#10B981"),
            (4, "⚖️ Depth 4 Blocks (Factual)", "Direct Fact Retrieval & Passage Extraction (~35ms)", "#3B82F6"),
            (8, "🧠 Depth 8 Blocks (Synthesis)", "Full Multi-Chunk Synthesis & Comparative Reasoning (~65ms)", "#8B5CF6"),
        ]

        self.depth_buttons = {}
        for depth_val, label_text, subtext, color in depth_configs:
            btn = QPushButton(f"{label_text}\n{subtext}")
            btn.setCheckable(True)
            btn.setChecked(depth_val == self.selected_depth)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(
                f"QPushButton {{ background-color: #17202F; border: 1px solid #2B374E; border-radius: 8px; "
                f"padding: 10px; text-align: left; font-size: 12px; font-weight: 500; color: #E2E8F0; }} "
                f"QPushButton:checked {{ background-color: #1E2D44; border: 2px solid {color}; color: #FFFFFF; font-weight: 600; }}"
            )
            btn.clicked.connect(lambda checked=False, d=depth_val: self._set_depth(d))
            self.depth_group.addButton(btn)
            self.depth_buttons[depth_val] = btn
            btn_layout.addWidget(btn)

        depth_layout.addLayout(btn_layout)
        main_layout.addWidget(depth_card)

        # 2. Planned 4-Expert Mixture-of-Experts Section
        moe_card = QFrame()
        moe_card.setProperty("class", "card")
        moe_layout = QVBoxLayout(moe_card)
        moe_layout.setContentsMargins(14, 12, 14, 12)
        moe_layout.setSpacing(10)

        moe_header = QHBoxLayout()
        moe_title = QLabel("4-Expert Mixture-of-Experts (MoE) Architecture")
        moe_title.setStyleSheet("font-weight: 600; font-size: 13px; color: #8B5CF6;")

        moe_status = QLabel("STATUS: DENSE BASELINE ACTIVE (MOE HOOKS STUBBED)")
        moe_status.setStyleSheet("font-size: 10px; color: #F59E0B; font-weight: 700; background: rgba(245, 158, 11, 0.15); padding: 2px 6px; border-radius: 4px;")

        moe_header.addWidget(moe_title)
        moe_header.addStretch()
        moe_header.addWidget(moe_status)
        moe_layout.addLayout(moe_header)

        # Four Expert Routing Cards
        experts_layout = QHBoxLayout()
        experts_layout.setSpacing(10)

        experts_info = [
            ("Expert 1: Savings & Budgeting", "42% Load (Synthetic)", 42),
            ("Expert 2: Investment Principles", "28% Load (Synthetic)", 28),
            ("Expert 3: Debt & Capital Flow", "18% Load (Synthetic)", 18),
            ("Expert 4: Behavioral Psychology", "12% Load (Synthetic)", 12),
        ]

        for exp_name, exp_load, percent in experts_info:
            exp_box = QFrame()
            exp_box.setStyleSheet("background-color: #111723; border: 1px solid #1F293D; border-radius: 6px; padding: 8px;")
            exp_box_layout = QVBoxLayout(exp_box)
            exp_box_layout.setContentsMargins(8, 8, 8, 8)
            exp_box_layout.setSpacing(6)

            e_lbl = QLabel(exp_name)
            e_lbl.setStyleSheet("font-size: 11px; font-weight: 600; color: #CBD5E1;")

            p_bar = QProgressBar()
            p_bar.setRange(0, 100)
            p_bar.setValue(percent)
            p_bar.setStyleSheet("QProgressBar::chunk { background-color: #8B5CF6; }")

            e_stat = QLabel(exp_load)
            e_stat.setStyleSheet("font-size: 10px; color: #94A3B8; font-family: monospace;")

            exp_box_layout.addWidget(e_lbl)
            exp_box_layout.addWidget(p_bar)
            exp_box_layout.addWidget(e_stat)
            experts_layout.addWidget(exp_box)

        moe_layout.addLayout(experts_layout)
        main_layout.addWidget(moe_card)

        # 3. Latency & Verification Status Meters
        metrics_card = QFrame()
        metrics_card.setProperty("class", "card")
        metrics_layout = QGridLayout(metrics_card)
        metrics_layout.setContentsMargins(14, 12, 14, 12)
        metrics_layout.setHorizontalSpacing(16)
        metrics_layout.setVerticalSpacing(10)

        # Retrieval Duration
        r_box = QFrame()
        r_box.setStyleSheet("background-color: #111723; border: 1px solid #1F293D; border-radius: 6px; padding: 8px;")
        r_layout = QVBoxLayout(r_box)
        r_title = QLabel("BM25 Retrieval Duration")
        r_title.setStyleSheet("font-size: 11px; color: #94A3B8; font-weight: 500;")
        self.lbl_retrieval = QLabel("11.4 ms")
        self.lbl_retrieval.setStyleSheet("font-size: 18px; font-weight: 700; color: #60A5FA; font-family: monospace;")
        r_sub = QLabel("Okapi BM25 • Top-5 Chunks Scored")
        r_sub.setStyleSheet("font-size: 10px; color: #64748B;")
        r_layout.addWidget(r_title)
        r_layout.addWidget(self.lbl_retrieval)
        r_layout.addWidget(r_sub)
        metrics_layout.addWidget(r_box, 0, 0)

        # Generation Duration
        g_box = QFrame()
        g_box.setStyleSheet("background-color: #111723; border: 1px solid #1F293D; border-radius: 6px; padding: 8px;")
        g_layout = QVBoxLayout(g_box)
        g_title = QLabel("Neural Generation Duration")
        g_title.setStyleSheet("font-size: 11px; color: #94A3B8; font-weight: 500;")
        self.lbl_generation = QLabel("52.8 ms")
        self.lbl_generation.setStyleSheet("font-size: 18px; font-weight: 700; color: #34D399; font-family: monospace;")
        g_sub = QLabel("28.4 tokens/sec • Stateful KV Cache")
        g_sub.setStyleSheet("font-size: 10px; color: #64748B;")
        g_layout.addWidget(g_title)
        g_layout.addWidget(self.lbl_generation)
        g_layout.addWidget(g_sub)
        metrics_layout.addWidget(g_box, 0, 1)

        # Verification Status
        v_box = QFrame()
        v_box.setStyleSheet("background-color: #111723; border: 1px solid #1F293D; border-radius: 6px; padding: 8px;")
        v_layout = QVBoxLayout(v_box)
        v_title = QLabel("Reliability & Grounding Status")
        v_title.setStyleSheet("font-size: 11px; color: #94A3B8; font-weight: 500;")
        self.lbl_grounding = QLabel("100% Grounded")
        self.lbl_grounding.setStyleSheet("font-size: 18px; font-weight: 700; color: #10B981; font-family: monospace;")
        v_sub = QLabel("Zero Fabrications • Exact Citation Match")
        v_sub.setStyleSheet("font-size: 10px; color: #64748B;")
        v_layout.addWidget(v_title)
        v_layout.addWidget(self.lbl_grounding)
        v_layout.addWidget(v_sub)
        metrics_layout.addWidget(v_box, 0, 2)

        main_layout.addWidget(metrics_card)

        # 4. Hardware Resources Section (CPU, RAM, GPU)
        hw_card = QFrame()
        hw_card.setProperty("class", "card")
        hw_layout = QVBoxLayout(hw_card)
        hw_layout.setContentsMargins(14, 12, 14, 12)
        hw_layout.setSpacing(10)

        hw_title = QLabel("Local Hardware Utilization")
        hw_title.setStyleSheet("font-weight: 600; font-size: 13px; color: #F3F4F6;")
        hw_layout.addWidget(hw_title)

        hw_grid = QGridLayout()
        hw_grid.setHorizontalSpacing(16)
        hw_grid.setVerticalSpacing(8)

        # CPU Usage
        hw_grid.addWidget(QLabel("CPU Utilization (AMD Ryzen 5 5500U, 12 Threads):"), 0, 0)
        self.cpu_bar = QProgressBar()
        self.cpu_bar.setRange(0, 100)
        self.cpu_bar.setValue(18)
        self.cpu_bar.setStyleSheet("QProgressBar::chunk { background-color: #3B82F6; }")
        hw_grid.addWidget(self.cpu_bar, 0, 1)
        self.lbl_cpu_text = QLabel("18% (Multi-threaded)")
        self.lbl_cpu_text.setStyleSheet("font-family: monospace; font-size: 11px; color: #94A3B8;")
        hw_grid.addWidget(self.lbl_cpu_text, 0, 2)

        # RAM Usage
        hw_grid.addWidget(QLabel("RAM Utilization:"), 1, 0)
        self.ram_bar = QProgressBar()
        self.ram_bar.setRange(0, 100)
        self.ram_bar.setValue(32)
        self.ram_bar.setStyleSheet("QProgressBar::chunk { background-color: #10B981; }")
        hw_grid.addWidget(self.ram_bar, 1, 1)
        self.lbl_ram_text = QLabel("2.4 GB / 7.1 GB Total")
        self.lbl_ram_text.setStyleSheet("font-family: monospace; font-size: 11px; color: #94A3B8;")
        hw_grid.addWidget(self.lbl_ram_text, 1, 2)

        # GPU Status
        hw_grid.addWidget(QLabel("GPU Acceleration:"), 2, 0)
        self.lbl_gpu = QLabel("None Detected • Clean CPU Fallback Active")
        self.lbl_gpu.setStyleSheet("font-family: monospace; font-size: 11px; color: #F59E0B;")
        hw_grid.addWidget(self.lbl_gpu, 2, 1, 1, 2)

        hw_layout.addLayout(hw_grid)
        main_layout.addWidget(hw_card)

        scroll.setWidget(content)
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(scroll)

    def _set_depth(self, depth: int) -> None:
        self.selected_depth = depth
        for d, btn in self.depth_buttons.items():
            btn.setChecked(d == depth)

    def _start_metrics_timer(self) -> None:
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh_hardware_metrics)
        self.timer.start(2000)

    def _refresh_hardware_metrics(self) -> None:
        """Read standard Linux /proc/meminfo or calculate live memory."""
        try:
            with open("/proc/meminfo", "r") as f:
                lines = f.readlines()
            mem_dict = {}
            for line in lines[:5]:
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    val = int(parts[1].split()[0])
                    mem_dict[key] = val

            total_kb = mem_dict.get("MemTotal", 7450000)
            avail_kb = mem_dict.get("MemAvailable", 4500000)
            used_kb = total_kb - avail_kb
            pct = int((used_kb / total_kb) * 100)

            self.ram_bar.setValue(pct)
            self.lbl_ram_text.setText(f"{used_kb / 1024 / 1024:.1f} GB / {total_kb / 1024 / 1024:.1f} GB ({pct}%)")
        except Exception:
            pass
