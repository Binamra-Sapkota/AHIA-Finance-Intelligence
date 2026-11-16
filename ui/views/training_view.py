"""Model Training View for AHIA-R2.

Features:
- Dataset status card
- Tokenizer training status card
- Neural model architectural configuration (8 blocks, 256 d_model, 4 heads, 1024 ff, 8000 vocab)
- Training start and stop controls
- Dynamic training loss chart (Depth 2, Depth 4, Depth 8, Total Loss)
- Checkpoint registry table
"""

from typing import Optional
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ui.components.loss_chart import LossChartWidget


class TrainingView(QWidget):
    """Model training dashboard with dataset status, model config, controls, and loss chart."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.is_training_active = False
        self._setup_ui()
        self._load_demo_checkpoints()

    def _setup_ui(self) -> None:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        content = QWidget()
        main_layout = QVBoxLayout(content)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(14)

        # Header Bar
        header_card = QFrame()
        header_card.setProperty("class", "card")
        header_layout = QHBoxLayout(header_card)
        header_layout.setContentsMargins(12, 10, 12, 10)

        title_layout = QVBoxLayout()
        view_title = QLabel("Pure PyTorch Neural Training Engine")
        view_title.setStyleSheet("font-size: 15px; font-weight: 600; color: #FFFFFF;")

        view_subtitle = QLabel("Trained entirely from random initialization • Zero pretrained weights")
        view_subtitle.setStyleSheet("font-size: 11px; color: #94A3B8;")

        title_layout.addWidget(view_title)
        title_layout.addWidget(view_subtitle)
        header_layout.addLayout(title_layout)

        header_layout.addStretch()

        demo_badge = QLabel("DEMONSTRATION SHELL")
        demo_badge.setProperty("class", "badge-demo")
        header_layout.addWidget(demo_badge)

        self.start_btn = QPushButton("▶ Start Training")
        self.start_btn.setProperty("class", "primary")
        self.start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_btn.clicked.connect(self._toggle_training)
        header_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("⏹ Stop Training")
        self.stop_btn.setProperty("class", "danger")
        self.stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._toggle_training)
        header_layout.addWidget(self.stop_btn)

        main_layout.addWidget(header_card)

        # Three Status Cards in a Row: Dataset, Tokenizer, Architecture
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(12)

        # 1. Dataset Status Card
        ds_card = QFrame()
        ds_card.setProperty("class", "card")
        ds_layout = QVBoxLayout(ds_card)
        ds_layout.setContentsMargins(12, 12, 12, 12)
        ds_layout.setSpacing(6)

        ds_title = QLabel("📊 Dataset Preparation")
        ds_title.setStyleSheet("font-weight: 600; font-size: 13px; color: #60A5FA;")
        ds_layout.addWidget(ds_title)

        ds_lbl1 = QLabel("• Chunks Prepared: 2,073 passages")
        ds_lbl2 = QLabel("• Estimated Tokens: 663,000 tokens")
        ds_lbl3 = QLabel("• Split: 90% Train / 10% Validation")
        ds_lbl4 = QLabel("• Status: Ready (Synthetic RAG + Finance)")
        for lbl in [ds_lbl1, ds_lbl2, ds_lbl3, ds_lbl4]:
            lbl.setStyleSheet("font-size: 11px; color: #94A3B8; font-family: monospace;")
            ds_layout.addWidget(lbl)
        cards_layout.addWidget(ds_card)

        # 2. Tokenizer Status Card
        tok_card = QFrame()
        tok_card.setProperty("class", "card")
        tok_layout = QVBoxLayout(tok_card)
        tok_layout.setContentsMargins(12, 12, 12, 12)
        tok_layout.setSpacing(6)

        tok_title = QLabel("🔤 Byte-Level BPE Tokenizer")
        tok_title.setStyleSheet("font-weight: 600; font-size: 13px; color: #10B981;")
        tok_layout.addWidget(tok_title)

        tok_lbl1 = QLabel("• Algorithm: Byte-Level BPE (From Scratch)")
        tok_lbl2 = QLabel("• Target Vocabulary: 8,000 tokens")
        tok_lbl3 = QLabel("• Base Alphabet: 256 raw bytes")
        tok_lbl4 = QLabel("• Status: Ready (Zero <unk> crashes)")
        for lbl in [tok_lbl1, tok_lbl2, tok_lbl3, tok_lbl4]:
            lbl.setStyleSheet("font-size: 11px; color: #94A3B8; font-family: monospace;")
            tok_layout.addWidget(lbl)
        cards_layout.addWidget(tok_card)

        # 3. Model Architecture Card
        model_card = QFrame()
        model_card.setProperty("class", "card")
        model_layout = QVBoxLayout(model_card)
        model_layout.setContentsMargins(12, 12, 12, 12)
        model_layout.setSpacing(6)

        model_title = QLabel("🧠 Neural Architecture")
        model_title.setStyleSheet("font-weight: 600; font-size: 13px; color: #F59E0B;")
        model_layout.addWidget(model_title)

        m_lbl1 = QLabel("• Transformer Blocks: 8 blocks")
        m_lbl2 = QLabel("• Hidden Dim: 256 | Heads: 4 (dk=64)")
        m_lbl3 = QLabel("• FF Dim: 1024 | Max Context: 512")
        m_lbl4 = QLabel("• Adaptive Depths: 2, 4, 8 | 4 Planned MoE")
        for lbl in [m_lbl1, m_lbl2, m_lbl3, m_lbl4]:
            lbl.setStyleSheet("font-size: 11px; color: #94A3B8; font-family: monospace;")
            model_layout.addWidget(lbl)
        cards_layout.addWidget(model_card)

        main_layout.addLayout(cards_layout)

        # Training Loss Chart Section
        chart_card = QFrame()
        chart_card.setProperty("class", "card")
        chart_layout = QVBoxLayout(chart_card)
        chart_layout.setContentsMargins(14, 12, 14, 12)
        chart_layout.setSpacing(10)

        chart_header_layout = QHBoxLayout()
        chart_title = QLabel("Multi-Depth Convergence Curve (Synthetic Demonstration)")
        chart_title.setStyleSheet("font-weight: 600; font-size: 13px; color: #F3F4F6;")

        chart_stats = QLabel("Current Step: 1,000 • Final Loss: 0.95 • LR: 4.0e-5")
        chart_stats.setStyleSheet("font-size: 11px; color: #34D399; font-family: monospace;")

        chart_header_layout.addWidget(chart_title)
        chart_header_layout.addStretch()
        chart_header_layout.addWidget(chart_stats)
        chart_layout.addLayout(chart_header_layout)

        # Embed custom vector chart
        self.loss_chart = LossChartWidget()
        chart_layout.addWidget(self.loss_chart)

        main_layout.addWidget(chart_card)

        # Checkpoints Section
        chk_card = QFrame()
        chk_card.setProperty("class", "card")
        chk_layout = QVBoxLayout(chk_card)
        chk_layout.setContentsMargins(14, 12, 14, 12)
        chk_layout.setSpacing(8)

        chk_title = QLabel("Saved Checkpoints (checkpoints/)")
        chk_title.setStyleSheet("font-weight: 600; font-size: 13px; color: #60A5FA;")
        chk_layout.addWidget(chk_title)

        self.chk_table = QTableWidget()
        self.chk_table.setColumnCount(6)
        self.chk_table.setHorizontalHeaderLabels([
            "Checkpoint File", "Step", "Loss", "Active Depths", "Timestamp", "Action"
        ])
        self.chk_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.chk_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.chk_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.chk_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.chk_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.chk_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self.chk_table.setMinimumHeight(150)
        chk_layout.addWidget(self.chk_table)

        main_layout.addWidget(chk_card)

        scroll.setWidget(content)
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(scroll)

    def _load_demo_checkpoints(self) -> None:
        checkpoints = [
            ("ahia_dense_step1000.pt", "1000", "0.95", "2, 4, 8 blocks", "2026-11-16 10:45", "Active"),
            ("ahia_dense_step500.pt", "500", "1.55", "2, 4, 8 blocks", "2026-11-16 09:30", "Restore"),
            ("ahia_dense_step250.pt", "250", "2.45", "2, 4, 8 blocks", "2026-11-16 08:15", "Restore"),
        ]
        self.chk_table.setRowCount(len(checkpoints))
        for row, (name, step, loss, depths, tstamp, action) in enumerate(checkpoints):
            self.chk_table.setItem(row, 0, QTableWidgetItem(name))
            self.chk_table.setItem(row, 1, QTableWidgetItem(step))
            self.chk_table.setItem(row, 2, QTableWidgetItem(loss))
            self.chk_table.setItem(row, 3, QTableWidgetItem(depths))
            self.chk_table.setItem(row, 4, QTableWidgetItem(tstamp))

            btn = QPushButton(action)
            btn.setStyleSheet(
                "QPushButton { padding: 3px 8px; font-size: 11px; } "
                "QPushButton:hover { background-color: #2563EB; }"
            )
            self.chk_table.setCellWidget(row, 5, btn)

    def _toggle_training(self) -> None:
        """Simulate training toggle for shell demonstration."""
        self.is_training_active = not self.is_training_active
        if self.is_training_active:
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.start_btn.setText("Training In Progress...")
        else:
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.start_btn.setText("▶ Start Training")
