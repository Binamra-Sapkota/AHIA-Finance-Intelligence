"""Settings View for AHIA-R2.

Features:
- Execution device selection (CPU / Auto / GPU Fallback)
- Model checkpoint selector
- Independent Resource Governor limits (Max Wall-Clock Time <= 10.0s)
- Deterministic reasoning loop ceiling (Max Iterations <= 3)
- Local storage directory configuration
"""

from pathlib import Path
from typing import Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class SettingsView(QWidget):
    """Configuration panel managing devices, checkpoints, storage, and governor limits."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._setup_ui()

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
        view_title = QLabel("System Settings & Governance Controls")
        view_title.setStyleSheet("font-size: 15px; font-weight: 600; color: #FFFFFF;")

        view_subtitle = QLabel("Hardware runtime, checkpoint selection, and independent resource bounds")
        view_subtitle.setStyleSheet("font-size: 11px; color: #94A3B8;")

        title_layout.addWidget(view_title)
        title_layout.addWidget(view_subtitle)
        header_layout.addLayout(title_layout)

        header_layout.addStretch()

        demo_badge = QLabel("DEMONSTRATION SHELL")
        demo_badge.setProperty("class", "badge-demo")
        header_layout.addWidget(demo_badge)

        main_layout.addWidget(header_card)

        # Settings Form Card
        form_card = QFrame()
        form_card.setProperty("class", "card")
        form_layout = QGridLayout(form_card)
        form_layout.setContentsMargins(16, 16, 16, 16)
        form_layout.setHorizontalSpacing(16)
        form_layout.setVerticalSpacing(14)

        # 1. Device Selection
        lbl_device = QLabel("Compute Device:")
        lbl_device.setStyleSheet("font-weight: 600; color: #E2E8F0;")
        self.device_combo = QComboBox()
        self.device_combo.addItems([
            "CPU (AMD Ryzen 5 5500U, 12 Threads)",
            "CPU (Multi-threaded Auto)",
            "GPU (CUDA / ROCm - Not Detected)",
        ])
        lbl_device_desc = QLabel("Pure PyTorch CPU execution optimized for AVX2 consumer hardware.")
        lbl_device_desc.setStyleSheet("color: #64748B; font-size: 11px;")

        form_layout.addWidget(lbl_device, 0, 0)
        form_layout.addWidget(self.device_combo, 0, 1)
        form_layout.addWidget(lbl_device_desc, 1, 1)

        # 2. Checkpoint Selection
        lbl_chk = QLabel("Active Model Checkpoint:")
        lbl_chk.setStyleSheet("font-weight: 600; color: #E2E8F0;")
        self.chk_combo = QComboBox()
        self.chk_combo.addItems([
            "checkpoints/ahia_dense_step1000.pt (15.2M Params, Loss: 0.95)",
            "checkpoints/ahia_dense_step500.pt (15.2M Params, Loss: 1.55)",
            "checkpoints/ahia_dense_step250.pt (15.2M Params, Loss: 2.45)",
            "Initialize Random Weights (Un-trained Baseline)",
        ])
        lbl_chk_desc = QLabel("Trained strictly from random initialization without external pretrained weights.")
        lbl_chk_desc.setStyleSheet("color: #64748B; font-size: 11px;")

        form_layout.addWidget(lbl_chk, 2, 0)
        form_layout.addWidget(self.chk_combo, 2, 1)
        form_layout.addWidget(lbl_chk_desc, 3, 1)

        # 3. Maximum Execution Time (Independent Governor Ceiling)
        lbl_time = QLabel("Governor Timeout Ceiling:")
        lbl_time.setStyleSheet("font-weight: 600; color: #E2E8F0;")
        self.time_spin = QDoubleSpinBox()
        self.time_spin.setRange(1.0, 10.0)
        self.time_spin.setSingleStep(0.5)
        self.time_spin.setValue(10.0)
        self.time_spin.setSuffix(" seconds")
        lbl_time_desc = QLabel("Independent watchdog thread enforces a hard ceiling (Maximum 10.0s, Rule 15).")
        lbl_time_desc.setStyleSheet("color: #64748B; font-size: 11px;")

        form_layout.addWidget(lbl_time, 4, 0)
        form_layout.addWidget(self.time_spin, 4, 1)
        form_layout.addWidget(lbl_time_desc, 5, 1)

        # 4. Maximum Reasoning Iterations (Deterministic Limit)
        lbl_iter = QLabel("Reasoning Iteration Limit:")
        lbl_iter.setStyleSheet("font-weight: 600; color: #E2E8F0;")
        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(1, 3)
        self.iter_spin.setValue(3)
        self.iter_spin.setSuffix(" iterations (T_max <= 3)")
        lbl_iter_desc = QLabel("Deterministic loop limit prevents unbounded retrieval and execution cycles (Rule 14).")
        lbl_iter_desc.setStyleSheet("color: #64748B; font-size: 11px;")

        form_layout.addWidget(lbl_iter, 6, 0)
        form_layout.addWidget(self.iter_spin, 6, 1)
        form_layout.addWidget(lbl_iter_desc, 7, 1)

        # 5. Local Storage Location
        lbl_storage = QLabel("Local Storage Directory:")
        lbl_storage.setStyleSheet("font-weight: 600; color: #E2E8F0;")
        storage_layout = QHBoxLayout()
        self.storage_edit = QLineEdit("data/")
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._on_browse_storage)
        storage_layout.addWidget(self.storage_edit, stretch=1)
        storage_layout.addWidget(self.browse_btn)
        lbl_storage_desc = QLabel("Local directory storing raw books, chunks, SQLite catalog, and BM25 indices.")
        lbl_storage_desc.setStyleSheet("color: #64748B; font-size: 11px;")

        form_layout.addWidget(lbl_storage, 8, 0)
        form_layout.addLayout(storage_layout, 8, 1)
        form_layout.addWidget(lbl_storage_desc, 9, 1)

        main_layout.addWidget(form_card)

        # Action Buttons (Save / Reset)
        btn_box = QFrame()
        btn_box_layout = QHBoxLayout(btn_box)
        btn_box_layout.setContentsMargins(0, 8, 0, 0)

        self.lbl_feedback = QLabel("")
        self.lbl_feedback.setStyleSheet("color: #10B981; font-weight: 600; font-size: 12px;")

        btn_box_layout.addWidget(self.lbl_feedback)
        btn_box_layout.addStretch()

        self.reset_btn = QPushButton("Reset Defaults")
        self.reset_btn.clicked.connect(self._on_reset_defaults)
        btn_box_layout.addWidget(self.reset_btn)

        self.save_btn = QPushButton("Save Configuration")
        self.save_btn.setProperty("class", "primary")
        self.save_btn.clicked.connect(self._on_save_config)
        btn_box_layout.addWidget(self.save_btn)

        main_layout.addWidget(btn_box)

        main_layout.addStretch()

        scroll.setWidget(content)
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(scroll)

    def _on_browse_storage(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Local Storage Directory", str(Path("data").resolve()))
        if folder:
            self.storage_edit.setText(folder)

    def _on_save_config(self) -> None:
        self.lbl_feedback.setText("✓ Configuration saved locally (demonstration mode).")

    def _on_reset_defaults(self) -> None:
        self.device_combo.setCurrentIndex(0)
        self.chk_combo.setCurrentIndex(0)
        self.time_spin.setValue(10.0)
        self.iter_spin.setValue(3)
        self.storage_edit.setText("data/")
        self.lbl_feedback.setText("✓ Defaults restored.")
