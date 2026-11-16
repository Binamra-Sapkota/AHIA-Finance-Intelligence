"""Automated tests for Phase 1: AHIA-R2 Desktop Application Shell."""

import ast
import os
from pathlib import Path
import pytest

# Ensure Qt runs offscreen for automated testing environments
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from main import create_application
from ui.main_window import MainWindow
from ui.views.chat_view import ChatView
from ui.views.library_view import LibraryView
from ui.views.training_view import TrainingView
from ui.views.monitor_view import MonitorView
from ui.views.settings_view import SettingsView


@pytest.fixture(scope="session")
def qapp():
    """Session-scoped QApplication fixture."""
    app = create_application()
    yield app


@pytest.fixture
def main_window(qapp):
    """Provide an initialized MainWindow instance."""
    win = MainWindow()
    win.show()
    yield win
    win.close()


def test_main_window_creation(main_window: MainWindow) -> None:
    """Verify that MainWindow instantiates cleanly with all 5 views."""
    assert main_window is not None
    assert main_window.stacked_views.count() == 5
    assert isinstance(main_window.chat_view, ChatView)
    assert isinstance(main_window.library_view, LibraryView)
    assert isinstance(main_window.training_view, TrainingView)
    assert isinstance(main_window.monitor_view, MonitorView)
    assert isinstance(main_window.settings_view, SettingsView)


def test_sidebar_navigation_switching(main_window: MainWindow) -> None:
    """Verify switching views via sidebar buttons."""
    sidebar = main_window.sidebar

    # Switch to Book Library (index 1)
    sidebar.set_active_index(1)
    assert main_window.stacked_views.currentIndex() == 1

    # Switch to Model Training (index 2)
    sidebar.set_active_index(2)
    assert main_window.stacked_views.currentIndex() == 2

    # Switch to Architecture Monitor (index 3)
    sidebar.set_active_index(3)
    assert main_window.stacked_views.currentIndex() == 3

    # Switch to Settings (index 4)
    sidebar.set_active_index(4)
    assert main_window.stacked_views.currentIndex() == 4

    # Switch back to AI Chat (index 0)
    sidebar.set_active_index(0)
    assert main_window.stacked_views.currentIndex() == 0


def test_chat_view_components_and_interaction(main_window: MainWindow) -> None:
    """Verify AI Chat controls, message submission, and simulated grounded responses."""
    chat = main_window.chat_view
    assert chat.input_edit is not None
    assert chat.send_btn is not None
    assert chat.message_list.count() >= 1  # Contains initial welcome message

    # Submit a query
    chat.input_edit.setText("What does Babylon say about 10%?")
    initial_count = chat.message_list.count()
    chat._on_send_clicked()

    # User message should be added immediately
    assert chat.message_list.count() == initial_count + 1

    # Simulate response generation
    chat._generate_simulated_response("What does Babylon say about 10%?")
    assert chat.message_list.count() == initial_count + 2
    assert chat.sources_content_layout.count() > 0


def test_library_view_components(main_window: MainWindow) -> None:
    """Verify Book Library table, documents, and metadata inspector."""
    lib = main_window.library_view
    assert lib.add_btn is not None
    assert lib.rebuild_btn is not None
    assert lib.table.rowCount() >= 6  # 7 personal finance books preloaded

    # Select first row and check inspector
    lib.table.selectRow(0)
    assert len(lib.lbl_meta_title.text()) > 0
    assert "Author:" in lib.lbl_meta_author.text()
    assert "Format:" in lib.lbl_meta_format.text()


def test_training_view_components(main_window: MainWindow) -> None:
    """Verify Model Training status cards, controls, loss chart, and checkpoints."""
    train = main_window.training_view
    assert train.start_btn is not None
    assert train.stop_btn is not None
    assert train.loss_chart is not None
    assert train.chk_table.rowCount() >= 3

    # Toggle training state
    assert train.is_training_active is False
    train._toggle_training()
    assert train.is_training_active is True
    assert train.stop_btn.isEnabled() is True
    train._toggle_training()
    assert train.is_training_active is False


def test_monitor_view_components(main_window: MainWindow) -> None:
    """Verify Architecture Monitor depth selector and hardware meters."""
    mon = main_window.monitor_view
    assert mon.lbl_retrieval is not None
    assert mon.lbl_generation is not None
    assert mon.lbl_grounding is not None
    assert mon.cpu_bar is not None
    assert mon.ram_bar is not None

    # Test depth selector
    mon._set_depth(2)
    assert mon.selected_depth == 2
    mon._set_depth(8)
    assert mon.selected_depth == 8


def test_settings_view_components(main_window: MainWindow) -> None:
    """Verify Settings view controls, governor ceilings, and default values."""
    settings = main_window.settings_view
    assert settings.device_combo.count() >= 2
    assert settings.chk_combo.count() >= 3

    # Check governor timeout ceiling (10.0s max)
    assert settings.time_spin.value() <= 10.0
    assert settings.time_spin.maximum() == 10.0

    # Check reasoning iteration ceiling (3 max)
    assert settings.iter_spin.value() <= 3
    assert settings.iter_spin.maximum() == 3

    # Test save and reset actions
    settings._on_save_config()
    assert "saved" in settings.lbl_feedback.text().lower()

    settings._on_reset_defaults()
    assert "restored" in settings.lbl_feedback.text().lower()


def test_strict_ui_decoupling_ast(repo_root: Path) -> None:
    """Verify zero PyTorch imports across all UI modules and components."""
    ui_files = list((repo_root / "ui").rglob("*.py"))
    assert len(ui_files) >= 8

    for py_file in ui_files:
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "torch" not in alias.name, (
                        f"FORBIDDEN IMPORT '{alias.name}' in {py_file}!"
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert "torch" not in node.module, (
                        f"FORBIDDEN IMPORT 'from {node.module}' in {py_file}!"
                    )
