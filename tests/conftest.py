"""Pytest fixtures for repository, ingestion, and integration testing."""

import os
from pathlib import Path
import sqlite3
import pytest

from database.connection import get_db_connection, initialize_database
from database.schema import SQL_CREATE_TABLES
from main import _preload_linux_xcb_cursor

# Ensure xcb cursor is preloaded on Linux
_preload_linux_xcb_cursor()

# Ensure offscreen Qt platform for testing
os.environ["QT_QPA_PLATFORM"] = "offscreen"


@pytest.fixture
def repo_root() -> Path:
    """Return the absolute path to the repository root directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def in_memory_db() -> sqlite3.Connection:
    """Provide an initialized in-memory SQLite database connection."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")

    for statement in SQL_CREATE_TABLES:
        conn.executescript(statement)
    yield conn
    conn.close()


@pytest.fixture
def sample_text_file(tmp_path: Path) -> Path:
    """Create a safe-to-redistribute sample personal-finance markdown document."""
    file_path = tmp_path / "financial_principles.md"
    content = (
        "# Principles of Personal Financial Intelligence\n\n"
        "Personal finance begins with a simple discipline: spending less than you earn. "
        "The surplus generated each month represents seed capital for your future autonomy. "
        "Without an initial margin between income and expenditure, investment is impossible.\n\n"
        "Building an emergency reserve of three to six months of living expenses forms the "
        "bedrock of capital preservation. This buffer prevents premature liquidation of long-term "
        "assets during unforeseen crises or economic downturns.\n\n"
        "Compound interest functions as an exponential force over extended time horizons. "
        "By reinvesting yields and dividends systematically, the return generated on past returns "
        "eventually exceeds the primary contributions. Patience is the investor's greatest asset.\n\n"
        "Diversification is an exercise in humility. Because no investor can predict market cycles "
        "with certainty, distributing capital across non-correlated asset classes mitigates "
        "catastrophic drawdown risks."
    )
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture(scope="session")
def qapp():
    """Session-wide QApplication instance for offscreen testing."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def sample_pdf_file(tmp_path: Path, qapp) -> Path:
    """Create a multi-page safe-to-redistribute sample PDF document."""
    from PySide6.QtGui import QPageSize, QPainter, QPdfWriter

    pdf_path = tmp_path / "sample_guide.pdf"

    def _write_file():
        writer = QPdfWriter(str(pdf_path))
        writer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
        painter = QPainter(writer)
        painter.drawText(
            100, 100,
            "Page 1: The foundational principle of wealth accumulation is retaining ten percent of earnings."
        )
        writer.newPage()
        painter.drawText(
            100, 100,
            "Page 2: Debt is a claim on future labor. High interest liabilities erode discretionary income."
        )
        painter.end()
        del painter
        del writer

    _write_file()
    return pdf_path


@pytest.fixture
def empty_text_file(tmp_path: Path) -> Path:
    """Create an empty text file."""
    empty_file = tmp_path / "empty_doc.txt"
    empty_file.write_text("", encoding="utf-8")
    return empty_file


@pytest.fixture
def corrupted_pdf_file(tmp_path: Path) -> Path:
    """Create a corrupted invalid PDF file."""
    corrupt_file = tmp_path / "corrupt.pdf"
    corrupt_file.write_bytes(b"%PDF-1.4\nINVALID_GARBAGE_BYTES_THAT_CANNOT_BE_PARSED_AS_PDF")
    return corrupt_file
