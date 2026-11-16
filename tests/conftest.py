"""Pytest fixtures for repository and integration testing."""

from pathlib import Path
import sqlite3
import pytest
from database.connection import get_db_connection, initialize_database


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
    from database.schema import SQL_CREATE_TABLES

    for statement in SQL_CREATE_TABLES:
        conn.executescript(statement)
    yield conn
    conn.close()


@pytest.fixture
def sample_chunk_dict() -> dict:
    """Provide sample chunk attributes for data validation tests."""
    return {
        "chunk_id": "BAB_CH01_0001",
        "book_id": "RICHEST_BABYLON",
        "chapter_id": "BAB_CH01",
        "chunk_index": 1,
        "char_start": 0,
        "char_end": 450,
        "word_count": 82,
        "token_count": 105,
        "content": (
            "In old Babylon there once lived a certain rich man named Arkad. "
            "Far and wide he was famed for his great wealth. Also was he famed "
            "for his liberality. He was generous in his charities."
        ),
    }
