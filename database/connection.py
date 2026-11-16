"""Database connection management and initialization for SQLite."""

from pathlib import Path
import sqlite3
from typing import Optional
from database.schema import SQL_CREATE_TABLES

DEFAULT_DB_PATH = Path("data/ahia_knowledge.db")


def get_db_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Create and configure a thread-safe connection to the SQLite database.

    Args:
        db_path: Path to SQLite database file. Defaults to data/ahia_knowledge.db.

    Returns:
        Configured sqlite3.Connection with Row factory and foreign keys enabled.
    """
    target_path = db_path or DEFAULT_DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(target_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn


def initialize_database(db_path: Optional[Path] = None) -> None:
    """Initialize SQLite tables and indexes.

    Args:
        db_path: Path to database. If omitted, uses default data/ahia_knowledge.db.
    """
    conn = get_db_connection(db_path)
    try:
        with conn:
            for statement in SQL_CREATE_TABLES:
                conn.executescript(statement)
    finally:
        conn.close()
