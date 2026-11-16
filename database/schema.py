"""SQLite DDL schemas and table definitions for AHIA-R2 knowledge library."""

SQL_CREATE_BOOKS_TABLE = """
CREATE TABLE IF NOT EXISTS books (
    book_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    author TEXT NOT NULL,
    filename TEXT DEFAULT '',
    source_path TEXT,
    format TEXT NOT NULL,
    file_hash TEXT UNIQUE,
    total_words INTEGER DEFAULT 0,
    total_pages INTEGER,
    license_type TEXT DEFAULT 'Personal Research',
    provenance TEXT DEFAULT 'User Uploaded',
    redistribution_permitted BOOLEAN NOT NULL DEFAULT 0,
    training_permitted BOOLEAN NOT NULL DEFAULT 0,
    authorized_by_user BOOLEAN NOT NULL DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

SQL_CREATE_CHAPTERS_TABLE = """
CREATE TABLE IF NOT EXISTS chapters (
    chapter_id TEXT PRIMARY KEY,
    book_id TEXT NOT NULL,
    chapter_number INTEGER,
    title TEXT NOT NULL,
    start_char INTEGER NOT NULL,
    end_char INTEGER NOT NULL,
    FOREIGN KEY(book_id) REFERENCES books(book_id) ON DELETE CASCADE
);
"""

SQL_CREATE_CHUNKS_TABLE = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    book_id TEXT NOT NULL,
    chapter_id TEXT,
    chunk_index INTEGER NOT NULL,
    page_start INTEGER,
    page_end INTEGER,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    word_count INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    content TEXT NOT NULL,
    FOREIGN KEY(book_id) REFERENCES books(book_id) ON DELETE CASCADE,
    FOREIGN KEY(chapter_id) REFERENCES chapters(chapter_id) ON DELETE SET NULL
);
"""

SQL_CREATE_BM25_TERMS_TABLE = """
CREATE TABLE IF NOT EXISTS bm25_terms (
    term TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    term_frequency INTEGER NOT NULL,
    PRIMARY KEY(term, chunk_id),
    FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE
);
"""

SQL_CREATE_CITATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS citations (
    citation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_text TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    quoted_text TEXT,
    verified BOOLEAN NOT NULL DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE
);
"""

SQL_CREATE_EXECUTION_LOGS_TABLE = """
CREATE TABLE IF NOT EXISTS execution_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_text TEXT NOT NULL,
    execution_depth INTEGER NOT NULL,
    iterations_used INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    tokens_generated INTEGER NOT NULL,
    governor_interrupted BOOLEAN NOT NULL DEFAULT 0,
    termination_reason TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

SQL_CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_books_hash ON books(file_hash);
CREATE INDEX IF NOT EXISTS idx_chunks_book_id ON chunks(book_id);
CREATE INDEX IF NOT EXISTS idx_chunks_chapter_id ON chunks(chapter_id);
CREATE INDEX IF NOT EXISTS idx_chunks_pages ON chunks(page_start, page_end);
CREATE INDEX IF NOT EXISTS idx_bm25_term ON bm25_terms(term);
CREATE INDEX IF NOT EXISTS idx_citations_chunk ON citations(chunk_id);
"""

SQL_CREATE_TABLES = [
    SQL_CREATE_BOOKS_TABLE,
    SQL_CREATE_CHAPTERS_TABLE,
    SQL_CREATE_CHUNKS_TABLE,
    SQL_CREATE_BM25_TERMS_TABLE,
    SQL_CREATE_CITATIONS_TABLE,
    SQL_CREATE_EXECUTION_LOGS_TABLE,
    SQL_CREATE_INDEXES,
]
