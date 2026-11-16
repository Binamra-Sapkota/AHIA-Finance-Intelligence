"""Typed Data Transfer Objects (DTOs) for cross-boundary communication.

Note: In accordance with engineering standards, this module MUST NEVER import
PyTorch or any tensor structures. All data crossing UI boundaries is strictly
typed standard Python primitives.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class CitationItem:
    """Individual citation entry displayed in the UI."""

    chunk_id: str
    book_title: str
    author: str
    chapter_title: str
    excerpt: str
    match_score: float
    char_start: int
    char_end: int


@dataclass(frozen=True)
class QueryRequest:
    """User query request submitted from the desktop UI."""

    prompt: str
    preferred_depth: Optional[int] = None  # None = adaptive, 2, 4, or 8
    max_tokens: int = 256
    filter_book_id: Optional[str] = None


@dataclass(frozen=True)
class QueryResult:
    """Complete grounded answer and execution statistics returned to UI."""

    query: str
    answer: str
    citations: List[CitationItem]
    execution_depth: int
    iterations_used: int
    latency_ms: float
    tokens_generated: int
    governor_interrupted: bool
    termination_reason: str
    is_grounded: bool


@dataclass(frozen=True)
class BookIngestRequest:
    """Request to parse and index a local book file."""

    file_path: Path
    title: str
    author: str
    book_id: str


@dataclass(frozen=True)
class BookIngestSummary:
    """Summary metrics of an ingested and chunked book."""

    book_id: str
    title: str
    total_words: int
    total_chunks: int
    total_chapters: int
    duration_ms: float


@dataclass(frozen=True)
class SystemDiagnostics:
    """System resource status and health metrics."""

    model_loaded: bool
    model_parameter_count: int
    vocab_size: int
    indexed_books_count: int
    indexed_chunks_count: int
    memory_footprint_mb: float
    device_type: str = "CPU"
