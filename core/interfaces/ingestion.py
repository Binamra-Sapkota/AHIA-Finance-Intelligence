"""Interface contracts and data schemas for book ingestion and chunking."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol


@dataclass(frozen=True)
class BookMetadata:
    """Metadata describing an ingested personal finance book."""

    book_id: str
    title: str
    author: str
    source_path: str
    format: str  # 'txt', 'pdf', 'epub'
    total_words: int = 0


@dataclass(frozen=True)
class Chunk:
    """Atomic text passage produced by sliding-window chunking."""

    chunk_id: str
    book_id: str
    chapter_id: Optional[str]
    chunk_index: int
    char_start: int
    char_end: int
    word_count: int
    token_count: int
    content: str


class DocumentParserProtocol(Protocol):
    """Protocol for parsing various raw document formats into clean structured text."""

    def parse(self, file_path: Path) -> str:
        """Parse raw document into clean text.

        Args:
            file_path: Path to raw input file.

        Returns:
            Normalized raw text representation.

        Raises:
            FileNotFoundError: If the document does not exist.
            ValueError: If the format is invalid or corrupt.
        """
        ...


class ChunkerProtocol(Protocol):
    """Protocol for sliding-window segmentation with metadata preservation."""

    def chunk_document(
        self,
        text: str,
        book_id: str,
        chapter_id: Optional[str] = None,
        target_words: int = 250,
        overlap_words: int = 50,
    ) -> List[Chunk]:
        """Segment a continuous document into overlapping chunks.

        Args:
            text: Normalized document text.
            book_id: Unique identifier of parent book.
            chapter_id: Optional identifier of containing chapter.
            target_words: Target word count per chunk.
            overlap_words: Word overlap between successive chunks.

        Returns:
            List of generated Chunk instances.
        """
        ...
