"""Interface contracts, exceptions, and data schemas for book ingestion and chunking."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Protocol


class IngestionError(Exception):
    """Base exception for document ingestion errors."""
    pass


class InvalidDocumentError(IngestionError):
    """Raised when an imported document is corrupted, unreadable, or invalid format."""
    pass


class EmptyDocumentError(IngestionError):
    """Raised when an imported document contains no extractable text."""
    pass


class DuplicateDocumentError(IngestionError):
    """Raised when attempting to import a document that is already indexed."""
    pass


class UnauthorizedDocumentError(IngestionError):
    """Raised when an ingestion attempt lacks explicit user permission/authorization."""
    pass


@dataclass(frozen=True)
class DocumentProvenance:
    """Legal permission, licensing, and provenance metadata.

    Note: Personal purchases do NOT imply permission for redistribution or
    unrestricted training. All flags default to strict conservative values.
    """

    license_type: str = "Personal Research"
    provenance: str = "User Uploaded"
    redistribution_permitted: bool = False
    training_permitted: bool = False
    authorized_by_user: bool = True


@dataclass(frozen=True)
class BookMetadata:
    """Metadata describing an ingested personal finance book or research document."""

    book_id: str
    title: str
    author: str
    filename: str
    source_path: str
    format: str  # 'pdf', 'txt', 'md'
    file_hash: str
    total_words: int = 0
    total_pages: Optional[int] = None
    provenance: DocumentProvenance = field(default_factory=DocumentProvenance)
    created_at: Optional[str] = None


@dataclass(frozen=True)
class PageExtract:
    """Extracted text unit from a specific page or section."""

    page_number: Optional[int]  # 1-indexed for PDF, None for unpaginated text
    text: str
    char_offset: int


@dataclass(frozen=True)
class Chunk:
    """Atomic text passage produced by sliding-window chunking."""

    chunk_id: str
    book_id: str
    chapter_id: Optional[str]
    chunk_index: int
    page_start: Optional[int]
    page_end: Optional[int]
    char_start: int
    char_end: int
    word_count: int
    token_count: int
    content: str


class DocumentParserProtocol(Protocol):
    """Protocol for parsing various raw document formats into structured page extracts."""

    def parse(self, file_path: Path) -> List[PageExtract]:
        """Parse raw document into a sequence of page extracts.

        Args:
            file_path: Path to raw input file.

        Returns:
            List of PageExtract instances.

        Raises:
            FileNotFoundError: If the document does not exist.
            InvalidDocumentError: If format is corrupted or unsupported.
            EmptyDocumentError: If file has no extractable text.
        """
        ...


class ChunkerProtocol(Protocol):
    """Protocol for sliding-window segmentation with metadata and page preservation."""

    def chunk_pages(
        self,
        pages: List[PageExtract],
        book_id: str,
        chapter_id: Optional[str] = None,
        target_words: int = 250,
        overlap_words: int = 50,
    ) -> List[Chunk]:
        """Segment a sequence of pages into overlapping chunks with page tracking.

        Args:
            pages: List of extracted page text blocks.
            book_id: Unique identifier of parent document.
            chapter_id: Optional chapter identifier.
            target_words: Target word count per chunk.
            overlap_words: Word overlap between successive chunks.

        Returns:
            List of generated Chunk instances.
        """
        ...
