"""Subsystem 2: Book Ingestion, parsing, normalization, and chunking."""

from core.ingestion.chunker import SlidingWindowChunker
from core.ingestion.normalizer import TextNormalizer
from core.ingestion.parser import DocumentParser
from core.ingestion.pipeline import IngestionPipeline
from core.ingestion.repository import BookRepository
from core.interfaces.ingestion import (
    BookMetadata,
    Chunk,
    ChunkerProtocol,
    DocumentParserProtocol,
    DocumentProvenance,
    DuplicateDocumentError,
    EmptyDocumentError,
    IngestionError,
    InvalidDocumentError,
    PageExtract,
    UnauthorizedDocumentError,
)

__all__ = [
    "BookMetadata",
    "Chunk",
    "DocumentProvenance",
    "PageExtract",
    "DocumentParserProtocol",
    "ChunkerProtocol",
    "IngestionError",
    "InvalidDocumentError",
    "EmptyDocumentError",
    "DuplicateDocumentError",
    "UnauthorizedDocumentError",
    "DocumentParser",
    "TextNormalizer",
    "SlidingWindowChunker",
    "BookRepository",
    "IngestionPipeline",
]
