"""Subsystem 2: Book Ingestion, parsing, normalization, and chunking."""

from core.interfaces.ingestion import BookMetadata, Chunk, ChunkerProtocol, DocumentParserProtocol

__all__ = ["BookMetadata", "Chunk", "DocumentParserProtocol", "ChunkerProtocol"]
