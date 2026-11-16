"""High-level ingestion pipeline coordinating parsing, chunking, SQLite, and BM25."""

import re
from pathlib import Path
from typing import List, Optional, Tuple

from core.ingestion.chunker import SlidingWindowChunker
from core.ingestion.parser import DocumentParser
from core.ingestion.repository import BookRepository
from core.interfaces.ingestion import (
    BookMetadata,
    Chunk,
    DocumentProvenance,
    UnauthorizedDocumentError,
)
from core.retrieval.bm25 import BM25Index


class IngestionPipeline:
    """100% offline local pipeline for document ingestion and retrieval preparation."""

    def __init__(
        self,
        repository: Optional[BookRepository] = None,
        bm25_index: Optional[BM25Index] = None,
        chunk_target_words: int = 250,
        chunk_overlap_words: int = 50,
    ) -> None:
        self.parser = DocumentParser()
        self.chunker = SlidingWindowChunker(
            target_words=chunk_target_words,
            overlap_words=chunk_overlap_words,
        )
        self.repo = repository or BookRepository()
        self.bm25 = bm25_index or BM25Index()

    def ingest_file(
        self,
        file_path: Path,
        title: Optional[str] = None,
        author: Optional[str] = None,
        provenance: Optional[DocumentProvenance] = None,
        allow_duplicate: bool = False,
    ) -> Tuple[BookMetadata, List[Chunk]]:
        """Ingest a single user-authorized document.

        Args:
            file_path: Path to target file (PDF, TXT, or MD).
            title: Optional title override.
            author: Optional author override.
            provenance: Optional DocumentProvenance (defaults to Personal Research, User Authorized).
            allow_duplicate: Whether to overwrite existing file with identical hash.

        Returns:
            Tuple of (BookMetadata, List[Chunk]).

        Raises:
            UnauthorizedDocumentError: If provenance lacks explicit user authorization.
            DuplicateDocumentError: If document hash already exists in database.
            InvalidDocumentError: If document is corrupt or unreadable.
            EmptyDocumentError: If document has no extractable text.
        """
        # 1. Parse document into pages and extracted metadata
        pages, raw_meta = self.parser.parse(file_path)

        # 2. Determine provenance and verify permission
        prov = provenance or DocumentProvenance(
            license_type="Personal Research",
            provenance="User Uploaded",
            redistribution_permitted=False,
            training_permitted=False,
            authorized_by_user=True,
        )

        if not prov.authorized_by_user:
            raise UnauthorizedDocumentError(
                f"Ingestion rejected for '{file_path.name}': Document lacks explicit user authorization."
            )

        doc_title = title or raw_meta.get("title", file_path.stem.title())
        doc_author = author or raw_meta.get("author", "Unknown Author")
        file_hash = raw_meta["file_hash"]
        short_hash = file_hash[:8]

        # Generate stable, human-readable document ID
        clean_slug = re.sub(r"[^A-Za-z0-9]+", "_", doc_title).strip("_").upper()[:24]
        doc_id = f"DOC_{clean_slug}_{short_hash}"

        # 3. Calculate word count
        total_words = sum(len(p.text.split()) for p in pages)

        book_meta = BookMetadata(
            book_id=doc_id,
            title=doc_title,
            author=doc_author,
            filename=file_path.name,
            source_path=str(file_path.resolve()),
            format=raw_meta["format"],
            file_hash=file_hash,
            total_words=total_words,
            total_pages=raw_meta.get("total_pages"),
            provenance=prov,
        )

        # 4. Chunk pages
        chunks = self.chunker.chunk_pages(pages, book_id=doc_id)

        # 5. Store metadata and chunks in SQLite
        self.repo.save_book(book_meta, chunks, allow_duplicate=allow_duplicate)

        # 6. Index into BM25
        self.bm25.index_chunks(chunks, doc_title=doc_title, doc_author=doc_author)

        return book_meta, chunks
