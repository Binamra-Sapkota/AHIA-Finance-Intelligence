"""Automated tests for Phase 2: Ingestion, Chunking, SQLite Storage, and BM25 Retrieval."""

from pathlib import Path
import pytest

from core.ingestion.chunker import SlidingWindowChunker
from core.ingestion.normalizer import TextNormalizer
from core.ingestion.parser import DocumentParser
from core.ingestion.pipeline import IngestionPipeline
from core.ingestion.repository import BookRepository
from core.interfaces.ingestion import (
    BookMetadata,
    Chunk,
    DocumentProvenance,
    DuplicateDocumentError,
    EmptyDocumentError,
    InvalidDocumentError,
    PageExtract,
    UnauthorizedDocumentError,
)
from core.retrieval.bm25 import BM25Index


def test_pdf_extraction(sample_pdf_file: Path) -> None:
    """Test extracting text page-by-page from a multi-page PDF document."""
    parser = DocumentParser()
    pages, meta = parser.parse(sample_pdf_file)

    assert len(pages) == 2, "Expected 2 extracted pages from sample PDF"
    assert pages[0].page_number == 1
    assert pages[1].page_number == 2
    assert "retaining ten percent" in pages[0].text
    assert "claim on future labor" in pages[1].text
    assert meta["format"] == "pdf"
    assert meta["total_pages"] == 2
    assert len(meta["file_hash"]) == 64  # Valid SHA-256


def test_text_extraction(sample_text_file: Path) -> None:
    """Test extracting text from Markdown/TXT and normalizing headings."""
    parser = DocumentParser()
    pages, meta = parser.parse(sample_text_file)

    assert len(pages) == 1
    assert pages[0].page_number is None  # Continuous text has no page numbers
    assert "Principles of Personal Financial Intelligence" in pages[0].text
    assert "compound interest" in pages[0].text.lower()
    assert meta["title"] == "Principles of Personal Financial Intelligence"
    assert meta["format"] == "md"
    assert len(meta["file_hash"]) == 64


def test_chunk_boundaries() -> None:
    """Test sliding-window chunking respecting target words, overlaps, and paragraph boundaries."""
    pages = [
        PageExtract(page_number=1, text="Paragraph one discusses income and budgeting.\n\nParagraph two covers saving.", char_offset=0),
        PageExtract(page_number=2, text="Paragraph three discusses investing.\n\nParagraph four covers risk.", char_offset=100),
    ]
    chunker = SlidingWindowChunker(target_words=10, overlap_words=3, min_chunk_words=3)
    chunks = chunker.chunk_pages(pages, book_id="TEST_DOC_001")

    assert len(chunks) >= 2, "Expected multiple chunks for test text"
    for idx, c in enumerate(chunks, start=1):
        assert c.chunk_id == f"TEST_DOC_001_CHUNK_{idx:04d}", "Stable ID pattern mismatch"
        assert c.word_count > 0
        assert c.char_start >= 0
        assert c.char_end > c.char_start

    # Verify page tracking
    assert chunks[0].page_start == 1


def test_text_normalizer_paragraph_preservation() -> None:
    """Verify that text normalizer cleans whitespace while preserving paragraph double-newlines."""
    raw = "Line one with soft wrap\nand extra   spaces.\n\nParagraph two with hyphen-\nated word."
    norm = TextNormalizer.normalize(raw)

    assert "\n\n" in norm, "Paragraph break must be preserved as double-newline"
    assert "hyphenated word" in norm, "Hyphenation across line break must be rejoined"
    assert "   " not in norm, "Excess whitespace must be collapsed"


def test_metadata_preservation(in_memory_db, sample_text_file: Path) -> None:
    """Verify that title, author, filename, hash, and provenance are strictly stored in SQLite."""
    repo = BookRepository(connection=in_memory_db)

    prov = DocumentProvenance(
        license_type="Public Domain",
        provenance="Gutenberg Archive",
        redistribution_permitted=True,
        training_permitted=True,
        authorized_by_user=True,
    )

    book = BookMetadata(
        book_id="DOC_FINANCE_001",
        title="Financial Intelligence Principles",
        author="Research Team",
        filename="financial_principles.md",
        source_path=str(sample_text_file),
        format="md",
        file_hash="abc123def456",
        total_words=350,
        total_pages=None,
        provenance=prov,
    )

    chunks = [
        Chunk(
            chunk_id="DOC_FINANCE_001_CHUNK_0001",
            book_id="DOC_FINANCE_001",
            chapter_id=None,
            chunk_index=1,
            page_start=None,
            page_end=None,
            char_start=0,
            char_end=150,
            word_count=25,
            token_count=32,
            content="Compound interest functions as an exponential force over extended time horizons.",
        )
    ]

    saved_id = repo.save_book(book, chunks)
    assert saved_id == "DOC_FINANCE_001"

    retrieved_book = repo.get_book("DOC_FINANCE_001")
    assert retrieved_book is not None
    assert retrieved_book.title == "Financial Intelligence Principles"
    assert retrieved_book.author == "Research Team"
    assert retrieved_book.file_hash == "abc123def456"
    assert retrieved_book.provenance.license_type == "Public Domain"
    assert retrieved_book.provenance.redistribution_permitted is True
    assert retrieved_book.provenance.training_permitted is True
    assert retrieved_book.provenance.authorized_by_user is True

    retrieved_chunks = repo.get_chunks("DOC_FINANCE_001")
    assert len(retrieved_chunks) == 1
    assert "Compound interest" in retrieved_chunks[0].content


def test_bm25_ranking(sample_text_file: Path) -> None:
    """Verify that BM25 accurately scores and ranks the most relevant passage first."""
    parser = DocumentParser()
    pages, _ = parser.parse(sample_text_file)

    chunker = SlidingWindowChunker(target_words=35, overlap_words=10)
    chunks = chunker.chunk_pages(pages, book_id="FIN_PRINCIPLES")

    bm25 = BM25Index()
    bm25.index_chunks(chunks, doc_title="Principles of Personal Financial Intelligence", doc_author="Research Team")

    # Search query about compound interest
    query = "How does compound interest work over time?"
    results = bm25.search(query, top_k=3)

    assert len(results) > 0, "Expected at least 1 BM25 retrieval match"
    top_result = results[0]
    assert "compound interest" in top_result.chunk.content.lower()
    assert top_result.score > 1.0
    assert "Principles of Personal Financial Intelligence" in top_result.reference_label

    # Search query about emergency buffer
    buffer_results = bm25.search("emergency reserve living expenses buffer", top_k=1)
    assert len(buffer_results) == 1
    assert "emergency reserve" in buffer_results[0].chunk.content.lower()


def test_bm25_page_reference_formatting(sample_pdf_file: Path) -> None:
    """Verify reliable page references [Document, p. X] and [Document, pp. X-Y]."""
    parser = DocumentParser()
    pages, _ = parser.parse(sample_pdf_file)

    # 1. Single page chunking (target_words = 12)
    chunker = SlidingWindowChunker(target_words=12, overlap_words=2, min_chunk_words=3)
    chunks = chunker.chunk_pages(pages, book_id="PDF_SAMPLE")

    bm25 = BM25Index()
    bm25.index_chunks(chunks, doc_title="Sample Wealth Guide", doc_author="Author")

    results = bm25.search("retaining ten percent", top_k=1)
    assert len(results) == 1
    assert "[Sample Wealth Guide, p. 1]" in results[0].reference_label

    debt_results = bm25.search("future labor debt liabilities", top_k=1)
    assert len(debt_results) == 1
    assert "[Sample Wealth Guide, p. 2]" in debt_results[0].reference_label

    # 2. Multi-page chunking (target_words = 40)
    multi_chunker = SlidingWindowChunker(target_words=40, overlap_words=5)
    multi_chunks = multi_chunker.chunk_pages(pages, book_id="PDF_MULTI")
    multi_bm25 = BM25Index()
    multi_bm25.index_chunks(multi_chunks, doc_title="Sample Wealth Guide", doc_author="Author")

    multi_results = multi_bm25.search("labor liabilities", top_k=1)
    assert len(multi_results) == 1
    assert "pp. 1" in multi_results[0].reference_label


def test_empty_documents(empty_text_file: Path) -> None:
    """Verify that importing an empty file raises EmptyDocumentError."""
    parser = DocumentParser()
    with pytest.raises(EmptyDocumentError):
        parser.parse(empty_text_file)


def test_duplicate_imports(in_memory_db, sample_text_file: Path) -> None:
    """Verify that importing an identical document hash twice raises DuplicateDocumentError."""
    repo = BookRepository(connection=in_memory_db)
    pipeline = IngestionPipeline(repository=repo)

    # First import succeeds
    meta1, chunks1 = pipeline.ingest_file(sample_text_file)
    assert meta1 is not None

    # Second import without allow_duplicate must raise DuplicateDocumentError
    with pytest.raises(DuplicateDocumentError):
        pipeline.ingest_file(sample_text_file, allow_duplicate=False)

    # Import with allow_duplicate=True succeeds
    meta2, chunks2 = pipeline.ingest_file(sample_text_file, allow_duplicate=True)
    assert meta2.file_hash == meta1.file_hash


def test_invalid_files(corrupted_pdf_file: Path, tmp_path: Path) -> None:
    """Verify that corrupted or unsupported documents raise InvalidDocumentError or FileNotFoundError."""
    parser = DocumentParser()

    # Corrupt PDF
    with pytest.raises(InvalidDocumentError):
        parser.parse(corrupted_pdf_file)

    # Non-existent file
    with pytest.raises(FileNotFoundError):
        parser.parse(tmp_path / "does_not_exist.pdf")

    # Unsupported file extension
    unsupported_file = tmp_path / "archive.zip"
    unsupported_file.write_bytes(b"PK\x03\x04")
    with pytest.raises(InvalidDocumentError):
        parser.parse(unsupported_file)


def test_unauthorized_documents(sample_text_file: Path, in_memory_db) -> None:
    """Verify that importing a document where authorized_by_user=False is rejected."""
    repo = BookRepository(connection=in_memory_db)
    pipeline = IngestionPipeline(repository=repo)

    unauthorized_prov = DocumentProvenance(
        license_type="Proprietary",
        provenance="Unauthorized Scraping",
        authorized_by_user=False,  # User did not grant authorization
    )

    with pytest.raises(UnauthorizedDocumentError):
        pipeline.ingest_file(sample_text_file, provenance=unauthorized_prov)
