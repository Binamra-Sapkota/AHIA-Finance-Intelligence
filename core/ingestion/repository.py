"""SQLite repository for document metadata, provenance, and chunk storage."""

from pathlib import Path
import sqlite3
from typing import List, Optional
from database.connection import get_db_connection, initialize_database
from core.interfaces.ingestion import (
    BookMetadata,
    Chunk,
    DocumentProvenance,
    DuplicateDocumentError,
    UnauthorizedDocumentError,
)


class BookRepository:
    """Manages transactional persistence of documents, chunks, and legal provenance in SQLite."""

    def __init__(
        self,
        db_path: Optional[Path] = None,
        connection: Optional[sqlite3.Connection] = None,
    ) -> None:
        self.db_path = db_path
        self._external_conn = connection

    def _get_conn(self) -> sqlite3.Connection:
        if self._external_conn is not None:
            return self._external_conn
        # Ensure database tables exist if creating new file
        if self.db_path is not None and not self.db_path.exists():
            initialize_database(self.db_path)
        return get_db_connection(self.db_path)

    def _close_conn(self, conn: sqlite3.Connection) -> None:
        """Close connection only if locally created."""
        if self._external_conn is None:
            conn.close()

    def save_book(
        self,
        book: BookMetadata,
        chunks: List[Chunk],
        allow_duplicate: bool = False,
    ) -> str:
        """Store book metadata and generated chunks into SQLite.

        Args:
            book: BookMetadata instance.
            chunks: List of generated Chunk objects.
            allow_duplicate: Whether to overwrite if hash exists.

        Returns:
            Saved book_id.

        Raises:
            UnauthorizedDocumentError: If user authorization flag is False.
            DuplicateDocumentError: If file hash already exists and allow_duplicate is False.
        """
        if not book.provenance.authorized_by_user:
            raise UnauthorizedDocumentError(
                f"Ingestion rejected: Document '{book.title}' has not been authorized by the user."
            )

        conn = self._get_conn()
        try:
            with conn:
                cursor = conn.cursor()

                # Check for duplicate file hash
                if book.file_hash:
                    cursor.execute(
                        "SELECT book_id, title FROM books WHERE file_hash = ?;",
                        (book.file_hash,),
                    )
                    existing = cursor.fetchone()
                    if existing:
                        if not allow_duplicate:
                            raise DuplicateDocumentError(
                                f"Document '{book.title}' is a duplicate of already indexed book '{existing['title']}' (ID: {existing['book_id']})."
                            )
                        # Remove prior entry if overwriting
                        cursor.execute("DELETE FROM books WHERE file_hash = ?;", (book.file_hash,))

                # Insert into books table
                cursor.execute(
                    """
                    INSERT INTO books (
                        book_id, title, author, filename, source_path, format,
                        file_hash, total_words, total_pages, license_type, provenance,
                        redistribution_permitted, training_permitted, authorized_by_user
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        book.book_id,
                        book.title,
                        book.author,
                        book.filename,
                        book.source_path,
                        book.format,
                        book.file_hash,
                        book.total_words,
                        book.total_pages,
                        book.provenance.license_type,
                        book.provenance.provenance,
                        1 if book.provenance.redistribution_permitted else 0,
                        1 if book.provenance.training_permitted else 0,
                        1 if book.provenance.authorized_by_user else 0,
                    ),
                )

                # Insert chunks
                chunk_tuples = [
                    (
                        c.chunk_id,
                        c.book_id,
                        c.chapter_id,
                        c.chunk_index,
                        c.page_start,
                        c.page_end,
                        c.char_start,
                        c.char_end,
                        c.word_count,
                        c.token_count,
                        c.content,
                    )
                    for c in chunks
                ]
                cursor.executemany(
                    """
                    INSERT INTO chunks (
                        chunk_id, book_id, chapter_id, chunk_index,
                        page_start, page_end, char_start, char_end,
                        word_count, token_count, content
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    chunk_tuples,
                )

            return book.book_id
        finally:
            self._close_conn(conn)

    def get_book(self, book_id: str) -> Optional[BookMetadata]:
        """Fetch book metadata by book_id."""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM books WHERE book_id = ?;", (book_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_book(row)
        finally:
            self._close_conn(conn)

    def get_book_by_hash(self, file_hash: str) -> Optional[BookMetadata]:
        """Fetch book metadata by SHA-256 hash."""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM books WHERE file_hash = ?;", (file_hash,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_book(row)
        finally:
            self._close_conn(conn)

    def list_books(self) -> List[BookMetadata]:
        """List all indexed books."""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM books ORDER BY created_at DESC;")
            rows = cursor.fetchall()
            return [self._row_to_book(r) for r in rows]
        finally:
            self._close_conn(conn)

    def get_chunks(self, book_id: str) -> List[Chunk]:
        """Fetch all chunks belonging to a specific book."""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM chunks WHERE book_id = ? ORDER BY chunk_index ASC;",
                (book_id,),
            )
            rows = cursor.fetchall()
            return [self._row_to_chunk(r) for r in rows]
        finally:
            self._close_conn(conn)

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """Fetch a specific chunk by chunk_id."""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM chunks WHERE chunk_id = ?;", (chunk_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_chunk(row)
        finally:
            self._close_conn(conn)

    def count_chunks(self) -> int:
        """Count total chunks across all indexed books."""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM chunks;")
            row = cursor.fetchone()
            return row["count"] if row else 0
        finally:
            self._close_conn(conn)

    def delete_book(self, book_id: str) -> bool:
        """Delete a book and all its associated chunks via foreign key cascade."""
        conn = self._get_conn()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM books WHERE book_id = ?;", (book_id,))
                return cursor.rowcount > 0
        finally:
            self._close_conn(conn)

    @staticmethod
    def _row_to_book(row: sqlite3.Row) -> BookMetadata:
        prov = DocumentProvenance(
            license_type=row["license_type"] or "Personal Research",
            provenance=row["provenance"] or "User Uploaded",
            redistribution_permitted=bool(row["redistribution_permitted"]),
            training_permitted=bool(row["training_permitted"]),
            authorized_by_user=bool(row["authorized_by_user"]),
        )
        return BookMetadata(
            book_id=row["book_id"],
            title=row["title"],
            author=row["author"],
            filename=row["filename"] if "filename" in row.keys() else "",
            source_path=row["source_path"] or "",
            format=row["format"],
            file_hash=row["file_hash"] or "",
            total_words=row["total_words"] or 0,
            total_pages=row["total_pages"],
            provenance=prov,
            created_at=row["created_at"],
        )

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> Chunk:
        return Chunk(
            chunk_id=row["chunk_id"],
            book_id=row["book_id"],
            chapter_id=row["chapter_id"],
            chunk_index=row["chunk_index"],
            page_start=row["page_start"],
            page_end=row["page_end"],
            char_start=row["char_start"],
            char_end=row["char_end"],
            word_count=row["word_count"],
            token_count=row["token_count"],
            content=row["content"],
        )
