"""Document parsing engine for PDF, TXT, and Markdown files."""

import hashlib
from pathlib import Path
import re
from typing import List, Optional, Tuple
import pypdf

from core.interfaces.ingestion import (
    EmptyDocumentError,
    InvalidDocumentError,
    PageExtract,
)


class DocumentParser:
    """Extracts text and page structure from local PDF, TXT, and Markdown documents."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}

    def parse(self, file_path: Path) -> Tuple[List[PageExtract], dict]:
        """Parse document into a sequence of PageExtracts and metadata dictionary.

        Args:
            file_path: Absolute or relative path to target document.

        Returns:
            Tuple of (List of PageExtracts, metadata dict).

        Raises:
            FileNotFoundError: If the file does not exist.
            InvalidDocumentError: If format is unsupported or file is corrupted.
            EmptyDocumentError: If extracted content is empty.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not path.is_file():
            raise InvalidDocumentError(f"Target path is not a regular file: {path}")

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise InvalidDocumentError(
                f"Unsupported file format '{ext}'. Supported: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}"
            )

        # Compute SHA-256 hash
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                sha256.update(block)
        file_hash = sha256.hexdigest()

        if ext == ".pdf":
            pages, meta = self._parse_pdf(path)
        else:
            pages, meta = self._parse_text_or_markdown(path)

        # Check for empty content
        total_text = "".join(p.text.strip() for p in pages)
        if not total_text:
            raise EmptyDocumentError(f"Document contains no readable text: {path.name}")

        meta["file_hash"] = file_hash
        meta["filename"] = path.name
        meta["format"] = ext.replace(".", "")
        meta["source_path"] = str(path)

        return pages, meta

    def _parse_pdf(self, path: Path) -> Tuple[List[PageExtract], dict]:
        """Extract text page-by-page using pypdf."""
        pages: List[PageExtract] = []
        meta = {
            "title": path.stem.replace("_", " ").title(),
            "author": "Unknown Author",
            "total_pages": 0,
        }

        try:
            reader = pypdf.PdfReader(str(path))
            meta["total_pages"] = len(reader.pages)

            # Attempt to extract metadata from PDF catalog
            if reader.metadata:
                if reader.metadata.title:
                    meta["title"] = str(reader.metadata.title).strip()
                if reader.metadata.author:
                    meta["author"] = str(reader.metadata.author).strip()

            char_offset = 0
            for idx, page in enumerate(reader.pages):
                page_num = idx + 1
                try:
                    raw_text = page.extract_text() or ""
                    # Normalize tabs and spaces
                    text = re.sub(r"[ \t]+", " ", raw_text)
                except Exception as extract_err:
                    text = ""

                pages.append(PageExtract(
                    page_number=page_num,
                    text=text,
                    char_offset=char_offset,
                ))
                char_offset += len(text)

        except Exception as e:
            if isinstance(e, (EmptyDocumentError, InvalidDocumentError)):
                raise
            raise InvalidDocumentError(f"Failed to parse PDF file '{path.name}': {e}") from e

        return pages, meta

    def _parse_text_or_markdown(self, path: Path) -> Tuple[List[PageExtract], dict]:
        """Extract text from plain text or markdown files."""
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        raw_text: Optional[str] = None

        for enc in encodings:
            try:
                with open(path, "r", encoding=enc) as f:
                    raw_text = f.read()
                break
            except (UnicodeDecodeError, OSError):
                continue

        if raw_text is None:
            raise InvalidDocumentError(f"Could not decode text file with supported encodings: {path.name}")

        # Derive initial metadata from markdown title / filename
        title = path.stem.replace("_", " ").title()
        author = "Unknown Author"

        # Check for markdown top heading (# Title)
        lines = raw_text.splitlines()
        for line in lines[:10]:
            stripped = line.strip()
            if stripped.startswith("# "):
                title = stripped[2:].strip()
                break

        pages = [PageExtract(
            page_number=None,
            text=raw_text,
            char_offset=0,
        )]

        meta = {
            "title": title,
            "author": author,
            "total_pages": None,
        }
        return pages, meta
