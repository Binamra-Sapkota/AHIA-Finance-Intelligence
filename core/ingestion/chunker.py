"""Sliding-window chunker with page tracking and boundary preservation."""

import re
from typing import List, Optional, Tuple
from core.ingestion.normalizer import TextNormalizer
from core.interfaces.ingestion import Chunk, PageExtract


class SlidingWindowChunker:
    """Segments page extracts into overlapping chunks with exact page tracking."""

    def __init__(
        self,
        target_words: int = 250,
        overlap_words: int = 50,
        min_chunk_words: int = 30,
    ) -> None:
        """Initialize chunker.

        Args:
            target_words: Nominal word count per chunk.
            overlap_words: Word overlap between consecutive chunks.
            min_chunk_words: Minimum word threshold to avoid tiny orphan chunks.
        """
        if overlap_words >= target_words:
            raise ValueError(f"overlap_words ({overlap_words}) must be less than target_words ({target_words})")

        self.target_words = target_words
        self.overlap_words = overlap_words
        self.min_chunk_words = min_chunk_words

    def chunk_pages(
        self,
        pages: List[PageExtract],
        book_id: str,
        chapter_id: Optional[str] = None,
    ) -> List[Chunk]:
        """Convert page extracts into structured, overlapping chunks.

        Args:
            pages: Sequence of PageExtracts.
            book_id: Parent document identifier.
            chapter_id: Optional chapter identifier.

        Returns:
            List of generated Chunk records with stable IDs.
        """
        if not pages:
            return []

        # 1. Normalize page texts and collect page-tagged paragraphs
        tagged_paragraphs: List[Tuple[Optional[int], str, int]] = []
        global_char_offset = 0

        for page in pages:
            normalized_page = TextNormalizer.normalize(page.text)
            if not normalized_page:
                continue

            # Split into individual paragraphs
            paragraphs = normalized_page.split("\n\n")
            for p in paragraphs:
                p_clean = p.strip()
                if p_clean:
                    tagged_paragraphs.append((page.page_number, p_clean, global_char_offset))
                    global_char_offset += len(p_clean) + 2

        if not tagged_paragraphs:
            return []

        # 2. Build word-level units with page tags
        # Each word token has (word_str, page_num, char_offset)
        tagged_words: List[Tuple[str, Optional[int]]] = []
        for page_num, para_text, _ in tagged_paragraphs:
            words = para_text.split()
            for w in words:
                tagged_words.append((w, page_num))

        if not tagged_words:
            return []

        total_words = len(tagged_words)
        chunks: List[Chunk] = []
        chunk_idx = 1
        start_idx = 0
        step = self.target_words - self.overlap_words

        while start_idx < total_words:
            end_idx = min(start_idx + self.target_words, total_words)

            # If remaining words are less than min_chunk_words and we already have chunks,
            # merge them into the previous chunk instead of creating an orphan
            if (end_idx - start_idx) < self.min_chunk_words and chunks:
                prev = chunks[-1]
                extra_words = [w for w, _ in tagged_words[start_idx:end_idx]]
                merged_content = prev.content + " " + " ".join(extra_words)
                merged_pages = [p for _, p in tagged_words[start_idx:end_idx] if p is not None]
                new_page_end = max(merged_pages) if merged_pages else prev.page_end

                updated_chunk = Chunk(
                    chunk_id=prev.chunk_id,
                    book_id=prev.book_id,
                    chapter_id=prev.chapter_id,
                    chunk_index=prev.chunk_index,
                    page_start=prev.page_start,
                    page_end=new_page_end,
                    char_start=prev.char_start,
                    char_end=prev.char_end + len(" ".join(extra_words)) + 1,
                    word_count=len(merged_content.split()),
                    token_count=int(len(merged_content.split()) * 1.3),
                    content=merged_content,
                )
                chunks[-1] = updated_chunk
                break

            slice_words = tagged_words[start_idx:end_idx]
            content = " ".join(w for w, _ in slice_words)

            # Determine page range
            pages_in_chunk = [p for _, p in slice_words if p is not None]
            page_start = min(pages_in_chunk) if pages_in_chunk else None
            page_end = max(pages_in_chunk) if pages_in_chunk else None

            # Generate stable chunk ID
            chunk_id = f"{book_id}_CHUNK_{chunk_idx:04d}"

            chunk = Chunk(
                chunk_id=chunk_id,
                book_id=book_id,
                chapter_id=chapter_id,
                chunk_index=chunk_idx,
                page_start=page_start,
                page_end=page_end,
                char_start=start_idx,
                char_end=start_idx + len(content),
                word_count=len(slice_words),
                token_count=int(len(slice_words) * 1.3),
                content=content,
            )
            chunks.append(chunk)
            chunk_idx += 1

            if end_idx >= total_words:
                break
            start_idx += step

        return chunks
