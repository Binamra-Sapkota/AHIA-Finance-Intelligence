"""Pure-Python Okapi BM25 retrieval engine with page-aware citation formatting."""

import math
import re
from typing import Dict, List, Optional, Set, Tuple
from core.interfaces.ingestion import Chunk
from core.interfaces.retrieval import ScoredChunk

# Minimal standard English stopwords to focus retrieval on informational terms
DEFAULT_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
}


class BM25Index:
    """Okapi BM25 inverted index with document length normalization and candidate scoring."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        stopwords: Optional[Set[str]] = None,
    ) -> None:
        """Initialize BM25 indexer.

        Args:
            k1: Term frequency saturation parameter (default 1.5).
            b: Document length normalization parameter (default 0.75).
            stopwords: Optional set of stopwords to filter out.
        """
        self.k1 = k1
        self.b = b
        self.stopwords = stopwords if stopwords is not None else DEFAULT_STOPWORDS

        # Storage structures
        # inverted_index: term -> {chunk_id: term_frequency}
        self.inverted_index: Dict[str, Dict[str, int]] = {}
        # doc_lengths: chunk_id -> token_length
        self.doc_lengths: Dict[str, int] = {}
        # chunk_lookup: chunk_id -> Chunk
        self.chunk_lookup: Dict[str, Chunk] = {}
        # doc_meta_lookup: chunk_id -> (title, author)
        self.doc_meta_lookup: Dict[str, Tuple[str, str]] = {}

        self.total_docs: int = 0
        self.avg_doc_len: float = 0.0

    def tokenize(self, text: str) -> List[str]:
        """Convert text into normalized term tokens."""
        if not text:
            return []
        raw_terms = re.findall(r"\b[a-zA-Z0-9_]{2,}\b", text.lower())
        return [t for t in raw_terms if t not in self.stopwords]

    def index_chunks(
        self,
        chunks: List[Chunk],
        doc_title: str = "",
        doc_author: str = "",
    ) -> int:
        """Add chunks to the inverted index.

        Args:
            chunks: List of Chunk objects.
            doc_title: Title of parent document.
            doc_author: Author of parent document.

        Returns:
            Number of newly indexed chunks.
        """
        count = 0
        for chunk in chunks:
            tokens = self.tokenize(chunk.content)
            doc_len = len(tokens)
            if doc_len == 0:
                continue

            self.chunk_lookup[chunk.chunk_id] = chunk
            self.doc_lengths[chunk.chunk_id] = doc_len
            self.doc_meta_lookup[chunk.chunk_id] = (doc_title, doc_author)

            # Calculate term frequencies
            tf_map: Dict[str, int] = {}
            for t in tokens:
                tf_map[t] = tf_map.get(t, 0) + 1

            for term, freq in tf_map.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = {}
                self.inverted_index[term][chunk.chunk_id] = freq

            count += 1

        self.total_docs = len(self.doc_lengths)
        if self.total_docs > 0:
            self.avg_doc_len = sum(self.doc_lengths.values()) / self.total_docs
        else:
            self.avg_doc_len = 0.0

        return count

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score_threshold: float = 0.5,
    ) -> List[ScoredChunk]:
        """Execute BM25 search over all indexed passages.

        Args:
            query: User search query.
            top_k: Maximum candidate passages to return.
            min_score_threshold: Minimum BM25 score required for candidate retention.

        Returns:
            List of ScoredChunk sorted by descending score with references.
        """
        query_terms = self.tokenize(query)
        if not query_terms or self.total_docs == 0:
            return []

        # Accumulate scores for candidate chunks
        scores: Dict[str, float] = {}

        for term in query_terms:
            if term not in self.inverted_index:
                continue

            posting_list = self.inverted_index[term]
            doc_freq = len(posting_list)

            # Okapi BM25 IDF formulation: ln((N - n + 0.5) / (n + 0.5) + 1)
            idf = math.log(((self.total_docs - doc_freq + 0.5) / (doc_freq + 0.5)) + 1.0)

            for chunk_id, freq in posting_list.items():
                doc_len = self.doc_lengths[chunk_id]
                # Length normalization denominator
                len_norm = 1.0 - self.b + self.b * (doc_len / self.avg_doc_len)
                term_score = idf * ((freq * (self.k1 + 1.0)) / (freq + self.k1 * len_norm))
                scores[chunk_id] = scores.get(chunk_id, 0.0) + term_score

        if not scores:
            return []

        # Filter by threshold and sort descending
        filtered_items = [
            (cid, score) for cid, score in scores.items() if score >= min_score_threshold
        ]
        filtered_items.sort(key=lambda x: x[1], reverse=True)
        top_items = filtered_items[:top_k]

        results: List[ScoredChunk] = []
        for rank, (cid, score) in enumerate(top_items, start=1):
            chunk = self.chunk_lookup[cid]
            title, author = self.doc_meta_lookup.get(cid, ("Unknown Title", "Unknown Author"))
            ref_label = self._format_reference(title, chunk.page_start, chunk.page_end, chunk.chunk_index)

            results.append(ScoredChunk(
                chunk=chunk,
                score=round(score, 3),
                rank=rank,
                doc_title=title,
                doc_author=author,
                reference_label=ref_label,
            ))

        return results

    @staticmethod
    def _format_reference(
        title: str,
        page_start: Optional[int],
        page_end: Optional[int],
        chunk_index: int,
    ) -> str:
        """Format reliable citation reference string."""
        if page_start is not None:
            if page_end is not None and page_end != page_start:
                return f"[{title}, pp. {page_start}–{page_end}]"
            return f"[{title}, p. {page_start}]"
        return f"[{title}, Section {chunk_index}]"
