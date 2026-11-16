"""Interface contracts for BM25 knowledge retrieval and candidate ranking."""

from dataclasses import dataclass
from typing import List, Protocol
from core.interfaces.ingestion import Chunk


@dataclass(frozen=True)
class ScoredChunk:
    """Retrieved chunk associated with its Okapi BM25 relevance score."""

    chunk: Chunk
    score: float
    rank: int


class BM25ScorerProtocol(Protocol):
    """Protocol for scoring term matches under Okapi BM25 / BM25+."""

    def score(self, query_terms: List[str], chunk_id: str) -> float:
        """Compute BM25 score for given query terms on a specific chunk.

        Args:
            query_terms: Preprocessed query terms.
            chunk_id: Target chunk identifier.

        Returns:
            Relevance score.
        """
        ...


class RetrievalIndexProtocol(Protocol):
    """Protocol for document candidate indexing and search."""

    def index_chunks(self, chunks: List[Chunk]) -> int:
        """Index a batch of chunks into the BM25 inverted index.

        Args:
            chunks: List of Chunk objects to index.

        Returns:
            Number of indexed chunks.
        """
        ...

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score_threshold: float = 1.0,
    ) -> List[ScoredChunk]:
        """Perform BM25 search over indexed passages.

        Args:
            query: User search query.
            top_k: Maximum number of candidate chunks to return.
            min_score_threshold: Minimum BM25 score required for candidate retention.

        Returns:
            List of ScoredChunk sorted by score descending.
        """
        ...
