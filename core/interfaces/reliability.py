"""Interface contracts for the Reliability and Grounding Engine."""

from dataclasses import dataclass, field
from typing import List, Protocol
from core.interfaces.ingestion import Chunk


@dataclass(frozen=True)
class CitationReference:
    """Detailed record of a verified citation in generated text."""

    chunk_id: str
    book_title: str
    chapter_title: str
    quoted_text: str
    exact_match: bool
    similarity_score: float


@dataclass(frozen=True)
class GroundingReport:
    """Comprehensive validation report for model-generated output."""

    is_grounded: bool
    verified_citations: List[CitationReference]
    unsupported_claims: List[str]
    hallucination_detected: bool
    sanitized_response: str


class CitationVerifierProtocol(Protocol):
    """Protocol for checking model output against retrieved knowledge chunks."""

    def verify_grounding(
        self,
        response_text: str,
        retrieved_chunks: List[Chunk],
        strictness_threshold: float = 0.85,
    ) -> GroundingReport:
        """Validate that all claims and quotations exist within the retrieved chunks.

        Args:
            response_text: Generated response text from neural model.
            retrieved_chunks: Ground truth chunks provided as context.
            strictness_threshold: Minimum match ratio for cited quotes.

        Returns:
            GroundingReport with verification status and sanitized text.
        """
        ...
