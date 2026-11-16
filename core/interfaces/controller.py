"""Interface contracts for the Adaptive Reasoning Controller."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Protocol
from core.interfaces.retrieval import ScoredChunk


class TerminationReason(str, Enum):
    """Reason for reasoning loop termination."""

    CONFIDENT_GROUNDING = "confident_grounding"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"
    GOVERNOR_TIMEOUT = "governor_timeout"
    NO_RELEVANT_CONTEXT = "no_relevant_context"
    VALIDATION_FAILURE = "validation_failure"


@dataclass(frozen=True)
class ReasoningStep:
    """Record of a single iteration step within the bounded controller."""

    iteration: int
    query_used: str
    depth_executed: int
    retrieved_count: int
    top_bm25_score: float
    grounding_valid: bool


@dataclass(frozen=True)
class ControllerResult:
    """Final output produced by the adaptive reasoning controller."""

    query: str
    response_text: str
    citations: List[str]
    retrieved_chunks: List[ScoredChunk]
    iterations_used: int
    execution_depth: int
    termination_reason: TerminationReason
    steps: List[ReasoningStep] = field(default_factory=list)


class AdaptiveControllerProtocol(Protocol):
    """Protocol for bounded iterative retrieval and reasoning controller."""

    @property
    def max_iterations(self) -> int:
        """Deterministic upper bound on reasoning loop iterations (must be <= 3)."""
        ...

    def execute(
        self,
        query: str,
        max_iterations: int = 3,
        forced_depth: Optional[int] = None,
    ) -> ControllerResult:
        """Execute bounded retrieval and reasoning workflow.

        Args:
            query: Input user query.
            max_iterations: Maximum iteration steps (capped at 3).
            forced_depth: Optional override for Transformer execution depth (2, 4, 8).

        Returns:
            ControllerResult containing answer, citations, and execution trace.
        """
        ...
