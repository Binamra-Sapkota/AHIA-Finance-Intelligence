"""Interface contracts for the Evaluation and Benchmarking Suite."""

from dataclasses import dataclass, field
from typing import Dict, List, Protocol


@dataclass(frozen=True)
class EvaluationReport:
    """Benchmark evaluation results."""

    retrieval_recall_at_1: float
    retrieval_recall_at_3: float
    retrieval_recall_at_5: float
    citation_precision: float
    hallucination_rate: float
    avg_tokens_per_sec: float
    avg_latency_ms: float
    deterministic_limit_passed: bool
    summary: str
    detailed_metrics: Dict[str, float] = field(default_factory=dict)


class EvaluationSuiteProtocol(Protocol):
    """Protocol for running offline automated benchmarks."""

    def evaluate_retrieval(self, test_queries: List[Dict[str, str]]) -> Dict[str, float]:
        """Evaluate BM25 recall and precision.

        Args:
            test_queries: List of query-to-ground-truth-chunk mappings.

        Returns:
            Dictionary of retrieval metrics.
        """
        ...

    def evaluate_grounding(self, test_cases: List[Dict[str, str]]) -> Dict[str, float]:
        """Evaluate citation fidelity and anti-hallucination efficacy.

        Args:
            test_cases: List of test cases.

        Returns:
            Dictionary of grounding metrics.
        """
        ...

    def run_full_suite(self) -> EvaluationReport:
        """Execute end-to-end evaluation pipeline and generate report."""
        ...
