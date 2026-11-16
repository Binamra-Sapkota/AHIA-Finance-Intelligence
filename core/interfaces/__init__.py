"""Module interface definitions and protocol contracts for AHIA-R2 subsystems."""

from core.interfaces.ingestion import (
    BookMetadata,
    Chunk,
    ChunkerProtocol,
    DocumentParserProtocol,
)
from core.interfaces.tokenizer import (
    SpecialTokens,
    TokenizerProtocol,
)
from core.interfaces.retrieval import (
    BM25ScorerProtocol,
    RetrievalIndexProtocol,
    ScoredChunk,
)
from core.interfaces.model import (
    AdaptiveExecutionDepth,
    KVStateProtocol,
    ModelConfig,
    ModelProtocol,
)
from core.interfaces.training import (
    TrainerProtocol,
    TrainingConfig,
)
from core.interfaces.controller import (
    AdaptiveControllerProtocol,
    ControllerResult,
    ReasoningStep,
)
from core.interfaces.reliability import (
    CitationReference,
    CitationVerifierProtocol,
    GroundingReport,
)
from core.interfaces.governor import (
    ExecutionTelemetry,
    GovernorAction,
    ResourceGovernorProtocol,
    ResourceQuota,
)
from core.interfaces.evaluation import (
    EvaluationReport,
    EvaluationSuiteProtocol,
)

__all__ = [
    "BookMetadata",
    "Chunk",
    "DocumentParserProtocol",
    "ChunkerProtocol",
    "TokenizerProtocol",
    "SpecialTokens",
    "RetrievalIndexProtocol",
    "BM25ScorerProtocol",
    "ScoredChunk",
    "ModelProtocol",
    "ModelConfig",
    "KVStateProtocol",
    "AdaptiveExecutionDepth",
    "TrainerProtocol",
    "TrainingConfig",
    "AdaptiveControllerProtocol",
    "ControllerResult",
    "ReasoningStep",
    "CitationVerifierProtocol",
    "CitationReference",
    "GroundingReport",
    "ResourceGovernorProtocol",
    "ResourceQuota",
    "ExecutionTelemetry",
    "GovernorAction",
    "EvaluationSuiteProtocol",
    "EvaluationReport",
]
