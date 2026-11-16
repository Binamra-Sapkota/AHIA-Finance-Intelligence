"""Service layer decoupling core algorithms from the PySide6 UI."""

from service.dto import (
    BookIngestRequest,
    BookIngestSummary,
    CitationItem,
    QueryRequest,
    QueryResult,
    SystemDiagnostics,
)

__all__ = [
    "BookIngestRequest",
    "BookIngestSummary",
    "CitationItem",
    "QueryRequest",
    "QueryResult",
    "SystemDiagnostics",
]
