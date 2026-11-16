"""Interface contracts and view specifications for the PySide6 UI.

CRITICAL RULE: This module and the entire ui/ package must never import PyTorch.
Data is exchanged purely using service DTOs.
"""

from typing import List, Optional, Protocol
from service.dto import CitationItem, QueryResult, SystemDiagnostics


class ChatViewProtocol(Protocol):
    """Protocol for the main conversation canvas view."""

    def append_user_message(self, message: str) -> None:
        """Render user input message bubble."""
        ...

    def start_streaming_response(self) -> None:
        """Initialize response container for incoming token stream."""
        ...

    def append_token(self, token_text: str) -> None:
        """Append streamed token to current response bubble."""
        ...

    def finalize_response(self, result: QueryResult) -> None:
        """Finalize message bubble with metadata chips and citation links."""
        ...


class CitationViewProtocol(Protocol):
    """Protocol for the slide-out citation and passage inspector."""

    def display_citations(self, citations: List[CitationItem]) -> None:
        """Populate citation drawer with source passages."""
        ...

    def highlight_citation(self, chunk_id: str) -> None:
        """Scroll to and visually highlight a specific citation passage."""
        ...


class LibraryViewProtocol(Protocol):
    """Protocol for managing indexed books and storage stats."""

    def update_book_list(self, books: List[dict]) -> None:
        """Refresh displayed inventory of local personal-finance books."""
        ...

    def set_indexing_progress(self, percent: int, status_message: str) -> None:
        """Update progress bar during book ingestion and indexing."""
        ...


class DiagnosticsViewProtocol(Protocol):
    """Protocol for real-time resource monitor bar."""

    def update_metrics(self, diagnostics: SystemDiagnostics) -> None:
        """Update CPU, latency, memory, and token generation speed indicators."""
        ...


class MainWindowProtocol(Protocol):
    """Protocol for the primary desktop application window."""

    def show_error(self, title: str, message: str) -> None:
        """Display error dialog to the user."""
        ...

    def set_busy_state(self, is_busy: bool) -> None:
        """Update cursor and disable input during model inference."""
        ...
