"""Repository consistency tests validating Phase 0 architecture, docs, and interfaces."""

import ast
from pathlib import Path
import sqlite3
import pytest

from core.interfaces.model import AdaptiveExecutionDepth, ModelConfig
from core.interfaces.governor import ResourceQuota
from core.interfaces.tokenizer import SpecialTokens
from core.interfaces.ingestion import BookMetadata, Chunk
from core.interfaces.retrieval import ScoredChunk
from core.interfaces.controller import ControllerResult, TerminationReason
from core.interfaces.reliability import GroundingReport
from core.interfaces.evaluation import EvaluationReport
from service.dto import QueryRequest, QueryResult, CitationItem


def test_core_documentation_files_exist(repo_root: Path) -> None:
    """Verify that all mandatory Phase 0 documentation and metadata files exist and are non-empty."""
    required_files = [
        "README.md",
        "AGENTS.md",
        "LICENSE",
        "pyproject.toml",
        "requirements.txt",
        "docs/architecture.md",
        "docs/training_plan.md",
        "docs/data_policy.md",
        "docs/development_roadmap.md",
    ]
    for file_rel in required_files:
        file_path = repo_root / file_rel
        assert file_path.exists(), f"Missing required file: {file_rel}"
        content = file_path.read_text(encoding="utf-8").strip()
        assert len(content) > 50, f"File appears too short or empty: {file_rel}"


def test_neural_model_configuration_matches_spec() -> None:
    """Verify that ModelConfig defaults strictly adhere to the neural target specifications."""
    config = ModelConfig()
    assert config.num_layers == 8, "Expected 8 Transformer blocks"
    assert config.hidden_dim == 256, "Expected hidden dimension of 256"
    assert config.num_heads == 4, "Expected 4 attention heads"
    assert config.ff_dim == 1024, "Expected feed-forward dimension of 1024"
    assert config.vocab_size == 8000, "Expected target vocabulary of 8000 tokens"
    assert config.max_context_len == 512, "Expected maximum context window of 512 tokens"
    assert config.supported_depths == (2, 4, 8), "Expected adaptive depths of (2, 4, 8)"
    assert config.num_experts == 4, "Expected 4 planned mixture-of-experts"
    assert config.moe_enabled is False, "Dense baseline must be active initially before MoE"


def test_adaptive_execution_depth_enum() -> None:
    """Verify the enum values for early-exit depths."""
    assert AdaptiveExecutionDepth.FAST == 2
    assert AdaptiveExecutionDepth.MEDIUM == 4
    assert AdaptiveExecutionDepth.DEEP == 8


def test_resource_governor_quotas_match_spec() -> None:
    """Verify independent resource governor quotas enforce safety limits."""
    quota = ResourceQuota()
    assert quota.max_wall_clock_seconds == 10.0, "Watchdog timeout must be 10.0s"
    assert quota.max_output_tokens == 256, "Max output tokens must be 256"
    assert quota.max_memory_mb > 0


def test_special_tokens_definitions() -> None:
    """Verify special token identifiers for the scratch tokenizer."""
    specials = SpecialTokens()
    assert specials.pad_id == 0
    assert specials.bos_id == 1
    assert specials.eos_id == 2
    assert specials.unk_id == 3
    assert specials.query_token == "<query>"
    assert specials.context_token == "<context>"
    assert specials.response_token == "<response>"
    assert specials.cite_token == "<cite>"
    assert specials.nocontext_token == "<nocontext>"


def test_sqlite_schema_initialization(in_memory_db: sqlite3.Connection) -> None:
    """Verify that SQLite tables and indexes initialize cleanly and accept foreign-keyed records."""
    cursor = in_memory_db.cursor()

    # Query sqlite_master for created tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = {row["name"] for row in cursor.fetchall()}
    expected_tables = {"books", "chapters", "chunks", "bm25_terms", "citations", "execution_logs"}
    assert expected_tables.issubset(tables), f"Missing tables. Found: {tables}"

    # Verify inserting a book and chunk with foreign key integrity
    cursor.execute(
        "INSERT INTO books (book_id, title, author, format, total_words) VALUES (?, ?, ?, ?, ?);",
        ("RICHEST_BABYLON", "The Richest Man in Babylon", "George S. Clason", "txt", 42000),
    )
    cursor.execute(
        """INSERT INTO chunks (chunk_id, book_id, chunk_index, char_start, char_end,
           word_count, token_count, content) VALUES (?, ?, ?, ?, ?, ?, ?, ?);""",
        ("BAB_0001", "RICHEST_BABYLON", 1, 0, 100, 20, 25, "Gold cometh gladly and in increasing quantity..."),
    )
    in_memory_db.commit()

    cursor.execute("SELECT COUNT(*) as count FROM chunks WHERE book_id = 'RICHEST_BABYLON';")
    assert cursor.fetchone()["count"] == 1


def test_subsystem_packages_importable() -> None:
    """Verify that all 10 subsystems can be cleanly imported."""
    import core.ingestion
    import core.tokenizer
    import core.dataset
    import core.model
    import core.training
    import core.controller
    import core.retrieval
    import core.reliability
    import core.governor
    import core.evaluation
    import service.dto
    import ui.interfaces

    assert core.model.ModelConfig is not None
    assert service.dto.QueryResult is not None
    assert ui.interfaces.ChatViewProtocol is not None


def test_strict_ui_and_dto_decoupling(repo_root: Path) -> None:
    """Ensure that ui/ and service/dto.py never import PyTorch (torch / torch.nn)."""
    paths_to_audit = list((repo_root / "ui").rglob("*.py")) + [repo_root / "service" / "dto.py"]

    for py_file in paths_to_audit:
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "torch" not in alias.name, (
                        f"FORBIDDEN IMPORT: '{alias.name}' in {py_file}. UI/DTO must never import PyTorch!"
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert "torch" not in node.module, (
                        f"FORBIDDEN IMPORT: 'from {node.module}' in {py_file}. UI/DTO must never import PyTorch!"
                    )


def test_data_transfer_objects_instantiation() -> None:
    """Verify typed DTO instantiation without PyTorch dependencies."""
    citation = CitationItem(
        chunk_id="BAB_0001",
        book_title="The Richest Man in Babylon",
        author="George S. Clason",
        chapter_title="Seven Cures for a Lean Purse",
        excerpt="A part of all you earn is yours to keep.",
        match_score=4.85,
        char_start=0,
        char_end=45,
    )
    assert citation.match_score == 4.85

    result = QueryResult(
        query="How much should I save?",
        answer="You should save at least one tenth of your earnings.",
        citations=[citation],
        execution_depth=4,
        iterations_used=1,
        latency_ms=18.5,
        tokens_generated=32,
        governor_interrupted=False,
        termination_reason="confident_grounding",
        is_grounded=True,
    )
    assert result.execution_depth == 4
    assert result.is_grounded is True
