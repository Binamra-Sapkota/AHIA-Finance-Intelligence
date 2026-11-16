"""Automated tests for Phase 3: Byte-Level BPE Tokenizer and Pretraining Dataset Pipeline."""

import json
from pathlib import Path
import pytest

from core.dataset.pipeline import (
    DatasetProvenanceRecord,
    PretrainDocument,
    PretrainingDatasetPipeline,
)
from core.interfaces.ingestion import DocumentProvenance, UnauthorizedDocumentError
from core.interfaces.tokenizer import SpecialTokens
from core.tokenizer.bpe import ByteLevelBPETokenizer


SAMPLE_TRAINING_CORPUS = [
    "The primary rule of financial intelligence is to save at least ten percent of all earnings.",
    "Compound interest functions as an exponential multiplier over extended investment horizons.",
    "An emergency reserve covering three to six months of expenses protects capital preservation.",
    "High-interest debt liabilities erode disposable income and impair future wealth accumulation.",
    "Diversification across non-correlated asset classes provides resilience against market volatility.",
    "Disciplined budgeting ensures that expenditures never exceed incoming revenue streams.",
]


def test_bpe_training_and_vocab_target() -> None:
    """Verify that BPE training from scratch reaches specified target vocabulary size."""
    target_vocab = 320  # Base bytes (256) + specials (9) + merges (55)
    tokenizer = ByteLevelBPETokenizer(target_vocab_size=target_vocab)
    tokenizer.train_from_corpus(SAMPLE_TRAINING_CORPUS, target_vocab_size=target_vocab, min_frequency=2)

    assert tokenizer.vocab_size <= target_vocab
    assert tokenizer.vocab_size > 265  # Base bytes + specials + at least some merges learned
    assert len(tokenizer.merges) > 0


def test_special_tokens_integrity() -> None:
    """Verify special control tokens are registered at constant IDs 0..8."""
    tokenizer = ByteLevelBPETokenizer()
    specials = tokenizer.special_tokens

    assert specials.pad_id == 0
    assert specials.bos_id == 1
    assert specials.eos_id == 2
    assert specials.unk_id == 3
    assert specials.query_id == 4
    assert specials.context_id == 5
    assert specials.response_id == 6
    assert specials.cite_id == 7
    assert specials.nocontext_id == 8

    # Verify ID to string mappings
    assert tokenizer.decode([specials.bos_id], skip_special_tokens=False) == "<bos>"
    assert tokenizer.decode([specials.eos_id], skip_special_tokens=False) == "<eos>"
    assert tokenizer.decode([specials.cite_id], skip_special_tokens=False) == "<cite>"


def test_encode_decode_roundtrip_fidelity() -> None:
    """Verify 100% round-trip string reconstruction on various text structures."""
    tokenizer = ByteLevelBPETokenizer(target_vocab_size=350)
    tokenizer.train_from_corpus(SAMPLE_TRAINING_CORPUS, target_vocab_size=350, min_frequency=2)

    test_samples = [
        "Simple plain text sentence.",
        "Financial figures: $1,250.50 with a 10.5% annual yield in 2026.",
        "Hyphenated compound-words and punctuation: semi-colons; colons: dashes—and ellipses...",
        "Whitespace test with multiple   spaces and \n newlines \n\n double breaks.",
        "Multi-byte Unicode: résumé café with accents, and currency symbols: €500, ¥10,000, £75.",
        "Emojis and symbols: 💰📈 $SPY & $BTC rockets 🚀!",
    ]

    for sample in test_samples:
        token_ids = tokenizer.encode(sample, add_special_tokens=False)
        reconstructed = tokenizer.decode(token_ids, skip_special_tokens=True)
        assert reconstructed == sample, f"Roundtrip failed for: '{sample}' != '{reconstructed}'"


def test_special_tokens_inclusion_in_encode() -> None:
    """Verify that add_special_tokens adds <bos> and <eos>."""
    tokenizer = ByteLevelBPETokenizer()
    text = "Save money systematically."

    tokens_with_specials = tokenizer.encode(text, add_special_tokens=True)
    assert tokens_with_specials[0] == tokenizer.special_tokens.bos_id
    assert tokens_with_specials[-1] == tokenizer.special_tokens.eos_id

    tokens_without = tokenizer.encode(text, add_special_tokens=False)
    assert tokens_without[0] != tokenizer.special_tokens.bos_id
    assert tokens_without[-1] != tokenizer.special_tokens.eos_id


def test_tokenizer_serialization_and_deserialization(tmp_path: Path) -> None:
    """Verify saving and restoring tokenizer configuration and vocabulary."""
    tokenizer = ByteLevelBPETokenizer(target_vocab_size=320)
    tokenizer.train_from_corpus(SAMPLE_TRAINING_CORPUS, min_frequency=2)

    save_path = tokenizer.save(tmp_path)
    assert save_path.exists()

    # Load back
    loaded_tokenizer = ByteLevelBPETokenizer.load(save_path)
    assert loaded_tokenizer.vocab_size == tokenizer.vocab_size
    assert len(loaded_tokenizer.merges) == len(tokenizer.merges)

    # Check identical encoding
    test_text = "Compound interest and emergency reserve."
    assert loaded_tokenizer.encode(test_text) == tokenizer.encode(test_text)


def test_vocabulary_statistics_report() -> None:
    """Verify vocabulary statistics generation."""
    tokenizer = ByteLevelBPETokenizer(target_vocab_size=320)
    tokenizer.train_from_corpus(SAMPLE_TRAINING_CORPUS, min_frequency=2)

    stats = tokenizer.get_vocab_stats()
    assert stats["total_vocab_size"] == tokenizer.vocab_size
    assert stats["special_tokens_count"] == 9
    assert stats["base_byte_tokens_count"] == 256
    assert stats["learned_merges_count"] == len(tokenizer.merges)
    assert stats["avg_token_length_bytes"] >= 1.0


def test_unknown_token_frequency_on_utf8() -> None:
    """Verify that Byte-Level BPE has 0.0% unknown tokens on any valid UTF-8 text."""
    tokenizer = ByteLevelBPETokenizer(target_vocab_size=320)
    tokenizer.train_from_corpus(SAMPLE_TRAINING_CORPUS, min_frequency=2)

    unseen_texts = [
        "Unseen financial term: quantitative tightening and liquidity ratios.",
        "Foreign terms: Weltanschauung, omakase, and नमस्ते.",
    ]
    unk_freq = tokenizer.compute_unk_frequency(unseen_texts)
    assert unk_freq == 0.0, "Byte-level BPE must have 0% unk on valid UTF-8 strings"


def test_pretraining_dataset_pipeline(tmp_path: Path) -> None:
    """Verify dataset pipeline generates tokenized sequences, splits, and provenance."""
    tokenizer = ByteLevelBPETokenizer(target_vocab_size=320)
    tokenizer.train_from_corpus(SAMPLE_TRAINING_CORPUS, min_frequency=2)

    prov = DocumentProvenance(
        license_type="Public Domain",
        provenance="Author Archive",
        redistribution_permitted=True,
        training_permitted=True,
        authorized_by_user=True,
    )

    docs = [
        PretrainDocument(
            doc_id=f"DOC_PUBLIC_{i:02d}",
            title=f"Public Financial Principles Vol {i}",
            author="George S. Clason",
            content=f"Article {i}: A part of all you earn is yours to keep. Reinvest yields diligently in year {2000 + i}.",
            provenance=prov,
        )
        for i in range(10)
    ]

    pipeline = PretrainingDatasetPipeline(
        tokenizer=tokenizer,
        max_context_len=64,
        val_ratio=0.20,
        seed=100,
    )

    output_dir = tmp_path / "pretrain_data"
    record = pipeline.prepare_dataset(docs, output_dir=output_dir)

    assert (output_dir / "train.jsonl").exists()
    assert (output_dir / "val.jsonl").exists()
    assert (output_dir / "dataset_provenance.json").exists()

    assert record.train_documents_count == 8
    assert record.val_documents_count == 2
    assert record.leakage_check_passed is True

    # Read back one sequence and verify structure
    with open(output_dir / "train.jsonl", "r", encoding="utf-8") as f:
        first_line = json.loads(f.readline())
        assert len(first_line["input_ids"]) == 64
        assert first_line["input_ids"][0] == tokenizer.special_tokens.bos_id


def test_dataset_pipeline_zero_leakage_and_deduplication(tmp_path: Path) -> None:
    """Verify that duplicate documents are dropped and no hash appears in both train and val."""
    tokenizer = ByteLevelBPETokenizer(target_vocab_size=320)
    tokenizer.train_from_corpus(SAMPLE_TRAINING_CORPUS, min_frequency=2)

    prov = DocumentProvenance(
        license_type="Public Domain",
        provenance="Gutenberg",
        training_permitted=True,
        authorized_by_user=True,
    )

    # 4 distinct docs + 2 exact duplicates
    docs = [
        PretrainDocument(doc_id="D1", title="Doc 1", author="A", content="Unique text alpha", provenance=prov),
        PretrainDocument(doc_id="D2", title="Doc 2", author="B", content="Unique text beta", provenance=prov),
        PretrainDocument(doc_id="D3", title="Doc 3", author="C", content="Unique text gamma", provenance=prov),
        PretrainDocument(doc_id="D4", title="Doc 4", author="D", content="Unique text delta", provenance=prov),
        PretrainDocument(doc_id="D1_DUP", title="Doc 1 Copy", author="A", content="Unique text alpha", provenance=prov),
    ]

    pipeline = PretrainingDatasetPipeline(tokenizer=tokenizer, max_context_len=32, val_ratio=0.25)
    record = pipeline.prepare_dataset(docs, output_dir=tmp_path / "dedup_data")

    # 4 unique docs total (duplicate D1_DUP was filtered)
    assert record.train_documents_count + record.val_documents_count == 4
    assert record.leakage_check_passed is True


def test_dataset_pipeline_rejects_unauthorized_copyrighted_material(tmp_path: Path) -> None:
    """Verify that documents with training_permitted=False are strictly rejected."""
    tokenizer = ByteLevelBPETokenizer()

    copyrighted_prov = DocumentProvenance(
        license_type="Commercial Copyright",
        provenance="Purchased Retail eBook",
        redistribution_permitted=False,
        training_permitted=False,  # Unrestricted training forbidden
        authorized_by_user=True,
    )

    copyrighted_doc = PretrainDocument(
        doc_id="DOC_COMMERCIAL_001",
        title="Modern Commercial Bestseller",
        author="Commercial Author",
        content="Full text of copyrighted book that must NOT be used for model pretraining.",
        provenance=copyrighted_prov,
    )

    pipeline = PretrainingDatasetPipeline(tokenizer=tokenizer)

    with pytest.raises(UnauthorizedDocumentError):
        pipeline.prepare_dataset([copyrighted_doc], output_dir=tmp_path / "illegal_data")
