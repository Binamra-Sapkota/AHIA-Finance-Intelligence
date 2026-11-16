"""Pretraining dataset pipeline with document-level splitting and strict provenance checks.

Ensures zero duplicate document leakage between training and validation splits.
Excludes unauthorized copyrighted materials from neural training.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Dict, List, Optional, Set, Tuple

from core.interfaces.ingestion import DocumentProvenance, UnauthorizedDocumentError
from core.tokenizer.bpe import ByteLevelBPETokenizer


@dataclass(frozen=True)
class PretrainDocument:
    """Document unit submitted for pretraining dataset compilation."""

    doc_id: str
    title: str
    author: str
    content: str
    provenance: DocumentProvenance

    @property
    def content_hash(self) -> str:
        """SHA-256 hash of document content."""
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TokenizedSequence:
    """Pre-processed fixed-length token sequence ready for Transformer pretraining."""

    sequence_id: str
    doc_id: str
    input_ids: List[int]
    token_count: int


@dataclass(frozen=True)
class DatasetProvenanceRecord:
    """Audit ledger verifying license legality and zero document leakage."""

    created_at: str
    tokenizer_vocab_size: int
    context_length: int
    train_documents_count: int
    val_documents_count: int
    train_sequences_count: int
    val_sequences_count: int
    train_tokens_total: int
    val_tokens_total: int
    leakage_check_passed: bool
    documents: List[Dict[str, str]]


class PretrainingDatasetPipeline:
    """Builds tokenized causal language modeling datasets from authorized corpora."""

    def __init__(
        self,
        tokenizer: ByteLevelBPETokenizer,
        max_context_len: int = 512,
        val_ratio: float = 0.15,
        seed: int = 42,
    ) -> None:
        """Initialize pipeline.

        Args:
            tokenizer: Trained ByteLevelBPETokenizer instance.
            max_context_len: Maximum sequence length (L_ctx).
            val_ratio: Fraction of documents allocated to validation (default 15%).
            seed: Random seed for deterministic document-level splitting.
        """
        self.tokenizer = tokenizer
        self.max_context_len = max_context_len
        self.val_ratio = val_ratio
        self.seed = seed

    def prepare_dataset(
        self,
        documents: List[PretrainDocument],
        output_dir: Path,
        strict_permissions: bool = True,
    ) -> DatasetProvenanceRecord:
        """Compile tokenized train and validation datasets with zero leakage.

        Args:
            documents: List of input PretrainDocument candidates.
            output_dir: Directory where train.jsonl, val.jsonl, and provenance are written.
            strict_permissions: If True, raises error on unauthorized texts.

        Returns:
            DatasetProvenanceRecord summarizing dataset metrics and legal audits.

        Raises:
            UnauthorizedDocumentError: If document training_permitted is False.
            ValueError: If no valid non-empty documents remain.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # 1. Filter permissions & deduplicate by content hash
        authorized_docs: List[PretrainDocument] = []
        seen_hashes: Set[str] = set()

        for doc in documents:
            if not doc.provenance.training_permitted or not doc.provenance.authorized_by_user:
                if strict_permissions:
                    raise UnauthorizedDocumentError(
                        f"Document '{doc.title}' cannot be used for training: "
                        f"training_permitted={doc.provenance.training_permitted}, "
                        f"authorized_by_user={doc.provenance.authorized_by_user}"
                    )
                continue

            h = doc.content_hash
            if h in seen_hashes:
                continue  # Skip duplicate documents
            seen_hashes.add(h)

            if len(doc.content.strip()) > 0:
                authorized_docs.append(doc)

        if not authorized_docs:
            raise ValueError("No authorized non-empty documents available for dataset preparation.")

        # 2. Document-level split (guarantees zero leakage)
        # Sort by doc_id first for reproducibility across environments
        authorized_docs.sort(key=lambda d: d.doc_id)
        rng = random.Random(self.seed)
        shuffled = list(authorized_docs)
        rng.shuffle(shuffled)

        num_val = max(1, int(len(shuffled) * self.val_ratio)) if len(shuffled) > 1 else 0
        val_docs = shuffled[:num_val]
        train_docs = shuffled[num_val:] if num_val > 0 else shuffled

        # Verify zero intersection between train and val hashes
        train_hashes = {d.content_hash for d in train_docs}
        val_hashes = {d.content_hash for d in val_docs}
        leakage_detected = bool(train_hashes.intersection(val_hashes))
        if leakage_detected and num_val > 0:
            raise RuntimeError("CRITICAL ERROR: Document hash collision detected between train and val splits!")

        # 3. Tokenize and segment into fixed-length context windows
        train_seqs = self._tokenize_documents(train_docs, "train")
        val_seqs = self._tokenize_documents(val_docs, "val")

        # 4. Serialize to JSONL files
        train_file = out_path / "train.jsonl"
        val_file = out_path / "val.jsonl"

        with open(train_file, "w", encoding="utf-8") as f:
            for seq in train_seqs:
                f.write(json.dumps(asdict(seq)) + "\n")

        with open(val_file, "w", encoding="utf-8") as f:
            for seq in val_seqs:
                f.write(json.dumps(asdict(seq)) + "\n")

        # 5. Build provenance audit record
        from datetime import datetime, timezone
        now_str = datetime.now(timezone.utc).isoformat()

        doc_audit: List[Dict[str, str]] = []
        for d in train_docs:
            doc_audit.append({
                "doc_id": d.doc_id,
                "title": d.title,
                "author": d.author,
                "license": d.provenance.license_type,
                "content_hash": d.content_hash,
                "split": "train",
            })
        for d in val_docs:
            doc_audit.append({
                "doc_id": d.doc_id,
                "title": d.title,
                "author": d.author,
                "license": d.provenance.license_type,
                "content_hash": d.content_hash,
                "split": "val",
            })

        train_token_total = sum(s.token_count for s in train_seqs)
        val_token_total = sum(s.token_count for s in val_seqs)

        record = DatasetProvenanceRecord(
            created_at=now_str,
            tokenizer_vocab_size=self.tokenizer.vocab_size,
            context_length=self.max_context_len,
            train_documents_count=len(train_docs),
            val_documents_count=len(val_docs),
            train_sequences_count=len(train_seqs),
            val_sequences_count=len(val_seqs),
            train_tokens_total=train_token_total,
            val_tokens_total=val_token_total,
            leakage_check_passed=not leakage_detected,
            documents=doc_audit,
        )

        provenance_file = out_path / "dataset_provenance.json"
        with open(provenance_file, "w", encoding="utf-8") as f:
            json.dump(asdict(record), f, indent=2)

        return record

    def _tokenize_documents(
        self,
        docs: List[PretrainDocument],
        split_name: str,
    ) -> List[TokenizedSequence]:
        """Convert documents into tokenized sequences of max_context_len."""
        sequences: List[TokenizedSequence] = []
        seq_idx = 1
        pad_id = self.tokenizer.special_tokens.pad_id

        for doc in docs:
            # Encode without BOS/EOS because we manage block boundaries explicitly
            raw_tokens = self.tokenizer.encode(doc.content, add_special_tokens=False)
            if not raw_tokens:
                continue

            # Segment into non-overlapping or stepped chunks of max_context_len - 2 (leaving room for BOS/EOS)
            chunk_step = self.max_context_len - 2
            for start_idx in range(0, len(raw_tokens), chunk_step):
                chunk_slice = raw_tokens[start_idx : start_idx + chunk_step]
                # Enclose with BOS and EOS
                seq_tokens = [self.tokenizer.special_tokens.bos_id] + chunk_slice + [self.tokenizer.special_tokens.eos_id]

                # Pad to max_context_len
                original_len = len(seq_tokens)
                if len(seq_tokens) < self.max_context_len:
                    seq_tokens.extend([pad_id] * (self.max_context_len - len(seq_tokens)))

                sequences.append(
                    TokenizedSequence(
                        sequence_id=f"SEQ_{split_name.upper()}_{seq_idx:05d}",
                        doc_id=doc.doc_id,
                        input_ids=seq_tokens,
                        token_count=original_len,
                    )
                )
                seq_idx += 1

        return sequences
