"""Pure-Python Byte-Level BPE (Byte Pair Encoding) Tokenizer.

Trained strictly from scratch on authorized text corpora without pretrained weights.
Guarantees 100% UTF-8 round-trip fidelity with zero out-of-vocabulary crashes.
"""

from collections import Counter
import json
from pathlib import Path
import re
from typing import Dict, List, Optional, Set, Tuple

from core.interfaces.tokenizer import SpecialTokens

# Pre-tokenization regex splitting into words, contractions, numbers, punctuation, and whitespace
WORD_SPLIT_REGEX = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+"""
)


class ByteLevelBPETokenizer:
    """Byte-Level BPE Tokenizer built from random initialization / scratch."""

    SPECIAL_TOKENS = SpecialTokens()
    BASE_BYTE_OFFSET = 9  # IDs 0..8 reserved for special tokens

    def __init__(self, target_vocab_size: int = 8000) -> None:
        """Initialize tokenizer.

        Args:
            target_vocab_size: Total vocabulary target (including special tokens and bytes).
        """
        self.target_vocab_size = target_vocab_size
        self._special_tokens = self.SPECIAL_TOKENS

        # Vocabulary maps
        # token_id -> bytes
        self.id_to_bytes: Dict[int, bytes] = {}
        # bytes -> token_id
        self.bytes_to_id: Dict[bytes, int] = {}

        # Merge rules: (id_a, id_b) -> merged_id
        self.merges: Dict[Tuple[int, int], int] = {}
        # Inverse merges: merged_id -> (id_a, id_b)
        self.merge_ranks: Dict[Tuple[int, int], int] = {}

        # Special token mappings
        self.special_token_to_id: Dict[str, int] = {
            self._special_tokens.pad_token: self._special_tokens.pad_id,
            self._special_tokens.bos_token: self._special_tokens.bos_id,
            self._special_tokens.eos_token: self._special_tokens.eos_id,
            self._special_tokens.unk_token: self._special_tokens.unk_id,
            self._special_tokens.query_token: self._special_tokens.query_id,
            self._special_tokens.context_token: self._special_tokens.context_id,
            self._special_tokens.response_token: self._special_tokens.response_id,
            self._special_tokens.cite_token: self._special_tokens.cite_id,
            self._special_tokens.nocontext_token: self._special_tokens.nocontext_id,
        }
        self.id_to_special_token: Dict[int, str] = {
            v: k for k, v in self.special_token_to_id.items()
        }

        # Initialize base 256 individual byte tokens
        self._init_base_vocabulary()

    def _init_base_vocabulary(self) -> None:
        """Populate the 256 base byte tokens (IDs 9..264)."""
        self.id_to_bytes.clear()
        self.bytes_to_id.clear()
        self.merges.clear()
        self.merge_ranks.clear()

        for b in range(256):
            token_id = self.BASE_BYTE_OFFSET + b
            byte_val = bytes([b])
            self.id_to_bytes[token_id] = byte_val
            self.bytes_to_id[byte_val] = token_id

    @property
    def vocab_size(self) -> int:
        """Total current vocabulary size."""
        return len(self.special_token_to_id) + len(self.id_to_bytes)

    @property
    def special_tokens(self) -> SpecialTokens:
        """Special control tokens definition."""
        return self._special_tokens

    def train_from_corpus(
        self,
        texts: List[str],
        target_vocab_size: Optional[int] = None,
        min_frequency: int = 2,
    ) -> None:
        """Train BPE merge pairs on a list of corpus documents.

        Args:
            texts: List of authorized document strings.
            target_vocab_size: Target vocabulary limit.
            min_frequency: Minimum pair frequency required for merging.
        """
        if target_vocab_size is not None:
            self.target_vocab_size = target_vocab_size

        self._init_base_vocabulary()

        # 1. Word frequency extraction
        word_counts: Counter[str] = Counter()
        for text in texts:
            words = WORD_SPLIT_REGEX.findall(text)
            word_counts.update(words)

        # 2. Represent each word as a tuple of base byte token IDs
        # word_tokens: tuple of token IDs -> frequency
        word_tokens: Dict[Tuple[int, ...], int] = {}
        for word, count in word_counts.items():
            byte_seq = word.encode("utf-8")
            token_ids = tuple(self.BASE_BYTE_OFFSET + b for b in byte_seq)
            word_tokens[token_ids] = word_tokens.get(token_ids, 0) + count

        next_token_id = self.BASE_BYTE_OFFSET + 256
        merge_rank = 0

        # 3. Iterative pair extraction and merging
        while self.vocab_size < self.target_vocab_size:
            # Count pair frequencies
            pair_counts: Counter[Tuple[int, int]] = Counter()
            for token_tuple, count in word_tokens.items():
                if len(token_tuple) < 2:
                    continue
                for i in range(len(token_tuple) - 1):
                    pair = (token_tuple[i], token_tuple[i + 1])
                    pair_counts[pair] += count

            if not pair_counts:
                break

            best_pair, best_freq = pair_counts.most_common(1)[0]
            if best_freq < min_frequency:
                break

            # Register new merge rule
            self.merges[best_pair] = next_token_id
            self.merge_ranks[best_pair] = merge_rank

            # Register concatenated bytes
            merged_bytes = self.id_to_bytes[best_pair[0]] + self.id_to_bytes[best_pair[1]]
            self.id_to_bytes[next_token_id] = merged_bytes
            self.bytes_to_id[merged_bytes] = next_token_id

            # Update words in word_tokens
            new_word_tokens: Dict[Tuple[int, ...], int] = {}
            for token_tuple, count in word_tokens.items():
                if len(token_tuple) < 2:
                    new_word_tokens[token_tuple] = count
                    continue

                new_tuple: List[int] = []
                idx = 0
                while idx < len(token_tuple):
                    if (
                        idx < len(token_tuple) - 1
                        and token_tuple[idx] == best_pair[0]
                        and token_tuple[idx + 1] == best_pair[1]
                    ):
                        new_tuple.append(next_token_id)
                        idx += 2
                    else:
                        new_tuple.append(token_tuple[idx])
                        idx += 1
                new_word_tokens[tuple(new_tuple)] = count

            word_tokens = new_word_tokens
            next_token_id += 1
            merge_rank += 1

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs.

        Args:
            text: Input string.
            add_special_tokens: If True, prepends <bos> and appends <eos>.

        Returns:
            List of integer token IDs.
        """
        if not text:
            return [self._special_tokens.bos_id, self._special_tokens.eos_id] if add_special_tokens else []

        words = WORD_SPLIT_REGEX.findall(text)
        token_ids: List[int] = []

        for word in words:
            # Check if word is an explicit special token
            if word in self.special_token_to_id:
                token_ids.append(self.special_token_to_id[word])
                continue

            # Convert word to base byte token IDs
            byte_seq = word.encode("utf-8")
            cur_tokens = [self.BASE_BYTE_OFFSET + b for b in byte_seq]

            # Iteratively apply lowest-rank merges
            while len(cur_tokens) >= 2:
                # Find all mergeable pairs and their ranks
                min_rank = float("inf")
                best_pair_idx = -1

                for i in range(len(cur_tokens) - 1):
                    pair = (cur_tokens[i], cur_tokens[i + 1])
                    if pair in self.merge_ranks:
                        rank = self.merge_ranks[pair]
                        if rank < min_rank:
                            min_rank = rank
                            best_pair_idx = i

                if best_pair_idx == -1:
                    break

                # Apply best merge
                pair_to_merge = (cur_tokens[best_pair_idx], cur_tokens[best_pair_idx + 1])
                merged_id = self.merges[pair_to_merge]
                cur_tokens = (
                    cur_tokens[:best_pair_idx]
                    + [merged_id]
                    + cur_tokens[best_pair_idx + 2 :]
                )

            token_ids.extend(cur_tokens)

        if add_special_tokens:
            return [self._special_tokens.bos_id] + token_ids + [self._special_tokens.eos_id]
        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to a UTF-8 string.

        Args:
            token_ids: Sequence of integer token IDs.
            skip_special_tokens: If True, omits special tokens from output.

        Returns:
            Decoded UTF-8 string.
        """
        if not token_ids:
            return ""

        byte_chunks: List[bytes] = []

        for tid in token_ids:
            if tid in self.id_to_special_token:
                if not skip_special_tokens:
                    spec_str = self.id_to_special_token[tid]
                    byte_chunks.append(spec_str.encode("utf-8"))
            elif tid in self.id_to_bytes:
                byte_chunks.append(self.id_to_bytes[tid])
            else:
                # Fallback for unknown IDs
                if not skip_special_tokens:
                    byte_chunks.append(self._special_tokens.unk_token.encode("utf-8"))

        full_bytes = b"".join(byte_chunks)
        return full_bytes.decode("utf-8", errors="replace")

    def save(self, directory: Path) -> Path:
        """Save vocabulary and merge rules to a JSON file.

        Args:
            directory: Directory to write tokenizer.json.

        Returns:
            Path to saved tokenizer.json.
        """
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        out_file = dir_path / "tokenizer.json"

        # Serialize merges as list of strings "id1 id2"
        merges_serialized = [f"{p[0]} {p[1]}" for p in sorted(self.merge_ranks, key=lambda x: self.merge_ranks[x])]

        data = {
            "version": "1.0",
            "target_vocab_size": self.target_vocab_size,
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_token_to_id,
            "merges": merges_serialized,
        }

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return out_file

    @classmethod
    def load(cls, file_or_dir: Path) -> "ByteLevelBPETokenizer":
        """Load tokenizer from tokenizer.json.

        Args:
            file_or_dir: Path to tokenizer.json or parent directory.

        Returns:
            Loaded ByteLevelBPETokenizer instance.
        """
        path = Path(file_or_dir)
        if path.is_dir():
            path = path / "tokenizer.json"

        if not path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls(target_vocab_size=data["target_vocab_size"])
        tokenizer._init_base_vocabulary()

        next_token_id = tokenizer.BASE_BYTE_OFFSET + 256
        for rank, merge_str in enumerate(data.get("merges", [])):
            parts = merge_str.split()
            pair = (int(parts[0]), int(parts[1]))
            tokenizer.merges[pair] = next_token_id
            tokenizer.merge_ranks[pair] = rank

            merged_bytes = tokenizer.id_to_bytes[pair[0]] + tokenizer.id_to_bytes[pair[1]]
            tokenizer.id_to_bytes[next_token_id] = merged_bytes
            tokenizer.bytes_to_id[merged_bytes] = next_token_id
            next_token_id += 1

        return tokenizer

    def get_vocab_stats(self) -> Dict[str, float]:
        """Compute structural statistics of the vocabulary."""
        byte_lengths = [len(b) for b in self.id_to_bytes.values()]
        avg_len = sum(byte_lengths) / len(byte_lengths) if byte_lengths else 0.0
        max_len = max(byte_lengths) if byte_lengths else 0

        return {
            "total_vocab_size": self.vocab_size,
            "special_tokens_count": len(self.special_token_to_id),
            "base_byte_tokens_count": 256,
            "learned_merges_count": len(self.merges),
            "avg_token_length_bytes": round(avg_len, 2),
            "max_token_length_bytes": max_len,
        }

    def compute_unk_frequency(self, texts: List[str]) -> float:
        """Calculate frequency of unknown tokens on a text sample.

        Note: Because Byte-Level BPE covers all 256 bytes, unk frequency for
        any valid UTF-8 string is mathematically 0.0%.
        """
        total_tokens = 0
        unk_tokens = 0

        for text in texts:
            tokens = self.encode(text, add_special_tokens=False)
            total_tokens += len(tokens)
            unk_tokens += sum(1 for t in tokens if t == self._special_tokens.unk_id)

        if total_tokens == 0:
            return 0.0
        return unk_tokens / total_tokens
