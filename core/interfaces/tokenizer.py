"""Interface contracts and special tokens for Byte-Level BPE tokenization."""

from dataclasses import dataclass
from typing import List, Protocol


@dataclass(frozen=True)
class SpecialTokens:
    """Special control token definitions and constant IDs."""

    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    unk_token: str = "<unk>"
    query_token: str = "<query>"
    context_token: str = "<context>"
    response_token: str = "<response>"
    cite_token: str = "<cite>"
    nocontext_token: str = "<nocontext>"

    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    unk_id: int = 3
    query_id: int = 4
    context_id: int = 5
    response_id: int = 6
    cite_id: int = 7
    nocontext_id: int = 8


class TokenizerProtocol(Protocol):
    """Protocol for Byte-Level BPE tokenizers trained from scratch."""

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including byte tokens, merges, and special tokens."""
        ...

    @property
    def special_tokens(self) -> SpecialTokens:
        """Special tokens mapping."""
        ...

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Convert a string into a list of token IDs.

        Args:
            text: UTF-8 string to encode.
            add_special_tokens: Whether to prepend <bos> and append <eos>.

        Returns:
            List of integer token IDs.
        """
        ...

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Reconstruct UTF-8 string from token IDs.

        Args:
            token_ids: List of integer token IDs.
            skip_special_tokens: Whether to omit special tokens from output.

        Returns:
            Decoded UTF-8 string.
        """
        ...

    def train_from_corpus(self, texts: List[str], target_vocab_size: int = 8000) -> None:
        """Train tokenizer merge pairs on raw corpus texts.

        Args:
            texts: List of document strings.
            target_vocab_size: Target total vocabulary size.
        """
        ...
