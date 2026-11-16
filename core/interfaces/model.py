"""Interface contracts and configuration dataclasses for the Small Decoder-Only Transformer."""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, List, Optional, Protocol, Tuple


class AdaptiveExecutionDepth(IntEnum):
    """Supported adaptive execution block depths."""

    FAST = 2       # 2 blocks: Low-complexity / refusal checks
    MEDIUM = 4     # 4 blocks: Factual queries
    DEEP = 8       # 8 blocks: Full multi-chunk synthesis (maximum depth)


@dataclass(frozen=True)
class ModelConfig:
    """Target architectural hyperparameters for AHIA-R2 Transformer."""

    num_layers: int = 8                    # 8 Transformer blocks
    hidden_dim: int = 256                  # Hidden dimension (d_model)
    num_heads: int = 4                     # 4 attention heads (head_dim = 64)
    ff_dim: int = 1024                     # Feed-forward expansion (4 * d_model)
    vocab_size: int = 8000                 # Vocabulary target (Byte-Level BPE)
    max_context_len: int = 512             # Maximum sequence length
    dropout: float = 0.1                   # Dropout rate
    norm_eps: float = 1e-6                 # RMSNorm epsilon
    use_rope: bool = True                  # Rotary Position Embeddings
    supported_depths: Tuple[int, ...] = (2, 4, 8)  # Adaptive depths

    # Planned Mixture-of-Experts (MoE) configuration (activated after dense baseline)
    num_experts: int = 4                   # 4 planned experts
    top_k_experts: int = 1                 # Top-1 routing per token
    moe_enabled: bool = False              # Dense baseline initially


class KVStateProtocol(Protocol):
    """Protocol for stateful Key-Value caching during generation."""

    def update(self, layer_idx: int, k: Any, v: Any) -> Tuple[Any, Any]:
        """Append new step key/value projections to layer cache."""
        ...

    def reset(self) -> None:
        """Clear cached states."""
        ...


class ModelProtocol(Protocol):
    """Protocol for the decoder-only Transformer model."""

    @property
    def config(self) -> ModelConfig:
        """Model architectural configuration."""
        ...

    def forward(
        self,
        input_ids: Any,
        depth: int = 8,
        kv_cache: Optional[KVStateProtocol] = None,
    ) -> Any:
        """Forward pass through specified number of Transformer blocks.

        Args:
            input_ids: Tensor of token IDs [batch_size, seq_len].
            depth: Number of layers to execute (must be in config.supported_depths).
            kv_cache: Optional stateful KV-cache for incremental decoding.

        Returns:
            Logits tensor [batch_size, seq_len, vocab_size].
        """
        ...

    def generate(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int = 256,
        depth: int = 8,
        temperature: float = 0.7,
        top_k: int = 40,
    ) -> List[int]:
        """Autoregressive token generation.

        Args:
            prompt_tokens: Input prompt token IDs.
            max_new_tokens: Maximum tokens to generate.
            depth: Execution depth (2, 4, or 8).
            temperature: Sampling temperature.
            top_k: Top-K filtering.

        Returns:
            List of generated token IDs.
        """
        ...
