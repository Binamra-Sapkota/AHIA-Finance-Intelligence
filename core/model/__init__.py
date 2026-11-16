"""Subsystem 4: Small Decoder-Only Transformer (Pure PyTorch, Random Initialization)."""

from core.interfaces.model import (
    AdaptiveExecutionDepth,
    KVStateProtocol,
    ModelConfig,
    ModelProtocol,
)

__all__ = [
    "AdaptiveExecutionDepth",
    "KVStateProtocol",
    "ModelConfig",
    "ModelProtocol",
]
