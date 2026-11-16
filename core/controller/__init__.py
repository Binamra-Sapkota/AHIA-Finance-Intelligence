"""Subsystem 6: Adaptive Reasoning Controller (Bounded Loops T_max <= 3, Depths 2/4/8)."""

from core.interfaces.controller import (
    AdaptiveControllerProtocol,
    ControllerResult,
    ReasoningStep,
    TerminationReason,
)

__all__ = [
    "AdaptiveControllerProtocol",
    "ControllerResult",
    "ReasoningStep",
    "TerminationReason",
]
