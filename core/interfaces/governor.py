"""Interface contracts for the Independent Resource Governor."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol


class GovernorAction(str, Enum):
    """Action taken by the resource governor."""

    ALLOW = "allow"
    THROTTLE = "throttle"
    TIMEOUT_ABORT = "timeout_abort"
    TOKEN_LIMIT_ABORT = "token_limit_abort"
    MEMORY_LIMIT_ABORT = "memory_limit_abort"


@dataclass(frozen=True)
class ResourceQuota:
    """Configured limits for system execution."""

    max_wall_clock_seconds: float = 10.0
    max_output_tokens: int = 256
    max_memory_mb: float = 4096.0


@dataclass(frozen=True)
class ExecutionTelemetry:
    """Metrics collected during a single execution run."""

    wall_clock_elapsed_sec: float
    tokens_generated: int
    memory_peak_mb: float
    action_taken: GovernorAction
    interrupted: bool


class ResourceGovernorProtocol(Protocol):
    """Protocol for independent supervisor overseeing resource limits."""

    @property
    def quota(self) -> ResourceQuota:
        """Active resource quota configuration."""
        ...

    def start_watchdog(self, timeout_seconds: Optional[float] = None) -> None:
        """Start outer watchdog timer thread.

        Args:
            timeout_seconds: Optional custom timeout overriding quota.
        """
        ...

    def check_token_limit(self, current_token_count: int) -> bool:
        """Verify whether generation token count is within quota.

        Args:
            current_token_count: Generated token count.

        Returns:
            True if within quota, False if ceiling exceeded.
        """
        ...

    def stop_watchdog(self, final_token_count: int) -> ExecutionTelemetry:
        """Stop watchdog timer and compute execution telemetry.

        Args:
            final_token_count: Total tokens generated.

        Returns:
            ExecutionTelemetry summarizing runtime and limits.
        """
        ...
