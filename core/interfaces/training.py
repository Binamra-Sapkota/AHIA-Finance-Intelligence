"""Interface contracts and configurations for the PyTorch training engine."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters for model training."""

    batch_size: int = 16
    learning_rate: float = 4e-4
    min_learning_rate: float = 4e-5
    warmup_steps: int = 500
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    num_epochs: int = 10
    save_interval_steps: int = 1000
    eval_interval_steps: int = 250
    checkpoint_dir: Path = Path("checkpoints")
    device: str = "cpu"

    # Multi-depth auxiliary loss weights
    alpha_depth_2: float = 0.2
    alpha_depth_4: float = 0.3
    alpha_depth_8: float = 0.5


class TrainerProtocol(Protocol):
    """Protocol for training engine orchestrating loss and optimization."""

    @property
    def config(self) -> TrainingConfig:
        """Active training configuration."""
        ...

    def train_step(self, batch: Any) -> Dict[str, float]:
        """Execute a single forward-backward optimization step.

        Args:
            batch: Dictionary or tensor tuple containing input_ids and targets.

        Returns:
            Dictionary mapping loss component names to float values.
        """
        ...

    def save_checkpoint(self, path: Path) -> None:
        """Serialize current model and optimizer states.

        Args:
            path: Destination file path.
        """
        ...

    def load_checkpoint(self, path: Path) -> None:
        """Restore model and optimizer states from checkpoint.

        Args:
            path: Source checkpoint file path.
        """
        ...
