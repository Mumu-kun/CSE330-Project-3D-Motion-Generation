"""
W&B Logger for external training monitoring.

Works seamlessly on both local machine and Kaggle notebooks.
Auto-detects Kaggle environment and uses secrets for authentication.

Usage:
    from utils.wandb_logger import WandbLogger

    logger = WandbLogger(
        project="motion-generation",
        config={"lr": 1e-4, "epochs": 100}
    )
    # Run name auto-generated as "run-YYYY-MM-DD_HH-MM-SS"

    logger.log({"loss": loss, "lr": lr}, step=global_step)
    logger.log_model("checkpoints/best.pt", "best-model")
    logger.finish()
"""

import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any

# Check if wandb is available
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None  # type: ignore


def is_kaggle_environment() -> bool:
    """Detect if running in Kaggle notebook."""
    return os.path.exists("/kaggle") or "kaggle" in sys.executable.lower()


def get_kaggle_secret(secret_name: str) -> Optional[str]:
    """Get secret from Kaggle secrets if available."""
    if not is_kaggle_environment():
        return None

    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        user_secrets = UserSecretsClient()
        return user_secrets.get_secret(secret_name)
    except Exception:
        return None


class WandbLogger:
    """
    W&B logger with Kaggle support and graceful fallback.

    Features:
    - Auto-detects Kaggle environment and authenticates via secrets
    - Graceful fallback when wandb is not installed
    - Simple API for logging metrics and model checkpoints

    Args:
        project: W&B project name
        name: Run name (optional, auto-generated if not provided)
        config: Dictionary of hyperparameters to log
        kaggle_secret_name: Name of the Kaggle secret containing W&B API key
        enabled: Set to False to disable logging (useful for testing)

    Example:
        >>> logger = WandbLogger(
        ...     project="motion-generation",
        ...     name="flow-predictor-v1",
        ...     config={"lr": 1e-4, "epochs": 100, "batch_size": 64}
        ... )
        >>> logger.log({"loss": 0.5, "lr": 1e-4}, step=100)
        >>> logger.finish()
    """

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        kaggle_secret_name: str = "WANDB_API_KEY",
        enabled: bool = True,
    ):
        self.project = project
        self.config = config or {}
        self.enabled = enabled and WANDB_AVAILABLE
        self.run = None

        # Auto-generate run name from datetime if not provided
        if name is None:
            self.name = "motion-generation-buet"
        else:
            self.name = name

        if not self.enabled:
            if not WANDB_AVAILABLE:
                print("[WandbLogger] wandb not installed. Logging disabled.")
            elif not enabled:
                print("[WandbLogger] Logging disabled by user.")
            return

        # Try to authenticate
        self._authenticate(kaggle_secret_name)

        # Initialize run
        try:
            self.run = wandb.init(
                project=project,
                entity="motion-generation-buet",
                config=config,
                reinit=True,
            )
            print(f"[WandbLogger] Initialized run: {self.run.name}")
            print(f"[WandbLogger] View at: {self.run.url}")
        except Exception as e:
            print(f"[WandbLogger] Failed to initialize: {e}")
            self.enabled = False

    def _authenticate(self, secret_name: str) -> None:
        """Authenticate with W&B, using Kaggle secret if available."""
        # Check for API key in environment or Kaggle secrets
        api_key = os.environ.get("WANDB_API_KEY")

        if api_key is None:
            api_key = get_kaggle_secret(secret_name)

        if api_key:
            try:
                wandb.login(key=api_key)
                print("[WandbLogger] Authenticated successfully.")
            except Exception as e:
                print(f"[WandbLogger] Authentication failed: {e}")
        else:
            print(
                "[WandbLogger] No API key found. Using existing login or anonymous mode."
            )

    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
    ) -> None:
        """
        Log metrics to W&B.

        Args:
            metrics: Dictionary of metric names and values
            step: Global step (optional, auto-incremented if not provided)
        """
        if not self.enabled or self.run is None:
            return

        try:
            wandb.log(metrics, step=step)
        except Exception as e:
            print(f"[WandbLogger] Failed to log metrics: {e}")

    def log_model(
        self,
        path: str,
        name: str,
        description: Optional[str] = None,
    ) -> None:
        """
        Log a model checkpoint as a W&B artifact.

        Args:
            path: Path to the checkpoint file
            name: Name for the artifact
            description: Optional description
        """
        if not self.enabled or self.run is None:
            return

        try:
            artifact = wandb.Artifact(name, type="model", description=description)
            artifact.add_file(path)
            self.run.log_artifact(artifact)
            print(f"[WandbLogger] Logged model artifact: {name}")
        except Exception as e:
            print(f"[WandbLogger] Failed to log model: {e}")

    def log_image(
        self,
        key: str,
        path: str,
        step: Optional[int] = None,
        caption: Optional[str] = None,
    ) -> None:
        """
        Log an image file to W&B so it appears in the run media.

        Args:
            key: Metric/media key shown in W&B
            path: Local path to the image file
            step: Optional global step for the log entry
            caption: Optional display caption
        """
        if not self.enabled or self.run is None:
            return

        try:
            wandb.log({key: wandb.Image(path, caption=caption)}, step=step)
        except Exception as e:
            print(f"[WandbLogger] Failed to log image: {e}")

    def log_summary(self, metrics: Dict[str, Any]) -> None:
        """
        Log final summary metrics (shown in W&B run summary).

        Args:
            metrics: Dictionary of final metric values
        """
        if not self.enabled or self.run is None:
            return

        try:
            for key, value in metrics.items():
                wandb.run.summary[key] = value  # type: ignore
        except Exception as e:
            print(f"[WandbLogger] Failed to log summary: {e}")

    def finish(self) -> None:
        """Finish the W&B run."""
        if not self.enabled or self.run is None:
            return

        try:
            wandb.finish()
            print("[WandbLogger] Run finished.")
        except Exception as e:
            print(f"[WandbLogger] Failed to finish run: {e}")

    def __enter__(self) -> "WandbLogger":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - automatically finish run."""
        self.finish()
