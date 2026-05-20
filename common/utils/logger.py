"""
TensorBoard-based logger for RL training metrics.

Usage:
    logger = Logger(log_dir="runs/dqn_cartpole")
    logger.log_scalar("loss/q_loss", 0.42, step=1000)
    logger.log_episode(ep_return=200.0, ep_length=500, step=1000)
    logger.close()
"""

import os
import time
from typing import Dict, Optional
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """
    Thin wrapper around TensorBoard SummaryWriter with convenience methods
    for logging common RL metrics.

    Args:
        log_dir:      Root directory for TensorBoard event files.
        run_name:     Optional sub-directory name (timestamp used if None).
        flush_secs:   How often (seconds) TensorBoard should flush to disk.
        print_freq:   Log to stdout every N episodes (0 = never).
    """

    def __init__(
        self,
        log_dir: str = "runs",
        run_name: Optional[str] = None,
        flush_secs: int = 10,
        print_freq: int = 10,
    ):
        if run_name is None:
            run_name = time.strftime("%Y%m%d_%H%M%S")

        self.log_dir = os.path.join(log_dir, run_name)
        self.print_freq = print_freq
        self._episode_count = 0

        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=flush_secs)

        print(f"[Logger] TensorBoard logs -> {self.log_dir}")
        print(f"[Logger] Run: tensorboard --logdir {log_dir}")

    # ------------------------------------------------------------------
    # Core logging methods
    # ------------------------------------------------------------------

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """
        Log a single scalar value.

        Args:
            tag:   Metric name (e.g. "loss/q_loss", "train/epsilon").
            value: Scalar value to record.
            step:  Global training step.
        """
        self.writer.add_scalar(tag, value, global_step=step)

    def log_scalars(self, tag_values: Dict[str, float], step: int) -> None:
        """Log multiple scalars at once."""
        for tag, value in tag_values.items():
            self.log_scalar(tag, value, step)

    def log_episode(
        self,
        ep_return: float,
        ep_length: int,
        step: int,
        extra: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Log end-of-episode statistics.

        Args:
            ep_return: Total undiscounted episode return.
            ep_length: Number of steps in the episode.
            step:      Global training step at episode end.
            extra:     Optional dict of additional metrics to log.
        """
        self._episode_count += 1
        self.writer.add_scalar("episode/return", ep_return, global_step=step)
        self.writer.add_scalar("episode/length", ep_length, global_step=step)
        self.writer.add_scalar("episode/count", self._episode_count, global_step=step)

        if extra:
            for k, v in extra.items():
                self.writer.add_scalar(f"episode/{k}", v, global_step=step)

        if self.print_freq > 0 and self._episode_count % self.print_freq == 0:
            print(
                f"[Ep {self._episode_count:5d} | Step {step:8d}] "
                f"Return: {ep_return:8.2f}  Length: {ep_length:5d}"
            )

    def log_histogram(self, tag: str, values, step: int) -> None:
        """Log a histogram of parameter or gradient values."""
        self.writer.add_histogram(tag, values, global_step=step)

    def log_hparams(self, hparams: dict, metrics: Optional[dict] = None) -> None:
        """Log hyperparameters (shown in TensorBoard HPARAMS tab)."""
        self.writer.add_hparams(hparams, metrics or {})

    def close(self) -> None:
        """Flush and close the TensorBoard writer."""
        self.writer.flush()
        self.writer.close()
        print(f"[Logger] Closed. Total episodes logged: {self._episode_count}")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
