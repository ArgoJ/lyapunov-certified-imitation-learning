from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class LyapunovTrainingConfig:
    """Configuration for Lyapunov training and optional certification.

    Parameters
    ----------
    state_dim : int
        Dimension of the system state.
    state_bounds : Sequence[float]
        Per-dimension absolute bounds for sampling and certification.
    sample_size : int
        Number of initial random samples used for training.
    batch_size : int
        Batch size for training iterations.
    outer_epochs : int
        Number of outer training epochs.
    steps_per_epoch : int
        Number of optimization steps per epoch.
    learning_rate : float
        Adam optimizer learning rate.
    seed : int | None
        Random seed for reproducibility.
    reg_scale : float
        Scale for Lyapunov decrease regularization.
    pos_scale : float
        Scale for positive definiteness regularization.
    reg_clamp_max : float
        Maximum clamp for the norm regularizer.
    counterexample_steps : int
        PGD steps for counterexample search.
    counterexample_every : int
        Frequency of counterexample mining in steps.
    max_buffer : int
        Maximum size of the training buffer.
    run_certification : bool
        Whether to run CROWN-based certification after training.
    cert_step : float
        Grid step size for certification region decomposition.
    cert_origin_exclusion : float | None
        Radius around the origin to skip during certification.
    """

    state_dim: int
    state_bounds: Sequence[float]
    sample_size: int = 1000
    batch_size: int = 512
    outer_epochs: int = 10
    steps_per_epoch: int = 500
    learning_rate: float = 1e-2
    seed: int | None = None
    reg_scale: float = 0.1
    pos_scale: float = 0.01
    reg_clamp_max: float = 5e-4
    counterexample_steps: int = 30
    counterexample_every: int = 50
    max_buffer: int = 10000
    run_certification: bool = True
    cert_step: float = 1.0
    cert_origin_exclusion: float | None = None
