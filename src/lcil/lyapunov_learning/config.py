from __future__ import annotations

from dataclasses import dataclass
from numpy.typing import NDArray


@dataclass(frozen=True)
class LyapunovTrainingConfig:
    """Configuration for Lyapunov training only.

    Parameters
    ----------
    state_dim : int
        Dimension of the system state.
    state_bounds : NDArray
        Lower and upper bounds for each state dimension, used for sampling and certification. [lbx, ubx]
    initial_sample_size : int
        Number of initial random samples used for training.
    batch_size : int
        Batch size for training iterations.
    outer_epochs : int
        Number of outer CEGIS epochs.
    steps_per_epoch : int
        Number of optimization steps per epoch.
    learning_rate : float
        Adam optimizer learning rate.
    train_policy_model : bool
        Whether to jointly optimize policy parameters with Lyapunov parameters.
        If False, only the Lyapunov model is updated.
    seed : int | None
        Random seed for reproducibility.
    kappa : float
        Exponential decay factor in the Lyapunov decrease condition.
    invariance_weight : float
        Weight of the set-invariance penalty term.
    reg_scale : float
        Deprecated compatibility field from the legacy trainer.
    pos_scale : float
        Weight for keeping V(0) near zero.
    reg_clamp_max : float
        Deprecated compatibility field from the legacy trainer.
    rho_growth_gamma : float
        Growth factor for estimating sublevel values from boundary points.
    rho_boundary_samples : int
        Number of boundary points used to estimate the sublevel value.
    rho_descent_steps : int
        Number of projected gradient steps for boundary-value descent.
    rho_step_size : float
        Relative step size for boundary-value descent.
    rho_min : float
        Minimum admissible sublevel value.
    adversarial_samples : int
        Number of PGD seeds for counterexample mining.
    counterexample_steps : int
        PGD steps for counterexample search.
    adversarial_step_size : float
        Relative PGD step size for counterexample mining.
    counterexample_every : int
        Frequency of counterexample mining in outer epochs.
    max_buffer : int
        Maximum size of the training buffer.
    roa_candidate_size : int
        Number of candidate states used in the ROA surrogate loss.
    roa_weight : float
        Weight for the ROA surrogate term.
    l1_weight : float
        Weight for the parameter l1 regularization.
    condition_tolerance : float
        Numerical tolerance for condition satisfaction.
    condition_margin : float
        Safety margin enforced on the verifier output during training.
    """

    state_dim: int
    state_bounds: NDArray
    initial_sample_size: int = 1000
    batch_size: int = 512
    outer_epochs: int = 10
    steps_per_epoch: int = 500
    learning_rate: float = 1e-2
    train_policy_model: bool = True
    seed: int | None = None
    kappa: float = 0.05
    invariance_weight: float = 1.0
    reg_scale: float = 0.1
    pos_scale: float = 0.01
    reg_clamp_max: float = 5e-4
    rho_growth_gamma: float = 1.1
    rho_boundary_samples: int = 512
    rho_descent_steps: int = 15
    rho_step_size: float = 0.05
    rho_min: float = 1e-6
    adversarial_samples: int = 1024
    counterexample_steps: int = 30
    adversarial_step_size: float = 0.05
    counterexample_every: int = 1
    max_buffer: int = 10000
    roa_candidate_size: int = 1024
    roa_weight: float = 0.1
    l1_weight: float = 1e-5
    condition_tolerance: float = 1e-6
    condition_margin: float = 0.0

    def __post_init__(self):
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive.")
        if self.roa_candidate_size <= 0:
            raise ValueError("ROA candidate size must be positive.")
        if self.outer_epochs <= 0:
            raise ValueError("Outer epochs must be positive.")
        if self.steps_per_epoch <= 0:
            raise ValueError("Steps per epoch must be positive.")
        if self.counterexample_every < 0:
            raise ValueError("Counterexample mining interval must be non-negative.")
        if self.rho_min <= 0:
            raise ValueError("Minimum rho estimate must be positive.")
        if self.state_bounds.shape[1] != self.state_dim:
            raise ValueError(
                "state_bounds must match state_dim. "
                f"Expected {self.state_dim}, got {self.state_bounds.shape[1]} (maybe transposed, bound shape: {self.state_bounds.shape})."
            )