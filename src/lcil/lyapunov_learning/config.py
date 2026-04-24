from __future__ import annotations
import os

from dataclasses import dataclass
from pathlib import Path
from numpy.typing import NDArray

from ..utils.base_config import ArgumentParserConfig, JsonDataclass, config_field


@dataclass(frozen=True)
class LyapunovTrainingConfig(JsonDataclass, ArgumentParserConfig):
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
    tb_log_dir : str | os.PathLike | None
        TensorBoard logging directory. If ``None``, TensorBoard logging is disabled.
    seed : int | None
        Random seed for reproducibility.
    kappa : float
        Exponential decay factor in the Lyapunov decrease condition.
    invariance_weight : float
        Weight of the set-invariance penalty term.
    equilibrium_weight : float
        Weight for keeping V(0) near zero.
    formal_positivity_weight : float
        Weight for the IBP-based positivity penalty over the full training box.
    rho_growth_gamma : float
        Growth factor for estimating sublevel values from boundary points.
    rho_boundary_samples : int
        Number of boundary points used to estimate the sublevel value.
    rho_boundary_buffer_size : int | None
        Optional cache size for retaining boundary points with the smallest
        Lyapunov values across outer iterations. If ``None``, a small multiple
        of ``rho_boundary_samples`` is used.
    rho_descent_steps : int
        Number of projected gradient steps for boundary-value descent.
    rho_step_size : float
        Relative step size for boundary-value descent.
    rho_estimate_quantile : float
        Quantile in ``(0, 1]`` used for robust boundary-value aggregation when estimating rho.
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

    state_dim: int = config_field(cli=False)
    state_bounds: NDArray = config_field(cli=False)
    initial_sample_size: int = config_field(default=1000, help="Number of initial random samples used for training.")
    batch_size: int = config_field(default=512, help="Batch size for training iterations.")
    outer_epochs: int = config_field(default=10, help="Number of outer CEGIS epochs.")
    steps_per_epoch: int = config_field(default=500, help="Number of optimization steps per outer epoch.")
    learning_rate: float = config_field(default=1e-2, help="Adam optimizer learning rate.")
    train_policy_model: bool = config_field(default=True, help="Whether to jointly optimize policy parameters with the Lyapunov model.")
    tb_log_dir: str | os.PathLike | None = config_field(default=None, help="TensorBoard logging directory.")
    seed: int | None = config_field(default=None, help="Random seed for reproducibility.")
    kappa: float = config_field(default=0.05, help="Exponential decay factor in the Lyapunov decrease condition.")
    invariance_weight: float = config_field(default=1.0, help="Weight of the set-invariance penalty term.")
    equilibrium_weight: float = config_field(default=0.01, help="Weight for keeping V(0) near zero.")
    formal_positivity_weight: float = config_field(default=1.0, help="Weight of the positivity penalty over the training box.")
    rho_growth_gamma: float = config_field(default=1.1, help="Growth factor used when estimating rho from boundary points.")
    rho_boundary_samples: int = config_field(default=512, help="Number of boundary points used to estimate rho.")
    rho_boundary_buffer_size: int | None = config_field(default=None, help="Optional cache size for retaining low-value boundary points.")
    rho_descent_steps: int = config_field(default=15, help="Number of projected gradient steps for boundary-value descent.")
    rho_step_size: float = config_field(default=0.05, help="Relative step size for boundary-value descent.")
    rho_estimate_quantile: float = config_field(default=0.1, help="Quantile used for robust boundary-value aggregation in rho estimation.")
    rho_min: float = config_field(default=1e-6, help="Minimum admissible rho value.")
    adversarial_samples: int = config_field(default=1024, help="Number of PGD seed states for counterexample mining.")
    counterexample_steps: int = config_field(default=30, help="PGD steps used during counterexample search.")
    adversarial_step_size: float = config_field(default=0.05, help="Relative PGD step size for counterexample mining.")
    counterexample_every: int = config_field(default=1, help="Frequency of counterexample mining in outer epochs.")
    max_buffer: int = config_field(default=10000, help="Maximum size of the training buffer.")
    roa_candidate_size: int = config_field(default=1024, help="Number of candidate states used in the ROA surrogate loss.")
    roa_weight: float = config_field(default=0.1, help="Weight for the ROA surrogate term.")
    l1_weight: float = config_field(default=1e-5, help="Weight for parameter L1 regularization.")
    condition_tolerance: float = config_field(default=1e-6, help="Numerical tolerance for Lyapunov condition satisfaction.")
    condition_margin: float = config_field(default=0.0, help="Safety margin enforced on the verifier output during training.")
    NP_ARRAY_FIELDS = ("state_bounds",)

    def __post_init__(self):
        if self.tb_log_dir is not None and not isinstance(self.tb_log_dir, (str, os.PathLike)):
            raise ValueError("tb_log_dir must be a string, os.PathLike, or None.")
        if self.tb_log_dir is not None:
            object.__setattr__(self, "tb_log_dir", Path(self.tb_log_dir).resolve())
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive.")
        if self.formal_positivity_weight < 0.0:
            raise ValueError("formal_positivity_weight must be non-negative.")
        if self.condition_margin < 0.0:
            raise ValueError("condition_margin must be non-negative.")
        if self.rho_boundary_samples <= 0:
            raise ValueError("rho_boundary_samples must be positive.")
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
        if self.rho_estimate_quantile <= 0.0 or self.rho_estimate_quantile > 1.0:
            raise ValueError("rho_estimate_quantile must be in the interval (0, 1].")
        if self.rho_boundary_buffer_size is None:
            object.__setattr__(
                self,
                "rho_boundary_buffer_size",
                max(self.rho_boundary_samples, 4 * self.rho_boundary_samples),
            )
        elif self.rho_boundary_buffer_size <= 0:
            raise ValueError("rho_boundary_buffer_size must be positive when provided.")
        if self.state_bounds.shape[1] != self.state_dim:
            raise ValueError(
                "state_bounds must match state_dim. "
                f"Expected {self.state_dim}, got {self.state_bounds.shape[1]} (maybe transposed, bound shape: {self.state_bounds.shape})."
            )
                
        lbx = self.state_bounds[0]
        ubx = self.state_bounds[1]
        if (lbx >= 0).any() or (ubx <= 0).any() or (lbx > ubx).any():
            raise ValueError(
                "State bounds do not appear to include the origin. " 
                "Ensure that state_bounds are correctly specified for Lyapunov training."
            )
