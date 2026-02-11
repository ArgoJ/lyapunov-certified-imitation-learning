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
    kappa : float
        Exponential decay factor in the Lyapunov decrease condition.
    invariance_weight : float
        Weight of the set-invariance penalty term.
    reg_scale : float
        Deprecated compatibility field from the legacy trainer.
    pos_scale : float
        Deprecated compatibility field from the legacy trainer.
    reg_clamp_max : float
        Maximum clamp for the norm regularizer.
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
        Frequency of counterexample mining in steps.
    max_buffer : int
        Maximum size of the training buffer.
    roa_candidate_size : int
        Number of candidate states used in the ROA surrogate loss.
    roa_weight : float
        Weight for the ROA surrogate term.
    l1_weight : float
        Weight for the parameter l1 regularization.
    run_certification : bool
        Whether to run CROWN-based certification after training.
    cert_step : float
        Grid step size for certification region decomposition.
    cert_origin_exclusion : float | None
        Radius around the origin to skip during certification.
    cert_rho_scaling : float
        Multiplicative factor used in rho scaling before bisection.
    cert_bisection_tol : float
        Tolerance used in rho bisection.
    cert_max_scale_steps : int
        Maximum rho scaling attempts during certification.
    cert_max_bisection_steps : int
        Maximum bisection iterations for certification.
    cert_method : str
        AutoLiRPA bound method (e.g., "crown", "alpha-crown").
    condition_tolerance : float
        Numerical tolerance for condition satisfaction.
    """

    state_dim: int
    state_bounds: Sequence[float]
    sample_size: int = 1000
    batch_size: int = 512
    outer_epochs: int = 10
    steps_per_epoch: int = 500
    learning_rate: float = 1e-2
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
    run_certification: bool = True
    cert_step: float = 1.0
    cert_origin_exclusion: float | None = None
    cert_rho_scaling: float = 1.2
    cert_bisection_tol: float = 1e-3
    cert_max_scale_steps: int = 20
    cert_max_bisection_steps: int = 40
    cert_method: str = "alpha-crown"
    condition_tolerance: float = 1e-6
