from __future__ import annotations
import os

from dataclasses import dataclass
from pathlib import Path
from numpy.typing import NDArray

from ..utils.base_config import (
    ArgumentParserConfig,
    JsonDataclass,
    config_field,
    optional_validator,
    positive_validator,
    non_negative_validator,
    fraction_validator,
    growth_rate_validator,
    pathlike_validator,
    run_field_validators,
)
from ..utils.constants import *


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
    policy_epochs : int | None
        Epochs jointly optimizing policy parameters with Lyapunov parameters.
        If ``None``, only the Lyapunov model is updated.
    seed : int | None
        Random seed for reproducibility.
    kappa : float
        Exponential decay factor in the Lyapunov decrease condition.
    dropout : float
        Dropout probability for the Lyapunov model. Must be in the range [0, 1].
    tb_log_dir : str | os.PathLike | None
        TensorBoard logging directory. If ``None``, TensorBoard logging is disabled.

    invariance_weight : float
        Weight of the set-invariance penalty term.
    equilibrium_weight : float
        Weight for keeping V(0) near zero.
    formal_positivity_weight : float
        Weight for the IBP-based positivity penalty over the full training box.
    roa_weight : float
        Weight for the ROA surrogate term.
    l1_weight : float
        Weight for the parameter l1 regularization.
    
    roa_candidate_size : int
        Number of candidate states used in the ROA surrogate loss.
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
    adversarial_step_size : float
        Relative PGD step size for counterexample mining.
    counterexample_steps : int
        PGD steps for counterexample search.
    counterexample_every : int
        Frequency of counterexample mining in outer epochs.
    cex_fraction_min : float
        Minimum fraction of counterexamples in a batch.
    cex_fraction_max : float
        Maximum fraction of counterexamples in a batch.
    cex_fraction_ema_decay : float
        Exponential moving average decay for the counterexample fraction, used to adaptively adjust the number of counterexamples in each batch.
    state_buffer_limit : int
        Maximum size of the training buffer for regular samples.
    cex_buffer_limit : int
        Maximum size of the training buffer for counterexamples.
    condition_tolerance : float
        Numerical tolerance for condition satisfaction.
    condition_margin : float
        Safety margin enforced on the verifier output during training.
    """

    state_dim: int = config_field(cli=False, validators=(positive_validator,))
    state_bounds: NDArray = config_field(cli=False)
    initial_sample_size: int = config_field(
        default=1000, 
        help="Number of initial random samples used for training.",
        display_alias="init_samples",
        validators=(positive_validator,),
    )
    batch_size: int = config_field(
        default=512,
        help="Batch size for training iterations.",
        validators=(positive_validator,),
    )
    outer_epochs: int = config_field(
        default=10,
        help="Number of outer CEGIS epochs.",
        display_alias="epochs",
        validators=(positive_validator,),
    )
    steps_per_epoch: int = config_field(
        default=500,
        help="Number of optimization steps per outer epoch.",
        display_alias="steps/epoch",
        validators=(positive_validator,),
    )
    learning_rate: float = config_field(
        default=1e-2,
        help="Adam optimizer learning rate.",
        display_alias="lr",
        validators=(positive_validator,),
    )
    policy_epochs: int | None = config_field(
        default=None,
        help="Number of final outer epochs jointly optimizing policy and lyapunov parameters, starting at outer_epochs - policy_epochs. If None, the policy is never updated.",
        display_alias="policy_epochs",
        validators=(optional_validator(positive_validator),)
    )
    seed: int | None = config_field(
        default=None,
        help="Random seed for reproducibility.",
        validators=(optional_validator(positive_validator),)
    )
    kappa: float = config_field(
        default=0.05,
        help="Exponential decay factor in the Lyapunov decrease condition.",
        display_alias="\u03BA",
    )
    dropout: float = config_field(
        default=0.0,
        help="Dropout probability for the Lyapunov model.",
        validators=(fraction_validator,),
    )
    roa_candidate_size: int = config_field(
        default=1024,
        help="Number of candidate states used in the ROA surrogate loss.",
        display_alias="roa_cand",
        validators=(positive_validator,),
    )
    detach_relative_denominator: bool = config_field(
        default=True,
        help="Whether to detach V(x) in the denominator of the relative decrease violation.",
    )
    scale_anchor: float = config_field(
        default=100.0,
        help="Anchor value for the Lyapunov scale loss, which encourages the Lyapunov function values to be close to this anchor for better numerical conditioning.",
        display_alias="scale_anchor",
        validators=(positive_validator,),
    )
    scale_anchor_num_points: int = config_field(
        default=1024,
        help="Number of anchor points used in the Lyapunov scale anchor loss.",
        display_alias="scale_anchor_points",
        validators=(positive_validator,),
    )
    tb_log_dir: str | os.PathLike | None = config_field(
        default=None,
        help="TensorBoard logging directory.",
        validators=(optional_validator(pathlike_validator),)
    )

    # Weights
    condition_weight: float = config_field(
        default=1.0,
        help="Weight of the Lyapunov decrease and set-invariance condition penalty term.",
        display_alias="condw",
        validators=(non_negative_validator,),
    )
    condition_ibp_weight: float = config_field(
        default=1.0,
        help="Weight of the IBP-based Lyapunov decrease penalty term.",
        display_alias="cond_ibp_w",
        validators=(non_negative_validator,),
    )
    invariance_weight: float = config_field(
        default=1.0,
        help="Weight of the set-invariance penalty term.",
        display_alias="invw",
        validators=(non_negative_validator,),
    )
    equilibrium_weight: float = config_field(
        default=0.1,
        help="Weight for keeping V(0) near zero.",
        display_alias="eqw",
        validators=(non_negative_validator,),
    )
    formal_positivity_weight: float = config_field(
        default=1.0,
        help="Weight of the positivity penalty over the training box.",
        display_alias="fpw",
        validators=(non_negative_validator,),
    )
    roa_weight: float = config_field(
        default=0.1,
        help="Weight for the ROA surrogate term.",
        display_alias="roaw",
        validators=(non_negative_validator,),
    )
    l1_weight: float = config_field(
        default=1e-5,
        help="Weight for parameter L1 regularization.",
        display_alias="l1w",
        validators=(non_negative_validator,),
    )
    scale_weight: float = config_field(
        default=0.1,
        help="Weight for the Lyapunov scale anchor loss, which encourages the Lyapunov function values to be close to a specified anchor value for better numerical conditioning.",
        display_alias="scalew",
        validators=(non_negative_validator,),
    )
    policy_regularization_weight: float = config_field(
        default=1.0,
        help="Weight for the policy regularization loss, which helps keep the policy close to the initial reference policy.",
        display_alias="policy_reg_w",
        validators=(non_negative_validator,),
    )

    # Rho estimation parameters
    rho_growth_gamma: float = config_field(
        default=1.1,
        help="Growth factor used when estimating rho from boundary points.",
        display_alias="\u03C1_fac",
        validators=(growth_rate_validator,),
    )
    rho_estimation_samples: int = config_field(
        default=512,
        help="Number of boundary points used to estimate rho.",
        display_alias="\u03C1_num",
        validators=(positive_validator,),
    )
    rho_boundary_buffer_size: int | None = config_field(
        default=None,
        help="Optional cache size for retaining low-value boundary points.",
        display_alias="\u03C1_buff",
        validators=(optional_validator(positive_validator),)
    )
    rho_descent_steps: int = config_field(
        default=15,
        help="Number of projected gradient steps for boundary-value descent.",
        display_alias="\u03C1_steps",
        validators=(positive_validator,),
    )
    rho_step_size: float = config_field(
        default=0.05,
        help="Relative step size for boundary-value descent.",
        display_alias="\u03C1_step",
        validators=(positive_validator,),
    )
    rho_estimate_quantile: float = config_field(
        default=0.1,
        help="Quantile used for robust boundary-value aggregation in rho estimation.",
        display_alias="\u03C1_quant",
        validators=(fraction_validator,),
    )
    rho_min: float = config_field(
        default=1e-6,
        help="Minimum admissible rho value.",
        display_alias="\u03C1_min",
        validators=(positive_validator,),
    )

    # PGD counterexample mining parameters
    adversarial_samples: int = config_field(
        default=4096,
        help="Number of PGD seed states for counterexample mining.",
        display_alias="cex_samples",
        validators=(positive_validator,),
    )
    adversarial_step_size: float = config_field(
        default=0.05,
        help="Relative PGD step size for counterexample mining.",
        display_alias="cex_step",
        validators=(positive_validator,),
    )
    counterexample_steps: int = config_field(
        default=10,
        help="PGD steps used during counterexample search.",
        display_alias="cex_steps",
        validators=(positive_validator,),
    )
    counterexample_every: int = config_field(
        default=1,
        help="Frequency of counterexample mining in outer epochs.",
        display_alias="cex_every",
        validators=(positive_validator,),
    )
    cex_fraction_min: float = config_field(
        default=0.2,
        help="Minimum fraction of counterexamples in a batch.",
        display_alias="cex_frac_min",
        validators=(fraction_validator,),
    )
    cex_fraction_max: float = config_field(
        default=0.5,
        help="Maximum fraction of counterexamples in a batch.",
        display_alias="cex_frac_max",
        validators=(fraction_validator,),
    )
    cex_fraction_ema_decay: float = config_field(
        default=0.8,
        help="Exponential moving average decay for counterexample fraction.",
        display_alias="cex_frac_ema_decay",
        validators=(fraction_validator,),
    )
    state_buffer_limit: int = config_field(
        default=10000,
        help="Maximum size of the training buffer.",
        display_alias="buff",
        validators=(positive_validator,),
    )
    cex_buffer_limit: int = config_field(
        default=10000,
        help="Maximum size of the counterexample buffer.",
        display_alias="cex_buff",
        validators=(positive_validator,),
    )

    # Tolerances
    condition_tolerance: float = config_field(
        default=1e-6,
        help="Numerical tolerance for Lyapunov condition satisfaction.",
        display_alias="cond_tol",
        validators=(positive_validator,),
    )
    condition_margin: float = config_field(
        default=0.0,
        help="Safety margin enforced on the verifier output during training.",
        display_alias="margin",
        validators=(non_negative_validator,),
    )
    relative_condition_eps: float = config_field(
        default=1e-4,
        help="Numerical epsilon used in relative condition normalization.",
        display_alias="rel_eps",
        validators=(positive_validator,),
    )
    NP_ARRAY_FIELDS = ("state_bounds",)
    DEFAULT_FILE_NAME = TRAINING_CONFIG_FILENAME

    def __post_init__(self):
        run_field_validators(self)

        lbx = self.state_bounds[0]
        ubx = self.state_bounds[1]
        if (lbx >= 0).any() or (ubx <= 0).any() or (lbx > ubx).any():
            raise ValueError(
                "State bounds do not appear to include the origin. " 
                "Ensure that state_bounds are correctly specified for Lyapunov training."
            )

        if self.tb_log_dir is not None:
            object.__setattr__(self, "tb_log_dir", Path(self.tb_log_dir).resolve())

        # Rho estimation parameters
        if self.rho_boundary_buffer_size is None:
            object.__setattr__(
                self,
                "rho_boundary_buffer_size",
                max(self.rho_estimation_samples, 4 * self.rho_estimation_samples),
            )

        if self.state_bounds.shape[1] != self.state_dim:
            raise ValueError(
                "state_bounds must match state_dim. "
                f"Expected {self.state_dim}, got {self.state_bounds.shape[1]} (maybe transposed, bound shape: {self.state_bounds.shape})."
            )

        # CEX mining parameters
        if self.cex_fraction_min > self.cex_fraction_max:
            raise ValueError("cex_fraction_min must be less than or equal to cex_fraction_max.")

