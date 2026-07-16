from __future__ import annotations
import os

from dataclasses import dataclass
from pathlib import Path
from numpy.typing import NDArray
from typing import Sequence

from ..utils.base_config import (
    ArgumentParserConfig,
    JsonDataclass,
    config_field,
    optional_validator,
    positive_validator,
    non_negative_validator,
    fraction_validator,
    sequence_validator,
    normalize_scalar_or_sequence,
    growth_rate_validator,
    pathlike_validator,
    run_field_validators,
    bounds_include_origin_validator,
    array_shape_validator,
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
    train_bounds : NDArray
        Lower and upper bounds for each state dimension, used for training. [lbx, ubx]
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
    policy_lr_factor : float
        Learning rate factor for policy optimization when policy_epochs is not None.
    seed : int | None
        Random seed for reproducibility.
    kappa : float
        Exponential decay factor in the Lyapunov decrease condition.
    dropout : float
        Dropout probability for the Lyapunov model. Must be in the range [0, 1].
    roa_candidate_size : int
        Number of candidate states used in the ROA surrogate loss.
    roa_max_age : int
        Maximum age for states in the ROA surrogate buffer before they are removed.
    regularization_num_samples : int
        Number of uniformly sampled states used for global loss regularizers (scale anchoring, policy regularization).
    regularization_resample_interval : int
        Number of optimization steps between resampling states for global loss regularizers. 
        If 0, states are only sampled once at initialization.
    bins_per_dim : int | Sequence[int]
        Number of bins per dimension for region discretization when enforcing conditions over a grid.
    origin_exclusion : float | Sequence[float]
        Origin exclusion per dimension of bounds, used to exclude a neighborhood around the origin.
    tb_log_dir : str | os.PathLike | None
        TensorBoard logging directory. If ``None``, TensorBoard logging is disabled.

    condition_weight : float
        Weight of the Lyapunov decrease and set-invariance condition penalty term.
    condition_lirpa_weight : float
        Weight of the LIRPA-based Lyapunov decrease penalty term.
    invariance_weight : float
        Weight of the set-invariance penalty term.
    equilibrium_weight : float
        Weight for keeping V(0) near zero.
    formal_positivity_weight : float
        Weight for the LIRPA-based positivity penalty over the full training box.
    roa_weight : float
        Weight for the ROA surrogate term.
    l1_weight : float
        Weight for the parameter l1 regularization.
    scale_weight : float
        Weight for the Lyapunov scale anchor loss.
    policy_regularization_weight : float
        Weight for the policy regularization loss, which encourages the policy to stay close to the initial policy.
    r_factor_fro_norm_weight : float
        Weight for the Frobenius norm regularization of the R factor in the Lyapunov model, if it exists. 
        This helps prevent the R factor from collapsing or growing too large.
    
    rho_growth_gamma : float
        Growth factor for estimating sublevel values from boundary points.
    rho_estimation_samples : int
        Number of boundary points used to estimate the sublevel value.
    roa_boundary_buffer_size : int | None
        Optional cache size for retaining boundary points with the smallest
        Lyapunov values across outer iterations. If ``None``, a small multiple
        of ``rho_estimation_samples`` is used.
    rho_descent_steps : int
        Number of projected gradient steps for boundary-value descent.
    rho_step_size : float
        Relative step size for boundary-value descent.
    rho_estimate_quantile : float
        Quantile in ``(0, 1]`` used for robust boundary-value aggregation when estimating rho.
    rho_min : float
        Minimum admissible sublevel value.
    cex_step_size : float
        Relative PGD step size for counterexample mining.
    cex_descent_steps : int
        PGD steps for counterexample search.
    cex_every : int
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
    rho_gate_sharpness : float
        Steepness of the sigmoid gate used to soft-weight samples by their
        distance to the ρ-sublevel boundary. Higher values approximate a hard
        gate; lower values allow more gradient signal from points outside ρ.
    rho_resample_margin : float
        Multiplicative margin over ρ for rejection-resampling the state buffer.
        States with ``V(x) ≤ rho_resample_margin * ρ`` are retained during
        periodic resampling to focus training on the relevant sublevel region.
    """

    state_dim: int = config_field(
        cli=False, 
        validators=(positive_validator,)
    )
    state_bounds: NDArray = config_field(
        cli=False,
        validators=(bounds_include_origin_validator,),
    )
    train_bounds: NDArray | None = config_field(
        default=None,
        help="Optional training bounds for sampling. If None, state_bounds are used.",
        validators=(optional_validator(bounds_include_origin_validator),)
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
        help="Number of final outer epochs jointly optimizing policy and lyapunov parameters, " \
            "starting at outer_epochs - policy_epochs. If None, the policy is never updated.",
        display_alias="policy_epochs",
        validators=(optional_validator(positive_validator),)
    )
    policy_lr_factor: float = config_field(
        default=0.01,
        help="Learning rate factor for policy optimization when policy_epochs is not None.",
        display_alias="policy_lr_factor",
        validators=(positive_validator,),
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
    condition_lirpa_weight: float = config_field(
        default=1.0,
        help="Weight of the LiRPA-based Lyapunov decrease penalty term.",
        display_alias="cond_lirpa_w",
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
        default=1.0,
        help="Weight for the Lyapunov scale anchor loss, which encourages the Lyapunov " \
            "function values to be close to a specified anchor value for better numerical conditioning.",
        display_alias="scalew",
        validators=(non_negative_validator,),
    )
    policy_regularization_weight: float = config_field(
        default=1.0,
        help="Weight for the policy regularization loss, which helps keep the policy close to " \
            "the initial reference policy.",
        display_alias="policy_reg_w",
        validators=(non_negative_validator,),
    )
    r_factor_fro_norm_weight: float = config_field(
        default=100.0,
        help="Weight for the Frobenius norm regularization of the R factor in the Lyapunov model, if it exists. " \
            "This helps prevent the R factor from collapsing or growing too large.",
        display_alias="r_factor_fro_norm_w",
        validators=(non_negative_validator,),
    )

    # Condition Loss
    use_relative_decrease: bool = config_field(
        default=True,
        help="Whether to use relative Lyapunov decrease violation.",
    )
    relative_condition_eps: float = config_field(
        default=1e-2,
        help="Numerical epsilon used in relative condition normalization.",
        display_alias="rel_eps",
        validators=(positive_validator,),
    )
    
    # Counterexample mining
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
    cex_step_size: float = config_field(
        default=0.05,
        help="Relative PGD step size for counterexample mining.",
        display_alias="cex_step_size",
        validators=(positive_validator,),
    )
    cex_descent_steps: int = config_field(
        default=10,
        help="PGD steps used during counterexample search.",
        validators=(positive_validator,),
    )
    cex_every: int = config_field(
        default=1,
        help="Frequency of counterexample mining in outer epochs.",
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
    cex_max_age: int = config_field(
        default=5,
        help="Maximum age for counterexamples in the buffer before they are automatically removed, " \
            "used to ensure that the training buffer contains up-to-date counterexamples.",
        display_alias="cex_age",
        validators=(positive_validator,),
    )

    # Lirpa Condition Loss
    bins_per_dim: int | Sequence[int] = config_field(
        default=10,
        help="Number of bins per dimension for region discretization.",
        validators=(sequence_validator(positive_validator),),
    )
    origin_exclusion: float | Sequence[float] = config_field(
        default=0.0,
        help="Origin exclusion per dimension of bounds.",
        validators=(sequence_validator(non_negative_validator),),
    )
    
    # ROA Loss
    roa_boundary_buffer_size: int | None = config_field(
        default=None,
        help="Optional cache size for retaining low-value boundary points.",
        display_alias="roa_buff",
        validators=(optional_validator(positive_validator),)
    )
    roa_candidate_size: int = config_field(
        default=1024,
        help="Number of candidate states used in the ROA surrogate loss.",
        display_alias="roa_cand",
        validators=(positive_validator,),
    )
    roa_max_age: int = config_field(
        default=15,
        help="Maximum age for states in the ROA surrogate buffer before they are removed, " \
            "used to ensure that the surrogate loss is computed based on up-to-date samples.",
        display_alias="roa_age",
        validators=(positive_validator,),
    )

    # Regularization Losses
    regularization_num_samples: int = config_field(
        default=1024,
        help="Number of uniformly sampled states used for global loss regularizers (scale anchoring, policy regularization).",
        display_alias="loss_num_samples",
        validators=(positive_validator,),
    )
    regularization_resample_interval: int = config_field(
        default=100,
        help="Number of optimization steps between resampling states " \
            "for global loss regularizers. If 0, states are only sampled once at initialization.",
        validators=(non_negative_validator,),
    )

    # Rho estimation
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
    rho_descent_steps: int = config_field(
        default=3,
        help="Number of projected gradient steps for boundary-value descent.",
        display_alias="\u03C1_steps",
        validators=(non_negative_validator,),
    )
    rho_step_size: float = config_field(
        default=0.01,
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
    rho_ema_decay: float = config_field(
        default=0.8,
        help="Exponential moving average decay for rho estimates across epochs, " \
            "used to stabilize training when rho estimates are noisy.",
        display_alias="\u03C1_ema_decay",
        validators=(fraction_validator,),
    )
    rho_gate_sharpness: float = config_field(
        default=10.0,
        help="Steepness of the sigmoid gate in the condition loss. "
            "Higher values approximate a hard gate; lower values yield softer weighting.",
        display_alias="\u03C1_gate_β",
        validators=(positive_validator,),
    )
    rho_resample_margin: float = config_field(
        default=1.5,
        help="Multiplicative margin over rho for rejection-resampling the state buffer. "
            "States with V(x) <= margin * rho are retained.",
        display_alias="\u03C1_resamp_m",
        validators=(positive_validator,),
    )
    
    NP_ARRAY_FIELDS = ("state_bounds", "train_bounds")
    DEFAULT_FILE_NAME = TRAINING_CONFIG_FILENAME

    def __post_init__(self):
        run_field_validators(self)

        bins_per_dim = normalize_scalar_or_sequence(
            self.bins_per_dim,
            state_dim=self.state_dim,
            name="bins_per_dim",
            caster=int,
        )
        normalized_origin_exclusion = normalize_scalar_or_sequence(
            self.origin_exclusion,
            state_dim=self.state_dim,
            name="origin_exclusion",
            caster=float,
        )

        # object.__setattr__, because of frozen=True
        object.__setattr__(self, "bins_per_dim", bins_per_dim)
        object.__setattr__(self, "origin_exclusion", normalized_origin_exclusion)

        if self.tb_log_dir is not None:
            object.__setattr__(self, "tb_log_dir", Path(self.tb_log_dir).resolve())

        # Rho estimation parameters
        if self.roa_boundary_buffer_size is None:
            object.__setattr__(
                self,
                "roa_boundary_buffer_size",
                max(self.rho_estimation_samples, 4 * self.rho_estimation_samples),
            )

        if self.train_bounds is None:
            object.__setattr__(self, "train_bounds", self.state_bounds)

        array_shape_validator((2, self.state_dim))(self.state_bounds, "state_bounds")
        array_shape_validator((2, self.state_dim))(self.train_bounds, "train_bounds")

        # Counterexample mining parameters
        if self.cex_fraction_min > self.cex_fraction_max:
            raise ValueError("cex_fraction_min must be less than or equal to cex_fraction_max.")

