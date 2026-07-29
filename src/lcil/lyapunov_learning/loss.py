from __future__ import annotations

import logging
import torch as th
import torch.nn as nn

from copy import deepcopy
from dataclasses import dataclass
from collections.abc import Callable, Iterable
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

from .config import LyapunovTrainingConfig
from .utils import get_th_lbx_ubx, get_center
from .sampling import sample_sobol_box

__logger__ = logging.getLogger(__name__)


def _has_learnable_r_factor(module: nn.Module) -> bool:
    """Check if the Lyapunov model has an R factor attribute."""
    return hasattr(module, "r_factor") and isinstance(module.r_factor, nn.Parameter)


@dataclass
class LyapunovTrainingLossParts:
    """Container for the individual loss components of the Lyapunov training objective."""
    condition_raw: th.Tensor
    roa_raw: th.Tensor
    condition_lirpa_raw: th.Tensor
    l1_raw: th.Tensor
    equilibrium_raw: th.Tensor
    formal_positivity_raw: th.Tensor
    scale_raw: th.Tensor
    policy_regularization_raw: th.Tensor
    r_factor_fro_norm_raw: th.Tensor

    condition_weight: float
    roa_weight: float
    condition_lirpa_weight: float
    l1_weight: float
    equilibrium_weight: float
    formal_positivity_weight: float
    scale_weight: float
    policy_regularization_weight: float
    r_factor_fro_norm_weight: float

    @property
    def condition(self) -> th.Tensor:
        return self.condition_raw * self.condition_weight

    @property
    def roa(self) -> th.Tensor:
        return self.roa_weight * self.roa_raw

    @property
    def condition_lirpa(self) -> th.Tensor:
        return self.condition_lirpa_raw * self.condition_lirpa_weight

    @property
    def l1(self) -> th.Tensor:
        return self.l1_weight * self.l1_raw

    @property
    def equilibrium(self) -> th.Tensor:
        return self.equilibrium_weight * self.equilibrium_raw

    @property
    def formal_positivity(self) -> th.Tensor:
        return self.formal_positivity_weight * self.formal_positivity_raw

    @property
    def scale(self) -> th.Tensor:
        return self.scale_weight * self.scale_raw

    @property
    def policy_regularization(self) -> th.Tensor:
        return self.policy_regularization_weight * self.policy_regularization_raw

    @property
    def r_factor_fro_norm(self) -> th.Tensor:
        return self.r_factor_fro_norm_weight * self.r_factor_fro_norm_raw

    @property
    def total(self) -> th.Tensor:
        return (
            self.condition
            + self.roa
            + self.condition_lirpa
            + self.l1
            + self.equilibrium
            + self.formal_positivity
            + self.scale
            + self.policy_regularization
            + self.r_factor_fro_norm
        )


def lyapunov_decrease(
    v_curr: th.Tensor,
    v_next: th.Tensor,
    kappa: float,
) -> th.Tensor:
    """Compute the one-step Lyapunov decrease value."""
    return v_next - (1.0 - kappa) * v_curr

def relative_denominator(v_curr: th.Tensor, eps: float) -> th.Tensor:
    return v_curr.detach().abs().clamp_min(eps)

def relative_lyapunov_decrease(
    v_curr: th.Tensor,
    v_next: th.Tensor,
    kappa: float,
    eps: float,
) -> th.Tensor:
    return lyapunov_decrease(v_curr, v_next, kappa) / relative_denominator(v_curr, eps)
    

def weighted_mean(values: th.Tensor, weights: th.Tensor) -> th.Tensor:
    """Compute a weighted mean of the given values with the provided weights."""
    weights = weights.to(dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)

def safe_rho(rho: float, rho_min: float):
    return max(float(rho), rho_min)


class StateBoundsModule(nn.Module):
    """Base module that stores the shared training-state bounds."""

    def __init__(self, state_bounds: th.Tensor, device: th.device | str = "cpu") -> None:
        super().__init__()
        self.device = th.device(device)

        lbx, ubx = get_th_lbx_ubx(state_bounds, device=self.device)
        self.register_buffer("lbx", lbx)
        self.register_buffer("ubx", ubx)


class BoundedStateSamplingModule(StateBoundsModule):
    """Base module that provides uniform sampling of states from the training bounds."""

    def __init__(
        self,
        state_bounds: th.Tensor,
        num_samples: int = 1024,
        resample_interval: int = 100,
        device: th.device | str = "cpu"
    ) -> None:
        super().__init__(state_bounds=state_bounds, device=device)
        self.num_samples = int(num_samples)
        self.resample_interval = int(resample_interval)
        self._max_resample_interval = int(3 * self.resample_interval)
        self._min_resample_interval = self.resample_interval // 3
        self._step_counter = 0

        state_dim = self.lbx.shape[0]
        self.sobol_engine = th.quasirandom.SobolEngine(dimension=state_dim, scramble=True)

        self.register_buffer(
            "samples",
            th.zeros((self.num_samples, self.lbx.shape[0]), device=self.device)
        )
        self._register_new_samples()

    def _needs_resample(self) -> bool:
        """Check if a new batch of samples should be drawn based on the resample interval."""
        if self.resample_interval <= 0 or not self.training:
            return False
        if self._step_counter >= self._max_resample_interval:
            return True

        resample_probability = 1.0 / self.resample_interval
        return (
            self._step_counter > self._min_resample_interval
            and bool(th.rand((), device=self.device) < resample_probability)
        )

    def _sample_uniform_states(self, num_points: int) -> th.Tensor:
        """Sample a batch of states uniformly from the training bounds."""
        return sample_sobol_box(
            sample_size=num_points,
            lb=self.lbx,
            ub=self.ubx,
            sobol_engine=self.sobol_engine,
            device=self.device,
        )

    @th.no_grad()
    def _register_new_samples(self) -> None:
        """Register a new batch of uniform samples."""
        self.samples.copy_(self._sample_uniform_states(self.num_samples))

    def step_sampling(self) -> bool:
        """Register new samples if it is at intervalle, otherwise increment the step counter."""
        if self._needs_resample():
            self._register_new_samples()
            self._step_counter = 0
            return True

        self._step_counter += 1
        return False



class LyapunovScaleAnchorLoss(BoundedStateSamplingModule):
    """Anchor the absolute scale of V on uniformly sampled reference states."""
    
    def __init__(
        self,
        lyap_model: nn.Module,
        state_bounds: th.Tensor,
        num_samples: int = 1000,
        resample_interval: int = 100,
        device: th.device | str = "cpu",
        eps: float = 1e-6
    ) -> None:
        super().__init__(
            state_bounds=state_bounds, num_samples=num_samples, resample_interval=resample_interval, device=device
        )
        self.lyap_model = lyap_model
        self.eps = float(eps)

        self.register_buffer(
            "target_log_value", 
            th.tensor(float("nan"), dtype=th.float32, device=self.device)
        )

    def forward(self) -> th.Tensor:
        self.step_sampling()
        v_reference = self.lyap_model(self.samples)
        ref_mean = v_reference.mean().clamp_min(self.eps)
        current_log_value = th.log(ref_mean)
        
        if th.isnan(self.target_log_value):
            self.target_log_value.copy_(current_log_value.detach())
            __logger__.info(
                "Auto-anchored Lyapunov scale at log(V) = %.4f (V ≈ %.4f)",
                self.target_log_value.item(),
                th.exp(self.target_log_value).item()
            )

        return (current_log_value - self.target_log_value).pow(2)
    

class LyapunovDecreaseViolation(nn.Module):
    """Compute the one-step Lyapunov decrease violation."""

    def __init__(self, kappa: float) -> None:
        super().__init__()
        self.kappa = float(kappa)

    def forward(self, v_curr: th.Tensor, v_next: th.Tensor) -> th.Tensor:
        return th.relu(lyapunov_decrease(v_curr, v_next, self.kappa))


class RelativeLyapunovDecreaseViolation(nn.Module):
    """Compute a relative version of the one-step Lyapunov decrease violation."""

    def __init__(
        self,
        kappa: float,
        relative_eps: float = 1e-2,
    ) -> None:
        super().__init__()
        self.kappa = float(kappa)
        self.relative_eps = float(relative_eps)

    def forward(self, v_curr: th.Tensor, v_next: th.Tensor) -> th.Tensor:
        return th.relu(relative_lyapunov_decrease(v_curr, v_next, self.kappa, self.relative_eps))


class InvarianceViolation(StateBoundsModule):
    """Compute the forward-invariance violation."""

    def __init__(self, state_bounds: th.Tensor, device: th.device | str = "cpu") -> None:
        super().__init__(state_bounds=state_bounds, device=device)

    def forward(self, x_next: th.Tensor) -> th.Tensor:
        upper_violation = th.relu(x_next - self.ubx).sum(dim=1, keepdim=True)
        lower_violation = th.relu(self.lbx - x_next).sum(dim=1, keepdim=True)
        return upper_violation + lower_violation


class RelativeInvarianceViolation(StateBoundsModule):
    """Compute a relative version of the forward-invariance violation."""

    def __init__(
        self,
        state_bounds: th.Tensor,
        device: th.device | str = "cpu",
        relative_eps: float = 1e-4,
    ) -> None:
        super().__init__(state_bounds=state_bounds, device=device)
        self.relative_eps = float(relative_eps)
        self.register_buffer("widths", (self.ubx - self.lbx).clamp_min(1e-6))

    def forward(self, x_next: th.Tensor) -> th.Tensor:
        upper_violation = th.relu(x_next - self.ubx) / self.widths
        lower_violation = th.relu(self.lbx - x_next) / self.widths
        return upper_violation.sum(dim=1, keepdim=True) + lower_violation.sum(dim=1, keepdim=True)


class RhoGatedConditionLoss(nn.Module):
    """Compute a rho-gated condition loss with soft sigmoid weighting.

    Instead of a hard binary mask ``V(x) ≤ ρ``, a smooth sigmoid gate
    ``σ(β · (ρ - V) / ρ)`` is used so that points near the sublevel boundary
    still contribute gradient signal.  The sharpness parameter ``β`` controls
    the transition width.
    """

    def __init__(self, config: LyapunovTrainingConfig, device: th.device | str = "cpu") -> None:
        super().__init__()
        self.rho_min = float(config.rho_min)
        self.relative_eps = float(config.relative_condition_eps)
        self.invariance_weight = float(config.invariance_weight)
        self.gate_sharpness = float(config.rho_gate_sharpness)
        
        self.decrease_violation = RelativeLyapunovDecreaseViolation(
            kappa=config.kappa, relative_eps=config.relative_condition_eps
        ) if config.use_relative_decrease else LyapunovDecreaseViolation(kappa=config.kappa)
            
        self.invariance_violation = RelativeInvarianceViolation(
            config.train_bounds, # type: ignore
            device=device,
            relative_eps=config.relative_condition_eps,
        )

    def condition_violation(
        self,
        v_curr: th.Tensor,
        v_next: th.Tensor,
        x_next: th.Tensor,
    ) -> th.Tensor:
        decrease = self.decrease_violation(v_curr=v_curr, v_next=v_next)
        invariance = self.invariance_violation(x_next=x_next)
        return decrease + self.invariance_weight * invariance

    def soft_sublevel_weight(self, v_curr: th.Tensor, rho_estimate: float, detach: bool = True) -> th.Tensor:
        """Smooth sigmoid weight: ≈1 inside ρ, ≈0.5 at ρ boundary, ≈0 far outside."""
        if detach:
            v_curr = v_curr.detach()
        rho_value = safe_rho(rho_estimate, self.rho_min)
        margin = (rho_value - v_curr) / max(rho_value, self.relative_eps)
        return th.sigmoid(self.gate_sharpness * margin)

    def forward(
        self,
        v_curr: th.Tensor,
        v_next: th.Tensor,
        x_next: th.Tensor,
        rho_estimate: float,
    ) -> th.Tensor:
        violation = self.condition_violation(
            v_curr=v_curr,
            v_next=v_next,
            x_next=x_next,
        )
        weights = self.soft_sublevel_weight(v_curr, rho_estimate)

        # logging
        # num_active = int((weights > 0.5).sum().item())
        # effective_weight_mass = float(weights.sum().item())
        # __logger__.info("active=%d, effective mass=%.3f", num_active, effective_weight_mass)
        return weighted_mean(violation, weights)



class SignedConditionMargin(nn.Module):
    def __init__(
        self,
        policy_model: nn.Module,
        lyap_model: nn.Module,
        dyn_model: nn.Module,
        config: LyapunovTrainingConfig,
    ) -> None:
        super().__init__()
        self.policy_model = policy_model
        self.lyap_model = lyap_model
        self.dyn_model = dyn_model
        
        self.kappa = float(config.kappa)

    def forward(self, x: th.Tensor) -> th.Tensor:
        v_curr = self.lyap_model(x)
        u = self.policy_model(x)
        x_next = self.dyn_model(x, u)
        v_next = self.lyap_model(x_next)

        signed_margin = (
            -lyapunov_decrease(v_curr, v_next, self.kappa)
        )

        return signed_margin


class ConditionLirpaLoss(StateBoundsModule):
    def __init__(
        self,
        policy_model: nn.Module,
        lyap_model: nn.Module,
        dyn_model: nn.Module,
        config: LyapunovTrainingConfig,
        device: th.device = th.device("cpu"),
    ):
        super().__init__(state_bounds=config.train_bounds, device=device)

        self.signed_margin_model = SignedConditionMargin(
            policy_model=policy_model,
            lyap_model=lyap_model,
            dyn_model=dyn_model,
            config=config,
        )
        self.relative_eps = float(config.relative_condition_eps)
        self.use_relative_decrease = bool(config.use_relative_decrease)
        self.bounded_model = self._build_bounded_module()

    def _build_bounded_module(self) -> BoundedModule:
        dummy_x = get_center(self.lbx, self.ubx).reshape(1, -1)
        return BoundedModule(
            self.signed_margin_model,
            (dummy_x,),
            device=self.device,
            verbose=False,
            bound_opts={"perturb_bound": True},
        )
    
    def _calculate_weight(self, centers: th.Tensor) -> th.Tensor:
        distances_sq = (centers ** 2).sum(dim=1)
        alpha = 0.1 
        return th.exp(-alpha * distances_sq).clamp(min=0.01)

    def bounded_input(self, x_L: th.Tensor, x_U: th.Tensor) -> BoundedTensor:
        x_center = 0.5 * (x_L + x_U)
        ptb = PerturbationLpNorm(norm=float("inf"), x_L=x_L, x_U=x_U)
        return BoundedTensor(x_center, ptb)

    def forward(self, regions: th.Tensor) -> th.Tensor:
        if regions.shape[0] == 0:
            return th.tensor(0.0, device=self.device, requires_grad=True)
        
        x_L = regions[:, 0, :]
        x_U = regions[:, 1, :]

        bounded_x = self.bounded_input(x_L=x_L, x_U=x_U)
        lb, _ = self.bounded_model.compute_bounds(
            x=(bounded_x,),
            method="IBP+backward",
            bound_upper=False,
        )
        raw_violations = th.relu(-lb).squeeze()
        violations = th.log1p(raw_violations)

        centers = 0.5 * (x_L + x_U)
        
        if self.use_relative_decrease:
            with th.no_grad():
                v_centers = self.signed_margin_model.lyap_model(centers).squeeze(-1)
            final_violations = violations / relative_denominator(v_centers, self.relative_eps)
        else:
            final_violations = violations
            
        weights = self._calculate_weight(centers=centers)
        weighted_violations = weighted_mean(final_violations, weights)
        return weighted_violations


class RoaSurrogateLoss(nn.Module):
    """Compute the auxiliary ROA surrogate used during training."""

    def __init__(self, config: LyapunovTrainingConfig) -> None:
        super().__init__()
        self.rho_min = float(config.rho_min)

    def forward(self, v_candidates: th.Tensor, rho_estimate: float) -> th.Tensor:
        rho_value = safe_rho(rho_estimate, self.rho_min)
        return th.log1p(th.relu(v_candidates / rho_value - 1.0)).mean()


class EquilibriumLoss(nn.Module):
    """Compute the origin consistency loss for the Lyapunov candidate."""

    def __init__(self, model: nn.Module, state_dim: int, device: th.device | str = "cpu") -> None:
        super().__init__()
        self.model = model
        self.device = th.device(device)
        self.register_buffer("origin", th.zeros(1, state_dim, dtype=th.float32, device=self.device))

    def forward(self) -> th.Tensor:
        return self.model(self.origin).pow(2).mean()


class FormalPositivityLoss(StateBoundsModule):
    """Compute the positivity loss induced by a lower bound on V."""
    def __init__(
        self, 
        lyap_model: nn.Module, 
        train_bounds: th.Tensor, 
        device: th.device | str = "cpu"
    ) -> None:
        super().__init__(state_bounds=train_bounds, device=device)

        self.lyap_model = lyap_model

        self.bounded_lyapunov = self._build_lyapunov_bounded_model()
        self.bounded_lyapunov_input = self._build_bounded_positivity_input()
    
    def _build_lyapunov_bounded_model(self) -> BoundedModule:
        if not isinstance(self.lyap_model, nn.Module):
            raise ValueError("Lyapunov model must be provided to construct bounds.")

        dummy_x = th.zeros(1, self.lbx.shape[0], device=self.device)
        return BoundedModule(
            self.lyap_model,
            (dummy_x,),
            device=self.device,
            verbose=False,
            bound_opts={"perturb_bound": True},
        )

    def _build_bounded_positivity_input(self) -> BoundedTensor:
        batch_lbs = self.lbx.reshape(1, -1)
        batch_ubs = self.ubx.reshape(1, -1)
        batch_centers = get_center(self.lbx, self.ubx).reshape(1, -1)
        ptb = PerturbationLpNorm(norm=float("inf"), x_L=batch_lbs, x_U=batch_ubs)
        return BoundedTensor(batch_centers, ptb)

    def compute_lyapunov_lower_bound(self, method: str = "backward") -> th.Tensor:
        lower, _ = self.bounded_lyapunov.compute_bounds(
            x=(self.bounded_lyapunov_input,),
            method=method,
        )
        return lower

    def forward(self) -> th.Tensor:
        lower_bound = self.compute_lyapunov_lower_bound()
        return th.relu(-lower_bound).mean()


class ParameterL1Loss(nn.Module):
    """Compute the l1 norm of an explicitly provided trainable parameter collection."""

    def __init__(self, params: Iterable[nn.Parameter], device: th.device | str = "cpu") -> None:
        super().__init__()
        self.device = th.device(device)
        self._train_params = tuple(params)
        self._base_num_weights = sum(p.numel() for p in self._train_params)
        self._dyn_weight = 1.0

    def set_train_params(self, params: Iterable[nn.Parameter]) -> None:
        """Explicitly set the parameters to be tracked by the L1 loss."""
        self._train_params = tuple(params)
        num_new_weights = sum(p.numel() for p in params)

        if num_new_weights > 0:
            if self._base_num_weights == 0:
                self._base_num_weights = num_new_weights
            self._dyn_weight = self._base_num_weights / num_new_weights
        else:
            self._dyn_weight = 0.0

        __logger__.info(
                "Set trainable parameters for L1 loss: %d -> %d (weight scaled by %.3f)",
                self._base_num_weights,
                num_new_weights,
                self._dyn_weight,
            )

    def forward(self) -> th.Tensor:
        if not self._train_params:
            return th.tensor(0.0, device=self.device)
        return sum(param.abs().sum() for param in self._train_params) * self._dyn_weight


class PolicyRegularizationLoss(BoundedStateSamplingModule):
    """Regularize the policy to stay close to an initial reference policy.

    Samples its own uniform evaluation points from the training bounds so
    that the regularization signal is unbiased across the domain, rather
    than concentrated on counterexample regions.
    """

    def __init__(
        self,
        policy: nn.Module,
        state_bounds: th.Tensor,
        num_samples: int = 1024,
        resample_interval: int = 100,
        device: th.device | str = "cpu",
    ) -> None:
        super().__init__(
            state_bounds=state_bounds, num_samples=num_samples, resample_interval=resample_interval, device=device
        )
        self.init_policy = deepcopy(policy).to(self.device)
        self.policy = policy
        self._set_init_policy_mode()

        with th.no_grad():
            initial_values = self._forward_maybe_raw(
                self.init_policy,
                self.samples,
            )

        self.register_buffer("init_policy_values", initial_values)

    def _set_init_policy_mode(self) -> None:
        for param in self.init_policy.parameters():
            param.requires_grad = False
        self.init_policy.eval()

    def _forward_maybe_raw(self, policy: nn.Module, x_batch: th.Tensor) -> th.Tensor:
        """Evaluate the policy on a batch of states, optionally returning raw outputs."""
        if hasattr(policy, "forward_raw") and callable(getattr(policy, "forward_raw")):
            return policy.forward_raw(x_batch)
        return policy(x_batch)

    @th.no_grad()
    def _update_init_policy_values(self) -> None:
        """Evaluates the initial policy on current samples and caches the result."""
        self.init_policy_values.copy_(
            self._forward_maybe_raw(self.init_policy, self.samples)
        )
    
    def step_sampling(self) -> bool:
        """Wrapper für step_sampling, der an das Resampling gekoppelt ist."""
        needs_update = super().step_sampling() or self.init_policy_values is None
        if needs_update:
            self._update_init_policy_values()
        return needs_update

    def forward(self) -> th.Tensor:
        self.step_sampling()
        out = self._forward_maybe_raw(self.policy, self.samples)
        mse = th.square(out - self.init_policy_values).mean()
        ref_energy = th.square(self.init_policy_values).mean()
        normalized_loss = mse / ref_energy.clamp_min(1e-3)
        return normalized_loss


class RFactorFrobeniusLoss(nn.Module):
    """Compute the Frobenius norm distance of the R factor to regularize its magnitude."""

    def __init__(self, lyap_model: nn.Module, device: th.device | str = "cpu") -> None:
        super().__init__()
        self.lyap_model = lyap_model
        self.device = th.device(device)

        with th.no_grad():
            init_norm = th.linalg.norm(self.lyap_model.r_factor, ord="fro").to(self.device)
            self.register_buffer("init_norm", init_norm)

    def forward(self) -> th.Tensor:
        current_norm = th.linalg.norm(self.lyap_model.r_factor, ord="fro")
        return th.square(current_norm - self.init_norm)


class LyapunovTrainingLoss(nn.Module):
    """Full Lyapunov training objective with embedded models and sub-losses."""

    def __init__(
        self,
        policy_model: nn.Module,
        lyap_model: nn.Module,
        dyn_model: nn.Module,
        config: LyapunovTrainingConfig,
        device: th.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.policy_model = policy_model
        self.lyap_model = lyap_model
        self.dyn_model = dyn_model
        self.config = config
        self.device = th.device(device)
        self.last_loss_parts: LyapunovTrainingLossParts | None = None
        self._setup_losses()


    def _setup_losses(self) -> None:
        """Set up the individual loss components based on the configuration. 
        Losses are instantiated if they have a non-zero weight or if diagnosis is enabled."""

        self.condition_loss = RhoGatedConditionLoss(self.config, device=self.device)
        self.roa_loss = RoaSurrogateLoss(self.config)

        # Lyapunov decrease loss with signed margin model
        self.condition_lirpa_loss = (
            ConditionLirpaLoss(
                policy_model=self.policy_model,
                lyap_model=self.lyap_model,
                dyn_model=self.dyn_model,
                config=self.config,
            )
            if (self.config.condition_lirpa_weight > 0.0 or self.config.enable_diagnosis) else None
        )
        
        # Equilibrium loss
        self.equilibrium_loss = (
            EquilibriumLoss(self.lyap_model, self.config.state_dim, device=self.device) 
            if (self.config.equilibrium_weight > 0.0 or self.config.enable_diagnosis) else None
        )

        # Positivity loss
        self.positivity_loss = (
            FormalPositivityLoss(
                lyap_model=self.lyap_model,
                train_bounds=self.config.train_bounds,
                device=self.device
            )
            if (self.config.formal_positivity_weight > 0.0 or self.config.enable_diagnosis) else None
        )

        # L1 regularization loss
        if self.config.l1_weight > 0.0 or self.config.enable_diagnosis:
            initial_l1_params = self._parameters_excluding_r_factor(list(self.lyap_model.parameters()))
            self.l1_loss = ParameterL1Loss(initial_l1_params, device=self.device)
        else:
            self.l1_loss = None

        # Scale anchor loss
        self.scale_loss = (
            LyapunovScaleAnchorLoss(
                lyap_model=self.lyap_model,
                state_bounds=self.config.train_bounds,
                num_samples=self.config.regularization_num_samples,
                resample_interval=self.config.regularization_resample_interval,
                device=self.device,
            ) 
            if (self.config.scale_weight > 0.0 or self.config.enable_diagnosis) else None
        )

        # Policy regularization loss
        self.policy_regularization_loss = (
            PolicyRegularizationLoss(
                policy=self.policy_model,
                state_bounds=self.config.state_bounds,
                num_samples=self.config.regularization_num_samples,
                resample_interval=self.config.regularization_resample_interval,
                device=self.device,
            )
            if (self.config.policy_regularization_weight > 0.0 or self.config.enable_diagnosis) else None
        )

        # R factor Frobenius loss
        self.r_factor_frobenius_loss = (
            RFactorFrobeniusLoss(lyap_model=self.lyap_model, device=self.device)
            if _has_learnable_r_factor(self.lyap_model) and (self.config.r_factor_fro_norm_weight > 0.0 or self.config.enable_diagnosis) else None
        )

    def _eval_loss_part(
        self,
        weight: float,
        compute_fn: nn.Module | Callable[[], th.Tensor] | None,
        enabled: bool = True,
    ) -> th.Tensor:
        if not enabled or compute_fn is None:
            return th.zeros((), dtype=th.float32, device=self.device)
        if weight > 0.0:
            return compute_fn()
        if self.config.enable_diagnosis:
            with th.no_grad():
                return compute_fn()
        return th.zeros((), dtype=th.float32, device=self.device)

    def _closed_loop_values(
        self,
        x_batch: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Return ``(V(x), x_next, V(x_next))`` for a batch of states."""
        v_batch = self.lyap_model(x_batch)
        u = self.policy_model(x_batch)
        x_next = self.dyn_model(x_batch, u)
        v_next = self.lyap_model(x_next)
        return v_batch, x_next, v_next
    
    def _parameters_excluding_r_factor(self, params: list[nn.Parameter]) -> list[nn.Parameter]:
        """Return a list of parameters excluding the R factor if it exists."""
        if _has_learnable_r_factor(self.lyap_model):
            return [p for p in params if p is not self.lyap_model.r_factor]
        return params
    
    def _get_active_regions(self, regions: th.Tensor, rho: float):
        """Uses """
        centers = 0.5 * (regions[:, 0, :] + regions[:, 1, :])
        x_L = regions[:, 0, :]
        x_U = regions[:, 1, :]

        with th.no_grad():
            v_centers = self.lyap_model(centers).squeeze(-1)
            v_l = self.lyap_model(x_L).squeeze(-1)
            v_u = self.lyap_model(x_U).squeeze(-1)
            v_test = th.minimum(v_centers, th.minimum(v_l, v_u))
        return regions[v_test <= rho]

    def set_explicit_l1_params(self, params: list[nn.Parameter]) -> None:
        """Update the explicitly tracked parameters for the L1 loss and adjust the weight accordingly."""
        if self.l1_loss is not None:
            filtered_params = self._parameters_excluding_r_factor(params)
            self.l1_loss.set_train_params(filtered_params)
    
    def _condition_violation(self, x_batch: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """Return the condition violation and the Lyapunov values of the current states."""
        v_curr, x_next, v_next = self._closed_loop_values(x_batch)
        violation = self.condition_loss.condition_violation(
            v_curr=v_curr, v_next=v_next, x_next=x_next)
        return violation, v_curr

    def get_counterexample_mask(
        self, 
        candidate_states: th.Tensor, 
        rho_estimate: float, 
        violation_tolerance: float = 1e-6
    ) -> tuple[th.Tensor, th.Tensor]:
        """Evaluate candidate counterexamples and return their raw condition violations and a boolean validity mask.
        
        A valid counterexample must be strictly inside the rho-sublevel set and have a positive violation.
        """
        if candidate_states.numel() == 0:
            return (
                th.empty(0, device=candidate_states.device),
                th.empty(0, dtype=th.bool, device=candidate_states.device),
            )

        with th.no_grad():
            violation, v_curr = self._condition_violation(candidate_states)
            violation = violation.squeeze(-1)
            inside_sublevel = v_curr.squeeze(-1) <= rho_estimate
            mask = inside_sublevel & (violation > violation_tolerance)
        return violation, mask

    def mining_objective(self, x_batch: th.Tensor, rho_estimate: float) -> th.Tensor:
        """Return the mining objective value for a batch of states, 
        used for prioritization in the replay buffer and PGD."""
        violation, v_batch = self._condition_violation(x_batch)
        soft_mask = self.condition_loss.soft_sublevel_weight(v_batch, rho_estimate)
        return - soft_mask * violation
    
    def buffer_sorting_objective(self, x_batch: th.Tensor, rho_estimate: float) -> th.Tensor:
        """Return the pure violation score for sorting in the buffer."""
        violation, v_batch = self._condition_violation(x_batch)
        soft_mask = self.condition_loss.soft_sublevel_weight(v_batch, rho_estimate * 1.3)
        return - soft_mask * violation
    
    def compute_loss_parts(
        self,
        x_batch: th.Tensor,
        roa_candidates: th.Tensor,
        rho_estimate: float,
        lirpa_regions: th.Tensor | None = None,
        active_policy_regularization: bool = False,
    ) -> LyapunovTrainingLossParts:
        """Compute the individual loss components without combining them into a total loss."""

        zero = th.zeros((), dtype=th.float32, device=self.device)

        if self.config.condition_weight > 0.0:
            v_batch, x_next, v_next = self._closed_loop_values(x_batch)
        elif self.config.enable_diagnosis:
            with th.no_grad():
                v_batch, x_next, v_next = self._closed_loop_values(x_batch)
        else:
            v_batch, x_next, v_next = None, None, None

        condition_loss_value = self._eval_loss_part(
            self.config.condition_weight,
            lambda: self.condition_loss(
                v_curr=v_batch,
                v_next=v_next,
                x_next=x_next,
                rho_estimate=rho_estimate,
            ),
            enabled=(v_batch is not None),
        )

        roa_loss_value = self._eval_loss_part(
            self.config.roa_weight,
            lambda: self.roa_loss(
                v_candidates=self.lyap_model(roa_candidates),
                rho_estimate=rho_estimate,
            ),
        )

        def _compute_lirpa():
            if self.condition_lirpa_loss is None or lirpa_regions is None:
                return zero
            active_regions = self._get_active_regions(
                lirpa_regions,
                rho=safe_rho(rho_estimate, self.config.rho_min),
            )
            if active_regions.shape[0] > 0:
                return self.condition_lirpa_loss(regions=active_regions)
            return zero

        condition_lirpa_loss_value = self._eval_loss_part(
            self.config.condition_lirpa_weight,
            _compute_lirpa,
            enabled=(self.condition_lirpa_loss is not None and lirpa_regions is not None),
        )

        origin_loss_value = self._eval_loss_part(
            self.config.equilibrium_weight,
            self.equilibrium_loss,
        )

        formal_positivity_loss_value = self._eval_loss_part(
            self.config.formal_positivity_weight,
            self.positivity_loss,
        )

        l1_loss_value = self._eval_loss_part(
            self.config.l1_weight,
            self.l1_loss,
        )

        scale_loss_value = self._eval_loss_part(
            self.config.scale_weight,
            self.scale_loss,
        )

        policy_regularization_loss_value = self._eval_loss_part(
            self.config.policy_regularization_weight,
            self.policy_regularization_loss,
            enabled=active_policy_regularization,
        )

        r_factor_frobenius_loss_value = self._eval_loss_part(
            self.config.r_factor_fro_norm_weight,
            self.r_factor_frobenius_loss,
        )

        parts = LyapunovTrainingLossParts(
            condition_raw=condition_loss_value,
            roa_raw=roa_loss_value,
            condition_lirpa_raw=condition_lirpa_loss_value,
            l1_raw=l1_loss_value,
            equilibrium_raw=origin_loss_value,
            formal_positivity_raw=formal_positivity_loss_value,
            scale_raw=scale_loss_value,
            policy_regularization_raw=policy_regularization_loss_value,
            r_factor_fro_norm_raw=r_factor_frobenius_loss_value,

            condition_weight=self.config.condition_weight,
            roa_weight=self.config.roa_weight,
            condition_lirpa_weight=self.config.condition_lirpa_weight,
            l1_weight=self.config.l1_weight,
            equilibrium_weight=self.config.equilibrium_weight,
            formal_positivity_weight=self.config.formal_positivity_weight,
            scale_weight=self.config.scale_weight,
            policy_regularization_weight=self.config.policy_regularization_weight,
            r_factor_fro_norm_weight=self.config.r_factor_fro_norm_weight,
        )
        self.last_loss_parts = parts
        return parts

    def forward(
        self,
        x_batch: th.Tensor,
        roa_candidates: th.Tensor,
        rho_estimate: float,
        lirpa_regions: th.Tensor | None = None,
        active_policy_regularization: bool = False,
    ) -> th.Tensor:
        parts = self.compute_loss_parts(
            x_batch=x_batch,
            roa_candidates=roa_candidates,
            rho_estimate=rho_estimate,
            lirpa_regions=lirpa_regions,
            active_policy_regularization=active_policy_regularization,
        )
        return parts.total