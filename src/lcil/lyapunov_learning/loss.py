from __future__ import annotations

import logging
import math
import torch as th
import torch.nn as nn

from copy import deepcopy
from dataclasses import dataclass
from collections.abc import Sequence
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

from .config import LyapunovTrainingConfig
from .utils import get_th_lbx_ubx, get_center


__logger__ = logging.getLogger(__name__)

@dataclass
class LyapunovTrainingLossParts:
    """Container for the individual loss components of the Lyapunov training objective."""
    condition_raw: th.Tensor
    roa_raw: th.Tensor
    condition_ibp_raw: th.Tensor
    l1_raw: th.Tensor
    equilibrium_raw: th.Tensor
    formal_positivity_raw: th.Tensor
    scale_raw: th.Tensor
    policy_regularization_raw: th.Tensor

    condition_weight: float
    roa_weight: float
    condition_ibp_weight: float
    l1_weight: float
    equilibrium_weight: float
    formal_positivity_weight: float
    scale_weight: float
    policy_regularization_weight: float

    @property
    def condition(self) -> th.Tensor:
        return self.condition_raw * self.condition_weight

    @property
    def roa(self) -> th.Tensor:
        return self.roa_weight * self.roa_raw

    @property
    def condition_ibp(self) -> th.Tensor:
        return self.condition_ibp_raw * self.condition_ibp_weight

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
    def total(self) -> th.Tensor:
        return (
            self.condition
            + self.roa
            + self.condition_ibp
            + self.l1
            + self.equilibrium
            + self.formal_positivity
            + self.scale
            + self.policy_regularization
        )


def lyapunov_decrease(
    v_curr: th.Tensor,
    v_next: th.Tensor,
    kappa: float,
    condition_margin: float = 0.0,
) -> th.Tensor:
    """Compute the one-step Lyapunov decrease value."""
    return v_next - (1.0 - kappa) * v_curr + condition_margin

def weighted_mean(values: th.Tensor, weights: th.Tensor) -> th.Tensor:
    """Compute a weighted mean of the given values with the provided weights."""
    weights = weights.to(dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


class StateBoundsModule(nn.Module):
    """Base module that stores the shared training-state bounds."""

    def __init__(self, state_bounds: th.Tensor, device: th.device | str = "cpu") -> None:
        super().__init__()
        self.device = th.device(device)

        lbx, ubx = get_th_lbx_ubx(state_bounds, device=self.device)
        self.register_buffer("lbx", lbx)
        self.register_buffer("ubx", ubx)


class LyapunovScaleAnchorLoss(StateBoundsModule):
    """Anchor the absolute scale of V on uniformly sampled reference states."""
    
    def __init__(
        self,
        lyap_model: nn.Module,
        state_bounds: th.Tensor,
        num_anchor_points: int = 1000,
        device: th.device | str = "cpu",
        eps: float = 1e-6
    ) -> None:
        super().__init__(state_bounds=state_bounds, device=device)
        self.lyap_model = lyap_model
        self.eps = float(eps)

        rand_uniform = th.rand((num_anchor_points, self.lbx.shape[0]), device=self.device, dtype=self.lbx.dtype)
        anchor_states = self.lbx + rand_uniform * (self.ubx - self.lbx)
        self.register_buffer("anchor_states", anchor_states)

        self.register_buffer(
            "target_log_value", 
            th.tensor(float("nan"), dtype=th.float32, device=self.device)
        )

    def forward(self) -> th.Tensor:
        v_reference = self.lyap_model(self.anchor_states)
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

    def __init__(self, kappa: float, condition_margin: float = 0.0) -> None:
        super().__init__()
        self.kappa = float(kappa)
        self.condition_margin = float(condition_margin)

    def forward(self, v_curr: th.Tensor, v_next: th.Tensor) -> th.Tensor:
        return th.relu(lyapunov_decrease(v_curr, v_next, self.kappa, self.condition_margin))


class RelativeLyapunovDecreaseViolation(LyapunovDecreaseViolation):
    """Compute a relative version of the one-step Lyapunov decrease violation."""

    def __init__(
        self,
        kappa: float,
        condition_margin: float = 0.0,
        relative_eps: float = 1e-4,
        detach_relative_denominator: bool = True
    ) -> None:
        super().__init__(kappa=kappa, condition_margin=condition_margin)
        self.relative_eps = float(relative_eps)
        self.detach_relative_denominator = bool(detach_relative_denominator)

    def _relative_denominator(self, v_curr: th.Tensor) -> th.Tensor:
        denom_source = v_curr.detach() if self.detach_relative_denominator else v_curr
        return denom_source.abs().clamp_min(self.relative_eps)

    def forward(self, v_curr: th.Tensor, v_next: th.Tensor) -> th.Tensor:
        decrease_violation = super().forward(v_curr=v_curr, v_next=v_next)
        denom_source = self._relative_denominator(v_curr)
        relative_violation = decrease_violation / denom_source
        return relative_violation


class InvarianceViolation(StateBoundsModule):
    """Compute the forward-invariance violation."""

    def __init__(self, state_bounds: th.Tensor, device: th.device | str = "cpu") -> None:
        super().__init__(state_bounds=state_bounds, device=device)

    def forward(self, x_next: th.Tensor) -> th.Tensor:
        upper_violation = th.relu(x_next - self.ubx).sum(dim=1, keepdim=True)
        lower_violation = th.relu(self.lbx - x_next).sum(dim=1, keepdim=True)
        return upper_violation + lower_violation


class RelativeInvarianceViolation(InvarianceViolation):
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
    """Compute a rho-gated relative condition loss with explicit V-size regularization."""

    def __init__(self, config: LyapunovTrainingConfig, device: th.device | str = "cpu") -> None:
        super().__init__()
        self.rho_min = float(config.rho_min)
        self.relative_eps = float(config.relative_condition_eps)
        self.invariance_weight = float(config.invariance_weight)
        self.relative_decrease_violation = RelativeLyapunovDecreaseViolation(
            kappa=config.kappa,
            condition_margin=config.condition_margin,
            relative_eps=config.relative_condition_eps,
            detach_relative_denominator=config.detach_relative_denominator,
        )
        self.relative_invariance_violation = RelativeInvarianceViolation(
            config.state_bounds,
            device=device,
            relative_eps=config.relative_condition_eps,
        )

    def relative_condition_violation(
        self,
        v_curr: th.Tensor,
        v_next: th.Tensor,
        x_next: th.Tensor,
    ) -> th.Tensor:
        rel_decrease = self.relative_decrease_violation(v_curr=v_curr, v_next=v_next)
        rel_invariance = self.relative_invariance_violation(x_next=x_next)
        return rel_decrease + self.invariance_weight * rel_invariance
    
    def rho_value(self, rho_estimate: float) -> float:
        return max(float(rho_estimate), self.rho_min)
    
    def inside_sublevel_mask(self, v_curr: th.Tensor, rho_estimate: float) -> th.Tensor:
        rho_value = self.rho_value(rho_estimate)
        return (v_curr <= rho_value).to(dtype=v_curr.dtype)

    def origin_focused_weight(
        self,
        v_curr: th.Tensor,
        rho_estimate: float,
    ) -> th.Tensor:
        rho_value = self.rho_value(rho_estimate)
        scaled_v = (v_curr.detach() / max(rho_value, self.relative_eps)).clamp(0.0, 1.0)
        return self.inside_sublevel_mask(v_curr=v_curr, rho_estimate=rho_estimate) * (1.0 - scaled_v)

    def forward(
        self,
        v_curr: th.Tensor,
        v_next: th.Tensor,
        x_next: th.Tensor,
        rho_estimate: float,
    ) -> th.Tensor:
        relative_violation = self.relative_condition_violation(
            v_curr=v_curr,
            v_next=v_next,
            x_next=x_next,
        )
        weights = self.origin_focused_weight(v_curr, rho_estimate)
        return weighted_mean(relative_violation, weights)


class SignedConditionMargin(nn.Module):
    def __init__(
        self,
        policy_model: nn.Module,
        lyap_model: nn.Module,
        dyn_model: nn.Module,
        config: LyapunovTrainingConfig,
        device="cpu"
    ) -> None:
        super().__init__()
        self.policy_model = policy_model
        self.lyap_model = lyap_model
        self.dyn_model = dyn_model

        self.invariance_violation = InvarianceViolation(config.state_bounds, device=device)
        
        self.kappa = float(config.kappa)
        self.condition_margin = float(config.condition_margin)
        self.invariance_weight = float(config.invariance_weight)

    def forward(self, x: th.Tensor) -> th.Tensor:
        v_curr = self.lyap_model(x)
        u = self.policy_model(x)
        x_next = self.dyn_model(x, u)
        v_next = self.lyap_model(x_next)

        inv = self.invariance_violation(x_next=x_next)
        signed_margin = (
            -lyapunov_decrease(v_curr, v_next, self.kappa, self.condition_margin)
            - self.invariance_weight * inv
        )

        return signed_margin


class ConditionIBPLoss(StateBoundsModule):
    def __init__(
        self,
        signed_margin_model: nn.Module,
        state_bounds: th.Tensor,
        relative_eps: float = 1e-4,
        device="cpu"
    ):
        super().__init__(state_bounds=state_bounds, device=device)
        self.signed_margin_model = signed_margin_model
        self.relative_eps = float(relative_eps)
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
            method="ibp",
            bound_upper=False,
        )
        violations = th.relu(-lb).squeeze()

        centers = 0.5 * (x_L + x_U)
        with th.no_grad():
            v_centers = self.signed_margin_model.lyap_model(centers).squeeze(-1)
            
        relative_violations = violations / v_centers.clamp_min(self.relative_eps)
        weights = self._calculate_weight(centers=centers)
        weighted_violations = weighted_mean(relative_violations, weights)
        return weighted_violations


class RoaSurrogateLoss(nn.Module):
    """Compute the auxiliary ROA surrogate used during training."""

    def __init__(self, config: LyapunovTrainingConfig) -> None:
        super().__init__()
        self.rho_min = float(config.rho_min)

    def forward(self, v_candidates: th.Tensor, rho_estimate: float) -> th.Tensor:
        rho_value = max(float(rho_estimate), self.rho_min)
        return th.relu(v_candidates / rho_value - 1.0).mean()


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
        state_bounds: th.Tensor, 
        device: th.device | str = "cpu"
    ) -> None:
        super().__init__(state_bounds=state_bounds, device=device)

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
    """Compute the l1 norm of a trainable parameter collection."""

    def __init__(
        self, 
        *models: nn.Module, 
        exclude_param: Sequence[str] = (), 
        device: th.device | str = "cpu"
    ) -> None:
        super().__init__()
        self.device = th.device(device)
        self.models = models
        self.exclude_param = exclude_param
        self._train_params: tuple[nn.Parameter, ...] = ()
        self.refresh_train_params()
    
    def refresh_train_params(self) -> None:
        """Refresh the cached set of currently trainable parameters."""
        train_params = []
        for model in self.models:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # keep excluded parameters out of L1-Loss
                    if any(exclude in name for exclude in self.exclude_param):
                        continue
                    train_params.append(param)
        self._train_params = tuple(train_params)

    def get_train_params(self) -> tuple[nn.Parameter, ...]:
        """Return the cached trainable parameters of the training objective."""
        return self._train_params

    def forward(self) -> th.Tensor:
        trainable_params = self.get_train_params()
        if not trainable_params:
            return th.tensor(0.0, device=self.device)
        return sum(param.abs().sum() for param in trainable_params)


class PolicyRegularizationLoss(nn.Module):
    """Regularize the policy to stay close to an initial reference policy."""

    def __init__(self, policy: nn.Module, device: th.device | str = "cpu", eps: float = 1e-8) -> None:
        super().__init__()
        self.init_policy = deepcopy(policy).to(device)
        self.policy = policy
        self.eps = eps
        self._set_init_policy_mode()

    def _set_init_policy_mode(self) -> None:
        for param in self.init_policy.parameters():
            param.requires_grad = False
        self.init_policy.eval()

    def forward(self, x: th.Tensor) -> th.Tensor:
        with th.no_grad():
            orig_out = self.init_policy(x)
        out = self.policy(x) 
        
        squared_diff = th.square(out - orig_out)
        orig_squared_norm = th.sum(th.square(orig_out), dim=-1, keepdim=True)
        normalized_loss = th.mean(squared_diff / (orig_squared_norm.detach() + self.eps))
        return normalized_loss

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

        self.condition_loss = RhoGatedConditionLoss(config, device=self.device)
        self.roa_loss = RoaSurrogateLoss(config)

        # Only initialize when weight is positive
        # =======================================

        # Condition IBP loss with signed margin model
        self.signed_condition_margin = (
            SignedConditionMargin(
                policy_model=policy_model,
                lyap_model=lyap_model,
                dyn_model=dyn_model,
                config=config,
                device=self.device,
            )
            if config.condition_ibp_weight > 0.0 else None
        )
        self.condition_ibp_loss = (
            ConditionIBPLoss(
                signed_margin_model=self.signed_condition_margin,
                state_bounds=config.state_bounds,
                device=self.device,
            )
            if self.signed_condition_margin is not None else None
        )
        
        # Equilibrium loss
        self.equilibrium_loss = (
            EquilibriumLoss(lyap_model, config.state_dim, device=self.device) 
            if config.equilibrium_weight > 0.0 else None
        )

        # Positivity loss
        self.positivity_loss = (
            FormalPositivityLoss(
                lyap_model=lyap_model,
                state_bounds=config.state_bounds,
                device=self.device
            )
            if config.formal_positivity_weight > 0.0 else None
        )

        # L1 regularization loss
        self.l1_loss = (
            ParameterL1Loss(
                lyap_model,
                policy_model,
                exclude_param=("r_factor",),
                device=self.device
            ) 
            if config.l1_weight > 0.0 else None
        )

        # Scale anchor loss
        self.scale_loss = (
            LyapunovScaleAnchorLoss(
                lyap_model=lyap_model,
                state_bounds=config.state_bounds,
                num_anchor_points=config.scale_anchor_num_points,
                device=self.device,
            ) 
            if config.scale_weight > 0.0 else None
        )

        # Policy regularization loss
        self.policy_regularization_loss = (
            PolicyRegularizationLoss(policy_model, device=device)
            if self.config.policy_regularization_weight > 0.0 else None
        )

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

    def refresh_trainable_parameters(self) -> None:
        if self.l1_loss is not None:
            self.l1_loss.refresh_train_params()

    def mining_objective(
        self,
        x_batch: th.Tensor,
        rho_estimate: float,
    ) -> th.Tensor:
        """Return the mining objective value for a batch of states, used for prioritization in the replay buffer."""
        v_batch, x_next, v_next = self._closed_loop_values(x_batch)
        relative_violation = self.condition_loss.relative_condition_violation(
            v_curr=v_batch,
            v_next=v_next,
            x_next=x_next,
        )
        rho_value = self.condition_loss.rho_value(rho_estimate)
        inside_mask = (v_batch <= rho_value).to(dtype=v_batch.dtype)
        outside_penalty = th.relu(v_batch - rho_value) / max(
            rho_value,
            self.condition_loss.relative_eps,
        )
        return outside_penalty - inside_mask * relative_violation
    
    def buffer_sorting_objective(self, x_batch: th.Tensor) -> th.Tensor:
        """Return the pure relative violation score for sorting in the buffer."""
        v_batch, x_next, v_next = self._closed_loop_values(x_batch)
        relative_violation = self.condition_loss.relative_condition_violation(
            v_curr=v_batch,
            v_next=v_next,
            x_next=x_next,
        )
        return -relative_violation
    
    def compute_loss_parts(
        self,
        x_batch: th.Tensor,
        roa_candidates: th.Tensor,
        rho_estimate: float,
        ibp_regions: th.Tensor | None = None,
        active_policy_regularization: bool = False,
    ) -> LyapunovTrainingLossParts:
        """Compute the individual loss components without combining them into a total loss."""

        v_batch, x_next, v_next = self._closed_loop_values(x_batch)
        condition_loss_value = self.condition_loss(
            v_curr=v_batch,
            v_next=v_next,
            x_next=x_next,
            rho_estimate=rho_estimate,
        )

        v_candidates = self.lyap_model(roa_candidates)
        roa_loss_value = self.roa_loss(
            v_candidates=v_candidates, 
            rho_estimate=rho_estimate
        )
        
        zero = th.zeros((), dtype=v_batch.dtype, device=v_batch.device)

        condition_ibp_loss_value = zero
        if self.condition_ibp_loss is not None and ibp_regions is not None:
            centers = 0.5 * (ibp_regions[:, 0, :] + ibp_regions[:, 1, :])

            with th.no_grad():
                v_centers = self.lyap_model(centers).squeeze(-1)
                
            rho_safe = max(float(rho_estimate), 1e-6)
            expansion_factor = 1.1 
            active_mask = v_centers <= (rho_safe * expansion_factor)

            active_regions = ibp_regions[active_mask]
            if active_regions.shape[0] > 0:
                condition_ibp_loss_value = self.condition_ibp_loss(regions=active_regions)

        origin_loss_value = (
            self.equilibrium_loss()
            if self.equilibrium_loss is not None else zero
        )

        formal_positivity_loss_value = (
            self.positivity_loss()
            if self.positivity_loss is not None else zero
        )

        l1_loss_value = (
            self.l1_loss() 
            if self.l1_loss is not None else zero
        )

        scale_loss_value = (
            self.scale_loss()
            if self.scale_loss is not None else zero
        )

        policy_regularization_loss_value = (
            self.policy_regularization_loss(x_batch)
            if active_policy_regularization and self.policy_regularization_loss is not None else zero
        )

        parts = LyapunovTrainingLossParts(
            condition_raw=condition_loss_value,
            roa_raw=roa_loss_value,
            condition_ibp_raw=condition_ibp_loss_value,
            l1_raw=l1_loss_value,
            equilibrium_raw=origin_loss_value,
            formal_positivity_raw=formal_positivity_loss_value,
            scale_raw=scale_loss_value,
            policy_regularization_raw=policy_regularization_loss_value,

            condition_weight=self.config.condition_weight,
            roa_weight=self.config.roa_weight,
            condition_ibp_weight=self.config.condition_ibp_weight,
            l1_weight=self.config.l1_weight,
            equilibrium_weight=self.config.equilibrium_weight,
            formal_positivity_weight=self.config.formal_positivity_weight,
            scale_weight=self.config.scale_weight,
            policy_regularization_weight=self.config.policy_regularization_weight,
        )
        self.last_loss_parts = parts
        return parts

    def forward(
        self,
        x_batch: th.Tensor,
        roa_candidates: th.Tensor,
        rho_estimate: float,
        ibp_regions: th.Tensor | None = None,
        active_policy_regularization: bool = False,
    ) -> th.Tensor:
        parts = self.compute_loss_parts(
            x_batch=x_batch,
            roa_candidates=roa_candidates,
            rho_estimate=rho_estimate,
            ibp_regions=ibp_regions,
            active_policy_regularization=active_policy_regularization,
        )
        return parts.total


