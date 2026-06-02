from __future__ import annotations

import math
import torch as th
import torch.nn as nn

from copy import deepcopy
from dataclasses import dataclass
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

from .config import LyapunovTrainingConfig



@dataclass
class LyapunovTrainingLossParts:
    """Container for the individual loss components of the Lyapunov training objective."""
    condition_raw: th.Tensor
    roa_raw: th.Tensor
    l1_raw: th.Tensor
    equilibrium_raw: th.Tensor
    formal_positivity_raw: th.Tensor
    scale_raw: th.Tensor
    policy_regularization_raw: th.Tensor

    roa_weight: float
    l1_weight: float
    equilibrium_weight: float
    formal_positivity_weight: float
    scale_weight: float
    policy_regularization_weight: float

    @property
    def condition(self) -> th.Tensor:
        return self.condition_raw

    @property
    def roa(self) -> th.Tensor:
        return self.roa_weight * self.roa_raw

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
            + self.l1
            + self.equilibrium
            + self.formal_positivity
            + self.scale
            + self.policy_regularization
        )


class LyapunovScaleAnchorLoss(nn.Module):
    """Anchor the absolute scale of V on reference states."""
    
    def __init__(self, target_value: float = 1.0, eps: float = 1e-6) -> None:
        super().__init__()
        if target_value <= 0.0:
            raise ValueError("target_value must be positive.")
        self.target_log_value = float(math.log(target_value))
        self.eps = float(eps)
        
    def forward(self, v_reference: th.Tensor) -> th.Tensor:
        ref_mean = v_reference.mean().clamp_min(self.eps)
        return (th.log(ref_mean) - self.target_log_value).pow(2)


class StateBoundsModule(nn.Module):
    """Base module that stores the shared training-state bounds."""

    def __init__(self, state_bounds: th.Tensor, device: th.device | str = "cpu") -> None:
        super().__init__()
        self.device = th.device(device)

        bounds = th.as_tensor(state_bounds, dtype=th.float32, device=self.device)
        if bounds.ndim != 2 or bounds.shape[0] != 2:
            raise ValueError("state_bounds must have shape (2, nx).")

        self.register_buffer("lbx", bounds[0].reshape(-1))
        self.register_buffer("ubx", bounds[1].reshape(-1))
        self.register_buffer("center", 0.5 * (self.lbx + self.ubx))
        self.register_buffer("widths", (self.ubx - self.lbx).clamp_min(1e-6))


class LyapunovDecreaseViolation(nn.Module):
    """Compute the one-step Lyapunov decrease violation."""

    def __init__(self, kappa: float, condition_margin: float = 0.0) -> None:
        super().__init__()
        self.kappa = float(kappa)
        self.condition_margin = float(condition_margin)

    def forward(self, v_curr: th.Tensor, v_next: th.Tensor) -> th.Tensor:
        return th.relu(v_next - (1.0 - self.kappa) * v_curr + self.condition_margin)


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

    def forward(self, x_next: th.Tensor) -> th.Tensor:
        upper_violation = th.relu(x_next - self.ubx) / self.widths
        lower_violation = th.relu(self.lbx - x_next) / self.widths
        return upper_violation.sum(dim=1, keepdim=True) + lower_violation.sum(dim=1, keepdim=True)

class RhoGatedConditionLoss(StateBoundsModule):
    """Compute a rho-gated relative condition loss with explicit V-size regularization."""

    def __init__(self, config: LyapunovTrainingConfig, device: th.device | str = "cpu") -> None:
        super().__init__(state_bounds=config.state_bounds, device=device)
        self.invariance_weight = float(config.invariance_weight)
        self.rho_min = float(config.rho_min)
        self.relative_eps = float(config.relative_condition_eps)
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

    @staticmethod
    def weighted_mean(values: th.Tensor, weights: th.Tensor) -> th.Tensor:
        weights = weights.to(dtype=values.dtype)
        return (values * weights).sum() / weights.sum().clamp_min(1e-6)
    
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

    def relative_condition_violation(
        self,
        v_curr: th.Tensor,
        v_next: th.Tensor,
        x_next: th.Tensor,
    ) -> th.Tensor:
        """Compute the relative condition violation as a weighted sum of the relative decrease violation and the relative invariance violation.

        Parameters
        ----------
        v_curr : th.Tensor
            Value at the current state, i.e., V(x).
        v_next : th.Tensor
            Value at the next state, i.e., V(x').
        x_next : th.Tensor
            Next state, i.e., x'.

        Returns
        -------
        th.Tensor
            Relative condition violation.
        """
        rel_decrease = self.relative_decrease_violation(v_curr=v_curr, v_next=v_next)
        rel_invariance = self.relative_invariance_violation(x_next=x_next)
        return rel_decrease + self.invariance_weight * rel_invariance

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
        return self.weighted_mean(relative_violation, weights)


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
        batch_centers = self.center.reshape(1, -1)
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

    def __init__(self, *models: nn.Module, device: th.device | str = "cpu") -> None:
        super().__init__()
        self.device = th.device(device)
        self.models = models
        self._train_params: tuple[nn.Parameter, ...] = ()
        self.refresh_train_params()
    
    def refresh_train_params(self) -> None:
        """Refresh the cached set of currently trainable parameters."""
        self._train_params = tuple(
            param
            for model in self.models
            for param in model.parameters()
            if param.requires_grad
        )

    def get_train_params(self) -> tuple[nn.Parameter, ...]:
        """Return the cached trainable parameters of the training objective."""
        return self._train_params

    def forward(self) -> th.Tensor:
        trainable_params = self.get_train_params()
        if not trainable_params:
            return th.zeros((), dtype=th.float32, device=self.device)
        return th.stack([param.abs().sum() for param in trainable_params]).sum()


class PolicyRegularizationLoss(nn.Module):
    """Regularize the policy to stay close to an initial reference policy."""

    def __init__(self, policy: nn.Module, device: th.device | str = "cpu") -> None:
        super().__init__()
        self.init_policy = deepcopy(policy).to(device)
        self.policy = policy
        self._set_init_policy_mode()

    def _set_init_policy_mode(self) -> None:
        for param in self.init_policy.parameters():
            param.requires_grad = False
        self.init_policy.eval()

    def forward(self, x: th.Tensor) -> th.Tensor:
        with th.no_grad():
            orig_out = self.init_policy(x)
        out = self.policy(x) 
        return th.nn.functional.mse_loss(out, orig_out)


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

        bounds = th.as_tensor(config.state_bounds, dtype=th.float32, device=self.device)
        self.register_buffer("lbx", bounds[0].reshape(-1))
        self.register_buffer("ubx", bounds[1].reshape(-1))
        self.register_buffer("center", 0.5 * (self.lbx + self.ubx))

        self.condition_loss = RhoGatedConditionLoss(config, device=self.device)
        self.roa_loss = RoaSurrogateLoss(config)
        self.equilibrium_loss = EquilibriumLoss(lyap_model, config.state_dim, device=self.device)
        self.positivity_loss = FormalPositivityLoss(
            lyap_model=lyap_model,
            state_bounds=config.state_bounds,
            device=self.device,
        )
        self.l1_loss = ParameterL1Loss(lyap_model, policy_model, device=self.device)
        self.scale_loss = LyapunovScaleAnchorLoss(target_value=config.rho_scale_anchor)
        self.policy_regularization_loss = None
        if self.config.policy_regularization_weight > 0.0:
            self.policy_regularization_loss = PolicyRegularizationLoss(policy_model, device=device)
        self.last_loss_parts: LyapunovTrainingLossParts | None = None

    def refresh_trainable_parameters(self) -> None:
        self.l1_loss.refresh_train_params()

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

    def compute_lyapunov_lower_bound(self, method: str = "backward") -> th.Tensor:
        return self.positivity_loss.compute_lyapunov_lower_bound(method=method)

    def formal_positivity_loss(self) -> th.Tensor:
        return self.positivity_loss()

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
        roa_loss_value = self.roa_loss(v_candidates=v_candidates, rho_estimate=rho_estimate)
        origin_loss_value = self.equilibrium_loss()
        formal_positivity_loss_value = self.positivity_loss()
        zero = th.zeros((), dtype=v_batch.dtype, device=v_batch.device)
        l1_loss_value = self.l1_loss() if self.config.l1_weight > 0.0 else zero
        scale_loss_value = self.scale_loss(v_candidates)
        policy_regularization_loss_value = (
            self.policy_regularization_loss(x_batch)
            if active_policy_regularization and self.policy_regularization_loss is not None else zero
        )

        parts = LyapunovTrainingLossParts(
            condition_raw=condition_loss_value,
            roa_raw=roa_loss_value,
            l1_raw=l1_loss_value,
            equilibrium_raw=origin_loss_value,
            formal_positivity_raw=formal_positivity_loss_value,
            scale_raw=scale_loss_value,
            policy_regularization_raw=policy_regularization_loss_value,
            roa_weight=self.config.roa_weight,
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
        active_policy_regularization: bool = False,
    ) -> th.Tensor:
        parts = self.compute_loss_parts(
            x_batch=x_batch,
            roa_candidates=roa_candidates,
            rho_estimate=rho_estimate,
            active_policy_regularization=active_policy_regularization,
        )
        return parts.total


