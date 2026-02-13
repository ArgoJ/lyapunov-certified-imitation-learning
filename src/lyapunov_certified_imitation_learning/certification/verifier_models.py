import torch as th
import torch.nn as nn


class ClosedLoopLyapunovConditionVerifier(nn.Module):
    """Verification graph for the relaxed condition in Eq. (16b)."""

    def __init__(
        self,
        policy_model: nn.Module,
        lyap_model: nn.Module,
        dyn_model: nn.Module,
        bounds: th.Tensor,
        kappa: float,
        invariance_weight: float,
        rho: float,
    ):
        super().__init__()
        self.policy = policy_model
        self.lyap = lyap_model
        self.dyn = dyn_model
        self.kappa = float(kappa)
        self.invariance_weight = float(invariance_weight)
        # Keep constants batch-free for AutoLiRPA operators.
        self.register_buffer("bounds", bounds.reshape(-1))
        self.register_buffer("rho", th.tensor(float(rho), dtype=th.float32))

    def set_rho(self, rho: float) -> None:
        self.rho.copy_(th.tensor(float(rho), dtype=self.rho.dtype, device=self.rho.device))

    def _invariance_violation(self, x_next: th.Tensor) -> th.Tensor:
        upper_violation = th.relu(x_next - self.bounds).sum(dim=1, keepdim=True)
        lower_violation = th.relu(-self.bounds - x_next).sum(dim=1, keepdim=True)
        return upper_violation + lower_violation

    def forward(self, x: th.Tensor) -> th.Tensor:
        v_curr = self.lyap(x)
        u = self.policy(x)
        x_next = self.dyn(x, u)
        v_next = self.lyap(x_next)

        f_term = v_next - (1.0 - self.kappa) * v_curr
        h_term = self._invariance_violation(x_next)
        decrease_or_invariance = th.relu(f_term) + self.invariance_weight * h_term
        sublevel_guard = self.rho - v_curr

        # min(a, b) = a - ReLU(a - b)
        relaxed_condition = decrease_or_invariance - th.relu(
            decrease_or_invariance - sublevel_guard
        )
        return relaxed_condition
