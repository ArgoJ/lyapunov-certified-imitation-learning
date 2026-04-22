import torch as th
import torch.nn as nn

from ..utils.base_models import ClosedLoopLyapunovConditionCore


class ClosedLoopLyapunovCertificationVerifier(nn.Module):
    """
    Verification graph for the relaxed Lyapunov condition in Eq. (16b).

    This wrapper adds the rho-sublevel gate on top of the shared closed-loop
    Lyapunov core so it can be traced by certification backends.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        lyap_model: nn.Module,
        dyn_model: nn.Module,
        lbx: th.Tensor,
        ubx: th.Tensor,
        invariance_weight: float,
        kappa: float,
    ):
        super().__init__()
        self.core = ClosedLoopLyapunovConditionCore(
            policy_model=policy_model,
            lyap_model=lyap_model,
            dyn_model=dyn_model,
            lbx=lbx,
            ubx=ubx,
            invariance_weight=invariance_weight,
            kappa=kappa,
        )

    @staticmethod
    def _reshape_guard(value: th.Tensor) -> th.Tensor:
        if value.ndim == 1 and value.numel() > 1:
            return value.unsqueeze(1)
        return value

    def forward(self, x: th.Tensor, rho: th.Tensor) -> th.Tensor:
        """
        Evaluates the relaxed Lyapunov verification condition.

        The condition enforces that for states within the rho-sublevel set 
        (V(x) <= rho), the Lyapunov value must decrease by a factor of (1 - kappa)
        and the successor state must remain within the valid state bounds.

        Parameters
        ----------
        x : th.Tensor
            A batch of current states of shape (batch_size, state_dim).
        rho : th.Tensor
            A tensor representing the target level set bound.

        Returns
        -------
        th.Tensor
            The evaluation of the relaxed condition. If the returned upper bound 
            is <= 0, the Lyapunov condition (decrease and invariance) is strictly 
            satisfied for the given input region.
        """
        v_curr, positivity_or_decrease_violation = self.core.condition_terms(x)
        
        rho = self._reshape_guard(rho)
        sublevel_guard = rho - v_curr
        sublevel_condition = th.minimum(positivity_or_decrease_violation, sublevel_guard)

        return sublevel_condition