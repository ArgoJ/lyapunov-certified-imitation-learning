import torch as th
import torch.nn as nn


class LyapunovVerifier(nn.Module):
    """
    Verification graph for the certification Lyapunov condition.

    The module evaluates positivity, decrease, and forward-invariance of the
    closed loop and combines them with a conservative rho-sublevel gate so the
    full certification condition can be traced by verification backends.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        lyap_model: nn.Module,
        dyn_model: nn.Module,
        lbx: th.Tensor,
        ubx: th.Tensor,
        kappa: float,
        sublevel_tolerance: float,
        condition_margin: float = 0.0,
    ):
        """
        Initialize the shared closed-loop Lyapunov core.

        Parameters
        ----------
        policy_model : nn.Module
            The neural network representing the control policy (u = pi(x)).
        lyap_model : nn.Module
            The neural network representing the Lyapunov function candidate (V(x)).
        dyn_model : nn.Module
            The forward dynamics model of the system (x_next = f(x, u)).
        lbx : th.Tensor
            Lower bounds of the admissible state space. Used to penalize out-of-bound 
            transitions to verify forward invariance.
        ubx : th.Tensor
            Upper bounds of the admissible state space.
        kappa : float
            The required proportional decay rate for the Lyapunov function (0 < kappa <= 1).
        sublevel_tolerance : float
            Conservative slack added to the rho-sublevel guard. This keeps
            boundary-touching states inside the checked set when using the
            single-output certification graph.
        condition_margin : float, optional
            Optional additive margin on the decrease condition. Positive values make
            certification stricter while keeping the same logical structure.
        """
        super().__init__()
        self.policy = policy_model
        self.lyap = lyap_model
        self.dyn = dyn_model
        self.kappa = float(kappa)
        self.sublevel_tolerance = float(sublevel_tolerance)
        self.condition_margin = float(condition_margin)

        self.register_buffer("lbx", lbx.reshape(-1))
        self.register_buffer("ubx", ubx.reshape(-1))

    def _closed_loop_terms(
        self,
        x: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Return ``(V(x), x_next, V(x_next))`` for the closed loop."""
        v_curr = self.lyap(x)
        u = self.policy(x)
        x_next = self.dyn(x, u)
        v_next = self.lyap(x_next)
        return v_curr, x_next, v_next

    def _invariance_violation(self, x_next: th.Tensor) -> th.Tensor:
        """Return the L1 forward-invariance violation of ``x_next``."""
        upper_violation = th.relu(x_next - self.ubx).sum(dim=1, keepdim=True)
        lower_violation = th.relu(self.lbx - x_next).sum(dim=1, keepdim=True)
        return upper_violation + lower_violation

    def condition_terms(
        self,
        x: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        """Return ``V(x)`` and the hard certification violation at ``x``."""
        v_curr, x_next, v_next = self._closed_loop_terms(x)

        decrease_violation = v_next - (1.0 - self.kappa) * v_curr + self.condition_margin
        invariance_violation = self._invariance_violation(x_next)
        hard_violation = th.maximum(decrease_violation, invariance_violation)
        positivity_or_hard_violation = th.maximum(hard_violation, -v_curr)

        return v_curr, positivity_or_hard_violation

    @staticmethod
    def _reshape_guard(value: th.Tensor) -> th.Tensor:
        if value.ndim == 1 and value.numel() > 1:
            return value.unsqueeze(1)
        return value

    def forward(self, x: th.Tensor, rho: th.Tensor) -> th.Tensor:
        """
        Evaluates the certification Lyapunov verification condition.

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
            A scalar violation signal. If the returned upper bound is <= 0, the
            positivity, decrease, and invariance conditions all hold for the
            relevant part of the rho-sublevel set.
        """
        v_curr, positivity_or_hard_violation = self.condition_terms(x)
        
        rho = self._reshape_guard(rho)
        sublevel_guard = rho + self.sublevel_tolerance - v_curr
        sublevel_condition = th.minimum(positivity_or_hard_violation, sublevel_guard)

        return sublevel_condition


class LyapunovMultiOutputVerifier(LyapunovVerifier):
    """ABCrown-specific multi-output verifier.

    The output layout follows the external verifier style:

    - output 0: decrease-condition margin, positive when the decrease condition holds
    - output 1: current Lyapunov value ``V(x)``
    - outputs 2..: successor state ``x_next``
    """

    def forward(self, x: th.Tensor, rho: th.Tensor) -> th.Tensor:
        """Return condition margin, ``V(x)``, and ``x_next``.

        ``rho`` is accepted for interface compatibility with the shared wrapper,
        but is intentionally not fused into the outputs. The ABCrown spec
        expresses the rho-sublevel gate separately.
        """
        del rho
        v_curr, x_next, v_next = self._closed_loop_terms(x)
        decrease_margin = (1.0 - self.kappa) * v_curr - v_next - self.condition_margin
        return th.cat((decrease_margin, v_curr, x_next), dim=1)