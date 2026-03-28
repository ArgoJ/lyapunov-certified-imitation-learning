import torch as th
import torch.nn as nn


class ClosedLoopLyapunovConditionVerifier(nn.Module):
    """
    Verification graph for the relaxed Lyapunov condition in Eq. (16b).
    
    This module constructs the computational graph to evaluate whether the 
    closed-loop system satisfies the Lyapunov decrease condition and forward 
    invariance within a specified sublevel set. It is specifically designed 
    to be traced by AutoLiRPA for formal certification.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        lyap_model: nn.Module,
        dyn_model: nn.Module,
        lbx: th.Tensor, 
        ubx: th.Tensor,
        invariance_weight: float
    ):
        """
        Initializes the verifier module.

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
        invariance_weight : float
            Penalty weight (lambda) applied to state constraint violations. 
            A higher weight strongly enforces forward invariance in the relaxed condition.
        """
        super().__init__()
        self.policy = policy_model
        self.lyap = lyap_model
        self.dyn = dyn_model
        self.invariance_weight = float(invariance_weight)
        
        self.register_buffer("lbx", lbx.reshape(-1))
        self.register_buffer("ubx", ubx.reshape(-1))

    def _invariance_violation(self, x_next: th.Tensor) -> th.Tensor:
        """
        Computes the state constraint violation penalty for the successor state.

        Parameters
        ----------
        x_next : th.Tensor
            The predicted successor states of shape (batch_size, state_dim).

        Returns
        -------
        th.Tensor
            A column tensor of shape (batch_size, 1) containing the L1-norm 
            of the constraint violations (using ReLU activations).
        """
        upper_violation = th.relu(x_next - self.ubx).sum(dim=1, keepdim=True)
        lower_violation = th.relu(self.lbx - x_next).sum(dim=1, keepdim=True)
        return upper_violation + lower_violation

    def forward(self, x: th.Tensor, rho: th.Tensor, kappa: th.Tensor) -> th.Tensor:
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
        kappa : th.Tensor
            A tensor representing the required proportional decay rate (0 < kappa <= 1).

        Returns
        -------
        th.Tensor
            The evaluation of the relaxed condition. If the returned upper bound 
            is <= 0, the Lyapunov condition (decrease and invariance) is strictly 
            satisfied for the given input region.
        """
        if rho.ndim == 1:
            rho = rho.unsqueeze(1)
        if kappa.ndim == 1:
            kappa = kappa.unsqueeze(1)

        v_curr = self.lyap(x)
        u = self.policy(x)
        x_next = self.dyn(x, u)
        v_next = self.lyap(x_next)

        f_term = v_next - (1.0 - kappa) * v_curr
        h_term = self._invariance_violation(x_next)
        decrease_or_invariance = th.relu(f_term) + self.invariance_weight * h_term
        sublevel_guard = rho - v_curr

        # min(a, b) = a - ReLU(a - b)
        relaxed_condition = decrease_or_invariance - th.relu(
            decrease_or_invariance - sublevel_guard
        )
        return relaxed_condition
