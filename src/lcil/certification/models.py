import torch as th
import torch.nn as nn


class LyapunovCoreVerifier(nn.Module):
    """ABCrown-specific multi-output verifier.

    The output layout follows the external verifier style:

    - output 0: decrease-condition margin, positive when the decrease condition holds
    - output 1: current Lyapunov value ``V(x)``
    - outputs 2..: successor state ``x_next``
    """

    def __init__(
        self,
        policy_model: nn.Module,
        lyap_model: nn.Module,
        dyn_model: nn.Module,
        kappa: float,
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
        kappa : float
            The required proportional decay rate for the Lyapunov function (0 < kappa <= 1).
        condition_margin : float, optional
            Optional additive margin on the decrease condition. Positive values make
            certification stricter while keeping the same logical structure.
        """
        super().__init__()
        self.policy = policy_model
        self.lyap = lyap_model
        self.dyn = dyn_model
        self.kappa = float(kappa)
        self.condition_margin = float(condition_margin)

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

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Return condition margin, ``V(x)``, and ``x_next``."""
        v_curr, x_next, v_next = self._closed_loop_terms(x)
        decrease_margin = (1.0 - self.kappa) * v_curr - v_next - self.condition_margin
        return th.cat((decrease_margin, v_curr, x_next), dim=1)


def build_safe_outside_sublevel_constraint(y, rho: float, sublevel_tolerance: float):
    r"""Build a strict outside-sublevel ABCrown predicate.

    Parameters
    ----------
    y : Any
        The output from the Lyapunov verifier for ABCrown.
    rho : float
        The sublevel value for the Lyapunov function.
    sublevel_tolerance : float
        The tolerance added to the sublevel value.

    Returns
    -------
    Any
        A boolean indicating whether the condition is satisfied in ABCrown.
    """
    return y > (rho + sublevel_tolerance)


def build_safe_inside_sublevel_constraint(y, rho: float, sublevel_tolerance: float):
    r"""Build a strict inside-sublevel ABCrown predicate.

    Parameters
    ----------
    y : Any
        The output from the Lyapunov verifier for ABCrown.
    rho : float
        The sublevel value for the Lyapunov function.
    sublevel_tolerance : float
        The tolerance added to the sublevel value.

    Returns
    -------
    Any
        A boolean indicating whether the condition is satisfied in ABCrown.
    """
    return y < (rho + sublevel_tolerance)

def build_safe_lyap_condition_constraint(y, condition_tolerance: float, lb: th.Tensor, ub: th.Tensor):
    """Builds a safe output for the Lyapunov condition: 
    \( V(x) > 0 \) and \( V(x_{\text{next}}) < V(x) \).

    Parameters
    ----------
    y : Any
        The output from the Lyapunov verifier for ABCrown.
    condition_tolerance : float
        The tolerance added to the condition values.
    lb : th.Tensor
        The lower bounds for the state variables.
    ub : th.Tensor
        The upper bounds for the state variables.

    Returns
    -------
    Any
        A boolean indicating whether the condition is satisfied in ABCrown.
    """
    safe_positive = y[1] > (-condition_tolerance)
    safe_decrease = y[0] > (-condition_tolerance)

    safe_x_next = None

    if lb.numel() != ub.numel() or lb.numel() != y.shape[0] - 2:
        raise ValueError("Dimension mismatch between bounds and verifier output.")
    
    state_dim = lb.numel()
    for idx in range(state_dim):
        coord_safe = (y[idx + 2] > (float(lb[idx]) - condition_tolerance)) & (
            y[idx + 2] < (float(ub[idx]) + condition_tolerance)
        )
        safe_x_next = coord_safe if safe_x_next is None else (safe_x_next & coord_safe)

    return safe_positive & safe_decrease & safe_x_next