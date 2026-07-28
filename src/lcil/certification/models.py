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
        """
        super().__init__()
        self.policy = policy_model
        self.lyap = lyap_model
        self.dyn = dyn_model
        self.kappa = float(kappa)

        remove_dropout(self.policy)
        remove_dropout(self.lyap)
        remove_dropout(self.dyn)

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
        decrease_margin = (1.0 - self.kappa) * v_curr - v_next
        return th.cat((decrease_margin, v_curr, x_next), dim=1)


def build_outside_sublevel_constraint(y, rho: float):
    r"""Build a strict outside-sublevel ABCrown predicate.

    Parameters
    ----------
    y : Any
        The output from the Lyapunov verifier for ABCrown.
    rho : float
        The sublevel value for the Lyapunov function.

    Returns
    -------
    Any
        A boolean indicating whether the condition is satisfied in ABCrown.
    """
    return y > rho


def build_positivity_constraint(
    y,
    lyap_output_index: int = 1,
):
    """Build the positivity-only ABCrown predicate for ``V(x)``.
    
    Parameters
    ----------
    y : Any
        The output from the Lyapunov verifier for ABCrown.
    lyap_output_index : int
        The index in the verifier output corresponding to the Lyapunov value.

    Returns
    -------
    Any
        A boolean indicating whether the condition is satisfied in ABCrown.
    """
    return y[lyap_output_index] > 0.0


def build_decrease_constraint(y: float):
    """Build the decrease-only ABCrown predicate for the Lyapunov margin.
    
    Parameters
    ----------
    y : Any
        The output from the Lyapunov verifier for ABCrown.
    
    Returns
    -------
    Any
        A boolean indicating whether the condition is satisfied in ABCrown.

    Limitations
    -----------
    Only can be use in combination with ``LyapunovCoreVerifier``.
    """
    return y[0] > 0.0


def build_invariance_constraint(
    y,
    lb: th.Tensor,
    ub: th.Tensor,
):
    """Build the state-invariance ABCrown predicate for ``x_next``.
    
    Parameters
    ----------
    y : Any
        The output from the Lyapunov verifier for ABCrown.
    lb : th.Tensor
        The lower bounds for the state variables.
    ub : th.Tensor
        The upper bounds for the state variables.

    Returns
    -------
    Any
        A boolean indicating whether the condition is satisfied in ABCrown.

    Limitations
    -----------
    Only can be use in combination with ``LyapunovCoreVerifier``.
    """
    safe_x_next = None

    if lb.numel() != ub.numel() or lb.numel() != y.shape[0] - 2:
        raise ValueError("Dimension mismatch between bounds and verifier output.")

    state_dim = lb.numel()
    for idx in range(state_dim):
        coord_safe = (
            y[idx + 2] > float(lb[idx])
        ) & (
            y[idx + 2] < float(ub[idx])
        )
        safe_x_next = coord_safe if safe_x_next is None else (safe_x_next & coord_safe)

    return safe_x_next


def build_condition_constraint(y: float, lb: th.Tensor, ub: th.Tensor):
    """Build the full Lyapunov-core predicate.

    Parameters
    ----------
    y : Any
        The output from the Lyapunov verifier for ABCrown.
    lb : th.Tensor
        The lower bounds for the state variables.
    ub : th.Tensor
        The upper bounds for the state variables.

    Returns
    -------
    Any
        A boolean indicating whether the condition is satisfied in ABCrown.

    Limitations
    -----------
    Only can be use in combination with ``LyapunovCoreVerifier``.
    """
    safe_positive = build_positivity_constraint(y)
    safe_decrease = build_decrease_constraint(y)
    safe_x_next = build_invariance_constraint(y, lb, ub)
    return safe_positive & safe_decrease & safe_x_next


def remove_dropout(module):
    """Recursively replace all nn.Dropout layers in the given module with nn.Identity."""
    for name, child in module.named_children():
        if isinstance(child, nn.Dropout):
            setattr(module, name, nn.Identity())
        else:
            remove_dropout(child)
            
    return module