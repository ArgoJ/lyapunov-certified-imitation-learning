"""
Gradient-based counterexample search for Lyapunov violations.

Implements ``FindCounterExamples`` from the paper using Projected
Gradient Descent (PGD) to find states where V(x_next) - V(x) >= 0
(decrease-condition violation).
"""
import torch

from ..models.dynamics import pvtol_dynamics, STATE_DIM


# ------------------------------------------------------------------
# Helper: compute V(f(x, pi(x))) - V(x)
# ------------------------------------------------------------------
def lyap_diff(policy_model, lyap_model, state, env_params=None):
    """
    Compute V(x_next) - V(x) for given states.

    Parameters
    ----------
    policy_model : nn.Module
        Controller network.
    lyap_model : nn.Module
        Lyapunov network.
    state : Tensor, shape (N, 6)
    env_params : dict, optional

    Returns
    -------
    diff : Tensor, shape (N, 1)
        V(f(x, pi(x))) - V(x).
    state : Tensor
        The same input (useful when ``state.requires_grad``).
    """
    V_x = lyap_model(state)
    action = policy_model(state)
    state_next = pvtol_dynamics(state, action, env_params)
    V_x_next = lyap_model(state_next)
    return V_x_next - V_x, state


# ------------------------------------------------------------------
# PGD counterexample search (maximise V-diff)
# ------------------------------------------------------------------
def _gradient_lyap_diff(policy_model, lyap_model, state, env_params=None):
    """Return sign(grad_x V_diff) for PGD step."""
    diff, _ = lyap_diff(policy_model, lyap_model, state, env_params)
    target = torch.sum(diff)
    policy_model.zero_grad()
    lyap_model.zero_grad()
    target.backward()
    return torch.sign(state.grad)


def find_counterexamples(
    policy_model,
    lyap_model,
    device,
    bounds=None,
    num_samples: int = 1024,
    steps: int = 30,
    tol: float = -1e-4,
    env_params=None,
) -> torch.Tensor:
    """
    PGD search for states where V(x_next) - V(x) >= ``tol``.

    Matches ``FindCounterExamples`` from the paper.

    Parameters
    ----------
    policy_model, lyap_model : nn.Module
    device : str
    bounds : Tensor, shape (6,), optional
        Per-dimension half-width (default 1.0 everywhere).
    num_samples : int
    steps : int
    tol : float
    env_params : dict, optional

    Returns
    -------
    counterexamples : Tensor, shape (M, 6)
        States where the Lyapunov decrease condition is violated.
    """
    if bounds is None:
        bounds = torch.ones(STATE_DIM)
    bounds = bounds.to(device)

    # Random initialisation in [-bounds, bounds]
    delta = torch.zeros(num_samples, STATE_DIM).uniform_(-1, 1)
    x = (delta * bounds).to(device)

    step_size = 1.0 / steps

    for _ in range(steps):
        x.requires_grad = True
        grad_sign = _gradient_lyap_diff(policy_model, lyap_model, x, env_params)

        x = (x + step_size * grad_sign).detach()
        # Project back onto box
        x = torch.max(torch.min(x, bounds), -bounds)

    # Evaluate and filter
    with torch.no_grad():
        diff, _ = lyap_diff(policy_model, lyap_model, x, env_params)

    mask = diff.flatten() >= tol
    return x[mask].clone().detach()

