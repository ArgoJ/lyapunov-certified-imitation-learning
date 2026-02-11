import torch
import torch.nn as nn

from .base_models import ICNN, MLP


class LyapunovNet(nn.Module):
    """Lyapunov function approximator using ICNN or MLP.

    Parameters
    ----------
    layer_dims : list[int]
        Layer sizes including input and output dimensions.
    activations : list[str]
        Activation names for each layer transition.
    use_icnn : bool
        Whether to use an ICNN (convex) or a plain MLP.
    """

    def __init__(
        self,
        layer_dims: list[int],
        activations: list[str],
        use_icnn: bool = True,
    ):
        super().__init__()
        if use_icnn:
            self.net = ICNN(layer_dims, activations)
        else:
            self.net = MLP(layer_dims, activations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NeuralLyapunovCandidate(nn.Module):
    """Lyapunov candidate from Eq. (9) in the paper.

    V(x) = |phi(x) - phi(x*)| + ||(eps I + R^T R)(x - x*)||_1
    """

    def __init__(
        self,
        feature_net: nn.Module,
        state_dim: int,
        epsilon: float = 1e-3,
        goal_state: torch.Tensor | None = None,
    ):
        super().__init__()
        self.feature_net = feature_net
        self.state_dim = state_dim
        self.epsilon = float(epsilon)
        self.r_factor = nn.Parameter(torch.eye(state_dim))
        if goal_state is None:
            goal_state = torch.zeros(state_dim, dtype=torch.float32)
        self.register_buffer("goal_state", goal_state.reshape(1, state_dim))

    def _pd_matrix(self) -> torch.Tensor:
        eye = torch.eye(
            self.state_dim,
            dtype=self.r_factor.dtype,
            device=self.r_factor.device,
        )
        return self.epsilon * eye + self.r_factor.transpose(0, 1) @ self.r_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        goal = self.goal_state.to(dtype=x.dtype, device=x.device)
        goal_batch = goal.expand(x.shape[0], -1)
        phi_x = self.feature_net(x)
        phi_goal = self.feature_net(goal_batch)
        feature_term = torch.abs(phi_x - phi_goal).sum(dim=1, keepdim=True)

        delta = x - goal_batch
        pd_matrix = self._pd_matrix()
        linear_term = torch.abs(delta @ pd_matrix.transpose(0, 1)).sum(
            dim=1,
            keepdim=True,
        )
        return feature_term + linear_term


class QuadraticLyapunovCandidate(nn.Module):
    """Quadratic Lyapunov candidate from Eq. (10) in the paper."""

    def __init__(
        self,
        state_dim: int,
        epsilon: float = 1e-3,
        goal_state: torch.Tensor | None = None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.epsilon = float(epsilon)
        self.r_factor = nn.Parameter(torch.eye(state_dim))
        if goal_state is None:
            goal_state = torch.zeros(state_dim, dtype=torch.float32)
        self.register_buffer("goal_state", goal_state.reshape(1, state_dim))

    def _pd_matrix(self) -> torch.Tensor:
        eye = torch.eye(
            self.state_dim,
            dtype=self.r_factor.dtype,
            device=self.r_factor.device,
        )
        return self.epsilon * eye + self.r_factor.transpose(0, 1) @ self.r_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        goal = self.goal_state.to(dtype=x.dtype, device=x.device)
        delta = x - goal.expand(x.shape[0], -1)
        pd_matrix = self._pd_matrix()
        value = (delta @ pd_matrix) * delta
        return value.sum(dim=1, keepdim=True)


class ClosedLoopLyapunovVerifier(nn.Module):
    """
    Dieses Modul repräsentiert den Graphen, der verifiziert werden soll.
    Output: V(f(x, pi(x))) - V(x)
    Ziel: Beweisen, dass der Output < 0 ist (Upper Bound < 0).
    """
    def __init__(
        self, 
        policy_model: nn.Module, 
        lyap_model: nn.Module, 
        dyn_model: nn.Module
    ):
        super(ClosedLoopLyapunovVerifier, self).__init__()
        self.policy = policy_model
        self.lyap = lyap_model
        self.dyn = dyn_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v_curr = self.lyap(x)
        u = self.policy(x)
        x_next = self.dyn(x, u)
        v_next = self.lyap(x_next)
        return v_next - v_curr


class ClosedLoopLyapunovConditionVerifier(nn.Module):
    """Verification graph for the relaxed condition in Eq. (16b)."""

    def __init__(
        self,
        policy_model: nn.Module,
        lyap_model: nn.Module,
        dyn_model: nn.Module,
        bounds: torch.Tensor,
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
        self.register_buffer("rho", torch.tensor(float(rho), dtype=torch.float32))

    def set_rho(self, rho: float) -> None:
        self.rho.copy_(torch.tensor(float(rho), dtype=self.rho.dtype, device=self.rho.device))

    def _invariance_violation(self, x_next: torch.Tensor) -> torch.Tensor:
        upper_violation = torch.relu(x_next - self.bounds).sum(dim=1, keepdim=True)
        lower_violation = torch.relu(-self.bounds - x_next).sum(dim=1, keepdim=True)
        return upper_violation + lower_violation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v_curr = self.lyap(x)
        u = self.policy(x)
        x_next = self.dyn(x, u)
        v_next = self.lyap(x_next)

        f_term = v_next - (1.0 - self.kappa) * v_curr
        h_term = self._invariance_violation(x_next)
        decrease_or_invariance = torch.relu(f_term) + self.invariance_weight * h_term
        sublevel_guard = self.rho - v_curr

        # min(a, b) = a - ReLU(a - b)
        relaxed_condition = decrease_or_invariance - torch.relu(
            decrease_or_invariance - sublevel_guard
        )
        return relaxed_condition
