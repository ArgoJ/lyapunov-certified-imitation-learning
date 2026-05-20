import numpy as np
import torch as th
import torch.nn as nn
from math import gamma, pi


def analytical_quadratic_level_set_measure(
    rho: float,
    p_matrix: np.ndarray,
) -> float:
    """Return the nD measure of the quadratic sublevel set ``x^T P x <= rho``."""
    if rho < 0.0:
        raise ValueError(f"rho must be non-negative, got {rho}.")

    p_array = np.asarray(p_matrix, dtype=np.float64)
    if p_array.ndim != 2 or p_array.shape[0] != p_array.shape[1]:
        raise ValueError(
            f"p_matrix must be a square matrix, got shape {p_array.shape}."
        )

    p_sym = 0.5 * (p_array + p_array.T)
    eigenvalues = np.linalg.eigvalsh(p_sym)
    if np.any(eigenvalues <= 0.0):
        raise ValueError("p_matrix must be positive definite.")

    num_states = p_sym.shape[0]
    unit_ball_measure = pi ** (0.5 * num_states) / gamma(0.5 * num_states + 1.0)
    return float(
        unit_ball_measure * rho ** (0.5 * num_states) / np.sqrt(np.prod(eigenvalues))
    )




class _ZeroPolicy(nn.Module):
    def __init__(self, nu: int = 1):
        super().__init__()
        self.nu = nu

    def forward(self, x: th.Tensor) -> th.Tensor:
        return th.zeros((x.shape[0], self.nu), dtype=x.dtype, device=x.device)


class _ZeroDynamics(nn.Module):
    def forward(self, x: th.Tensor, u: th.Tensor) -> th.Tensor:
        del u
        return th.zeros_like(x)


class _IdentityDynamics(nn.Module):
    def forward(self, x: th.Tensor, u: th.Tensor) -> th.Tensor:
        del u
        return x


class _ShiftDynamics(nn.Module):
    def __init__(self, shift: float):
        super().__init__()
        self.shift = float(shift)

    def forward(self, x: th.Tensor, u: th.Tensor) -> th.Tensor:
        del u
        return x + self.shift


class _DirectionalScaleDynamics(nn.Module):
    def __init__(self, base_scale: float = 0.8, axis_gain: float = 0.4):
        super().__init__()
        self.base_scale = float(base_scale)
        self.axis_gain = float(axis_gain)

    def forward(self, x: th.Tensor, u: th.Tensor) -> th.Tensor:
        del u
        scale = self.base_scale + self.axis_gain * x[:, :1]
        return scale * x


class _DoubleDynamics(nn.Module):
    def forward(self, x: th.Tensor, u: th.Tensor) -> th.Tensor:
        del u
        return 2.0 * x


class _QuadraticLyapunov(nn.Module):
    def forward(self, x: th.Tensor) -> th.Tensor:
        return (x * x).sum(dim=1, keepdim=True)


class _ZeroLyapunov(nn.Module):
    def forward(self, x: th.Tensor) -> th.Tensor:
        return th.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)


class _IdentityLyapunov(nn.Module):
    def forward(self, x: th.Tensor) -> th.Tensor:
        return x[:, :1]


class _OffsetLyapunov(nn.Module):
    def forward(self, x: th.Tensor) -> th.Tensor:
        return (x * x).sum(dim=1, keepdim=True) + 2.0


class _EllipticalLyapunov(nn.Module):
    def forward(self, x: th.Tensor) -> th.Tensor:
        weights = th.tensor([1.0, 4.0], dtype=th.float32, device=x.device)
        return th.sum(weights * x**2, dim=1, keepdim=True)


class _NonFiniteOutsideUnitBallLyapunov(nn.Module):
    def forward(self, x: th.Tensor) -> th.Tensor:
        radius = th.linalg.norm(x, dim=1, keepdim=True)
        values = th.sum(x**2, dim=1, keepdim=True)
        nan_values = th.full_like(values, float("nan"))
        return th.where(radius <= 1.0, values, nan_values)


class _NonMonotonicRadialLyapunov(nn.Module):
    def forward(self, x: th.Tensor) -> th.Tensor:
        radius_sq = th.sum(x**2, dim=1, keepdim=True)
        return radius_sq**2 - radius_sq + 1.0


class _NegativeQuadraticLyapunov(nn.Module):
    def forward(self, x: th.Tensor) -> th.Tensor:
        return -(x * x).sum(dim=1, keepdim=True)


class _MixedLyapunov(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 1.5):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)

    def forward(self, x: th.Tensor) -> th.Tensor:
        r2 = (x * x).sum(dim=1, keepdim=True)
        return self.beta * r2 - self.alpha * (r2 * r2)


class _DescendingLinearLyapunov(nn.Module):
    def forward(self, x: th.Tensor) -> th.Tensor:
        return 2.0 - x[:, :1]


class _FirstCoordinateValue(nn.Module):
    def forward(self, x: th.Tensor) -> th.Tensor:
        return x[:, :1]


class _TrainableQuadraticLyapunov(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(th.ones(1, dtype=th.float32))

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.scale * x.pow(2).sum(dim=1, keepdim=True)


class _LinearValue(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)
        with th.no_grad():
            self.linear.weight.fill_(1.0)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.linear(x)


class _RecordingSequencePolicy(nn.Module):
    def __init__(self, max_seq_len: int, *, output_mode: str = "zeros") -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.output_mode = output_mode
        self.seen_inputs: list[np.ndarray] = []
        self.last_forward_input: th.Tensor | None = None
        self.last_forward_raw_input: th.Tensor | None = None

    def forward(self, x: th.Tensor) -> th.Tensor:
        self.last_forward_input = x.detach().cpu().clone()
        self.seen_inputs.append(x.detach().cpu().numpy().copy())
        if self.output_mode == "zeros":
            return th.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
        if self.output_mode == "sum":
            seq_x = x.unsqueeze(1) if x.ndim == 2 else x
            return seq_x.sum(dim=-1, keepdim=True)
        raise ValueError(f"Unsupported output_mode {self.output_mode!r}.")

    def forward_raw(self, x: th.Tensor) -> th.Tensor:
        self.last_forward_raw_input = x.detach().cpu().clone()
        seq_x = x.unsqueeze(1) if x.ndim == 2 else x
        return 2.0 * seq_x[..., :1]