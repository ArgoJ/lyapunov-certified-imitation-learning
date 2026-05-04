import itertools
import importlib
import sys
import types
import unittest
from dataclasses import dataclass, field
from typing import Any
from unittest import mock

import numpy as np
import torch as th
import torch.nn as nn


_ORIG_ABCROWN = sys.modules.get("abcrown")

_abcrown_stub = types.ModuleType("abcrown")
_abcrown_stub.ABCrownSolver = object
_abcrown_stub.VerificationSpec = object
_abcrown_stub.ConfigBuilder = object
_abcrown_stub.input_vars = lambda *_args, **_kwargs: None
_abcrown_stub.output_vars = lambda *_args, **_kwargs: None


def _install_abcrown_stub() -> None:
    sys.modules["abcrown"] = _abcrown_stub


_install_abcrown_stub()

bisect_certifier_module = importlib.import_module("lcil.certification.bisect_certifier")
BisectCertifier = bisect_certifier_module.BisectCertifier
RecursiveCertificationResult = bisect_certifier_module.RecursiveCertificationResult
LyapunovCertificationConfig = importlib.import_module(
    "lcil.certification.config"
).LyapunovCertificationConfig
LyapunovRegionBounds = importlib.import_module(
    "lcil.certification.lirpa_lyapunov_bounds"
).LyapunovRegionBounds
ABCrownRegionCertifier = importlib.import_module(
    "lcil.certification.abcrown_region_certifier"
).ABCrownRegionCertifier
cert_models_module = importlib.import_module("lcil.certification.models")
LyapunovCoreVerifier = cert_models_module.LyapunovCoreVerifier


@dataclass
class _FakeInputConstraint:
    lb: th.Tensor | None = None
    ub: th.Tensor | None = None

    def __and__(self, other: "_FakeInputConstraint") -> "_FakeInputConstraint":
        lb = self.lb if self.lb is not None else other.lb
        ub = self.ub if self.ub is not None else other.ub
        return _FakeInputConstraint(lb=lb, ub=ub)


class _FakeInputVars:
    def __init__(self, dim: int):
        self.dim = dim

    def __ge__(self, lb: th.Tensor) -> _FakeInputConstraint:
        return _FakeInputConstraint(lb=th.as_tensor(lb, dtype=th.float32).reshape(self.dim))

    def __le__(self, ub: th.Tensor) -> _FakeInputConstraint:
        return _FakeInputConstraint(ub=th.as_tensor(ub, dtype=th.float32).reshape(self.dim))


@dataclass
class _FakeOutputConstraint:
    predicate: object

    def evaluate(self, values: th.Tensor) -> th.Tensor:
        values_2d = values if values.ndim > 1 else values.unsqueeze(1)
        result = self.predicate(values_2d)
        return th.as_tensor(result, dtype=th.bool, device=values_2d.device).reshape(-1)

    def __and__(self, other: "_FakeOutputConstraint") -> "_FakeOutputConstraint":
        return _FakeOutputConstraint(
            predicate=lambda values: self.evaluate(values) & other.evaluate(values)
        )

    def __or__(self, other: "_FakeOutputConstraint") -> "_FakeOutputConstraint":
        return _FakeOutputConstraint(
            predicate=lambda values: self.evaluate(values) | other.evaluate(values)
        )


class _FakeOutputVar:
    def __init__(self, idx: int):
        self.idx = idx

    @staticmethod
    def _to_float(value) -> float:
        return float(th.as_tensor(value, dtype=th.float32).reshape(()).item())

    def __lt__(self, rhs: float) -> _FakeOutputConstraint:
        rhs_value = self._to_float(rhs)
        return _FakeOutputConstraint(predicate=lambda values: values[:, self.idx] < rhs_value)

    def __le__(self, rhs: float) -> _FakeOutputConstraint:
        rhs_value = self._to_float(rhs)
        return _FakeOutputConstraint(predicate=lambda values: values[:, self.idx] <= rhs_value)

    def __gt__(self, rhs: float) -> _FakeOutputConstraint:
        rhs_value = self._to_float(rhs)
        return _FakeOutputConstraint(predicate=lambda values: values[:, self.idx] > rhs_value)

    def __ge__(self, rhs: float) -> _FakeOutputConstraint:
        rhs_value = self._to_float(rhs)
        return _FakeOutputConstraint(predicate=lambda values: values[:, self.idx] >= rhs_value)


class _FakeOutputVars:
    def __init__(self, dim: int):
        self.dim = dim

    @property
    def shape(self) -> tuple[int]:
        return (self.dim,)

    def __getitem__(self, idx: int) -> _FakeOutputVar:
        if idx < 0 or idx >= self.dim:
            raise IndexError(idx)
        return _FakeOutputVar(idx)


@dataclass
class _FakeVerificationSpecData:
    lb: th.Tensor
    ub: th.Tensor
    output_constraint: _FakeOutputConstraint


class _FakeVerificationSpec:
    @staticmethod
    def build_spec(
        input_vars: _FakeInputVars,
        output_vars: _FakeOutputVars,
        input_constraint: _FakeInputConstraint,
        output_constraint: _FakeOutputConstraint,
    ) -> _FakeVerificationSpecData:
        del input_vars, output_vars
        if input_constraint.lb is None or input_constraint.ub is None:
            raise ValueError("Both lower and upper input bounds are required.")
        return _FakeVerificationSpecData(
            lb=input_constraint.lb,
            ub=input_constraint.ub,
            output_constraint=output_constraint,
        )


@dataclass
class _FakeSolveResult:
    status: str
    stats: dict[str, Any] = field(default_factory=dict)


class _FakeABCrownSolver:
    def __init__(
        self,
        spec: _FakeVerificationSpecData,
        computing_graph: nn.Module,
        config: dict,
    ):
        del config
        self.spec = spec
        self.computing_graph = computing_graph

    @staticmethod
    def _sample_points(lb: th.Tensor, ub: th.Tensor) -> th.Tensor:
        coords_per_dim = []
        for idx in range(lb.numel()):
            mid = 0.5 * (lb[idx] + ub[idx])
            coords_per_dim.append((lb[idx].item(), mid.item(), ub[idx].item()))
        grid = list(itertools.product(*coords_per_dim))
        return th.as_tensor(grid, dtype=lb.dtype, device=lb.device)

    def solve(self) -> _FakeSolveResult:
        points = self._sample_points(self.spec.lb, self.spec.ub)
        with th.no_grad():
            values = self.computing_graph(points)
        safe_mask = self.spec.output_constraint.evaluate(values)
        status = "verified" if bool(safe_mask.all().item()) else "unsafe"
        return _FakeSolveResult(status=status)


class _FakeConfigBuilder:
    def __init__(self):
        self.config: dict[str, str] = {}

    @staticmethod
    def from_defaults() -> "_FakeConfigBuilder":
        return _FakeConfigBuilder()

    def set(self, **kwargs) -> "_FakeConfigBuilder":
        self.config.update(kwargs)
        return self

    def __call__(self) -> dict:
        return dict(self.config)


def _fake_input_vars(dim: int) -> _FakeInputVars:
    return _FakeInputVars(dim)


def _fake_output_vars(dim: int) -> _FakeOutputVars:
    return _FakeOutputVars(dim)


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


class _QuadraticLyapunov(nn.Module):
    def forward(self, x: th.Tensor) -> th.Tensor:
        return (x * x).sum(dim=1, keepdim=True)


class _IdentityLyapunov(nn.Module):
    def forward(self, x: th.Tensor) -> th.Tensor:
        return x[:, :1]


class _DirectionalScaleDynamics(nn.Module):
    def __init__(self, base_scale: float = 0.8, axis_gain: float = 0.4):
        super().__init__()
        self.base_scale = float(base_scale)
        self.axis_gain = float(axis_gain)

    def forward(self, x: th.Tensor, u: th.Tensor) -> th.Tensor:
        del u
        scale = self.base_scale + self.axis_gain * x[:, :1]
        return scale * x


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


class CertificationMockedABCrownTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        _install_abcrown_stub()
        cls._patchers = [
            mock.patch.object(_abcrown_stub, "ABCrownSolver", _FakeABCrownSolver),
            mock.patch.object(_abcrown_stub, "VerificationSpec", _FakeVerificationSpec),
            mock.patch.object(_abcrown_stub, "ConfigBuilder", _FakeConfigBuilder),
            mock.patch.object(_abcrown_stub, "input_vars", _fake_input_vars),
            mock.patch.object(_abcrown_stub, "output_vars", _fake_output_vars),
        ]
        for patcher in cls._patchers:
            patcher.start()

    @classmethod
    def tearDownClass(cls) -> None:
        for patcher in getattr(cls, "_patchers", []):
            patcher.stop()

        sys.modules.pop("lcil.certification.bisect_certifier", None)
        sys.modules.pop("lcil.certification.abcrown_region_certifier", None)
        sys.modules.pop("lcil.certification.config", None)

        if _ORIG_ABCROWN is None:
            sys.modules.pop("abcrown", None)
        else:
            sys.modules["abcrown"] = _ORIG_ABCROWN
        super().tearDownClass()

    @staticmethod
    def make_config(
        *,
        state_dim: int,
        cert_bounds: list[list[float]] | np.ndarray | None = None,
        kappa: float = 0.1,
        rho_min: float = 1e-6,
        bins_per_dim: int | tuple[int, ...] = 4,
        center_refinement_factor: float | tuple[float, ...] = 1.0,
        origin_exclusion: float | tuple[float, ...] = 0.0,
        rho_scaling: float = 1.2,
        bisection_tol: float = 1e-3,
        max_scale_steps: int = 6,
        max_bisection_steps: int = 6,
        cert_method: str = "alpha-crown",
        sublevel_tolerance: float = 1e-6,
        condition_tolerance: float = 1e-6,
        condition_margin: float = 0.0,
        suppress_native_output: bool = True,
        batch_size: int = 512,
        max_recursion_depth: int = 3,
    ) -> Any:
        bounds = cert_bounds
        if bounds is None:
            bounds = np.array(
                [
                    [-2.0] * state_dim,
                    [2.0] * state_dim,
                ],
                dtype=np.float32,
            )
        else:
            bounds = np.asarray(bounds, dtype=np.float32)

        return LyapunovCertificationConfig(
            state_dim=state_dim,
            cert_bounds=bounds,
            kappa=kappa,
            rho_min=rho_min,
            bins_per_dim=bins_per_dim,
            center_refinement_factor=center_refinement_factor,
            origin_exclusion=origin_exclusion,
            rho_scaling=rho_scaling,
            bisection_tol=bisection_tol,
            max_scale_steps=max_scale_steps,
            max_bisection_steps=max_bisection_steps,
            cert_method=cert_method,
            sublevel_tolerance=sublevel_tolerance,
            condition_tolerance=condition_tolerance,
            condition_margin=condition_margin,
            suppress_native_output=suppress_native_output,
            batch_size=batch_size,
            max_recursion_depth=max_recursion_depth,
        )

    @classmethod
    def make_bisect_certifier(
        cls,
        *,
        policy_model: nn.Module | None = None,
        lyap_model: nn.Module | None = None,
        dyn_model: nn.Module | None = None,
        config: Any = None,
        **config_kwargs,
    ) -> Any:
        if config is None:
            state_dim = int(config_kwargs.pop("state_dim", 3))
            config = cls.make_config(state_dim=state_dim, **config_kwargs)
        return BisectCertifier(
            policy_model=_ZeroPolicy() if policy_model is None else policy_model,
            lyap_model=_QuadraticLyapunov() if lyap_model is None else lyap_model,
            dyn_model=_ZeroDynamics() if dyn_model is None else dyn_model,
            config=config,
            device=th.device("cpu"),
        )

    @classmethod
    def make_abcrown_region_certifier(
        cls,
        *,
        policy_model: nn.Module | None = None,
        lyap_model: nn.Module | None = None,
        dyn_model: nn.Module | None = None,
        config: Any = None,
        **config_kwargs,
    ) -> Any:
        if config is None:
            state_dim = int(config_kwargs.pop("state_dim", 1))
            config = cls.make_config(state_dim=state_dim, **config_kwargs)
        return ABCrownRegionCertifier(
            policy_model=_ZeroPolicy() if policy_model is None else policy_model,
            lyap_model=_QuadraticLyapunov() if lyap_model is None else lyap_model,
            dyn_model=_IdentityDynamics() if dyn_model is None else dyn_model,
            config=config,
            device=th.device("cpu"),
        )