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
from shared_utils import (
    _IdentityDynamics,
    _QuadraticLyapunov,
    _ZeroDynamics,
    _ZeroPolicy,
)


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
recursive_certifier_module = importlib.import_module("lcil.certification.recursive_certifier")
BisectCertifier = bisect_certifier_module.BisectCertifier
RecursiveCertifier = recursive_certifier_module.RecursiveCertifier
RecursiveCertificationResult = recursive_certifier_module.RecursiveCertificationResult
LyapunovCertificationConfig = importlib.import_module(
    "lcil.certification.config"
).LyapunovCertificationConfig
LyapunovRegionBounds = importlib.import_module(
    "lcil.certification.lirpa_lyapunov_bounds"
).LyapunovRegionBounds
CompleteABCrownCertifier = importlib.import_module(
    "lcil.certification.abcrown_region_certifier"
).CompleteABCrownCertifier
cert_models_module = importlib.import_module("lcil.certification.models")
LyapunovCoreVerifier = cert_models_module.LyapunovCoreVerifier


@dataclass(frozen=True)
class _MockVerificationResult:
    verified: bool
    counterexample_found: bool
    status: str


@dataclass(frozen=True)
class _MockBatchVerification:
    verified_mask: th.Tensor
    counterexample_mask: th.Tensor
    unknown_mask: th.Tensor

    @property
    def failed_mask(self) -> th.Tensor:
        return self.counterexample_mask | self.unknown_mask

    @property
    def any_counterexample(self) -> bool:
        return bool(self.counterexample_mask.any().item())


class _StatusAwareMockRegionCertifier:
    def __init__(self, results: list[_MockVerificationResult]):
        self.results = list(results)
        self.calls = 0

    def verify_region(
        self,
        region: th.Tensor,
        rho: float,
        *,
        is_leaf: bool = False,
        **kwargs,
    ) -> _MockVerificationResult:
        del region, rho, is_leaf, kwargs
        if self.calls >= len(self.results):
            raise AssertionError("verify_region called more often than expected.")
        result = self.results[self.calls]
        self.calls += 1
        return result

    def certify_regions(
        self,
        regions: th.Tensor,
        rho: float,
        *,
        early_exit: bool = False,
        progress=None,
        is_leaf: bool = False,
        **kwargs,
    ) -> _MockBatchVerification:
        verified_mask = th.zeros((len(regions),), dtype=th.bool)
        counterexample_mask = th.zeros((len(regions),), dtype=th.bool)
        unknown_mask = th.zeros((len(regions),), dtype=th.bool)

        for idx, region in enumerate(regions):
            result = self.verify_region(region, rho, is_leaf=is_leaf)
            verified_mask[idx] = result.verified
            counterexample_mask[idx] = result.counterexample_found
            unknown_mask[idx] = (not result.verified and not result.counterexample_found)
            if early_exit and result.counterexample_found:
                break

        return _MockBatchVerification(
            verified_mask=verified_mask,
            counterexample_mask=counterexample_mask,
            unknown_mask=unknown_mask,
        )


class _RecordingMockRegionCertifier(_StatusAwareMockRegionCertifier):
    def __init__(self, results: list[_MockVerificationResult]):
        super().__init__(results)
        self.batches: list[th.Tensor] = []
        self.is_leaf_calls: list[bool] = []

    def certify_regions(
        self,
        regions: th.Tensor,
        rho: float,
        *,
        early_exit: bool = False,
        progress=None,
        is_leaf: bool = False,
        **kwargs,
    ) -> _MockBatchVerification:
        self.batches.append(regions.clone())
        self.is_leaf_calls.append(is_leaf)
        return super().certify_regions(
            regions,
            rho,
            early_exit=early_exit,
            progress=progress,
            is_leaf=is_leaf,
            **kwargs,
        )


def _complete_candidate_partition(regions: th.Tensor):
    from lcil.certification.region_manager import CertificationRegionPartition

    empty = regions[:0]
    return CertificationRegionPartition(
        irrelevant_regions=empty,
        cached_complete_safe_regions=empty,
        cached_core_safe_regions=empty,
        cached_inside_counterexample_regions=empty,
        cached_inside_unknown_regions=empty,
        inside_core_unchecked_regions=empty,
        boundary_core_unchecked_regions=empty,
        boundary_complete_candidate_regions=regions,
    )


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
    output_spec: Any = None


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
            output_spec=types.SimpleNamespace(clauses=[output_constraint]),
        )

    @staticmethod
    def build_from_input_bounds(
        lower: th.Tensor,
        upper: th.Tensor,
        clauses: Any,
    ) -> _FakeVerificationSpecData:
        return _FakeVerificationSpecData(
            lb=lower.squeeze(0),
            ub=upper.squeeze(0),
            output_constraint=clauses,
            output_spec=types.SimpleNamespace(clauses=[clauses]),
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
        sys.modules.pop("lcil.certification.recursive_certifier", None)
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
        lirpa_method: str = "alpha-crown",
        sublevel_tolerance: float = 1e-6,
        condition_tolerance: float = 1e-6,
        suppress_native_output: bool = True,
        batch_size: int = 512,
        max_recursion_depth: int = 3,
        skip_boundary_core_cert: bool = False,
        abcrown_timeout: float | None = None,
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
            lirpa_method=lirpa_method,
            sublevel_tolerance=sublevel_tolerance,
            condition_tolerance=condition_tolerance,
            suppress_native_output=suppress_native_output,
            batch_size=batch_size,
            max_recursion_depth=max_recursion_depth,
            skip_boundary_core_cert=skip_boundary_core_cert,
            abcrown_timeout=abcrown_timeout,
        )

    @classmethod
    def make_bisect_certifier(
        cls,
        *,
        policy_model: nn.Module | None = None,
        lyap_model: nn.Module | None = None,
        dyn_model: nn.Module | None = None,
        config: Any = None,
        progress_level: int = 0,
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
            progress_level=progress_level,
        )

    @classmethod
    def make_recursive_certifier(
        cls,
        *,
        policy_model: nn.Module | None = None,
        lyap_model: nn.Module | None = None,
        dyn_model: nn.Module | None = None,
        config: Any = None,
        progress_level: int = 0,
        **config_kwargs,
    ) -> Any:
        if config is None:
            state_dim = int(config_kwargs.pop("state_dim", 3))
            config = cls.make_config(state_dim=state_dim, **config_kwargs)
        return RecursiveCertifier(
            policy_model=_ZeroPolicy() if policy_model is None else policy_model,
            lyap_model=_QuadraticLyapunov() if lyap_model is None else lyap_model,
            dyn_model=_ZeroDynamics() if dyn_model is None else dyn_model,
            config=config,
            device=th.device("cpu"),
            progress_level=progress_level,
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
        return CompleteABCrownCertifier(
            policy_model=_ZeroPolicy() if policy_model is None else policy_model,
            lyap_model=_QuadraticLyapunov() if lyap_model is None else lyap_model,
            dyn_model=_IdentityDynamics() if dyn_model is None else dyn_model,
            config=config,
            device=th.device("cpu"),
        )