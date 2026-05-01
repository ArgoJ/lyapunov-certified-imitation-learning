import itertools
import importlib
import sys
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import numpy as np
import torch as th
import torch.nn as nn
from plot_assertions_mixin import PlotAssertionsMixin

_ORIG_ABCROWN = sys.modules.get("abcrown")
_ORIG_LCIL_CERTIFICATION = sys.modules.get("lcil.certification")

_abcrown_stub = types.ModuleType("abcrown")
_abcrown_stub.ABCrownSolver = object
_abcrown_stub.VerificationSpec = object
_abcrown_stub.ConfigBuilder = object
_abcrown_stub.input_vars = lambda *_args, **_kwargs: None
_abcrown_stub.output_vars = lambda *_args, **_kwargs: None
sys.modules["abcrown"] = _abcrown_stub

_certification_stub = types.ModuleType("lcil.certification")
_certification_stub.__path__ = [
    str(Path(__file__).resolve().parents[1] / "src" / "lcil" / "certification")
]
sys.modules["lcil.certification"] = _certification_stub

bisect_certifier_module = importlib.import_module("lcil.certification.bisect_certifier")
BisectCertifier = bisect_certifier_module.BisectCertifier
LyapunovCertificationConfig = importlib.import_module(
    "lcil.certification.config"
).LyapunovCertificationConfig
RecursiveCertificationResult = bisect_certifier_module.RecursiveCertificationResult
region_bounds_module = importlib.import_module("lcil.certification.lirpa_lyapunov_bounds")
LyapunovRegionBounds = region_bounds_module.LyapunovRegionBounds
plot_module = importlib.import_module("lcil.utils.plot")
certified_regions_2d = plot_module.certified_regions_2d
lyapunov_cert_regions = plot_module.lyapunov_cert_regions
cert_models_module = importlib.import_module("lcil.certification.models")
LyapunovVerifier = cert_models_module.LyapunovVerifier
LyapunovMultiOutputVerifier = cert_models_module.LyapunovMultiOutputVerifier


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
        return _FakeOutputConstraint(
            predicate=lambda values: values[:, self.idx] < rhs_value
        )

    def __le__(self, rhs: float) -> _FakeOutputConstraint:
        rhs_value = self._to_float(rhs)
        return _FakeOutputConstraint(
            predicate=lambda values: values[:, self.idx] <= rhs_value
        )

    def __gt__(self, rhs: float) -> _FakeOutputConstraint:
        rhs_value = self._to_float(rhs)
        return _FakeOutputConstraint(
            predicate=lambda values: values[:, self.idx] > rhs_value
        )

    def __ge__(self, rhs: float) -> _FakeOutputConstraint:
        rhs_value = self._to_float(rhs)
        return _FakeOutputConstraint(
            predicate=lambda values: values[:, self.idx] >= rhs_value
        )


class _FakeOutputVars:
    def __init__(self, dim: int):
        self.dim = dim

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


class _StaticRegionBounder:
    def __init__(self, lower: th.Tensor, upper: th.Tensor):
        self.bounds = LyapunovRegionBounds(lower=lower, upper=upper)

    def compute_bounds_for_regions(self, regions: th.Tensor, *, method: str | None = None):
        del regions, method
        return self.bounds


class _RecursiveMockCertifier(BisectCertifier):
    def __init__(self, *args, width_limit: float, rho_limit: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.width_limit = float(width_limit)
        self.rho_limit = float(rho_limit)
        self.certify_calls: list[tuple[int, bool]] = []

    def setup_backend(self, *args, **kwargs) -> None:
        del args, kwargs

    def _partition_regions_by_sublevel(
        self,
        bs: th.Tensor,
        rho: float,
    ) -> tuple[th.Tensor, th.Tensor]:
        del rho
        return bs, bs[:0]

    def _certify_batched_regions(
        self,
        bs: th.Tensor,
        rho: float,
        early_exit: bool = True,
        *args,
        **kwargs,
    ) -> th.Tensor:
        del args, kwargs
        self.certify_calls.append((len(bs), early_exit))
        widths = (bs[:, 1] - bs[:, 0]).amax(dim=1)
        return (widths <= self.width_limit) & (rho <= self.rho_limit)


class TestBisectCertifier(PlotAssertionsMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
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
        for patcher in cls._patchers:
            patcher.stop()

        # Drop cached submodules that may have captured stubbed abcrown symbols.
        sys.modules.pop("lcil.certification.bisect_certifier", None)
        sys.modules.pop("lcil.certification.config", None)

        # Restore original modules so later tests can import real integrations.
        if _ORIG_ABCROWN is None:
            sys.modules.pop("abcrown", None)
        else:
            sys.modules["abcrown"] = _ORIG_ABCROWN

        if _ORIG_LCIL_CERTIFICATION is None:
            sys.modules.pop("lcil.certification", None)
        else:
            sys.modules["lcil.certification"] = _ORIG_LCIL_CERTIFICATION

    @staticmethod
    def _make_certifier(
        lyap_model: nn.Module,
        *,
        dyn_model: nn.Module | None = None,
        kappa: float = 0.1,
        rho_min: float = 1e-6,
    ) -> BisectCertifier:
        config = LyapunovCertificationConfig(
            state_dim=3,
            cert_bounds=np.array([[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]], dtype=np.float32),
            kappa=kappa,
            rho_min=rho_min,
            bins_per_dim=4,
            center_refinement_factor=0.7,
            origin_exclusion=0.0,
            max_scale_steps=6,
            max_bisection_steps=6,
            cert_method="alpha-crown",
            condition_tolerance=1e-6,
            max_recursion_depth=3,
        )
        return BisectCertifier(
            policy_model=_ZeroPolicy(),
            lyap_model=lyap_model,
            dyn_model=_ZeroDynamics() if dyn_model is None else dyn_model,
            config=config,
            device=th.device("cpu"),
        )

    def _assert_region_plot_written(
        self,
        certification_result,
        stem: str,
    ) -> None:
        self._assert_plot_written(
            plot_fn=certified_regions_2d,
            stem=stem,
            plot_kwargs={
                "certification_result": certification_result,
                "state_labels": ["x0", "x1", "x2"],
            },
        )

    @staticmethod
    def _to_numpy_lyapunov(
        lyap_model: nn.Module,
        state_dim: int,
    ):
        def _lyapunov_func(x: np.ndarray) -> np.ndarray | float:
            x_array = np.asarray(x, dtype=np.float32)

            if x_array.shape[-1] == state_dim:
                x_lifted = x_array
            elif x_array.shape[-1] < state_dim:
                # mpc_datagen may evaluate pairwise projections with reduced tail dimensions.
                # Preserve existing coordinates and pad missing higher dimensions with zeros.
                x_lifted = np.zeros((*x_array.shape[:-1], state_dim), dtype=np.float32)
                x_lifted[..., : x_array.shape[-1]] = x_array
            else:
                raise ValueError(
                    f"Lyapunov input has invalid shape {x_array.shape}; expected last dim "
                    f"<= {state_dim}."
                )

            x_tensor = th.as_tensor(x_lifted, dtype=th.float32)
            if x_tensor.ndim == 1:
                x_tensor = x_tensor.unsqueeze(0)

            with th.no_grad():
                values = lyap_model(x_tensor).reshape(-1)

            values_np = values.detach().cpu().numpy()
            if x_lifted.ndim == 1:
                return float(values_np[0])
            return values_np

        return _lyapunov_func

    def _assert_lyapunov_plot_written(
        self,
        lyap_model: nn.Module,
        certification_result,
        stem: str,
    ) -> None:
        lyap_func = self._to_numpy_lyapunov(
            lyap_model=lyap_model,
            state_dim=3,
        )
        self._assert_plot_written(
            plot_fn=lyapunov_cert_regions,
            stem=stem,
            plot_kwargs={
                "dataset": None,
                "lyapunov_func": lyap_func,
                "state_labels": ["x0", "x1", "x2"],
                "certification_result": certification_result,
            },
        )

    def test_quadratic_lyapunov_with_identity_dynamics_certifies_all_regions(self) -> None:
        certifier = self._make_certifier(
            _QuadraticLyapunov(),
            dyn_model=_IdentityDynamics(),
            kappa=0.0,
        )
        result = certifier.certify(rho_estimate=1.0)

        self.assertTrue(result.global_success)
        self.assertTrue(result.partial_success)
        self.assertGreaterEqual(result.rho, 1.0)
        self.assertEqual(result.failed_regions.shape[0], 0)
        self.assertGreater(result.certified_regions.shape[0], 0)
        self._assert_region_plot_written(
            certification_result=result,
            stem="quadratic_regions",
        )
        self._assert_lyapunov_plot_written(
            lyap_model=certifier.lyap_model,
            certification_result=result,
            stem="quadratic_lyapunov",
        )

    def test_negative_quadratic_produces_counterexamples(self) -> None:
        certifier = self._make_certifier(_NegativeQuadraticLyapunov())
        result = certifier.certify(rho_estimate=1.0)

        self.assertFalse(result.global_success)
        self.assertFalse(result.partial_success)
        self.assertAlmostEqual(result.rho, 1.0, places=6)
        self.assertEqual(result.certified_regions.shape[0], 0)
        self.assertGreaterEqual(result.failed_regions.shape[0], 0)
        self.assertGreaterEqual(result.outside_sublevel_regions.shape[0], 0)
        self._assert_region_plot_written(
            certification_result=result,
            stem="negative_regions",
        )
        self._assert_lyapunov_plot_written(
            lyap_model=certifier.lyap_model,
            certification_result=result,
            stem="negative_lyapunov",
        )

    def test_mixed_lyapunov_has_safe_and_unsafe_regions(self) -> None:
        certifier = self._make_certifier(
            _QuadraticLyapunov(),
            dyn_model=_DirectionalScaleDynamics(base_scale=0.8, axis_gain=0.4),
            kappa=0.0,
            rho_min=0.9,
        )
        result = certifier.certify(rho_estimate=1.0)

        self.assertFalse(result.global_success)
        self.assertTrue(result.partial_success)
        self.assertGreater(result.certified_regions.shape[0], 0)
        self.assertGreater(result.failed_regions.shape[0], 0)

        certified_centers = result.certified_regions.mean(axis=1)
        failed_centers = result.failed_regions.mean(axis=1)
        self.assertTrue(np.any(certified_centers[:, 0] < 0.0))
        self.assertTrue(np.any(failed_centers[:, 0] > 0.0))
        self._assert_region_plot_written(
            certification_result=result,
            stem="mixed_regions",
        )
        self._assert_lyapunov_plot_written(
            lyap_model=certifier.lyap_model,
            certification_result=result,
            stem="mixed_lyapunov",
        )

    def test_negative_values_are_not_hidden_by_origin_guard(self) -> None:
        certification_verifier = LyapunovVerifier(
            policy_model=_ZeroPolicy(),
            lyap_model=_NegativeQuadraticLyapunov(),
            dyn_model=_IdentityDynamics(),
            lbx=th.tensor([[-2.0]], dtype=th.float32),
            ubx=th.tensor([[2.0]], dtype=th.float32),
            kappa=0.1,
            sublevel_tolerance=1e-6,
        )

        x = th.tensor([[1.0]], dtype=th.float32)
        rho = th.tensor([[2.0]], dtype=th.float32)

        self.assertGreater(float(certification_verifier(x, rho).item()), 0.0)

    def test_certification_verifier_enforces_hard_invariance(self) -> None:
        certification_verifier = LyapunovVerifier(
            policy_model=_ZeroPolicy(),
            lyap_model=_DescendingLinearLyapunov(),
            dyn_model=_ShiftDynamics(shift=2.0),
            lbx=th.tensor([[0.0]], dtype=th.float32),
            ubx=th.tensor([[2.0]], dtype=th.float32),
            kappa=0.0,
            sublevel_tolerance=1e-6,
        )

        x = th.tensor([[1.0]], dtype=th.float32)
        rho = th.tensor([[2.0]], dtype=th.float32)

        self.assertGreater(float(certification_verifier(x, rho).item()), 0.0)

    def test_certification_verifier_checks_levelset_boundary_conservatively(self) -> None:
        certification_verifier = LyapunovVerifier(
            policy_model=_ZeroPolicy(),
            lyap_model=_DescendingLinearLyapunov(),
            dyn_model=_ShiftDynamics(shift=3.0),
            lbx=th.tensor([[0.0]], dtype=th.float32),
            ubx=th.tensor([[2.0]], dtype=th.float32),
            kappa=0.0,
            sublevel_tolerance=1e-6,
        )

        x = th.tensor([[0.0]], dtype=th.float32)
        rho = th.tensor([[2.0]], dtype=th.float32)

        self.assertGreater(float(certification_verifier(x, rho).item()), 0.0)

    def test_abcrown_multi_output_verifier_returns_condition_v_and_xnext(self) -> None:
        verifier = LyapunovMultiOutputVerifier(
            policy_model=_ZeroPolicy(),
            lyap_model=_DescendingLinearLyapunov(),
            dyn_model=_ShiftDynamics(shift=2.0),
            lbx=th.tensor([[0.0]], dtype=th.float32),
            ubx=th.tensor([[2.0]], dtype=th.float32),
            kappa=0.0,
            sublevel_tolerance=1e-6,
        )

        outputs = verifier(
            th.tensor([[1.0]], dtype=th.float32),
            th.tensor([[2.0]], dtype=th.float32),
        )

        self.assertEqual(outputs.shape, (1, 3))
        self.assertAlmostEqual(float(outputs[0, 0]), 2.0, places=6)
        self.assertAlmostEqual(float(outputs[0, 1]), 1.0, places=6)
        self.assertAlmostEqual(float(outputs[0, 2]), 3.0, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
