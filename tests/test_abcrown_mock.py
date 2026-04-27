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

abcrown_wrapper = importlib.import_module("lcil.certification.abcrown_wrapper")
ABCrownCertifier = abcrown_wrapper.ABCrownCertifier
LyapunovCertificationConfig = importlib.import_module(
    "lcil.certification.config"
).LyapunovCertificationConfig
certifier_base_module = importlib.import_module("lcil.certification.certifier_base")
BaseCertifier = certifier_base_module.BaseCertifier
RecursiveCertificationResult = certifier_base_module.RecursiveCertificationResult
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


class _StaticUpperBoundFilter:
    def __init__(self, ub_out: th.Tensor):
        self.ub_out = ub_out

    def compute_bounds(self, x, method: str):
        del x, method
        return None, self.ub_out


class _RecursiveMockCertifier(BaseCertifier):
    def __init__(self, *args, width_limit: float, rho_limit: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.width_limit = float(width_limit)
        self.rho_limit = float(rho_limit)
        self.certify_calls: list[tuple[int, bool]] = []

    def setup_backend(self, *args, **kwargs) -> None:
        del args, kwargs

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


class TestABCrownCertifier(PlotAssertionsMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._patchers = [
            mock.patch.object(abcrown_wrapper, "ABCrownSolver", _FakeABCrownSolver),
            mock.patch.object(abcrown_wrapper, "VerificationSpec", _FakeVerificationSpec),
            mock.patch.object(abcrown_wrapper, "ConfigBuilder", _FakeConfigBuilder),
            mock.patch.object(abcrown_wrapper, "input_vars", _fake_input_vars),
            mock.patch.object(abcrown_wrapper, "output_vars", _fake_output_vars),
        ]
        for patcher in cls._patchers:
            patcher.start()

    @classmethod
    def tearDownClass(cls) -> None:
        for patcher in cls._patchers:
            patcher.stop()

        # Drop cached submodules that may have captured stubbed abcrown symbols.
        sys.modules.pop("lcil.certification.abcrown_wrapper", None)
        sys.modules.pop("lcil.certification.config", None)
        sys.modules.pop("lcil.certification.certifier_base", None)

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
    ) -> ABCrownCertifier:
        config = LyapunovCertificationConfig(
            state_dim=3,
            cert_bounds=np.array([[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]], dtype=np.float32),
            kappa=kappa,
            bins_per_dim=4,
            center_refinement_factor=0.7,
            origin_exclusion=0.0,
            max_scale_steps=6,
            max_bisection_steps=6,
            cert_method="alpha-crown",
            condition_tolerance=1e-6,
            max_recursion_depth=3,
        )
        return ABCrownCertifier(
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
        certifier = self._make_certifier(_MixedLyapunov(alpha=0.3, beta=2.0))
        result = certifier.certify(rho_estimate=1.0)

        self.assertFalse(result.global_success)
        self.assertTrue(result.partial_success)
        self.assertAlmostEqual(result.rho, 1.0, places=6)
        self.assertGreater(result.certified_regions.shape[0], 0)
        self.assertGreater(result.failed_regions.shape[0], 0)

        certified_centers = result.certified_regions.mean(axis=1)
        failed_centers = result.failed_regions.mean(axis=1)
        self.assertTrue(np.any(np.linalg.norm(certified_centers, axis=1) < 1.0))
        self.assertTrue(np.any(np.linalg.norm(failed_centers, axis=1) > 1.5))
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

    def test_abcrown_multi_output_spec_rejects_out_of_bounds_successor(self) -> None:
        config = LyapunovCertificationConfig(
            state_dim=1,
            cert_bounds=np.array([[0.0], [2.0]], dtype=np.float32),
            kappa=0.0,
            bins_per_dim=1,
            origin_exclusion=0.0,
            cert_method="alpha-crown",
            sublevel_tolerance=0.0,
            condition_tolerance=1e-6,
            use_ibp_filter=False,
            max_recursion_depth=0,
        )
        certifier = ABCrownCertifier(
            policy_model=_ZeroPolicy(),
            lyap_model=_DescendingLinearLyapunov(),
            dyn_model=_ShiftDynamics(shift=2.0),
            config=config,
            device=th.device("cpu"),
        )

        certifier.verifier = certifier._setup_verifier()
        certifier.regions = certifier._build_regions()
        certifier.setup_backend()

        is_safe = certifier._certify_batched_regions(certifier.regions, rho=2.0, early_exit=False)
        self.assertFalse(bool(is_safe.all().item()))

    def test_conservative_pre_verifier_only_sends_uncertain_regions_to_abcrown(self) -> None:
        certifier = self._make_certifier(_QuadraticLyapunov())
        certifier.setup_backend()

        bs = th.tensor(
            [
                [[-2.0], [-1.0]],
                [[-1.0], [0.0]],
                [[0.0], [1.0]],
            ],
            dtype=th.float32,
        )
        conservative_safe = th.tensor([True, False, True], dtype=th.bool)

        with mock.patch.object(
            certifier,
            "_certify_with_conservative_verifier",
            return_value=conservative_safe,
        ) as precheck_mock, mock.patch.object(
            certifier,
            "_solve_box_with_model",
            return_value=False,
        ) as solve_mock:
            is_safe = certifier._certify_batched_regions(bs, rho=1.0, early_exit=False)

        self.assertTrue(th.equal(is_safe, conservative_safe))
        precheck_mock.assert_called_once_with(bs, 1.0)
        solve_mock.assert_called_once()
        self.assertTrue(th.equal(solve_mock.call_args.kwargs["lb"], bs[1, 0]))
        self.assertTrue(th.equal(solve_mock.call_args.kwargs["ub"], bs[1, 1]))

    def test_setup_backend_uses_configured_batch_size(self) -> None:
        config = LyapunovCertificationConfig(
            state_dim=1,
            cert_bounds=np.array([[0.0], [2.0]], dtype=np.float32),
            kappa=0.0,
            bins_per_dim=1,
            origin_exclusion=0.0,
            cert_method="alpha-crown",
            condition_tolerance=1e-6,
            batch_size=123,
            max_recursion_depth=0,
        )
        certifier = ABCrownCertifier(
            policy_model=_ZeroPolicy(),
            lyap_model=_QuadraticLyapunov(),
            dyn_model=_ZeroDynamics(),
            config=config,
            device=th.device("cpu"),
        )

        certifier.setup_backend()

        self.assertEqual(certifier.abcrown_config["solver__batch_size"], 123)
        self.assertEqual(certifier.abcrown_config["general__complete_verifier"], "input_bab")
        self.assertTrue(certifier.abcrown_config["bab__branching__input_split__enable"])
        self.assertEqual(certifier.abcrown_config["bab__branching__input_split__split_partitions"], 2)

    def test_abcrown_build_regions_only_splits_origin_hole(self) -> None:
        config = LyapunovCertificationConfig(
            state_dim=2,
            cert_bounds=np.array([[-2.0, -3.0], [2.0, 3.0]], dtype=np.float32),
            kappa=0.0,
            bins_per_dim=8,
            center_refinement_factor=0.5,
            origin_exclusion=0.5,
            cert_method="alpha-crown",
            condition_tolerance=1e-6,
        )
        certifier = ABCrownCertifier(
            policy_model=_ZeroPolicy(),
            lyap_model=_QuadraticLyapunov(),
            dyn_model=_ZeroDynamics(),
            config=config,
            device=th.device("cpu"),
        )

        regions = certifier._build_regions().cpu().numpy()

        expected = np.array(
            [
                [[-2.0, -3.0], [-0.5, 3.0]],
                [[0.5, -3.0], [2.0, 3.0]],
                [[-0.5, -3.0], [0.5, -0.5]],
                [[-0.5, 0.5], [0.5, 3.0]],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(regions, expected)

    def test_abcrown_rho_search_checks_root_regions_once(self) -> None:
        certifier = self._make_certifier(_QuadraticLyapunov())
        certifier.regions = th.tensor(
            [
                [[-2.0, -2.0, -2.0], [0.0, 2.0, 2.0]],
                [[0.0, -2.0, -2.0], [2.0, 2.0, 2.0]],
            ],
            dtype=th.float32,
        )
        root_success = RecursiveCertificationResult(
            certified=certifier.regions[:1],
            failed=certifier.regions[:0],
            outside_sublevel=certifier.regions[1:],
        )

        with mock.patch.object(certifier, "_process_regions", return_value=root_success) as process_mock:
            self.assertTrue(certifier.is_rho_certified(rho=1.0))

        process_mock.assert_called_once_with(certifier.regions, 1.0, early_exit=True)

        root_failure = RecursiveCertificationResult(
            certified=certifier.regions[:1],
            failed=certifier.regions[1:],
            outside_sublevel=certifier.regions[:0],
        )
        with mock.patch.object(certifier, "_process_regions", return_value=root_failure) as process_mock:
            self.assertFalse(certifier.is_rho_certified(rho=1.0))

        process_mock.assert_called_once_with(certifier.regions, 1.0, early_exit=True)

    def test_ibp_filter_keeps_boundary_touching_boxes(self) -> None:
        config = LyapunovCertificationConfig(
            state_dim=1,
            cert_bounds=np.array([[0.0], [2.0]], dtype=np.float32),
            kappa=0.0,
            bins_per_dim=1,
            origin_exclusion=0.0,
            cert_method="alpha-crown",
            condition_tolerance=1e-3,
            use_ibp_filter=True,
        )
        certifier = ABCrownCertifier(
            policy_model=_ZeroPolicy(),
            lyap_model=_QuadraticLyapunov(),
            dyn_model=_ZeroDynamics(),
            config=config,
            device=th.device("cpu"),
        )
        bs = th.tensor([[[0.0], [2.0]]], dtype=th.float32)

        certifier.negative_filter = _StaticUpperBoundFilter(
            th.tensor([[-1.0]], dtype=th.float32)
        )
        kept_bs, filtered_bs = certifier._filter_sublevel_regions(bs, rho=1.0)
        self.assertEqual(len(kept_bs), 1)
        self.assertEqual(len(filtered_bs), 0)

        certifier.negative_filter = _StaticUpperBoundFilter(
            th.tensor([[-1.0005]], dtype=th.float32)
        )
        kept_bs, filtered_bs = certifier._filter_sublevel_regions(bs, rho=1.0)
        self.assertEqual(len(kept_bs), 0)
        self.assertEqual(len(filtered_bs), 1)

    def test_find_max_rho_uses_recursive_region_splitting(self) -> None:
        config = LyapunovCertificationConfig(
            state_dim=1,
            cert_bounds=np.array([[0.0], [2.0]], dtype=np.float32),
            kappa=0.0,
            rho_min=1e-6,
            bins_per_dim=1,
            origin_exclusion=0.0,
            rho_scaling=2.0,
            bisection_tol=1e-4,
            max_scale_steps=4,
            max_bisection_steps=8,
            cert_method="alpha-crown",
            condition_tolerance=1e-6,
            use_ibp_filter=False,
            max_recursion_depth=1,
        )
        certifier = _RecursiveMockCertifier(
            policy_model=_ZeroPolicy(),
            lyap_model=_QuadraticLyapunov(),
            dyn_model=_IdentityDynamics(),
            config=config,
            device=th.device("cpu"),
            width_limit=1.0,
            rho_limit=2.0,
        )

        best_rho = certifier.find_max_rho(rho_estimate=1.0)

        self.assertGreater(best_rho, 1.5)
        self.assertLessEqual(best_rho, 2.0)

    def test_is_rho_certified_uses_leaf_only_early_exit(self) -> None:
        config = LyapunovCertificationConfig(
            state_dim=1,
            cert_bounds=np.array([[0.0], [2.0]], dtype=np.float32),
            kappa=0.0,
            rho_min=1e-6,
            bins_per_dim=1,
            origin_exclusion=0.0,
            rho_scaling=2.0,
            bisection_tol=1e-4,
            max_scale_steps=4,
            max_bisection_steps=8,
            cert_method="alpha-crown",
            condition_tolerance=1e-6,
            use_ibp_filter=False,
            max_recursion_depth=1,
        )
        certifier = _RecursiveMockCertifier(
            policy_model=_ZeroPolicy(),
            lyap_model=_QuadraticLyapunov(),
            dyn_model=_IdentityDynamics(),
            config=config,
            device=th.device("cpu"),
            width_limit=1.0,
            rho_limit=2.0,
        )
        certifier.regions = certifier._build_regions()

        self.assertTrue(certifier.is_rho_certified(rho=1.0))
        self.assertEqual(certifier.certify_calls, [(1, False), (2, True)])

        certifier.certify_calls.clear()

        self.assertFalse(certifier.is_rho_certified(rho=3.0))
        self.assertEqual(certifier.certify_calls, [(1, False), (2, True)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
