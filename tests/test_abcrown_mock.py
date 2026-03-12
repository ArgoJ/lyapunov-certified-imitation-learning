import itertools
import importlib
import os
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import numpy as np
import torch as th
import torch.nn as nn

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
certifier_base = importlib.import_module("lcil.certification.certifier_base")
plot_module = importlib.import_module("lcil.utils.plot")
certified_regions_2d = plot_module.certified_regions_2d
lyapunov = plot_module.lyapunov


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
    tol: float


class _FakeOutputVar:
    def __lt__(self, tol: float) -> _FakeOutputConstraint:
        return _FakeOutputConstraint(tol=float(tol))


class _FakeOutputVars:
    def __init__(self, dim: int):
        self.dim = dim

    def __getitem__(self, idx: int) -> _FakeOutputVar:
        if idx < 0 or idx >= self.dim:
            raise IndexError(idx)
        return _FakeOutputVar()


@dataclass
class _FakeVerificationSpecData:
    lb: th.Tensor
    ub: th.Tensor
    tol: float


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
            tol=output_constraint.tol,
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
            values = self.computing_graph(points).reshape(-1)
        status = "verified" if float(values.max().item()) < self.spec.tol else "unsafe"
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


class _FakeProgressBar:
    def __init__(self, iterable):
        self.iterable = iterable

    def __enter__(self) -> "_FakeProgressBar":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, _payload: dict) -> None:
        return None


def _fake_tqdm(iterable, desc: str | None = None) -> _FakeProgressBar:
    del desc
    return _FakeProgressBar(iterable)


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


class TestABCrownCertifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._patchers = [
            mock.patch.object(abcrown_wrapper, "ABCrownSolver", _FakeABCrownSolver),
            mock.patch.object(abcrown_wrapper, "VerificationSpec", _FakeVerificationSpec),
            mock.patch.object(abcrown_wrapper, "ConfigBuilder", _FakeConfigBuilder),
            mock.patch.object(abcrown_wrapper, "input_vars", _fake_input_vars),
            mock.patch.object(abcrown_wrapper, "output_vars", _fake_output_vars),
            mock.patch.object(certifier_base.__logger__, "tqdm", _fake_tqdm),
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
    def _make_certifier(lyap_model: nn.Module) -> ABCrownCertifier:
        config = LyapunovCertificationConfig(
            state_dim=3,
            state_bounds=np.array([[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]], dtype=np.float32),
            kappa=0.1,
            invariance_weight=1.0,
            cert_bins_per_dim=4,
            origin_exclusion=0.0,
            max_scale_steps=6,
            max_bisection_steps=6,
            cert_method="alpha-crown",
            condition_tolerance=1e-6,
        )
        return ABCrownCertifier(
            policy_model=_ZeroPolicy(),
            lyap_model=lyap_model,
            dyn_model=_ZeroDynamics(),
            config=config,
            device=th.device("cpu"),
        )

    @staticmethod
    def _project_regions_to_2d(
        regions: np.ndarray,
        state_indices: tuple[int, int] = (0, 1),
    ) -> np.ndarray:
        return regions[:, :, list(state_indices)]

    def _assert_region_plot_written(
        self,
        certified_regions: np.ndarray,
        uncertified_regions: np.ndarray,
        stem: str,
    ) -> None:
        certified_regions_2d_view = self._project_regions_to_2d(certified_regions)
        uncertified_regions_2d_view = self._project_regions_to_2d(uncertified_regions)

        plot_dir = os.environ.get("LCIL_TEST_PLOT_DIR")
        if plot_dir:
            output_dir = Path(plot_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            html_path = output_dir / f"{stem}.html"
            certified_regions_2d(
                certified_regions=certified_regions_2d_view,
                uncertified_regions=uncertified_regions_2d_view,
                state_labels=["x0", "x1"],
                html_path=str(html_path),
            )
            self.assertTrue(html_path.exists())
            self.assertGreater(html_path.stat().st_size, 0)
            return

        with tempfile.TemporaryDirectory() as tmp_dir:
            html_path = Path(tmp_dir) / f"{stem}.html"
            certified_regions_2d(
                certified_regions=certified_regions_2d_view,
                uncertified_regions=uncertified_regions_2d_view,
                state_labels=["x0", "x1"],
                html_path=str(html_path),
            )
            self.assertTrue(html_path.exists())
            self.assertGreater(html_path.stat().st_size, 0)

    @staticmethod
    def _to_numpy_lyapunov(
        lyap_model: nn.Module,
        state_dim: int,
        plot_state_indices: tuple[int, int],
    ):
        def _lyapunov_func(x: np.ndarray) -> np.ndarray | float:
            x_array = np.asarray(x, dtype=np.float32)

            if x_array.shape[-1] == len(plot_state_indices):
                x_lifted = np.zeros((*x_array.shape[:-1], state_dim), dtype=np.float32)
                x_lifted[..., plot_state_indices[0]] = x_array[..., 0]
                x_lifted[..., plot_state_indices[1]] = x_array[..., 1]
            elif x_array.shape[-1] == state_dim:
                x_lifted = x_array
            else:
                raise ValueError(
                    f"Lyapunov input has invalid shape {x_array.shape}; expected last dim "
                    f"{len(plot_state_indices)} or {state_dim}."
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
        certified_regions: np.ndarray,
        uncertified_regions: np.ndarray,
        stem: str,
    ) -> None:
        plot_state_indices = (0, 1)
        lyap_func = self._to_numpy_lyapunov(
            lyap_model=lyap_model,
            state_dim=3,
            plot_state_indices=plot_state_indices,
        )
        certified_regions_2d_view = self._project_regions_to_2d(
            certified_regions,
            state_indices=plot_state_indices,
        )
        uncertified_regions_2d_view = self._project_regions_to_2d(
            uncertified_regions,
            state_indices=plot_state_indices,
        )

        plot_dir = os.environ.get("LCIL_TEST_PLOT_DIR")
        if plot_dir:
            output_dir = Path(plot_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            html_path = output_dir / f"{stem}.html"
            lyapunov(
                dataset=None,
                lyapunov_func=lyap_func,
                state_labels=["x0", "x1"],
                certified_regions=certified_regions_2d_view,
                uncertified_regions=uncertified_regions_2d_view,
                html_path=str(html_path),
            )
            self.assertTrue(html_path.exists())
            self.assertGreater(html_path.stat().st_size, 0)
            return

        with tempfile.TemporaryDirectory() as tmp_dir:
            html_path = Path(tmp_dir) / f"{stem}.html"
            lyapunov(
                dataset=None,
                lyapunov_func=lyap_func,
                state_labels=["x0", "x1"],
                certified_regions=certified_regions_2d_view,
                uncertified_regions=uncertified_regions_2d_view,
                html_path=str(html_path),
            )
            self.assertTrue(html_path.exists())
            self.assertGreater(html_path.stat().st_size, 0)

    def test_quadratic_lyapunov_certifies_all_regions(self) -> None:
        certifier = self._make_certifier(_QuadraticLyapunov())
        rho_certified, result = certifier.certify(rho_estimate=1.0)

        self.assertTrue(result.success)
        self.assertGreaterEqual(rho_certified, 1.0)
        self.assertEqual(result.failed_regions.shape[0], 0)
        self.assertGreater(result.certified_regions.shape[0], 0)
        self._assert_region_plot_written(
            certified_regions=result.certified_regions,
            uncertified_regions=result.failed_regions,
            stem="quadratic_regions",
        )
        self._assert_lyapunov_plot_written(
            lyap_model=certifier.lyap_model,
            certified_regions=result.certified_regions,
            uncertified_regions=result.failed_regions,
            stem="quadratic_lyapunov",
        )

    def test_negative_quadratic_produces_counterexamples(self) -> None:
        certifier = self._make_certifier(_NegativeQuadraticLyapunov())
        rho_certified, result = certifier.certify(rho_estimate=1.0)

        self.assertFalse(result.success)
        self.assertLessEqual(rho_certified, certifier.config.rho_min)
        self.assertEqual(result.certified_regions.shape[0], 0)
        self.assertGreater(result.failed_regions.shape[0], 0)
        self.assertGreater(result.counter_examples.shape[0], 0)
        self._assert_region_plot_written(
            certified_regions=result.certified_regions,
            uncertified_regions=result.failed_regions,
            stem="negative_regions",
        )
        self._assert_lyapunov_plot_written(
            lyap_model=certifier.lyap_model,
            certified_regions=result.certified_regions,
            uncertified_regions=result.failed_regions,
            stem="negative_lyapunov",
        )

    def test_mixed_lyapunov_has_safe_and_unsafe_regions(self) -> None:
        certifier = self._make_certifier(_MixedLyapunov(alpha=0.3, beta=2.0))
        rho_certified, result = certifier.certify(rho_estimate=1.0)

        self.assertFalse(result.success)
        self.assertLessEqual(rho_certified, certifier.config.rho_min)
        self.assertGreater(result.certified_regions.shape[0], 0)
        self.assertGreater(result.failed_regions.shape[0], 0)

        certified_centers = result.certified_regions.mean(axis=1)
        failed_centers = result.failed_regions.mean(axis=1)
        self.assertTrue(np.any(np.linalg.norm(certified_centers, axis=1) < 1.0))
        self.assertTrue(np.any(np.linalg.norm(failed_centers, axis=1) > 1.5))
        self._assert_region_plot_written(
            certified_regions=result.certified_regions,
            uncertified_regions=result.failed_regions,
            stem="mixed_regions",
        )
        self._assert_lyapunov_plot_written(
            lyap_model=certifier.lyap_model,
            certified_regions=result.certified_regions,
            uncertified_regions=result.failed_regions,
            stem="mixed_lyapunov",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
