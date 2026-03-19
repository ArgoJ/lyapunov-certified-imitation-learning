import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch as th
import torch.nn as nn
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete
from dataclasses import dataclass

plot_module = importlib.import_module("lcil.utils.plot")
base_models_module = importlib.import_module("lcil.utils.base_models")
certified_regions_2d = plot_module.certified_regions_2d
lyapunov = plot_module.lyapunov


@dataclass
class PendulumOnCartConfig:
    """
    Configuration class for the inverted pendulum on cart system.

    Parameters
    ----------
    m_cart : float, optional
        Mass of the cart, by default 1.0
    m_pole : float, optional
        Mass of the pendulum, by default 0.1
    length : float, optional
        Length of the pendulum, by default 0.5
    gravity : float, optional
        Gravitational acceleration, by default 9.81
    damping : float, optional
        Damping coefficient, by default 1.0
    dt : float
        Sampling time step for discretization, by default 0.01
    """
    m_cart: float = 1.0
    m_pole: float = 0.1
    length: float = 0.5
    gravity: float = 9.81
    damping: float = 1.0
    dt: float = 0.01


def _discrete_inverted_pendulum_on_cart_matrices(
        config: PendulumOnCartConfig
) -> tuple[np.ndarray, np.ndarray]:    
    """Discretize the linearized inverted pendulum on cart dynamics 
    around the upright (`s=1`) or down (`s=-1`) equilibrium 
    with optional damping and sign flip.

    P

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Discrete-time state transition matrix (Ad) and control input matrix (Bd)
    """
    m_c = config.m_cart
    m_p = config.m_pole
    l = config.length
    g = config.gravity
    d = config.damping
    ac = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -d / m_c, -(m_p * g) / m_c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, - d / (m_c * l), ((m_c + m_p) * g) / (m_c * l), 0.0],
        ],
        dtype=np.float64,
    )
    bc = np.array(
        [
            [0.0],
            [1.0 / m_c],
            [0.0],
            [1.0 / (m_c * l)],
        ],
        dtype=np.float64,
    )

    ad, bd, _, _, _ = cont2discrete(
        (ac, bc, np.eye(4, dtype=np.float64), np.zeros((4, 1), dtype=np.float64)),
        config.dt,
    )
    return ad, bd


def _riccati_gain_and_value_matrix(
    sys_cfg: PendulumOnCartConfig,
    q: np.ndarray | None = None,
    r: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the discrete-time LQR gain matrix K and value matrix P 
    for the linearized inverted pendulum on cart dynamics.

    Parameters
    ----------
    sys_cfg : PendulumOnCartConfig
        Configuration for the inverted pendulum on cart system
    q : np.ndarray | None, optional
        State cost matrix, by default None
    r : np.ndarray | None, optional
        Control cost matrix, by default None

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        LQR gain matrix K and value matrix P for the discretized inverted pendulum on cart
    """
    ad, bd = _discrete_inverted_pendulum_on_cart_matrices(sys_cfg)
    if q is None:
        q = np.diag([5.0, 1.0, 20.0, 50.0]).astype(np.float64)
    if r is None:
        r = np.array([[2.0]], dtype=np.float64)

    p = solve_discrete_are(ad, bd, q, r)
    k = np.linalg.solve(r + bd.T @ p @ bd, bd.T @ p @ ad)
    return k, p


class _RiccatiPolicy(nn.Module):
    def __init__(self, k_gain: np.ndarray):
        super().__init__()
        self.register_buffer("k_gain", th.as_tensor(k_gain, dtype=th.float32))

    def forward(self, x: th.Tensor) -> th.Tensor:
        return -(x @ self.k_gain.transpose(0, 1))


class _NonlinearInvertedPendulumOnCartDynamics(nn.Module):
    def __init__(self, cfg: PendulumOnCartConfig):
        super().__init__()
        self.cfg = cfg

    def _continuous_dynamics(self, x: th.Tensor, u: th.Tensor) -> th.Tensor:
        p = x[:, 0]
        p_dot = x[:, 1]
        theta = x[:, 2]
        theta_dot = x[:, 3]
        force = u[:, 0]

        m_c = self.cfg.m_cart
        m_p = self.cfg.m_pole
        l = self.cfg.length
        g = self.cfg.gravity
        d = self.cfg.damping

        sin_theta = th.sin(theta)
        cos_theta = th.cos(theta)
        total_mass = m_c + m_p

        effective_force = force - d * p_dot

        denom = total_mass - m_p * cos_theta * cos_theta # always positive for m_c > 0 and m_p > 0

        p_ddot = (
            effective_force
            + m_p * l * theta_dot.pow(2) * sin_theta
            - m_p * g * sin_theta * cos_theta
        ) / denom

        theta_ddot = (
            effective_force * cos_theta
            + m_p * l * theta_dot.pow(2) * sin_theta * cos_theta
            + total_mass * g * sin_theta
        ) / (l * denom)

        return th.stack([p_dot, p_ddot, theta_dot, theta_ddot], dim=1)

    def forward(self, x: th.Tensor, u: th.Tensor) -> th.Tensor:
        x_dot = self._continuous_dynamics(x, u)
        return x + self.cfg.dt * x_dot


class _RiccatiQuadraticLyapunov(nn.Module):
    def __init__(self, p_matrix: np.ndarray):
        super().__init__()
        p_sym = 0.5 * (p_matrix + p_matrix.T)
        self.register_buffer("p_matrix", th.as_tensor(p_sym, dtype=th.float32))

    def forward(self, x: th.Tensor) -> th.Tensor:
        return ((x @ self.p_matrix) * x).sum(dim=1, keepdim=True)


class TestABCrownInvertedPendulumOnCartIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if os.environ.get("LCIL_RUN_ABCROWN_INTEGRATION", "0") != "1":
            raise unittest.SkipTest(
                "Set LCIL_RUN_ABCROWN_INTEGRATION=1 to run real ABCrown integration tests."
            )

        try:
            abcrown = importlib.import_module("abcrown")

            is_stubbed_module = (
                getattr(abcrown, "__file__", None) is None
                or not hasattr(getattr(abcrown, "ConfigBuilder", object), "from_defaults")
            )
            if is_stubbed_module:
                sys.modules.pop("abcrown", None)
                abcrown = importlib.import_module("abcrown")
        except Exception as exc:  # pragma: no cover - depends on local environment
            raise unittest.SkipTest(f"abcrown is not importable: {exc}") from exc

        required_symbols = [
            "ABCrownSolver",
            "VerificationSpec",
            "ConfigBuilder",
            "input_vars",
            "output_vars",
        ]
        missing = [name for name in required_symbols if not hasattr(abcrown, name)]
        if missing:
            raise unittest.SkipTest(
                f"abcrown is missing required symbols for wrapper integration: {missing}"
            )

        try:
            # Unit tests may import this module with stubbed abcrown symbols first.
            # Ensure we bind to the real abcrown package for integration testing.
            sys.modules.pop("lcil.certification.abcrown_wrapper", None)
            abcrown_wrapper = importlib.import_module("lcil.certification.abcrown_wrapper")
            cls.ABCrownCertifier = abcrown_wrapper.ABCrownCertifier
            cls.LyapunovCertificationConfig = importlib.import_module(
                "lcil.certification.config"
            ).LyapunovCertificationConfig
            cls.certifier_base = importlib.import_module("lcil.certification.certifier_base")
        except Exception as exc:  # pragma: no cover - depends on local environment
            raise unittest.SkipTest(f"Could not import certifier modules: {exc}") from exc

    @classmethod
    def _make_certifier(cls):
        sys_cfg = PendulumOnCartConfig()
        k_gain, p_value = _riccati_gain_and_value_matrix(sys_cfg)
        lyap_model = _RiccatiQuadraticLyapunov(p_value)
        theta_bound = np.pi * 0.999 # TODO: maybe increas to 2/3 pi to test more challenging region shapes?
        config = cls.LyapunovCertificationConfig(
            state_dim=4,
            state_bounds=np.array(
                [
                    [-5.0, -1.0, -theta_bound, -1.0],
                    [ 5.0,  1.0,  theta_bound,  1.0]
                ],
                dtype=np.float32,
            ),
            kappa=0.001,
            invariance_weight=1.0,
            rho_scaling=1.1,
            cert_bins_per_dim=(3, 4, 6, 7),
            cert_center_refinement_factor=(0.6, 0.6, 0.5, 0.6),
            origin_exclusion=0.01,
            max_scale_steps=20,
            max_bisection_steps=10,
            cert_method="alpha-crown",
            condition_tolerance=1e-5,
        )
        return cls.ABCrownCertifier(
            policy_model=_RiccatiPolicy(k_gain),
            lyap_model=lyap_model,
            dyn_model=_NonlinearInvertedPendulumOnCartDynamics(sys_cfg),
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
        plot_state_indices = (2, 3)
        state_labels = ["theta", "theta_dot"]

        certified_regions_2d_view = self._project_regions_to_2d(
            certified_regions,
            state_indices=plot_state_indices,
        )
        uncertified_regions_2d_view = self._project_regions_to_2d(
            uncertified_regions,
            state_indices=plot_state_indices,
        )

        self._assert_plot_written(
            plot_fn=certified_regions_2d,
            stem=stem,
            plot_kwargs={
                "certified_regions": certified_regions_2d_view,
                "uncertified_regions": uncertified_regions_2d_view,
                "state_labels": state_labels,
                "state_indices": plot_state_indices
            },
        )

    def _assert_plot_written(self, plot_fn, stem: str, plot_kwargs: dict) -> None:
        plot_dir = os.environ.get("LCIL_TEST_PLOT_DIR")
        if plot_dir:
            output_dir = Path(plot_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            html_path = output_dir / f"{stem}.html"
            plot_fn(**plot_kwargs, html_path=str(html_path))
            self.assertTrue(html_path.exists())
            self.assertGreater(html_path.stat().st_size, 0)
            return

        with tempfile.TemporaryDirectory() as tmp_dir:
            html_path = Path(tmp_dir) / f"{stem}.html"
            plot_fn(**plot_kwargs, html_path=str(html_path))
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
        plot_state_indices = (2, 3)
        state_labels = ["theta", "theta_dot"]

        lyap_func = self._to_numpy_lyapunov(
            lyap_model=lyap_model,
            state_dim=4,
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

        self._assert_plot_written(
            plot_fn=lyapunov,
            stem=stem,
            plot_kwargs={
                "dataset": None,
                "lyapunov_func": lyap_func,
                "state_indices": list(plot_state_indices),
                "state_labels": state_labels,
                "certified_regions": certified_regions_2d_view,
                "uncertified_regions": uncertified_regions_2d_view,
            },
        )

    def test_inverted_pendulum_on_cart_lqr(self) -> None:
        certifier = self._make_certifier()
        rho_certified, result = certifier.certify(rho_estimate=3.7)

        self.assertIsInstance(float(rho_certified), float)
        self.assertGreaterEqual(rho_certified, certifier.config.rho_min)
        self._assert_region_plot_written(
            certified_regions=result.certified_regions,
            uncertified_regions=result.failed_regions,
            stem="inverted_pendulum_on_cart_riccati_regions_integration",
        )
        self._assert_lyapunov_plot_written(
            lyap_model=certifier.lyap_model,
            certified_regions=result.certified_regions,
            uncertified_regions=result.failed_regions,
            stem="inverted_pendulum_on_cart_riccati_lyapunov_integration",
        )
        self.assertGreater(result.failed_regions.shape[0], 0)
        self.assertGreater(result.certified_regions.shape[0], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
