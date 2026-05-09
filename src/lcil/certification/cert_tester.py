import logging
from dataclasses import dataclass

import numpy as np
import torch as th
import torch.nn as nn
from numpy.typing import NDArray

from .bisect_certifier import RegionCertificationResult
from .config import LyapunovCertificationConfig
from .models import LyapunovCoreVerifier
from ..utils.base_config import JsonDataclass

__logger__ = logging.getLogger(__name__)


@dataclass(frozen=True)
class CertificationCategoryTestResult(JsonDataclass):
    """Empirical center-rollout summary for one certification region category.

    These numbers summarize hard Lyapunov-condition violations along rollouts
    started from region centers. They do not constitute a formal proof over the
    full boxes in the category.
    """

    NP_ARRAY_FIELDS = ("violations_per_step", "rollout_states")

    violation_rate: float
    max_violation: float
    num_regions: int
    violations_per_step: NDArray | None = None
    rollout_states: NDArray | None = None


@dataclass(frozen=True)
class CertificationTesterResult(JsonDataclass):
    """Container for empirical center-rollout checks across result categories.

    This is a post-hoc diagnostic view of a certification result. It complements
    the formal box-wise certification output, but does not replace it.
    """

    DEFAULT_FILE_NAME = "CertificationTesterResult.json"

    certified: CertificationCategoryTestResult
    failed: CertificationCategoryTestResult
    outside_sublevel: CertificationCategoryTestResult
    rollout_steps: int
    rho: float
    kappa: float
    tolerance: float


class CertificationResultTester:
    """
    Empirically audit a certification result via center-point rollouts.

    For each certified, failed, and outside-sublevel box, the tester starts a
    closed-loop rollout from the box center and measures hard Lyapunov-condition
    violations along the resulting trajectory. This is a representative rollout
    check, not a formal guarantee over the full box.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        lyap_model: nn.Module,
        dyn_model: nn.Module,
        config: LyapunovCertificationConfig,
        device: th.device = th.device("cpu"),
    ):
        """
        Initialize the empirical center-rollout tester.

        Parameters
        ----------
        policy_model : nn.Module
            Control policy network ``u = pi(x)``.
        lyap_model : nn.Module
            Candidate Lyapunov network ``V(x)``.
        dyn_model : nn.Module
            Dynamics model for one-step state propagation in closed loop.
        config : LyapunovCertificationConfig
            Configuration providing rollout bounds and Lyapunov-condition
            tolerances used for the empirical audit.
        device : th.device
            Device to run simulations on.
        """
        self.device = device
        self.config = config

        self.policy_model = policy_model.to(self.device).eval()
        self.lyap_model = lyap_model.to(self.device).eval()
        self.dyn_model = dyn_model.to(self.device).eval()
        self.bounds = th.as_tensor(self.config.cert_bounds, dtype=th.float32, device=self.device)

        self.verifier = LyapunovCoreVerifier(
            policy_model=self.policy_model,
            lyap_model=self.lyap_model,
            dyn_model=self.dyn_model,
            kappa=config.kappa,
            condition_margin=config.condition_margin,
        ).to(self.device).eval()

    def _hard_condition_violation(self, verifier_output: th.Tensor) -> th.Tensor:
        """Return the maximum raw Lyapunov-core deficit per sample.

        The verifier output packs the decrease margin, current Lyapunov value,
        and successor state. This helper converts that fused output into a
        non-negative scalar violation magnitude matching the core certification
        conditions: decrease, positivity, and state invariance.
        """
        decrease_margin = verifier_output[:, :1]
        lyap_value = verifier_output[:, 1:2]
        x_next = verifier_output[:, 2:]

        lower_bounds = self.bounds[0].unsqueeze(0)
        upper_bounds = self.bounds[1].unsqueeze(0)
        violation_terms = th.cat(
            (
                th.clamp_min(-decrease_margin, 0.0),
                th.clamp_min(-lyap_value, 0.0),
                th.clamp_min(lower_bounds - x_next, 0.0),
                th.clamp_min(x_next - upper_bounds, 0.0),
            ),
            dim=1,
        )
        return violation_terms.max(dim=1, keepdim=True).values

    def _evaluate_regions(
        self,
        regions: NDArray | None,
        *,
        name: str,
        tolerance: float,
        rollout_steps: int,
    ) -> CertificationCategoryTestResult:
        """Roll out from each region center and record hard-condition violations."""
        if rollout_steps <= 0:
            raise ValueError(f"rollout_steps must be positive, got {rollout_steps}.")

        if regions is None or len(regions) == 0:
            return CertificationCategoryTestResult(
                violation_rate=0.0,
                max_violation=0.0,
                num_regions=0,
                violations_per_step=None,
                rollout_states=None,
            )

        # One representative state per region: the geometric center of the box.
        centers = 0.5 * (regions[:, 0, :] + regions[:, 1, :])
        x_t = th.as_tensor(centers, dtype=th.float32, device=self.device)

        violations_over_time = []
        rollout_states = [x_t.cpu().numpy()]
        with th.no_grad():
            for _ in range(rollout_steps):
                verifier_output = self.verifier(x_t)
                hard_violation = self._hard_condition_violation(verifier_output)
                violations_over_time.append(hard_violation.squeeze(-1).cpu().numpy())
                x_t = verifier_output[:, 2:]
                rollout_states.append(x_t.cpu().numpy())

        violations_np = np.stack(violations_over_time, axis=0)
        rollout_states_np = np.stack(rollout_states, axis=1)
        violation_mask = violations_np > tolerance
        traj_has_violation = violation_mask.any(axis=0)

        violation_rate = float(traj_has_violation.mean())
        max_viol = float(violations_np.max()) if bool(traj_has_violation.any()) else 0.0

        __logger__.info(
            "Tested %d %s regions: violation rate = %.2f%%, max violation = %.2e",
            len(regions),
            name,
            violation_rate * 100.0,
            max_viol,
        )

        return CertificationCategoryTestResult(
            violation_rate=violation_rate,
            max_violation=float(np.maximum(max_viol, 0.0)),
            num_regions=len(regions),
            violations_per_step=violations_np,
            rollout_states=rollout_states_np,
        )

    def test_result(
        self,
        cert_result: RegionCertificationResult,
        rollout_steps: int = 50,
    ) -> CertificationTesterResult:
        """
        Run empirical center-point rollouts for each region category.

        Parameters
        ----------
        cert_result : RegionCertificationResult
            Formal certification result containing certified, failed, and
            outside-sublevel boxes.
        rollout_steps : int, optional
            Number of closed-loop steps per center rollout. Defaults to 50.

        Returns
        -------
        CertificationTesterResult
            Structured empirical diagnostics for the 'certified', 'failed', and
            'outside_sublevel' categories. These diagnostics summarize rollout
            behavior at box centers and are not formal certification statements.
        """
        if rollout_steps <= 0:
            raise ValueError(f"rollout_steps must be positive, got {rollout_steps}.")

        tolerance = self.config.condition_tolerance

        certified = self._evaluate_regions(
            cert_result.certified_sublevel_regions,
            name="certified",
            tolerance=tolerance,
            rollout_steps=rollout_steps,
        )
        failed = self._evaluate_regions(
            cert_result.uncertified_regions,
            name="failed",
            tolerance=tolerance,
            rollout_steps=rollout_steps,
        )
        outside_sublevel = self._evaluate_regions(
            cert_result.outside_sublevel_regions,
            name="outside_sublevel",
            tolerance=tolerance,
            rollout_steps=rollout_steps,
        )

        return CertificationTesterResult(
            certified=certified,
            failed=failed,
            outside_sublevel=outside_sublevel,
            rollout_steps=rollout_steps,
            rho=float(cert_result.rho),
            kappa=float(self.config.kappa),
            tolerance=float(tolerance),
        )

