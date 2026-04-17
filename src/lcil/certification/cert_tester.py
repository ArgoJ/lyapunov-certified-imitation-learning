import logging
from typing import Any

import numpy as np
import torch as th
import torch.nn as nn
from numpy.typing import NDArray

from .certifier_base import RegionCertificationResult
from .config import LyapunovCertificationConfig
from .models import ClosedLoopLyapunovConditionVerifier

__logger__ = logging.getLogger(__name__)


class CertificationResultTester:
    """
    Tests region certification results by performing closed-loop rollouts
    from the center of the certified, failed, and counter-example regions,
    and evaluating the empirical satisfaction of the Lyapunov condition.
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
        Initialize the tester.

        Parameters
        ----------
        policy_model : nn.Module
            Control policy network ``u = pi(x)``.
        lyap_model : nn.Module
            Candidate Lyapunov network ``V(x)``.
        dyn_model : nn.Module
            Dynamics model for one-step state propagation in closed loop.
        config : LyapunovCertificationConfig
            Configuration including bounds, tolerance, kappa, etc.
        device : th.device
            Device to run simulations on.
        """
        self.device = device
        self.config = config

        self.policy_model = policy_model.to(self.device).eval()
        self.lyap_model = lyap_model.to(self.device).eval()
        self.dyn_model = dyn_model.to(self.device).eval()

        lbx = th.tensor(config.cert_bounds[0], dtype=th.float32, device=self.device).unsqueeze(0)
        ubx = th.tensor(config.cert_bounds[1], dtype=th.float32, device=self.device).unsqueeze(0)

        self.verifier = ClosedLoopLyapunovConditionVerifier(
            policy_model=self.policy_model,
            lyap_model=self.lyap_model,
            dyn_model=self.dyn_model,
            lbx=lbx,
            ubx=ubx,
            invariance_weight=config.invariance_weight,
        ).to(self.device).eval()

    def test_result(
        self,
        cert_result: RegionCertificationResult,
        rollout_steps: int = 50,
    ) -> dict[str, dict[str, Any]]:
        """
        Run closed-loop rollouts from the center of all regions to test condition satisfaction.

        Parameters
        ----------
        cert_result : RegionCertificationResult
            The result containing certified, failed, and counterexample regions.
        rollout_steps : int, optional
            Number of closed-loop interaction steps to simulate per point. Defaults to 50.

        Returns
        -------
        dict[str, dict[str, Any]]
            A dictionary tracking violations for each category of regions:
            'certified', 'failed', 'counter_examples'.
        """
        results = {}
        regions_dict = {
            "certified": cert_result.certified_regions,
            "failed": cert_result.failed_regions,
            "counter_examples": cert_result.counter_examples,
        }

        rho_tensor = th.tensor(cert_result.rho, dtype=th.float32, device=self.device)
        kappa_tensor = th.tensor(self.config.kappa, dtype=th.float32, device=self.device)
        tolerance = self.config.condition_tolerance

        for name, regions in regions_dict.items():
            if regions is None or len(regions) == 0:
                results[name] = {
                    "violation_rate": 0.0,
                    "max_violation": 0.0,
                    "num_regions": 0,
                }
                continue

            # Compute the center point of the boxes
            # regions shape: (N, 2, state_dim) -> axis 1: 0=lower bound, 1=upper bound
            centers = 0.5 * (regions[:, 0, :] + regions[:, 1, :])
            x_t = th.tensor(centers, dtype=th.float32, device=self.device)

            violations_over_time = []

            with th.no_grad():
                for _ in range(rollout_steps):
                    # `condition_val <= 0` means mathematically proven safe (with tolerance relaxation)
                    condition_val = self.verifier(x_t, rho=rho_tensor, kappa=kappa_tensor).squeeze(-1)
                    
                    # Store conditions across batch
                    violations_over_time.append(condition_val.cpu().numpy())

                    # Forward step to next state
                    u_t = self.policy_model(x_t)
                    x_t = self.dyn_model(x_t, u_t)

            # shape: (rollout_steps, num_regions)
            violations_np = np.stack(violations_over_time, axis=0)

            # A state trajectory has a violation if at ANY step the condition > tolerance
            violation_mask = violations_np > tolerance
            traj_has_violation = violation_mask.any(axis=0)
            
            violation_rate = traj_has_violation.mean()
            # To compute max_violation, filter to just violations, baseline is 0
            max_viol = violations_np.max() if traj_has_violation.any() else 0.0

            results[name] = {
                "violation_rate": float(violation_rate),
                "max_violation": float(np.maximum(max_viol, 0.0)),
                "num_regions": len(regions),
                "violations_per_step": violations_np,
            }

            __logger__.info(
                "Tested %d %s regions: violation rate = %.2f%%, max violation = %.2e",
                len(regions),
                name,
                violation_rate * 100.0,
                max_viol,
            )

        return results

