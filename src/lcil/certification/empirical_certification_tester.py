from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch as th
import torch.nn as nn
from numpy.typing import NDArray

from .config import LyapunovCertificationConfig
from .models import LyapunovCoreVerifier
from ..utils.base_config import JsonDataclass

__logger__ = logging.getLogger(__name__)


@dataclass(frozen=True)
class CertificationCategoryTestResult(JsonDataclass):
    """Empirical rollout summary for sampled states near the rho boundary."""

    NP_ARRAY_FIELDS = (
        "sampled_states",
        "sampled_values",
        "violations_per_step",
        "rollout_states",
    )

    violation_rate: float
    max_violation: float
    num_samples: int
    sampled_states: NDArray | None = None
    sampled_values: NDArray | None = None
    violations_per_step: NDArray | None = None
    rollout_states: NDArray | None = None


@dataclass(frozen=True)
class CertificationTesterResult(JsonDataclass):
    """Container for empirical rho-boundary rollout checks."""

    DEFAULT_FILE_NAME = "CertificationTesterResult.json"

    rho_boundary: CertificationCategoryTestResult
    sample_size: int
    rollout_steps: int
    rho: float
    kappa: float
    tolerance: float


class CertificationResultTester:
    """Empirically audit the closed loop around the sampled rho boundary.

    The certifier resolves boxes either because they are outside ``V(x) <= rho``
    or because the Lyapunov core conditions hold inside the sublevel set. This
    tester therefore ignores certification boxes entirely and instead samples
    states directly from the certification bounds, keeps the ones with
    ``V(x) <= rho`` that are closest to ``rho`` from below, and evaluates their
    rollout violations with the shared verifier.
    """

    _CANDIDATE_MULTIPLIER = 64
    _MAX_SAMPLING_ROUNDS = 8

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
        """Return the maximum raw Lyapunov-core deficit per sample."""
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

    def _sample_uniform_states(self, sample_size: int) -> th.Tensor:
        if sample_size <= 0:
            raise ValueError(f"sample_size must be positive, got {sample_size}.")

        lower_bounds = self.bounds[0].unsqueeze(0)
        upper_bounds = self.bounds[1].unsqueeze(0)
        return lower_bounds + th.rand(
            sample_size,
            self.config.state_dim,
            device=self.device,
        ) * (upper_bounds - lower_bounds)

    def _sample_near_rho_within_sublevel(
        self,
        rho: float,
        sample_size: int,
    ) -> tuple[th.Tensor, th.Tensor]:
        if rho < 0.0:
            raise ValueError(f"rho must be non-negative, got {rho}.")
        if sample_size <= 0:
            raise ValueError(f"sample_size must be positive, got {sample_size}.")

        kept_states = th.empty((0, self.config.state_dim), dtype=th.float32, device=self.device)
        kept_values = th.empty((0,), dtype=th.float32, device=self.device)
        candidate_batch_size = max(sample_size * self._CANDIDATE_MULTIPLIER, sample_size)
        rho_scalar = float(rho)

        with th.no_grad():
            for _ in range(self._MAX_SAMPLING_ROUNDS):
                candidates = self._sample_uniform_states(candidate_batch_size)
                candidate_values = self.lyap_model(candidates).reshape(-1)
                inside_mask = candidate_values <= rho_scalar

                if not bool(inside_mask.any()):
                    continue

                inside_states = candidates[inside_mask]
                inside_values = candidate_values[inside_mask]

                combined_states = th.cat((kept_states, inside_states), dim=0)
                combined_values = th.cat((kept_values, inside_values), dim=0)
                keep_count = min(sample_size, combined_values.shape[0])
                keep_idx = th.topk(combined_values, k=keep_count, largest=True).indices
                kept_states = combined_states[keep_idx]
                kept_values = combined_values[keep_idx]

                if kept_states.shape[0] >= sample_size:
                    break

        if kept_states.shape[0] == 0:
            __logger__.warning(
                "Could not find any sampled states inside V(x) <= rho=%.6f within the certification bounds.",
                rho_scalar,
            )
            return kept_states, kept_values

        rho_gap = rho_scalar - kept_values
        __logger__.info(
            "Sampled %d rho-boundary states inside V(x) <= %.6f; rho gap stats min=%.3e, mean=%.3e, max=%.3e.",
            kept_states.shape[0],
            rho_scalar,
            float(rho_gap.min().item()),
            float(rho_gap.mean().item()),
            float(rho_gap.max().item()),
        )
        if kept_states.shape[0] < sample_size:
            __logger__.warning(
                "Requested %d rho-boundary samples but found only %d inside the sublevel set.",
                sample_size,
                kept_states.shape[0],
            )

        return kept_states, kept_values

    def _evaluate_rho(
        self,
        rho: float,
        *,
        sample_size: int,
        tolerance: float,
        rollout_steps: int,
    ) -> CertificationCategoryTestResult:
        """Roll out from sampled states near the rho-sublevel boundary."""
        if rollout_steps <= 0:
            raise ValueError(f"rollout_steps must be positive, got {rollout_steps}.")
        if sample_size <= 0:
            raise ValueError(f"sample_size must be positive, got {sample_size}.")

        x_t, sampled_values = self._sample_near_rho_within_sublevel(
            rho=float(rho),
            sample_size=sample_size,
        )
        if x_t.shape[0] == 0:
            return CertificationCategoryTestResult(
                violation_rate=0.0,
                max_violation=0.0,
                num_samples=0,
                sampled_states=None,
                sampled_values=None,
                violations_per_step=None,
                rollout_states=None,
            )

        sampled_states_np = x_t.detach().cpu().numpy()
        sampled_values_np = sampled_values.detach().cpu().numpy()
        violations_over_time = []
        rollout_states = [sampled_states_np]
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
            "Tested %d rho-boundary samples: violation rate = %.2f%%, max violation = %.2e",
            sampled_states_np.shape[0],
            violation_rate * 100.0,
            max_viol,
        )

        return CertificationCategoryTestResult(
            violation_rate=violation_rate,
            max_violation=float(np.maximum(max_viol, 0.0)),
            num_samples=int(sampled_states_np.shape[0]),
            sampled_states=sampled_states_np,
            sampled_values=sampled_values_np,
            violations_per_step=violations_np,
            rollout_states=rollout_states_np,
        )

    def test_result(
        self,
        rho: float,
        sample_size: int = 256,
        rollout_steps: int = 50,
    ) -> CertificationTesterResult:
        """Run empirical rollouts from states near the rho-sublevel boundary."""
        if rollout_steps <= 0:
            raise ValueError(f"rollout_steps must be positive, got {rollout_steps}.")
        if sample_size <= 0:
            raise ValueError(f"sample_size must be positive, got {sample_size}.")

        tolerance = self.config.condition_tolerance
        rho_boundary = self._evaluate_rho(
            rho=float(rho),
            sample_size=sample_size,
            tolerance=tolerance,
            rollout_steps=rollout_steps,
        )

        return CertificationTesterResult(
            rho_boundary=rho_boundary,
            sample_size=int(sample_size),
            rollout_steps=rollout_steps,
            rho=float(rho),
            kappa=float(self.config.kappa),
            tolerance=float(tolerance),
        )