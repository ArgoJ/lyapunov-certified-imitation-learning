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
from ..utils.constants import *

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

    DEFAULT_FILE_NAME = CERTIFICATION_TESTER_RESULTS_FILENAME

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

    def _inside_origin_exclusion_mask(self, states: th.Tensor) -> th.Tensor:
        """Return a boolean mask for states inside the origin-exclusion box."""
        exclusion = th.as_tensor(self.config.origin_exclusion, dtype=states.dtype, device=self.device)
        if exclusion.ndim == 0:
            exclusion = exclusion.repeat(states.shape[-1])
        else:
            exclusion = exclusion.reshape(-1)

        if exclusion.numel() != states.shape[-1]:
            raise ValueError(
                f"origin_exclusion dimension {exclusion.numel()} does not match state dimension {states.shape[-1]}."
            )

        active_dims = exclusion > 0
        if not bool(active_dims.any().item()):
            return th.zeros(states.shape[:-1], dtype=th.bool, device=states.device)

        exclusion = exclusion.reshape(*([1] * (states.ndim - 1)), exclusion.numel())
        active_dims = active_dims.reshape(*([1] * (states.ndim - 1)), active_dims.numel())
        return ((states.abs() <= exclusion) | (~active_dims)).all(dim=-1)

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
                inside_origin_exclusion = self._inside_origin_exclusion_mask(candidates)
                valid_mask = inside_mask & (~inside_origin_exclusion)

                if not bool(valid_mask.any()):
                    continue

                inside_states = candidates[valid_mask]
                inside_values = candidate_values[valid_mask]

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

        if kept_states.shape[0] < sample_size:
            __logger__.warning(
                "Requested %d rho-boundary samples but found only %d inside the sublevel set.",
                sample_size,
                kept_states.shape[0],
            )

        return kept_states, kept_values

    def _rollout_from_states(self, initial_states: th.Tensor, rollout_steps: int) -> tuple[th.Tensor, th.Tensor]:
        """Roll out from initial states for a given number of steps.

        Parameters
        ----------
        initial_states : th.Tensor
            The initial states to start the rollout from.
        rollout_steps : int
            The number of steps to roll out.

        Returns
        -------
        tuple[th.Tensor, th.Tensor]
            A tuple containing the rollout states and the violations.

        Raises
        ------
        ValueError
            If rollout_steps is not positive.
        """
        if rollout_steps <= 0:
            raise ValueError(f"rollout_steps must be positive, got {rollout_steps}.")

        x_t = initial_states
        states = th.empty((x_t.shape[0], rollout_steps + 1, x_t.shape[1]), dtype=th.float32, device=self.device)
        states[:, 0, :] = initial_states
        violations = th.empty((x_t.shape[0], rollout_steps), dtype=th.float32, device=self.device)
        with th.no_grad():
            for step_idx in range(rollout_steps):
                verifier_output = self.verifier(x_t)
                hard_violation = self._hard_condition_violation(verifier_output)
                violations[:, step_idx] = hard_violation.squeeze(-1)
                x_t = verifier_output[:, 2:]
                states[:, step_idx + 1, :] = x_t

        return states, violations

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

        rollout_states, violations = self._rollout_from_states(x_t, rollout_steps=rollout_steps)

        inside_origin_exclusion = self._inside_origin_exclusion_mask(rollout_states)
        ignored_rollout_state_mask = inside_origin_exclusion
        ignored_violation_mask = inside_origin_exclusion[:, :-1]

        isinside = rollout_states[ignored_rollout_state_mask]
        if isinside.numel() > 0:
            largest_inside_idx = isinside.abs().amax(dim=1).argmax()
            largest_inside_state = isinside[largest_inside_idx]
            __logger__.info(
                "Largest rollout state inside origin exclusion: state=%s, max_abs=%.6e, origin_exclusion=%s",
                largest_inside_state.detach().cpu().tolist(),
                float(largest_inside_state.abs().max().item()),
                self.config.origin_exclusion,
            )
        else:
            __logger__.info(
                "No rollout states entered the origin exclusion region (origin_exclusion=%s).",
                self.config.origin_exclusion,
            )

        rollout_states = rollout_states.clone()
        violations = violations.clone()
        rollout_states[ignored_rollout_state_mask] = th.nan
        violations[ignored_violation_mask] = th.nan

        violation_mask = violations > tolerance
        valid_traj_mask = (~ignored_violation_mask).any(dim=1)
        traj_has_violation = violation_mask.any(dim=1) & valid_traj_mask

        num_valid_trajectories = int(valid_traj_mask.sum().item())
        violation_rate = (
            float(traj_has_violation.sum().item() / num_valid_trajectories)
            if num_valid_trajectories > 0
            else 0.0
        )
        max_viol = float(th.nan_to_num(violations, nan=0.0).max().item()) if num_valid_trajectories > 0 else 0.0

        __logger__.info(
            "Tested %d rho-boundary samples: violation rate = %.2f%%, max violation = %.2e",
            x_t.shape[0],
            violation_rate * 100.0,
            max_viol,
        )

        return CertificationCategoryTestResult(
            violation_rate=violation_rate,
            max_violation=float(np.maximum(max_viol, 0.0)),
            num_samples=int(x_t.shape[0]),
            sampled_states=x_t.cpu().numpy(),
            sampled_values=sampled_values.cpu().numpy(),
            violations_per_step=violations.cpu().numpy(),
            rollout_states=rollout_states.cpu().numpy(),
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
