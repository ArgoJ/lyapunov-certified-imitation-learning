from __future__ import annotations

import logging

import torch as th
import torch.nn as nn

from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from typing import Any
from contextlib import nullcontext
from dataclasses import dataclass
from pkg_logger import suppress_native_output

from .models import LyapunovMultiOutputVerifier
from .config import LyapunovCertificationConfig

__logger__ = logging.getLogger(__name__)


@dataclass(frozen=True)
class ABCrownRegionVerification:
    """Status-aware verification result for a single packed region."""

    status: str
    verified: bool
    counterexample_found: bool


@dataclass(frozen=True)
class ABCrownRegionBatchVerification:
    """Status masks for a batched region certification pass."""

    verified_mask: th.Tensor
    counterexample_mask: th.Tensor
    unknown_mask: th.Tensor

    @property
    def failed_mask(self) -> th.Tensor:
        return self.counterexample_mask | self.unknown_mask

    @property
    def processed_mask(self) -> th.Tensor:
        return self.verified_mask | self.counterexample_mask | self.unknown_mask

    @property
    def any_counterexample(self) -> bool:
        return bool(self.counterexample_mask.any().item())

    @classmethod
    def empty(cls, length: int) -> ABCrownRegionBatchVerification:
        return cls(
            verified_mask=th.zeros(length, dtype=th.bool),
            counterexample_mask=th.zeros(length, dtype=th.bool),
            unknown_mask=th.zeros(length, dtype=th.bool),
        )


@dataclass(frozen=True)
class _ABCrownAPI:
    solver_cls: Any
    verification_spec_cls: Any
    config_builder_cls: Any
    input_vars_fn: Any
    output_vars_fn: Any


class _ABCrownModelWrapper(nn.Module):
    """Freeze rho so ABCrown sees a clean map ``x -> y``."""

    def __init__(self, verifier: nn.Module, device: th.device):
        super().__init__()
        self.verifier = verifier
        self.register_buffer("rho", th.tensor(0.0, dtype=th.float32, device=device))

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.verifier(x, self.rho)


class ABCrownRegionCertifier:
    """Shared ABCrown backend for certifying packed regions one by one."""

    def __init__(
        self,
        policy_model: nn.Module,
        lyap_model: nn.Module,
        dyn_model: nn.Module,
        config: LyapunovCertificationConfig,
        device: th.device = th.device("cpu"),
    ) -> None:
        self.config = config
        self.device = device

        self.policy_model = policy_model.to(self.device).eval()
        self.lyap_model = lyap_model.to(self.device).eval()
        self.dyn_model = dyn_model.to(self.device).eval()
        self.bounds = th.as_tensor(self.config.cert_bounds, dtype=th.float32, device=self.device)

        self._abcrown_api: _ABCrownAPI | None = None
        self.abcrown_config: Any = None
        self.wrapped_model: _ABCrownModelWrapper | None = None
        self.setup_backend()

    def _get_abcrown_api(self) -> _ABCrownAPI:
        if self._abcrown_api is None:
            from abcrown import (
                ABCrownSolver,
                ConfigBuilder,
                VerificationSpec,
                input_vars,
                output_vars,
            )

            self._abcrown_api = _ABCrownAPI(
                solver_cls=ABCrownSolver,
                verification_spec_cls=VerificationSpec,
                config_builder_cls=ConfigBuilder,
                input_vars_fn=input_vars,
                output_vars_fn=output_vars,
            )
        return self._abcrown_api

    def _get_suppress_ctx(self):
        if self.config.suppress_native_output:
            return suppress_native_output(suppress_stderr=True)
        return nullcontext()

    @staticmethod
    def _normalize_status(status: str) -> str:
        return str(status).strip().lower()

    @classmethod
    def _is_verified_status(cls, status: str) -> bool:
        normalized = cls._normalize_status(status)
        return normalized == "verified" or normalized.startswith("safe")

    @classmethod
    def _is_counterexample_status(cls, status: str) -> bool:
        normalized = str(status).strip().lower()
        return normalized.startswith("unsafe")

    @staticmethod
    def _format_bab_violation(value: Any) -> str:
        if isinstance(value, th.Tensor):
            if value.numel() != 1:
                return str(value)
            return f"{float(value.detach().cpu().item()):.3f}"
        if isinstance(value, (int, float)):
            return f"{float(value):.3f}"
        return str(value)

    @staticmethod
    def _extract_bab_violation(bab_vals: Any) -> str:
        if isinstance(bab_vals, (list, tuple)) and len(bab_vals) > 0 and isinstance(bab_vals[0], (list, tuple)) and len(bab_vals[0]) > 1:
            return f"{float(bab_vals[0][1].detach().cpu().item()):.3f}"
        return "N/A"

    def _setup_verifier(self) -> LyapunovMultiOutputVerifier:
        return LyapunovMultiOutputVerifier(
            policy_model=self.policy_model,
            lyap_model=self.lyap_model,
            dyn_model=self.dyn_model,
            lbx=self.bounds[0].unsqueeze(0),
            ubx=self.bounds[1].unsqueeze(0),
            kappa=self.config.kappa,
            sublevel_tolerance=self.config.sublevel_tolerance,
            condition_margin=self.config.condition_margin,
        )

    def setup_backend(self) -> None:
        if self.wrapped_model is not None and self.abcrown_config is not None:
            return

        abcrown_api = self._get_abcrown_api()
        self.abcrown_config = (
            abcrown_api.config_builder_cls.from_defaults()
            .set(general__device=self.device.type)
            .set(general__complete_verifier="input_bab")
            .set(general__enable_incomplete_verification=False)
            .set(solver__batch_size=int(self.config.batch_size))
            .set(solver__bound_prop_method="crown")
            .set(bab__branching__method="sb")
            .set(bab__branching__input_split__enable=True)
            .set(bab__branching__input_split__ibp_enhancement=True)
            .set(bab__branching__input_split__compare_with_old_bounds=True)
            .set(bab__branching__input_split__adv_check=-1)
            .set(bab__branching__input_split__split_partitions=2)
            .set(attack__pgd_order="before")
            .set(bab__decision_thresh=-float(self.config.condition_tolerance))
            ()
        )

        verifier = self._setup_verifier()
        self.wrapped_model = _ABCrownModelWrapper(verifier, self.device)
        self.wrapped_model.eval()

        __logger__.info(
            "Configured ABCrown region backend with solver_batch_size=%d on device=%s.",
            int(self.config.batch_size),
            self.device.type,
        )

    def _build_safe_output_constraint(self, y, rho: float):
        safe_outside_sublevel = y[1] > (rho + self.config.sublevel_tolerance)
        safe_positive = y[1] > (-self.config.condition_tolerance)
        safe_decrease = y[0] > (-self.config.condition_tolerance)

        safe_x_next = None
        global_lb = self.bounds[0]
        global_ub = self.bounds[1]
        for idx in range(self.config.state_dim):
            coord_safe = (y[idx + 2] > (float(global_lb[idx]) - self.config.condition_tolerance)) & (
                y[idx + 2] < (float(global_ub[idx]) + self.config.condition_tolerance)
            )
            safe_x_next = coord_safe if safe_x_next is None else (safe_x_next & coord_safe)

        return safe_outside_sublevel | (safe_positive & safe_decrease & safe_x_next)

    def verify_region(self, region: th.Tensor, rho: float) -> ABCrownRegionVerification:
        if region.shape != (2, self.config.state_dim):
            raise ValueError(
                f"region must have shape (2, {self.config.state_dim}); got {tuple(region.shape)}."
            )

        if self.wrapped_model is None or self.abcrown_config is None:
            raise RuntimeError("Adaptive ABCrown backend is not initialized.")

        abcrown_api = self._get_abcrown_api()
        region = region.to(device=self.device, dtype=th.float32)
        lb = region[0]
        ub = region[1]
        self.wrapped_model.rho.fill_(float(rho))

        with self._get_suppress_ctx():
            x = abcrown_api.input_vars_fn(self.config.state_dim)
            y = abcrown_api.output_vars_fn(2 + self.config.state_dim)

            input_constraint = (x >= lb) & (x <= ub)
            output_constraint = self._build_safe_output_constraint(y=y, rho=float(rho))

            spec = abcrown_api.verification_spec_cls.build_spec(
                input_vars=x,
                output_vars=y,
                input_constraint=input_constraint,
                output_constraint=output_constraint,
            )
            solver = abcrown_api.solver_cls(
                spec=spec,
                computing_graph=self.wrapped_model,
                config=self.abcrown_config,
            )
            result = solver.solve()

        status = str(result.status).strip()
        elapsed = str(result.stats.get("elapsed", "N/A"))
        bab_vals = result.stats.get("bab", list())
        bab_violation = self._extract_bab_violation(bab_vals)
        __logger__.info("ABCrown solver status: %s after %ss with violation %s", status, elapsed, bab_violation)
        return ABCrownRegionVerification(
            status=status,
            verified=self._is_verified_status(status),
            counterexample_found=self._is_counterexample_status(status),
        )

    def certify_regions(
        self,
        regions: th.Tensor,
        rho: float,
        *,
        early_exit: bool = False,
        show_progress: bool = False,
    ) -> ABCrownRegionBatchVerification:
        if regions.ndim != 3 or regions.shape[1] != 2 or regions.shape[2] != self.config.state_dim:
            raise ValueError(
                f"regions must have shape (N, 2, {self.config.state_dim}); got {tuple(regions.shape)}."
            )

        if len(regions) == 0:
            return ABCrownRegionBatchVerification.empty(0)

        verified_mask = th.zeros((len(regions),), dtype=th.bool, device=self.device)
        counterexample_mask = th.zeros((len(regions),), dtype=th.bool, device=self.device)
        unknown_mask = th.zeros((len(regions),), dtype=th.bool, device=self.device)

        progress = None
        task = None
        if show_progress:
            progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            )
            progress.__enter__()
            task = progress.add_task(
                f"Certifying {len(regions)} regions with ABCrown (rho={rho:.6g})...",
                total=len(regions),
            )

        try:
            for idx, region in enumerate(regions):
                verification_result = self.verify_region(region, rho)
                verified_mask[idx] = verification_result.verified
                counterexample_mask[idx] = verification_result.counterexample_found
                unknown_mask[idx] = (
                    not verification_result.verified
                    and not verification_result.counterexample_found
                )
                if show_progress: progress.update(task, advance=1)
                if early_exit and verification_result.counterexample_found:
                    if show_progress: progress.update(task, completed=len(regions))
                    break
        finally:
            if show_progress:
                progress.__exit__(None, None, None)

        return ABCrownRegionBatchVerification(
            verified_mask=verified_mask,
            counterexample_mask=counterexample_mask,
            unknown_mask=unknown_mask,
        )