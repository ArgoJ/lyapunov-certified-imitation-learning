from __future__ import annotations

import logging

import torch as th
import torch.nn as nn

from abc import ABC, abstractmethod
from typing import Any, Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import IntEnum
from pkg_logger import suppress_native_output

from .config import LyapunovCertificationConfig
from .models import (
    LyapunovCoreVerifier,
    build_outside_sublevel_constraint,
    build_decrease_constraint,
    build_condition_constraint,
    build_invariance_constraint,
    build_positivity_constraint,
)
from .progress import CertificationProgress


__logger__ = logging.getLogger(__name__)


# ========================================================
# HELPER
# ========================================================
def _normalize_status(status: str) -> str:
    return status.strip().lower()

def _is_safe_status(status: str) -> bool:
    normalized = _normalize_status(status)
    return normalized.startswith("safe") or normalized.startswith("verified")

def _is_counterexample_status(status: str) -> bool:
    normalized = _normalize_status(status)
    return normalized.startswith("unsafe")

def _is_unknown_status(status: str) -> bool:
    normalized = _normalize_status(status)
    return normalized.startswith("unknown")


# ========================================================
# DATACLASSES & ENUMS
# ========================================================
class PGDMode(IntEnum):
    NOT = 0
    COMBINED = 1
    ONLY = 2


class EarlyExitLevel(IntEnum):
    NONE = 0
    ON_COUNTEREXAMPLE = 1
    ON_UNKNOWN = 2


@dataclass(frozen=True)
class ABCrownRegionVerification:
    """Status-aware verification result for a single packed region."""

    status: str
    verified: bool
    counterexample_found: bool
    counterexample: th.Tensor | None = None


@dataclass(frozen=True)
class ABCrownRegionBatchVerification:
    """Status masks for a batched region certification pass."""

    verified_mask: th.Tensor
    counterexample_mask: th.Tensor
    unknown_mask: th.Tensor
    counterexamples: list[th.Tensor] = field(default_factory=list)

    @property
    def unresolved_mask(self) -> th.Tensor:
        return ~self.verified_mask

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


# ========================================================
# BASE ABCROWN CERTIFIER
# ========================================================
class BaseABCrownCertifier(ABC):
    """Shared ABCrown backend for certifying packed regions one by one."""

    def __init__(
        self,
        config: LyapunovCertificationConfig,
        device: th.device = th.device("cpu"),
    ) -> None:
        self.config = config
        self.device = device

        self._abcrown_api: _ABCrownAPI | None = None
        self.abcrown_pgd_config: Any = None
        self.abcrown_combined_config: Any = None
        self.abcrown_leaf_combined_config: Any = None
        self.abcrown_no_pgd_config: Any = None
        self.abcrown_leaf_no_pgd_config: Any = None
        self.verifier: nn.Module | None = None
        self._cached_clauses: dict[float, Any] = {}

    progress_description: str = "Certify Regions"

    @abstractmethod
    def _setup_verifier(self) -> nn.Module:
        """Set up and return a verifier module instance for ABCrown."""

    @abstractmethod
    def _output_dim(self) -> int:
        """Return the verifier output dimension expected by the ABCrown spec."""

    @abstractmethod
    def _build_safe_output_constraint(self, y, rho: float):
        """Build a safe output constraint for ABCrown based on the verifier output y and sublevel rho."""

    def _get_abcrown_api(self) -> _ABCrownAPI:
        with self._get_suppress_ctx():
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
    def _extract_bab_violation(bab_vals: Any) -> str:
        if isinstance(bab_vals, (list, tuple)) and len(bab_vals) > 0 and isinstance(bab_vals[0], (list, tuple)) and len(bab_vals[0]) > 1:
            return f"{float(bab_vals[0][1].detach().cpu().item()):.3f}"
        return "N/A"

    @staticmethod
    def _extract_counterexample(result: Any) -> th.Tensor | None:
        stats = getattr(result, "stats", {}) or {}
        ref = getattr(result, "reference", {}) or {}
        for c in (getattr(result, "c", None), stats.get("attack_examples"), ref.get("attack_examples")):
            if c is not None:
                cex = c[0] if isinstance(c, (list, tuple)) and len(c) > 0 else c
                if isinstance(cex, th.Tensor):
                    cex = cex.detach().cpu()
                    if cex.ndim == 2:
                        margins = stats.get("attack_margins")
                        if isinstance(margins, th.Tensor) and margins.numel() == cex.shape[0]:
                            cex = cex[th.argmin(margins)]
                        else:
                            cex = cex[0]
                    return cex.squeeze()
        return None

    def _build_abcrown_config(
        self,
        is_leaf: bool = False,
        pgd: PGDMode = PGDMode.COMBINED,
    ) -> Any:
        abcrown_api = self._get_abcrown_api()
        complete_verifier = "skip" if pgd == PGDMode.ONLY else "input_bab"
        pgd_order = "skip" if pgd == PGDMode.NOT else "before"
        enable_incomplete = (pgd == PGDMode.ONLY)
        config_builder = (
            abcrown_api.config_builder_cls.from_defaults()
            .set(general__device=self.device.type)
            .set(general__complete_verifier=complete_verifier)
            .set(general__enable_incomplete_verification=enable_incomplete)
            .set(solver__batch_size=self.config.batch_size)
            .set(solver__bound_prop_method="crown")
            .set(bab__branching__method="sb")
            .set(bab__branching__input_split__enable=True)
            .set(bab__branching__input_split__ibp_enhancement=True)
            .set(bab__branching__input_split__compare_with_old_bounds=True)
            .set(bab__branching__input_split__adv_check=-1)
            .set(bab__branching__input_split__split_partitions=self.config.abcrown_input_split_partitions)
            .set(attack__pgd_order=pgd_order)
            .set(bab__decision_thresh=-float(self.config.condition_tolerance))
        )
        if self.config.abcrown_timeout is not None:
            config_builder = config_builder.set(
                bab__timeout=float(self.config.abcrown_timeout if not is_leaf else self.config.abcrown_timeout * 5) 
            )
        if self.config.abcrown_max_domains is not None:
            config_builder = config_builder.set(bab__max_domains=self.config.abcrown_max_domains)
        return config_builder()


    def setup_backend(self) -> None:
        self._cached_clauses.clear()
        if (
            self.verifier is not None
            and self.abcrown_combined_config is not None
            and self.abcrown_leaf_combined_config is not None
            and self.abcrown_pgd_config is not None
            and self.abcrown_no_pgd_config is not None
            and self.abcrown_leaf_no_pgd_config is not None
        ):
            return

        self.abcrown_pgd_config = self._build_abcrown_config(pgd=PGDMode.ONLY)
        self.abcrown_combined_config = self._build_abcrown_config(is_leaf=False, pgd=PGDMode.COMBINED)
        self.abcrown_leaf_combined_config = self._build_abcrown_config(is_leaf=True, pgd=PGDMode.COMBINED)
        self.abcrown_no_pgd_config = self._build_abcrown_config(is_leaf=False, pgd=PGDMode.NOT)
        self.abcrown_leaf_no_pgd_config = self._build_abcrown_config(is_leaf=True, pgd=PGDMode.NOT)

        self.verifier = self._setup_verifier()
        self.verifier.eval()

        __logger__.debug(
            "Configured ABCrown region backend with solver_batch_size=%d on device=%s (timeout=%s, max_domains=%s).",
            self.config.batch_size,
            self.device.type,
            (
                f"{float(self.config.abcrown_timeout):.1f}s"
                if self.config.abcrown_timeout is not None
                else "default"
            ),
            (
                str(self.config.abcrown_max_domains)
                if self.config.abcrown_max_domains is not None
                else "default"
            ),
        )

    def verify_region(
        self,
        region: th.Tensor,
        rho: float,
        *,
        is_leaf: bool = False,
        pgd: PGDMode = PGDMode.COMBINED,
    ) -> ABCrownRegionVerification:
        if region.shape != (2, self.config.state_dim):
            raise ValueError(
                f"region must have shape (2, {self.config.state_dim}); got {tuple(region.shape)}."
            )

        if self.verifier is None or self.abcrown_combined_config is None:
            raise RuntimeError("Adaptive ABCrown backend is not initialized.")

        abcrown_api = self._get_abcrown_api()
        region = region.to(device=self.device, dtype=th.float32)
        lb = region[0]
        ub = region[1]

        with self._get_suppress_ctx():
            rho_key = round(float(rho), 8)
            if rho_key not in self._cached_clauses:
                x = abcrown_api.input_vars_fn(self.config.state_dim)
                y = abcrown_api.output_vars_fn(self._output_dim())
                dummy_lb = th.zeros(self.config.state_dim, device=self.device)
                dummy_ub = th.ones(self.config.state_dim, device=self.device)
                dummy_input_constraint = (x >= dummy_lb) & (x <= dummy_ub)
                output_constraint = self._build_safe_output_constraint(y=y, rho=float(rho))
                init_spec = abcrown_api.verification_spec_cls.build_spec(
                    input_vars=x,
                    output_vars=y,
                    input_constraint=dummy_input_constraint,
                    output_constraint=output_constraint,
                )
                self._cached_clauses[rho_key] = init_spec.output_spec.clauses

            clauses = self._cached_clauses[rho_key]
            num_clauses = len(clauses) if isinstance(clauses, (list, tuple)) else 1
            lower = lb.repeat((num_clauses, 1)) if lb.ndim == 1 else lb.repeat(num_clauses, *(1 for _ in range(lb.ndim - 1)))
            upper = ub.repeat((num_clauses, 1)) if ub.ndim == 1 else ub.repeat(num_clauses, *(1 for _ in range(ub.ndim - 1)))
            spec = abcrown_api.verification_spec_cls.build_from_input_bounds(
                lower=lower,
                upper=upper,
                clauses=clauses,
            )
            if pgd == PGDMode.ONLY and self.abcrown_pgd_config is not None:
                solver_config = self.abcrown_pgd_config
            elif pgd == PGDMode.NOT and self.abcrown_no_pgd_config is not None:
                solver_config = (
                    self.abcrown_leaf_no_pgd_config
                    if is_leaf and self.abcrown_leaf_no_pgd_config is not None
                    else self.abcrown_no_pgd_config
                )
            else:
                solver_config = (
                    self.abcrown_leaf_combined_config
                    if is_leaf and self.abcrown_leaf_combined_config is not None
                    else self.abcrown_combined_config
                )
            solver = abcrown_api.solver_cls(
                spec=spec,
                computing_graph=self.verifier,
                config=solver_config,
            )
            result = solver.solve()

        status = str(result.status).strip()
        elapsed = f"{result.stats.get('elapsed', -1.0):.3f}s"
        bab_vals = result.stats.get("bab", list())
        bab_violation = f"violation={self._extract_bab_violation(bab_vals)}." if _is_unknown_status(status) else "no violation."
        __logger__.debug("ABCrown solver status: %s after %s with %s", status, elapsed, bab_violation)

        cex = self._extract_counterexample(result) if _is_counterexample_status(status) else None
        if _is_counterexample_status(status):
            extra_info = ""
            if cex is not None and hasattr(self, "lyap_model") and self.lyap_model is not None:
                try:
                    v_val = self.lyap_model(cex.unsqueeze(0).to(self.device)).item()
                    extra_info = f" (V(x) = {v_val:.5f}, rho = {float(rho):.5f})"
                except Exception:
                    pass
            x_str = f" x = {[round(float(v), 5) for v in cex.flatten()]}{extra_info}" if cex is not None else ""
            __logger__.info("Counterexample found by solver (%s) after %s:%s", status, elapsed, x_str)

        return ABCrownRegionVerification(
            status=status,
            verified=_is_safe_status(status),
            counterexample_found=_is_counterexample_status(status),
            counterexample=cex,
        )

    def certify_regions(
        self,
        regions: th.Tensor,
        rho: float,
        *,
        description: str | None = None,
        early_exit: EarlyExitLevel | bool = EarlyExitLevel.NONE,
        progress: CertificationProgress | None = None,
        is_leaf: bool = False,
    ) -> ABCrownRegionBatchVerification:
        if isinstance(early_exit, bool):
            early_exit = EarlyExitLevel.ON_COUNTEREXAMPLE if early_exit else EarlyExitLevel.NONE

        if regions.ndim != 3 or regions.shape[1] != 2 or regions.shape[2] != self.config.state_dim:
            raise ValueError(
                f"regions must have shape (N, 2, {self.config.state_dim}); got {tuple(regions.shape)}."
            )

        if len(regions) == 0:
            return ABCrownRegionBatchVerification.empty(0)

        verified_mask = th.zeros((len(regions),), dtype=th.bool, device=self.device)
        counterexample_mask = th.zeros((len(regions),), dtype=th.bool, device=self.device)
        unknown_mask = th.zeros((len(regions),), dtype=th.bool, device=self.device)
        counterexamples: list[th.Tensor] = []

        if progress is not None:
            task_description = description if description is not None else self.progress_description
            progress.start_certify(description=task_description, total=len(regions))


        try:
            # PGD Screening
            if early_exit != EarlyExitLevel.NONE:
                for idx, region in enumerate(regions):
                    res = self.verify_region(region, rho, pgd=PGDMode.ONLY)
                    if res.counterexample_found:
                        counterexample_mask[idx] = True
                        if res.counterexample is not None:
                            counterexamples.append(res.counterexample)
                        if progress is not None:
                            progress.step_certify(False, True)
                        return ABCrownRegionBatchVerification(
                            verified_mask=verified_mask,
                            counterexample_mask=counterexample_mask,
                            unknown_mask=unknown_mask,
                            counterexamples=counterexamples,
                        )

            # Formal BaB
            pgd_mode: PGDMode = PGDMode.NOT if early_exit != EarlyExitLevel.NONE else PGDMode.COMBINED
            for idx, region in enumerate(regions):
                verification_result = self.verify_region(
                    region,
                    rho,
                    is_leaf=is_leaf,
                    pgd=pgd_mode,
                )
                verified_mask[idx] = verification_result.verified
                counterexample_mask[idx] = verification_result.counterexample_found
                unknown_mask[idx] = (
                    not verification_result.verified
                    and not verification_result.counterexample_found
                )
                if verification_result.counterexample is not None:
                    counterexamples.append(verification_result.counterexample)

                if progress is not None:
                    progress.step_certify(
                        verification_result.verified,
                        verification_result.counterexample_found,
                    )

                if early_exit != EarlyExitLevel.NONE and verification_result.counterexample_found:
                    break
                if early_exit == EarlyExitLevel.ON_UNKNOWN and not verification_result.verified:
                    break
        finally:
            if progress is not None:
                progress.stop_certify()

        return ABCrownRegionBatchVerification(
            verified_mask=verified_mask,
            counterexample_mask=counterexample_mask,
            unknown_mask=unknown_mask,
            counterexamples=counterexamples,
        )


# ========================================================
# ABCROWN CERTIFIER VARIANTS
# ========================================================
class BaseLyapunovCoreABCrownCertifier(BaseABCrownCertifier):
    """Shared ABCrown backend for Lyapunov-core predicates."""

    progress_description: str = "Core Check"

    def __init__(
        self,
        policy_model: nn.Module,
        lyap_model: nn.Module,
        dyn_model: nn.Module,
        config: LyapunovCertificationConfig,
        device: th.device = th.device("cpu"),
    ) -> None:
        super().__init__(config=config, device=device)
        self.policy_model = policy_model.to(self.device).eval()
        self.lyap_model = lyap_model.to(self.device).eval()
        self.dyn_model = dyn_model.to(self.device).eval()
        self.bounds = th.as_tensor(self.config.cert_bounds, dtype=th.float32, device=self.device)
        self.setup_backend()

    def _setup_verifier(self) -> nn.Module:
        return LyapunovCoreVerifier(
            policy_model=self.policy_model,
            lyap_model=self.lyap_model,
            dyn_model=self.dyn_model,
            kappa=self.config.kappa,
        )

    def _output_dim(self) -> int:
        return 2 + self.config.state_dim


class CompleteABCrownCertifier(BaseLyapunovCoreABCrownCertifier):
    """ABCrown certifier combining sublevel set certification and core Lyapunov condition verification."""

    progress_description: str = "Complete Certification"

    def _build_safe_output_constraint(self, y, rho: float):
        safe_outside_sublevel = build_outside_sublevel_constraint(y[1], rho)
        safe_condition = build_condition_constraint(y, self.bounds[0], self.bounds[1])
        return safe_outside_sublevel | safe_condition


class CoreABCrownCertifier(BaseLyapunovCoreABCrownCertifier):
    """ABCrown certifier focused on verifying the core Lyapunov condition without explicit sublevel constraints."""

    progress_description: str = "Core Check"

    def _build_safe_output_constraint(self, y, rho: float):
        del rho
        return build_condition_constraint(y, self.bounds[0], self.bounds[1])

    def certify_regions(
        self,
        regions: th.Tensor,
        rho: float,
        *,
        description: str | None = None,
        early_exit: EarlyExitLevel | bool = EarlyExitLevel.NONE,
        progress: CertificationProgress | None = None,
        is_leaf: bool = False,
    ) -> ABCrownRegionBatchVerification:
        # Core check must run through all regions completely and never abort early on counterexamples.
        del early_exit
        return super().certify_regions(
            regions=regions,
            rho=rho,
            description=description,
            early_exit=EarlyExitLevel.NONE,
            progress=progress,
            is_leaf=is_leaf,
        )


class PositivityABCrownCertifier(BaseABCrownCertifier):
    """ABCrown certifier for Lyapunov positivity only."""

    progress_description: str = "Positivity Check"

    def __init__(
        self,
        lyap_model: nn.Module,
        config: LyapunovCertificationConfig,
        device: th.device = th.device("cpu"),
    ) -> None:
        super().__init__(config=config, device=device)
        self.lyap_model = lyap_model.to(self.device).eval()
        self.setup_backend()

    def _setup_verifier(self) -> nn.Module:
        return self.lyap_model

    def _output_dim(self) -> int:
        return 1

    def _build_safe_output_constraint(self, y, rho: float):
        del rho
        return build_positivity_constraint(
            y,
            lyap_output_index=0,
        )


class DecreaseABCrownCertifier(BaseLyapunovCoreABCrownCertifier):
    """ABCrown certifier for the Lyapunov decrease condition only."""

    progress_description: str = "Decrease Check"

    def _build_safe_output_constraint(self, y, rho: float):
        del rho
        return build_decrease_constraint(y)


class InvarianceABCrownCertifier(BaseLyapunovCoreABCrownCertifier):
    """ABCrown certifier for state invariance only."""

    progress_description: str = "Invariance Check"

    def _build_safe_output_constraint(self, y, rho: float):
        del rho
        return build_invariance_constraint(
            y,
            self.bounds[0],
            self.bounds[1],
        )