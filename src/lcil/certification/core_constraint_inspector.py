from __future__ import annotations

import logging

import torch as th
import torch.nn as nn

from dataclasses import dataclass
from typing import Callable, Sequence

from .abcrown_region_certifier import (
    ABCrownRegionBatchVerification,
    DecreaseABCrownCertifier,
    InvarianceABCrownCertifier,
    PositivityABCrownCertifier,
)
from .config import LyapunovCertificationConfig
from .region_manager import RegionManager
from ..utils.region_builder import RegionBuilder

__logger__ = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConstraintInspectionResult:
    """Region-wise status partition for one individual core constraint."""

    name: str
    verified_regions: th.Tensor
    counterexample_regions: th.Tensor
    unknown_regions: th.Tensor

    @property
    def num_regions(self) -> int:
        return len(self.verified_regions) + len(self.counterexample_regions) + len(self.unknown_regions)

    @property
    def global_success(self) -> bool:
        return len(self.counterexample_regions) == 0 and len(self.unknown_regions) == 0 and self.num_regions > 0

    def log_summary(self) -> None:
        __logger__.info(
            "%s summary: total=%d verified=%d counterexample=%d unknown=%d success=%s.",
            self.name,
            self.num_regions,
            len(self.verified_regions),
            len(self.counterexample_regions),
            len(self.unknown_regions),
            self.global_success,
        )


@dataclass(frozen=True)
class CoreConstraintInspectionResult:
    """Grouped inspection result for positivity, decrease, and invariance."""

    regions: th.Tensor
    positivity: ConstraintInspectionResult
    decrease: ConstraintInspectionResult
    invariance: ConstraintInspectionResult

    def log_summary(self) -> None:
        __logger__.info("Core constraint inspection summary:")
        self.positivity.log_summary()
        self.decrease.log_summary()
        self.invariance.log_summary()


class CoreConstraintInspector:
    """Inspect individual Lyapunov-core constraints over certification regions."""

    def __init__(
        self,
        policy_model: nn.Module,
        lyap_model: nn.Module,
        dyn_model: nn.Module,
        config: LyapunovCertificationConfig,
        device: th.device = th.device("cpu"),
    ):
        self.config = config
        self.device = device

        self.policy_model = policy_model.to(self.device).eval()
        self.lyap_model = lyap_model.to(self.device).eval()
        self.dyn_model = dyn_model.to(self.device).eval()

        self.bounds = self._resolve_bounds(config.cert_bounds, device)

        self.region_builder: RegionBuilder | None = None
        self.positivity_certifier: PositivityABCrownCertifier | None = None
        self.decrease_certifier: DecreaseABCrownCertifier | None = None
        self.invariance_certifier: InvarianceABCrownCertifier | None = None
        self.region_manager = RegionManager(
            region_builder=self._get_region_builder(),
        )

    @staticmethod
    def _resolve_bounds(bounds: Sequence[float], device: th.device) -> th.Tensor:
        bounds = th.as_tensor(bounds, dtype=th.float32, device=device)
        if bounds.ndim != 2 or bounds.shape[0] != 2:
            raise ValueError("bounds must be a sequence of shape (2, nx) [lb, ub].")
        return bounds

    def _build_region_builder(self) -> RegionBuilder:
        return RegionBuilder(
            bounds=self.bounds,
            bins_per_dim=self.config.bins_per_dim,
            center_refinement_factor=self.config.center_refinement_factor,
            origin_exclusion=self.config.origin_exclusion,
            device=self.device,
        )

    def _get_region_builder(self) -> RegionBuilder:
        if self.region_builder is None:
            self.region_builder = self._build_region_builder()
        return self.region_builder

    def _build_positivity_certifier(self) -> PositivityABCrownCertifier:
        return PositivityABCrownCertifier(
            lyap_model=self.lyap_model,
            config=self.config,
            device=self.device,
        )

    def _get_positivity_certifier(self) -> PositivityABCrownCertifier:
        if self.positivity_certifier is None:
            self.positivity_certifier = self._build_positivity_certifier()
        return self.positivity_certifier

    def _build_decrease_certifier(self) -> DecreaseABCrownCertifier:
        return DecreaseABCrownCertifier(
            policy_model=self.policy_model,
            lyap_model=self.lyap_model,
            dyn_model=self.dyn_model,
            config=self.config,
            device=self.device,
        )

    def _get_decrease_certifier(self) -> DecreaseABCrownCertifier:
        if self.decrease_certifier is None:
            self.decrease_certifier = self._build_decrease_certifier()
        return self.decrease_certifier

    def _build_invariance_certifier(self) -> InvarianceABCrownCertifier:
        return InvarianceABCrownCertifier(
            policy_model=self.policy_model,
            lyap_model=self.lyap_model,
            dyn_model=self.dyn_model,
            config=self.config,
            device=self.device,
        )

    def _get_invariance_certifier(self) -> InvarianceABCrownCertifier:
        if self.invariance_certifier is None:
            self.invariance_certifier = self._build_invariance_certifier()
        return self.invariance_certifier

    @staticmethod
    def _pack_constraint_result(
        name: str,
        regions: th.Tensor,
        batch_result: ABCrownRegionBatchVerification,
    ) -> ConstraintInspectionResult:
        return ConstraintInspectionResult(
            name=name,
            verified_regions=regions[batch_result.verified_mask],
            counterexample_regions=regions[batch_result.counterexample_mask],
            unknown_regions=regions[batch_result.unknown_mask],
        )

    def _run_constraint_check(
        self,
        regions: th.Tensor,
        *,
        name: str,
        certifier_getter: Callable[[], PositivityABCrownCertifier | DecreaseABCrownCertifier | InvarianceABCrownCertifier],
        show_progress: bool,
    ) -> ConstraintInspectionResult:
        __logger__.info("Inspecting %s on %d regions.", name, len(regions))
        batch_result = certifier_getter().certify_regions(
            regions=regions,
            rho=0.0, # Not used for core constraint checks, but required by the certifier interface.
            early_exit=False,
            show_progress=show_progress,
        )
        result = self._pack_constraint_result(name, regions, batch_result)
        result.log_summary()
        return result

    def inspect(
        self,
        *,
        show_progress: bool = False,
    ) -> CoreConstraintInspectionResult:
        """Run positivity, decrease, and invariance checks on one region batch."""
        target_regions = self.region_manager.ensure_regions()

        positivity = self._run_constraint_check(
            target_regions,
            name="positivity",
            certifier_getter=self._get_positivity_certifier,
            show_progress=show_progress,
        )
        decrease = self._run_constraint_check(
            target_regions,
            name="decrease",
            certifier_getter=self._get_decrease_certifier,
            show_progress=show_progress,
        )
        invariance = self._run_constraint_check(
            target_regions,
            name="invariance",
            certifier_getter=self._get_invariance_certifier,
            show_progress=show_progress,
        )

        return CoreConstraintInspectionResult(
            regions=target_regions,
            positivity=positivity,
            decrease=decrease,
            invariance=invariance,
        )