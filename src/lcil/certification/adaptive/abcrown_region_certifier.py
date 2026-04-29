from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import logging
from typing import Any

import torch as th
import torch.nn as nn
from pkg_logger import suppress_native_output

from ..models import LyapunovMultiOutputVerifier
from .config import AdaptiveCertificationConfig

__logger__ = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ABCrownAPI:
    solver_cls: Any
    verification_spec_cls: Any
    config_builder_cls: Any
    input_vars_fn: Any
    output_vars_fn: Any


class _AdaptiveABCrownModelWrapper(nn.Module):
    """Freeze rho so ABCrown sees a clean map ``x -> y``."""

    def __init__(self, verifier: nn.Module, device: th.device):
        super().__init__()
        self.verifier = verifier
        self.register_buffer("rho", th.tensor(0.0, dtype=th.float32, device=device))

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.verifier(x, self.rho)


class AdaptiveABCrownRegionCertifier:
    """Minimal ABCrown wrapper that certifies adaptive regions one by one."""

    def __init__(
        self,
        policy_model: nn.Module,
        lyap_model: nn.Module,
        dyn_model: nn.Module,
        config: AdaptiveCertificationConfig,
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
        self.verifier: LyapunovMultiOutputVerifier | None = None
        self.wrapped_model: _AdaptiveABCrownModelWrapper | None = None

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
    def _is_verified_status(status: str) -> bool:
        normalized = str(status).strip().lower()
        return normalized == "verified" or normalized.startswith("safe")

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
            .set(bab__branching__input_split__enable=True)
            .set(attack__pgd_order="skip")
            .set(bab__decision_thresh=-float(self.config.condition_tolerance))
            ()
        )

        self.verifier = self._setup_verifier()
        self.wrapped_model = _AdaptiveABCrownModelWrapper(self.verifier, self.device)
        self.wrapped_model.eval()

        __logger__.info(
            "Configured adaptive ABCrown backend with solver_batch_size=%d on device=%s.",
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

    def certify_region(self, region: th.Tensor, rho: float) -> bool:
        if region.shape != (2, self.config.state_dim):
            raise ValueError(
                f"region must have shape (2, {self.config.state_dim}); got {tuple(region.shape)}."
            )

        self.setup_backend()
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

        return self._is_verified_status(result.status)

    def certify_regions(
        self,
        regions: th.Tensor,
        rho: float,
        *,
        early_exit: bool = False,
    ) -> th.Tensor:
        if regions.ndim != 3 or regions.shape[1] != 2 or regions.shape[2] != self.config.state_dim:
            raise ValueError(
                f"regions must have shape (N, 2, {self.config.state_dim}); got {tuple(regions.shape)}."
            )

        if len(regions) == 0:
            return th.empty((0,), dtype=th.bool, device=self.device)

        is_certified = th.zeros((len(regions),), dtype=th.bool, device=self.device)
        for idx, region in enumerate(regions):
            is_certified[idx] = self.certify_region(region, rho)
            if early_exit and not bool(is_certified[idx].item()):
                break

        return is_certified