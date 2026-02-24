from __future__ import annotations

import torch as th
import torch.nn as nn

from typing import Sequence
from contextlib import nullcontext
from dataclasses import dataclass, replace
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

from .config import LyapunovCertificationConfig
from .models import ClosedLoopLyapunovConditionVerifier
from ..utils.package_logger import get_package_logger, PackageLogger

__logger__ = get_package_logger(__name__)


@dataclass(frozen=True)
class RegionCertificationResult:
    """Result container for a full-region certification pass."""
    success: bool
    counter_examples: list[th.Tensor]
    failed_regions: list[tuple[list[float], list[float]]]
    certified_regions: list[tuple[list[float], list[float]]]


class LiRPACertifier:
    def __init__(
            self, 
            policy_model: nn.Module,
            lyap_model: nn.Module,
            dyn_model: nn.Module,
            config: LyapunovCertificationConfig,
            device: th.device = th.device("cpu"),
    ):
        self.config = self._resolve_config(config)
        self.device = device

        self.policy_model = policy_model.to(self.device).eval()
        self.lyap_model = lyap_model.to(self.device).eval()
        self.dyn_model = dyn_model.to(self.device).eval()

        self.bounds = self._resolve_bounds(config.state_bounds, device)
        self.regions = self._build_regions()

        self._lirpa_model = None
        self._verifier_module = None

    def _resolve_config(self, config: LyapunovCertificationConfig) -> LyapunovCertificationConfig:
        resolved_config = replace(
            config,
            cert_method=config.cert_method.strip().lower(),
            cert_rho_scaling=max(config.cert_rho_scaling, 1.01),
        )
        if resolved_config.cert_step <= 0:
            raise ValueError("cert_step must be positive.")
        if resolved_config.cert_method not in {"alpha-crown", "crown", "crown-ibp", "ibp"}:
            raise ValueError("cert_method must be one of 'alpha-crown', 'crown', 'crown-ibp', or 'ibp'.")
        return resolved_config

    @staticmethod
    def _resolve_bounds(state_bounds: Sequence[float], device: th.device) -> th.Tensor:
        bounds = th.as_tensor(state_bounds, dtype=th.float32, device=device)
        if bounds.ndim != 2 or bounds.shape[0] != 2:
            raise ValueError("state_bounds must be a sequence of shape (2, nx) [lb, ub].")
        return bounds

    def _setup_lirpa_model(self, verifier: nn.Module) -> BoundedModule:
        dummy_input = th.zeros(1, self.config.state_dim, device=self.device)
        lirpa_model = BoundedModule(
            verifier,
            dummy_input,
            device=self.device,
            verbose=False,
            bound_opts={'perturb_bound': True},
        )
        return lirpa_model

    def _setup_verifier(self, rho: float) -> ClosedLoopLyapunovConditionVerifier:
        lbx, ubx = self.bounds[0], self.bounds[1]
        verifier = ClosedLoopLyapunovConditionVerifier(
            policy_model=self.policy_model,
            lyap_model=self.lyap_model,
            dyn_model=self.dyn_model,
            lbx=lbx,
            ubx=ubx,
            kappa=self.config.kappa,
            invariance_weight=self.config.invariance_weight,
            rho=max(self.config.rho_min, rho),
        ).to(self.device)
        verifier.eval()
        return verifier

    def _build_regions(self) -> list[tuple[list[float], list[float]]]:
        if self.config.state_dim != 2:
            raise ValueError("certification currently supports state_dim == 2.")

        if self.config.cert_step <= 0:
            raise ValueError("cert_step must be positive.")

        # Use maximum absolute bound for origin exclusion heuristics.
        train_diameter = th.max(th.abs(self.bounds)).item()
        if self.config.cert_origin_exclusion is None:
            origin_exclusion = min(train_diameter * 0.01, 0.1)
        else:
            origin_exclusion = self.config.cert_origin_exclusion

        lbx, ubx = self.bounds[0], self.bounds[1]
        step = self.config.cert_step

        x_vals = th.arange(lbx[0].item(), ubx[0].item(), step, device=self.device)
        y_vals = th.arange(lbx[1].item(), ubx[1].item(), step, device=self.device)
        grid_y, grid_x = th.meshgrid(y_vals, x_vals, indexing="ij")
        
        grid_x = grid_x.flatten()
        grid_y = grid_y.flatten()

        mask = ~((th.abs(grid_x) < origin_exclusion) & (th.abs(grid_y) < origin_exclusion))
        valid_x = grid_x[mask].tolist()
        valid_y = grid_y[mask].tolist()
        regions: list[tuple[list[float], list[float]]] = [
            ([x, y], [x + step, y + step]) for x, y in zip(valid_x, valid_y)
        ]
        return regions

    def certify_batched_regions(
        self,
        regions: list[tuple[list[float], list[float]]]
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Certify a batch of regions at once using LiRPA. 
        Returns a boolean mask of certified regions, the centers of the regions, 
        and the maximum upper bound for each region.

        Parameters
        ----------
        regions : list[tuple[list[float], list[float]]]
            _description_

        Returns
        -------
        tuple[th.Tensor, th.Tensor, th.Tensor]
            _description_
        """
        # 1. Listen in batched Tensoren konvertieren
        lbs = th.tensor([r[0] for r in regions], device=self.device, dtype=th.float32)
        ubs = th.tensor([r[1] for r in regions], device=self.device, dtype=th.float32)
        centers = (lbs + ubs) / 2.0

        # 2. Perturbation für das gesamte Batch erstellen
        ptb = PerturbationLpNorm(norm=float("inf"), x_L=lbs, x_U=ubs)
        bounded_input = BoundedTensor(centers, ptb)

        normalized_method = self.config.cert_method.strip().lower()
        fallback_methods = [normalized_method]
        if normalized_method == "alpha-crown":
            fallback_methods.extend(["crown", "crown-ibp", "ibp"])

        ub_out: th.Tensor | None = None
        last_exc: Exception | None = None

        ctx = PackageLogger.suppress_native_output() if self.config.cert_suppress_native_output else nullcontext()
        
        # 3. LiRPA Forward-Pass für hunderte Boxen gleichzeitig
        for candidate_method in fallback_methods:
            try:
                with ctx:
                    if candidate_method == "alpha-crown":
                        _, ub_out = self.lirpa_model.compute_bounds(x=(bounded_input,), method=candidate_method)
                    else:
                        with th.no_grad():
                            _, ub_out = self.lirpa_model.compute_bounds(x=(bounded_input,), method=candidate_method)
                break
            except Exception as exc:
                last_exc = exc
                __logger__.warning(
                    f"LiRPA method '{candidate_method}' failed with {exc} on a batch of {len(regions)} regions."
                )

        if ub_out is None:
            __logger__.error(
                f"All LiRPA methods failed on batch. Marking batch as uncertified. Last error: {last_exc}"
            )
            # Falls alles fehlschlägt, gelten alle Boxen im Batch als unsicher
            return th.zeros(len(regions), dtype=th.bool, device=self.device), centers, th.full((len(regions),), float("inf"))

        # 4. Resultate evaluieren
        max_uppers = ub_out.flatten()
        is_certified = max_uppers <= self.config.condition_tolerance
        
        return is_certified, centers, max_uppers

    def certify_regions(
            self, 
            regions: list[tuple[list[float], list[float]]] = None, 
            collect_details: bool = True,
            depth: int = 0,
            max_depth: int = 3
    ) -> RegionCertificationResult:
        """
        Certify a list of regions and optionally collect details on 
        certified/failed regions and counterexamples.

        Parameters
        ----------
        regions : list of tuples (lb, ub)
            List of regions to certify, where each region is defined by a 
            lower bound (lb) and an upper bound (ub). If None, uses self.regions.
        collect_details : bool
            If True, collects detailed information about certified and failed regions, 
            as well as counterexamples.

        Returns
        -------
        RegionCertificationResult
            A dataclass containing the overall success status, list of counterexamples, 
            failed regions, and certified regions.
        """
        if regions is None:
            regions = self.regions
            
        if not regions:
            return RegionCertificationResult(True, [], [], [])

        is_safe, centers, _ = self.certify_batched_regions(regions)
        safe_mask = is_safe.cpu().tolist()

        failed_regions: list[tuple[list[float], list[float]]] = []
        certified_regions: list[tuple[list[float], list[float]]] = []
        counter_examples: list[th.Tensor] = []
        sub_regions_to_check: list[tuple[list[float], list[float]]] = []

        for i, safe in enumerate(safe_mask):
            lb, ub = regions[i]
            if safe:
                if collect_details:
                    certified_regions.append((lb, ub))
            else:
                if not collect_details:
                    # Early Exit
                    return RegionCertificationResult(False, [], [], [])

                counter_examples.append(centers[i].detach().cpu())

                # Hard to certify -> create 4 subdivisions (mit nativem Python)
                mid_x = (lb[0] + ub[0]) / 2.0
                mid_y = (lb[1] + ub[1]) / 2.0
                
                sub_regions_to_check.extend([
                    ([lb[0], lb[1]], [mid_x, mid_y]),
                    ([mid_x, lb[1]], [ub[0], mid_y]),
                    ([lb[0], mid_y], [mid_x, ub[1]]),
                    ([mid_x, mid_y], [ub[0], ub[1]]),
                ])

        # Recursive call for hard regions
        if sub_regions_to_check and collect_details:
            sub_result = self.certify_regions(
                regions=sub_regions_to_check,
                collect_details=True,
                depth=depth + 1,
                max_depth=max_depth
            )
            certified_regions.extend(sub_result.certified_regions)
            failed_regions.extend(sub_result.failed_regions)
            counter_examples.extend(sub_result.counter_examples)

        return RegionCertificationResult(
            success=len(failed_regions) == 0 and len(sub_regions_to_check) == 0,
            counter_examples=counter_examples,
            failed_regions=failed_regions,
            certified_regions=certified_regions,
        )

    def is_rho_certified(self, rho: float) -> bool:
        self.verifier.set_rho(rho)
        result = self.certify_regions(collect_details=False) 
        return result.success

    def certify(
            self,
            rho_estimate: float,
    ) -> tuple[float, RegionCertificationResult]:
        """Find the largest certified rho using scaling and bisection."""
        __logger__.info("Starting Lyapunov certification with method: %s", self.config.cert_method)

        self.verifier = self._setup_verifier(rho_estimate)
        self.lirpa_model = self._setup_lirpa_model(self.verifier)

        initial_rho = max(self.config.rho_min, float(rho_estimate))
        initial_ok = self.is_rho_certified(rho=initial_rho)

        # Initial rho passed, scale up to find an upper bound.
        if initial_ok:
            rho_lo = initial_rho
            rho_up = initial_rho
            found_upper_failure = False
            
            with __logger__.tqdm(range(self.config.cert_max_scale_steps), desc="Scale up: upper rho") as pbar:
                for _ in pbar:
                    trial = rho_up * self.config.cert_rho_scaling
                    if self.is_rho_certified(rho=trial):
                        rho_lo = trial
                        rho_up = trial
                    else:
                        rho_up = trial
                        found_upper_failure = True
                        break

            if not found_upper_failure:
                self.verifier.set_rho(rho_lo)
                details = self.certify_regions(collect_details=True)
                return rho_lo, details

        # Initial rho failed, scale down to find a certified rho.
        if not initial_ok:
            rho_up = initial_rho
            rho_lo: float | None = None
            trial = initial_rho
            
            with __logger__.tqdm(range(self.config.cert_max_scale_steps), desc="Scale down: lower rho") as pbar:
                for _ in pbar:
                    trial = max(self.config.rho_min, trial / self.config.cert_rho_scaling)
                    if self.is_rho_certified(rho=trial):
                        rho_lo = trial
                        break

                    rho_up = trial
                    if trial <= self.config.rho_min:
                        break

            if rho_lo is None:
                rho_min_ok = self.is_rho_certified(rho=self.config.rho_min)
                if not rho_min_ok:
                    self.verifier.set_rho(self.config.rho_min)
                    details = self.certify_regions(collect_details=True)
                    return self.config.rho_min, details

                rho_lo = self.config.rho_min
                rho_up = max(rho_up, initial_rho)

        # Bisection between rho_lo and rho_up to find the largest certified rho within tolerance.
        with __logger__.tqdm(range(self.config.cert_max_bisection_steps), desc="Bisection: max rho") as pbar:
            for _ in pbar:
                if rho_up - rho_lo <= self.config.cert_bisection_tol:
                    break

                rho_mid = 0.5 * (rho_lo + rho_up)
                if self.is_rho_certified(rho=rho_mid):
                    rho_lo = rho_mid
                else:
                    rho_up = rho_mid

        self.verifier.set_rho(rho_lo)
        details = self.certify_regions(collect_details=True)
        return rho_lo, details

