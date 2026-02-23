from __future__ import annotations

import numpy as np
import torch as th
import torch.nn as nn

from dataclasses import dataclass
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from contextlib import nullcontext

from .config import LyapunovCertificationConfig
from .models import ClosedLoopLyapunovConditionVerifier
from .counterexample import _bounds_tensor
from ..utils.package_logger import get_package_logger, PackageLogger

__logger__ = get_package_logger(__name__)


@dataclass(frozen=True)
class RegionCertificationResult:
    """Result container for a full-region certification pass."""
    success: bool
    counter_examples: list[th.Tensor]
    failed_regions: list[tuple[list[float], list[float]]]
    certified_regions: list[tuple[list[float], list[float]]]


def certify_with_crown(
    verifier_module: nn.Module,
    lb_box: list[float],
    ub_box: list[float],
    method: str = "alpha-crown",
    tolerance: float = 1e-6,
    device: th.device = th.device("cpu"),
    suppress_native_output: bool = True,
) -> tuple[bool, np.ndarray, float]:
    """Certify that verifier output <= tolerance on a single axis-aligned box."""
    lb = th.tensor(lb_box, device=device, dtype=th.float32).unsqueeze(0)
    ub = th.tensor(ub_box, device=device, dtype=th.float32).unsqueeze(0)
    center = (lb + ub) / 2
    counter_example = center.squeeze(0).detach().cpu().numpy()

    def _compute_upper_bound(bound_method: str) -> th.Tensor:
        ptb = PerturbationLpNorm(norm=float("inf"), x_L=lb, x_U=ub)
        bounded_input = BoundedTensor(center, ptb)
        lirpa_model = BoundedModule(verifier_module, center, device=device, verbose=False)

        ctx = PackageLogger.suppress_native_output() if suppress_native_output else nullcontext()
        with ctx:
            if bound_method == "alpha-crown":
                _, ub_out_local = lirpa_model.compute_bounds(x=(bounded_input,), method=bound_method)
            else:
                with th.no_grad():
                    _, ub_out_local = lirpa_model.compute_bounds(x=(bounded_input,), method=bound_method)
        return ub_out_local

    normalized_method = method.strip().lower()
    fallback_methods = [normalized_method]
    if normalized_method == "alpha-crown":
        fallback_methods.extend(["crown", "crown-ibp", "ibp"])

    ub_out: th.Tensor | None = None
    last_exc: Exception | None = None
    for candidate_method in fallback_methods:
        try:
            ub_out = _compute_upper_bound(candidate_method)
            break
        except Exception as exc:
            last_exc = exc
            __logger__.warning(
                "LiRPA method '%s' failed with %s on region lb=%s ub=%s",
                candidate_method,
                exc,
                lb_box,
                ub_box,
            )

    if ub_out is None:
        __logger__.error(
            "All LiRPA methods failed on region lb=%s ub=%s (last error: %s). "
            "Marking region as uncertified.",
            lb_box,
            ub_box,
            last_exc,
        )
        return False, counter_example, float("inf")

    max_upper = ub_out.max().item()
    is_certified = max_upper <= tolerance
    return is_certified, counter_example, max_upper


def _build_regions(
    config: LyapunovCertificationConfig,
) -> list[tuple[list[float], list[float]]]:
    if config.state_dim != 2:
        raise ValueError("certification currently supports state_dim == 2.")

    bounds = np.asarray(config.state_bounds, dtype=float)
    if bounds.ndim != 2 or bounds.shape != (2, config.state_dim):
        raise ValueError("state_bounds must be shape (2, nx) as [lbx, ubx].")
    if config.cert_step <= 0:
        raise ValueError("cert_step must be positive.")

    # Use maximum absolute bound for origin exclusion heuristics.
    train_diameter = np.max(np.abs(bounds))
    if config.cert_origin_exclusion is None:
        origin_exclusion = min(train_diameter * 0.01, 0.1)
    else:
        origin_exclusion = config.cert_origin_exclusion

    lbx = bounds[0]
    ubx = bounds[1]
    x_lb, x_ub = float(lbx[0]), float(ubx[0])
    y_lb, y_ub = float(lbx[1]), float(ubx[1])
    regions: list[tuple[list[float], list[float]]] = []
    for x in np.arange(x_lb, x_ub, config.cert_step):
        for y in np.arange(y_lb, y_ub, config.cert_step):
            if abs(x) < origin_exclusion and abs(y) < origin_exclusion:
                continue
            regions.append(([x, y], [x + config.cert_step, y + config.cert_step]))
    return regions


def certify_regions(
    verifier: nn.Module,
    regions: list[tuple[list[float], list[float]]],
    method: str,
    tolerance: float,
    device: th.device,
    collect_details: bool,
    suppress_native_output: bool = True,
) -> RegionCertificationResult:
    failed_regions: list[tuple[list[float], list[float]]] = []
    certified_regions: list[tuple[list[float], list[float]]] = []
    counter_examples: list[th.Tensor] = []

    for lb, ub in regions:
        safe, cex, _ = certify_with_crown(
            verifier_module=verifier,
            lb_box=lb,
            ub_box=ub,
            method=method,
            tolerance=tolerance,
            device=device,
            suppress_native_output=suppress_native_output,
        )
        if safe:
            if collect_details:
                certified_regions.append((lb, ub))
            continue

        # One-level subdivision to tighten bounds in hard regions.
        mid = ((np.array(lb) + np.array(ub)) / 2.0).tolist()
        sub_regions = [
            ([lb[0], lb[1]], [mid[0], mid[1]]),
            ([mid[0], lb[1]], [ub[0], mid[1]]),
            ([lb[0], mid[1]], [mid[0], ub[1]]),
            ([mid[0], mid[1]], [ub[0], ub[1]]),
        ]

        region_failed = False
        for sub_lb, sub_ub in sub_regions:
            sub_safe, sub_cex, _ = certify_with_crown(
                verifier_module=verifier,
                lb_box=sub_lb,
                ub_box=sub_ub,
                method=method,
                tolerance=tolerance,
                device=device,
                suppress_native_output=suppress_native_output,
            )
            if not sub_safe:
                region_failed = True
                if collect_details:
                    failed_regions.append((sub_lb, sub_ub))
                    counter_examples.append(th.tensor(sub_cex, dtype=th.float32))
            elif collect_details:
                certified_regions.append((sub_lb, sub_ub))

        if region_failed and not collect_details:
            return RegionCertificationResult(
                success=False,
                counter_examples=[],
                failed_regions=[],
                certified_regions=[],
            )

        if not region_failed and collect_details:
            certified_regions.append((lb, ub))

        if region_failed and collect_details and not counter_examples:
            counter_examples.append(th.tensor(cex, dtype=th.float32))

    return RegionCertificationResult(
        success=len(failed_regions) == 0,
        counter_examples=counter_examples,
        failed_regions=failed_regions,
        certified_regions=certified_regions,
    )


def is_rho_certified(
    verifier: ClosedLoopLyapunovConditionVerifier,
    rho: float,
    regions: list[tuple[list[float], list[float]]],
    method: str,
    tolerance: float,
    device: th.device,
    suppress_native_output: bool = True,
) -> bool:
    verifier.set_rho(rho)
    result = certify_regions(
        verifier=verifier,
        regions=regions,
        method=method,
        tolerance=tolerance,
        device=device,
        collect_details=False,
        suppress_native_output=suppress_native_output,
    )
    return result.success

    
def certify_lyapunov(
    policy_model: nn.Module,
    lyap_model: nn.Module,
    dyn_model: nn.Module,
    config: LyapunovCertificationConfig,
    rho_estimate: float,
    device: th.device = th.device("cpu"),
) -> tuple[float, RegionCertificationResult]:
    """Find the largest certified rho using scaling and bisection."""
    __logger__.info("Starting Lyapunov certification with method: %s", config.cert_method)
    
    bounds = _bounds_tensor(config.state_bounds, device)
    lbx, ubx = bounds[0], bounds[1]
    verifier = ClosedLoopLyapunovConditionVerifier(
        policy_model=policy_model,
        lyap_model=lyap_model,
        dyn_model=dyn_model,
        lbx=lbx,
        ubx=ubx,
        kappa=config.kappa,
        invariance_weight=config.invariance_weight,
        rho=max(config.rho_min, rho_estimate),
    ).to(device)
    verifier.eval()

    regions = _build_regions(config)
    method = config.cert_method.strip().lower()
    tolerance = config.condition_tolerance
    rho_scale = max(config.cert_rho_scaling, 1.01)

    initial_rho = max(config.rho_min, float(rho_estimate))
    initial_ok = is_rho_certified(
        verifier=verifier,
        rho=initial_rho,
        regions=regions,
        method=method,
        tolerance=tolerance,
        device=device,
        suppress_native_output=config.cert_suppress_native_output,
    )

    # Initial rho passed, scale up to find an upper bound.
    if initial_ok:
        rho_lo = initial_rho
        rho_up = initial_rho
        found_upper_failure = False
        
        with __logger__.tqdm(range(config.cert_max_scale_steps), desc="Scale up: upper rho") as pbar:
            for _ in pbar:
                trial = rho_up * rho_scale
                if is_rho_certified(
                    verifier,
                    trial,
                    regions,
                    method,
                    tolerance,
                    device,
                    suppress_native_output=config.cert_suppress_native_output,
                ):
                    rho_lo = trial
                    rho_up = trial
                else:
                    rho_up = trial
                    found_upper_failure = True
                    break

        if not found_upper_failure:
            verifier.set_rho(rho_lo)
            details = certify_regions(
                verifier=verifier,
                regions=regions,
                method=method,
                tolerance=tolerance,
                device=device,
                collect_details=True,
            )
            return rho_lo, details

    # Initial rho failed, scale down to find a certified rho.
    if not initial_ok:
        rho_up = initial_rho
        rho_lo: float | None = None
        trial = initial_rho
        
        with __logger__.tqdm(range(config.cert_max_scale_steps), desc="Scale down: lower rho") as pbar:
            for _ in pbar:
                trial = max(config.rho_min, trial / rho_scale)
                if is_rho_certified(
                    verifier,
                    trial,
                    regions,
                    method,
                    tolerance,
                    device,
                    suppress_native_output=config.cert_suppress_native_output,
                ):
                    rho_lo = trial
                    break
                rho_up = trial
                if trial <= config.rho_min:
                    break

        if rho_lo is None:
            rho_min_ok = is_rho_certified(
                verifier=verifier,
                rho=config.rho_min,
                regions=regions,
                method=method,
                tolerance=tolerance,
                device=device,
            )
            if not rho_min_ok:
                verifier.set_rho(config.rho_min)
                details = certify_regions(
                    verifier=verifier,
                    regions=regions,
                    method=method,
                    tolerance=tolerance,
                    device=device,
                    collect_details=True,
                )
                return config.rho_min, details
            rho_lo = config.rho_min
            rho_up = max(rho_up, initial_rho)

    # Bisection between rho_lo and rho_up to find the largest certified rho within tolerance.
    with __logger__.tqdm(range(config.cert_max_bisection_steps), desc="Bisection: max rho") as pbar:
        for _ in pbar:
            if rho_up - rho_lo <= config.cert_bisection_tol:
                break
            rho_mid = 0.5 * (rho_lo + rho_up)
            if is_rho_certified(
                verifier,
                rho_mid,
                regions,
                method,
                tolerance,
                device,
                suppress_native_output=config.cert_suppress_native_output,
            ):
                rho_lo = rho_mid
            else:
                rho_up = rho_mid

    verifier.set_rho(rho_lo)
    details = certify_regions(
        verifier=verifier,
        regions=regions,
        method=method,
        tolerance=tolerance,
        device=device,
        collect_details=True,
        suppress_native_output=config.cert_suppress_native_output,
    )
    return rho_lo, details