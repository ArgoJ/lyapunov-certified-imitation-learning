import torch as th
import torch.nn as nn
import numpy as np

from dataclasses import dataclass

from abcrown import (
    ABCrownSolver,
    ConfigBuilder,
    VerificationSpec,
    input_vars,
    output_vars,
)

from lcil.certification.models import (
    ClosedLoopLyapunovConditionVerifier,
)
from lcil.certification import LyapunovCertificationConfig
from lcil.lyapunov_learning import (
    LyapunovTrainingConfig,
    NeuralLyapunovCandidate,
)
from lcil.utils.package_logger import get_package_logger
from lcil.utils import ICNN, MLP

__logger__ = get_package_logger(__name__)


@dataclass(frozen=True)
class RegionCertificationResult:
    success: bool
    counter_examples: list[th.Tensor]
    failed_regions: list[tuple[list[float], list[float]]]
    certified_regions: list[tuple[list[float], list[float]]]


class DoubleIntegratorDynamics(nn.Module):
    """Discrete-time double integrator dynamics."""

    def __init__(self, dt: float = 0.1):
        super().__init__()
        self.dt = dt

    def forward(self, x: th.Tensor, u: th.Tensor) -> th.Tensor:
        if u.ndim == 1:
            u = u.unsqueeze(1)
        x_pos = x[:, 0:1]
        x_vel = x[:, 1:2]
        x_next_pos = x_pos + self.dt * x_vel
        x_next_vel = x_vel + self.dt * u
        return th.cat([x_next_pos, x_next_vel], dim=1)


def _build_regions(
    config: LyapunovCertificationConfig,
) -> list[tuple[list[float], list[float]]]:
    if config.state_dim != 2:
        raise ValueError("certification currently supports state_dim == 2.")
    if len(config.state_bounds) != config.state_dim:
        raise ValueError("state_bounds must match state_dim.")
    if config.cert_step <= 0:
        raise ValueError("cert_step must be positive.")

    train_diameter = max(config.state_bounds)
    if config.cert_origin_exclusion is None:
        origin_exclusion = min(train_diameter * 0.01, 0.1)
    else:
        origin_exclusion = config.cert_origin_exclusion

    x_bound, y_bound = config.state_bounds[0], config.state_bounds[1]
    regions: list[tuple[list[float], list[float]]] = []
    for x in np.arange(-x_bound, x_bound, config.cert_step):
        for y in np.arange(-y_bound, y_bound, config.cert_step):
            if abs(x) < origin_exclusion and abs(y) < origin_exclusion:
                continue
            regions.append(([x, y], [x + config.cert_step, y + config.cert_step]))
    return regions


def _certify_single_box_abcrown(
    verifier_module: nn.Module,
    lb_box: list[float],
    ub_box: list[float],
    tolerance: float,
    config: dict,
) -> tuple[bool, np.ndarray]:
    x = input_vars(2)
    y = output_vars(1)

    lower = th.tensor(lb_box, dtype=th.float32)
    upper = th.tensor(ub_box, dtype=th.float32)

    spec = VerificationSpec.build_spec(
        input_vars=x,
        output_vars=y,
        input_constraint=(x > lower) & (x < upper),
        output_constraint=(y[0] < tolerance),
    )

    result = ABCrownSolver(spec, verifier_module, config=config).solve()
    center = ((lower + upper) / 2.0).cpu().numpy()
    return bool(result.success), center


def _certify_regions_abcrown(
    verifier: nn.Module,
    regions: list[tuple[list[float], list[float]]],
    tolerance: float,
    abcrown_config: dict,
    collect_details: bool,
) -> RegionCertificationResult:
    failed_regions: list[tuple[list[float], list[float]]] = []
    certified_regions: list[tuple[list[float], list[float]]] = []
    counter_examples: list[th.Tensor] = []

    for lb, ub in regions:
        safe, cex = _certify_single_box_abcrown(
            verifier_module=verifier,
            lb_box=lb,
            ub_box=ub,
            tolerance=tolerance,
            config=abcrown_config,
        )
        if safe:
            if collect_details:
                certified_regions.append((lb, ub))
            continue

        mid = ((np.array(lb) + np.array(ub)) / 2.0).tolist()
        sub_regions = [
            ([lb[0], lb[1]], [mid[0], mid[1]]),
            ([mid[0], lb[1]], [ub[0], mid[1]]),
            ([lb[0], mid[1]], [mid[0], ub[1]]),
            ([mid[0], mid[1]], [ub[0], ub[1]]),
        ]

        region_failed = False
        for sub_lb, sub_ub in sub_regions:
            sub_safe, sub_cex = _certify_single_box_abcrown(
                verifier_module=verifier,
                lb_box=sub_lb,
                ub_box=sub_ub,
                tolerance=tolerance,
                config=abcrown_config,
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


def _is_rho_certified_abcrown(
    verifier: ClosedLoopLyapunovConditionVerifier,
    rho: float,
    regions: list[tuple[list[float], list[float]]],
    tolerance: float,
    abcrown_config: dict,
) -> bool:
    verifier.set_rho(rho)
    result = _certify_regions_abcrown(
        verifier=verifier,
        regions=regions,
        tolerance=tolerance,
        abcrown_config=abcrown_config,
        collect_details=False,
    )
    return result.success


def certify_lyapunov_with_abcrown(
    policy_model: nn.Module,
    lyap_model: nn.Module,
    dyn_model: nn.Module,
    cert_config: LyapunovCertificationConfig,
    rho_estimate: float,
    device: th.device,
) -> tuple[float, RegionCertificationResult]:
    bounds = th.tensor(cert_config.state_bounds, dtype=th.float32, device=device)
    verifier = ClosedLoopLyapunovConditionVerifier(
        policy_model=policy_model,
        lyap_model=lyap_model,
        dyn_model=dyn_model,
        bounds=bounds,
        kappa=cert_config.kappa,
        invariance_weight=cert_config.invariance_weight,
        rho=max(cert_config.rho_min, float(rho_estimate)),
    ).to(device)
    verifier.eval()

    if cert_config.cert_method.strip().lower() != "alpha-crown":
        raise ValueError(
            "This direct demo currently supports cert_method='alpha-crown' only."
        )

    abcrown_config = (
        ConfigBuilder.from_defaults()
        .set(general__device="cpu")
        .set(general__deterministic=True)
        ()
    )

    regions = _build_regions(cert_config)
    tolerance = cert_config.condition_tolerance
    rho_scale = max(cert_config.cert_rho_scaling, 1.01)

    initial_rho = max(cert_config.rho_min, float(rho_estimate))
    initial_ok = _is_rho_certified_abcrown(
        verifier=verifier,
        rho=initial_rho,
        regions=regions,
        tolerance=tolerance,
        abcrown_config=abcrown_config,
    )

    if initial_ok:
        rho_lo = initial_rho
        rho_up = initial_rho
        found_upper_failure = False
        for _ in range(cert_config.cert_max_scale_steps):
            trial = rho_up * rho_scale
            if _is_rho_certified_abcrown(
                verifier,
                trial,
                regions,
                tolerance,
                abcrown_config,
            ):
                rho_lo = trial
                rho_up = trial
            else:
                rho_up = trial
                found_upper_failure = True
                break

        if not found_upper_failure:
            verifier.set_rho(rho_lo)
            details = _certify_regions_abcrown(
                verifier=verifier,
                regions=regions,
                tolerance=tolerance,
                abcrown_config=abcrown_config,
                collect_details=True,
            )
            return rho_lo, details
    else:
        rho_up = initial_rho
        rho_lo: float | None = None
        trial = initial_rho
        for _ in range(cert_config.cert_max_scale_steps):
            trial = max(cert_config.rho_min, trial / rho_scale)
            if _is_rho_certified_abcrown(
                verifier,
                trial,
                regions,
                tolerance,
                abcrown_config,
            ):
                rho_lo = trial
                break
            rho_up = trial
            if trial <= cert_config.rho_min:
                break

        if rho_lo is None:
            rho_min_ok = _is_rho_certified_abcrown(
                verifier=verifier,
                rho=cert_config.rho_min,
                regions=regions,
                tolerance=tolerance,
                abcrown_config=abcrown_config,
            )
            if not rho_min_ok:
                verifier.set_rho(cert_config.rho_min)
                details = _certify_regions_abcrown(
                    verifier=verifier,
                    regions=regions,
                    tolerance=tolerance,
                    abcrown_config=abcrown_config,
                    collect_details=True,
                )
                return cert_config.rho_min, details
            rho_lo = cert_config.rho_min
            rho_up = max(rho_up, initial_rho)

    for _ in range(cert_config.cert_max_bisection_steps):
        if rho_up - rho_lo <= cert_config.cert_bisection_tol:
            break
        rho_mid = 0.5 * (rho_lo + rho_up)
        if _is_rho_certified_abcrown(
            verifier,
            rho_mid,
            regions,
            tolerance,
            abcrown_config,
        ):
            rho_lo = rho_mid
        else:
            rho_up = rho_mid

    verifier.set_rho(rho_lo)
    details = _certify_regions_abcrown(
        verifier=verifier,
        regions=regions,
        tolerance=tolerance,
        abcrown_config=abcrown_config,
        collect_details=True,
    )
    return rho_lo, details


def main() -> None:
    device = th.device("cpu")

    policy_model = MLP([2, 16, 16, 1], ["tanh", "tanh", "identity"]).to(device)
    lyap_feature = ICNN([2, 32, 1], ["relu", "identity"]).to(device)
    lyap_model = NeuralLyapunovCandidate(
        feature_net=lyap_feature,
        state_dim=2,
        epsilon=1e-3,
    ).to(device)
    dyn_model = DoubleIntegratorDynamics(dt=0.1).to(device)

    # Optional: uncomment if you want to verify trained checkpoints.
    # policy_model.load_state_dict(
    #     th.load(
    #         "results/models/double_integrator_lyap_policy_5912354.pt",
    #         map_location=device,
    #     )
    # )
    # lyap_model.load_state_dict(
    #     th.load(
    #         "results/models/double_integrator_lyap_lyap_5912354.pt",
    #         map_location=device,
    #     )
    # )

    training_config = LyapunovTrainingConfig(
        state_dim=2,
        state_bounds=(2.0, 2.0),
        sample_size=1000,
        batch_size=512,
        outer_epochs=10,
        steps_per_epoch=5,
        counterexample_every=10,
        learning_rate=1e-2,
        seed=5912354,
        kappa=0.05,
        invariance_weight=1.0,
        rho_growth_gamma=1.1,
        roa_weight=0.1,
        l1_weight=1e-6,
    )

    cert_config = LyapunovCertificationConfig.from_training_config(
        training_config,
        cert_step=1.0,
        cert_origin_exclusion=None,
        cert_rho_scaling=1.2,
        cert_bisection_tol=1e-3,
        cert_max_scale_steps=20,
        cert_max_bisection_steps=40,
        cert_method="alpha-crown",
    )

    rho_estimate = 0.2
    rho_certified, cert_results = certify_lyapunov_with_abcrown(
        policy_model=policy_model,
        lyap_model=lyap_model,
        dyn_model=dyn_model,
        cert_config=cert_config,
        rho_estimate=rho_estimate,
        device=device,
    )

    __logger__.info("Certified rho (ABCrown API): %.6f", rho_certified)
    __logger__.info(
        "Certified regions: %d | Failed regions: %d | Counterexamples: %d",
        len(cert_results.certified_regions),
        len(cert_results.failed_regions),
        len(cert_results.counter_examples),
    )


if __name__ == "__main__":
    main()
