import numpy as np
import torch as th
import torch.nn as nn

# Auto_LiRPA Imports für CROWN
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

from ..models.lyapunov import ClosedLoopLyapunovVerifier
from ..training.lyapunov_config import LyapunovTrainingConfig
from ..utils.package_logger import PackageLogger

__logger__ = PackageLogger.get_logger(__name__)


def certify_with_crown(
    verifier_module: nn.Module,
    LB_box: list[float],
    UB_box: list[float],
    device: th.device = th.device("cpu"),
) -> tuple[bool, np.ndarray]:
    __logger__.debug(
        "CROWN certify call with bounds: lb=%s, ub=%s, device=%s",
        LB_box,
        UB_box,
        device,
    )
    # Define input bounds (Perturbation)
    lb = th.tensor(LB_box, device=device, dtype=th.float32).unsqueeze(0)
    ub = th.tensor(UB_box, device=device, dtype=th.float32).unsqueeze(0)
    
    # Center for initialization
    center = (lb + ub) / 2
    
    # Define perturbation (L-infinity Norm / Box Constraints)
    ptb = PerturbationLpNorm(norm=float("inf"), x_L=lb, x_U=ub)
    bounded_input = BoundedTensor(center, ptb)
    
    # Initialization of the BoundedModule (only needed on first call, but stateless here for simplicity)
    # bound_opts={'relu': 'adaptive'} enables alpha-CROWN optimizations for ReLUs
    lirpa_model = BoundedModule(verifier_module, center, device=device, verbose=False)
    
    # Compute bounds (CROWN method)
    # We want the UPPER bound of V_next - V_curr.
    with th.no_grad():
        # method='CROWN' (fast, linear bounds) or 'alpha-CROWN' (more accurate, iterative)
        # FFor quick iteration, we use CROWN-IBP or CROWN.
        lb_out, ub_out = lirpa_model.compute_bounds(x=(bounded_input,), method='CROWN')
    
    # Verification
    # The condition is certified if the upper bound < 0 (including tolerance)
    # Tolerance corresponds to -0.00001 in the original code
    max_violation = ub_out.max().item()
    is_certified = max_violation < -1e-5
    __logger__.debug(
        "CROWN bounds result: lb_max=%.6f, ub_max=%.6f, certified=%s",
        lb_out.max().item(),
        max_violation,
        is_certified,
    )
    
    # If not certified, return a "worst-case" point (center or refinable via gradients)
    # CROWN does not provide exact counter-examples, but guarantees bounds.
    # We use the center as a proxy for subdivision.
    counter_example = center.squeeze(0).cpu().numpy()
    
    return is_certified, counter_example


def certify_list_all(
    policy_model: nn.Module,
    lyap_model: nn.Module,
    dyn_model: nn.Module,
    config: LyapunovTrainingConfig,
    device: th.device = th.device("cpu"),
) -> tuple[
    list[th.Tensor],
    list[tuple[list[float], list[float]]],
    list[tuple[list[float], list[float]]],
    bool,
]:
    """Run CROWN certification on a grid of state-space boxes."""
    if config.state_dim != 2:
        raise ValueError("certify_list_all currently supports state_dim == 2.")
    if len(config.state_bounds) != config.state_dim:
        raise ValueError("state_bounds must match state_dim.")

    # Create Verifier
    verifier = ClosedLoopLyapunovVerifier(policy_model, lyap_model, dyn_model).to(device)
    verifier.eval()
    
    # Initial Regions
    regions_to_check = []
    
    step = config.cert_step
    train_diameter = max(config.state_bounds)
    if config.cert_origin_exclusion is None:
        err_origin = min(train_diameter * 0.01, 0.1)
    else:
        err_origin = config.cert_origin_exclusion

    theta_bound, theta_d_bound = config.state_bounds[0], config.state_bounds[1]
    for theta in np.arange(-theta_bound, theta_bound, step):
        for theta_d in np.arange(-theta_d_bound, theta_d_bound, step):
            # Ignore region around origin
            if abs(theta) < err_origin and abs(theta_d) < err_origin:
                continue
            regions_to_check.append(([theta, theta_d], [theta+step, theta_d+step]))

    not_certified_regions = []
    certified_regions = []
    certify_counter_example = []
    
    __logger__.debug(
        "Certification config: step=%.6f, origin_exclusion=%.6f, bounds=%s",
        step,
        err_origin,
        config.state_bounds,
    )
    
    for lb, ub in regions_to_check:
        is_safe, cex = certify_with_crown(verifier, lb, ub, device)
        
        if not is_safe:
            __logger__.debug("Region not certified, subdividing: lb=%s, ub=%s", lb, ub)
            # If alpha-CROWN fails, subdivide (Subdivision / Branching)
            # Simple heuristic: split box into 4 sub-boxes
            mid = (np.array(lb) + np.array(ub)) / 2
            sub_regions = [
                ([lb[0], lb[1]], [mid[0], mid[1]]),
                ([mid[0], lb[1]], [ub[0], mid[1]]),
                ([lb[0], mid[1]], [mid[0], ub[1]]),
                ([mid[0], mid[1]], [ub[0], ub[1]])
            ]
            
            for l_sub, u_sub in sub_regions:
                # Recursive or iterative check (here simply flat for demo)
                safe_sub, cex_sub = certify_with_crown(verifier, l_sub, u_sub, device)
                if not safe_sub:
                    not_certified_regions.append((l_sub, u_sub))
                    certify_counter_example.append(th.FloatTensor(cex_sub))
                    __logger__.debug(
                        "Sub-region not certified: lb=%s, ub=%s",
                        l_sub,
                        u_sub,
                    )
                else:
                    certified_regions.append((l_sub, u_sub))
        else:
            certified_regions.append((lb, ub))
    
    success = len(not_certified_regions) == 0
    __logger__.info(
        "Certification done: success=%s, failed_regions=%d",
        success,
        len(not_certified_regions),
    )
    return certify_counter_example, not_certified_regions, certified_regions, success