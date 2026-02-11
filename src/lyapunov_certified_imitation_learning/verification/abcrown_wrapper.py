import logging
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
    # 1. Definition der Eingabeschranken (Perturbation)
    lb = th.tensor(LB_box, device=device, dtype=th.float32).unsqueeze(0)
    ub = th.tensor(UB_box, device=device, dtype=th.float32).unsqueeze(0)
    
    # Mittelpunkt für die Initialisierung
    center = (lb + ub) / 2
    
    # Perturbation definieren (L-infinity Norm / Box Constraints)
    ptb = PerturbationLpNorm(norm=float("inf"), x_L=lb, x_U=ub)
    bounded_input = BoundedTensor(center, ptb)
    
    # 2. Initialisierung des BoundedModules (nur beim ersten Aufruf nötig, aber hier stateless für Einfachheit)
    # bound_opts={'relu': 'adaptive'} aktiviert alpha-CROWN Optimierungen für ReLUs
    lirpa_model = BoundedModule(verifier_module, center, device=device, verbose=False)
    
    # 3. Berechnung der Bounds (CROWN Methode)
    # Wir wollen die OBERE Schranke von V_next - V_curr wissen.
    with th.no_grad():
        # method='CROWN' (schnell, lineare Bounds) oder 'alpha-CROWN' (genauer, iterativ)
        # Für schnelle Iteration nehmen wir CROWN-IBP oder CROWN.
        lb_out, ub_out = lirpa_model.compute_bounds(x=(bounded_input,), method='CROWN')
    
    # 4. Überprüfung
    # Die Bedingung ist zertifiziert, wenn die obere Schranke < 0 ist (inkl. Toleranz)
    # Toleranz entspricht -0.00001 im Originalcode
    max_violation = ub_out.max().item()
    is_certified = max_violation < -1e-5
    
    # Falls nicht zertifiziert, geben wir einen "Worst-Case" Punkt zurück (Mittelpunkt oder über Gradienten refinebar)
    # CROWN liefert keine exakten Counter-Examples, sondern garantiert Bounds.
    # Wir nutzen den Mittelpunkt als Proxy für die Subdivision.
    counter_example = center.squeeze(0).cpu().numpy()
    
    return is_certified, counter_example


def certify_list_all(
    policy_model: nn.Module,
    lyap_model: nn.Module,
    dyn_model: nn.Module,
    config: LyapunovTrainingConfig,
    device: th.device = th.device("cpu"),
) -> tuple[list[th.Tensor], list[tuple[list[float], list[float]]], bool]:
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
    certify_counter_example = []
    
    __logger__.info("Start certification with CROWN on %d regions...", len(regions_to_check))
    
    for lb, ub in regions_to_check:
        is_safe, cex = certify_with_crown(verifier, lb, ub, device)
        
        if not is_safe:
            # If CROWN fails, subdivide (Subdivision / Branching)
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
    
    success = len(not_certified_regions) == 0
    return certify_counter_example, not_certified_regions, success