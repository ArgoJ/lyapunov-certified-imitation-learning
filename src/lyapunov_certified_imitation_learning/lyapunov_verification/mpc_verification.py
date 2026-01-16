import numpy as np
import scipy.linalg
from typing import Dict, Any, Tuple

from ..data_generation.mpc_data import MPCDataset
from ..utils.package_logger import PackageLogger

__logger__ = PackageLogger.get_logger(__name__)


def verify_mpc_asymptotic_stability(
    dataset: MPCDataset, 
    Q: np.ndarray, 
    R: np.ndarray,
    mode: str = 'empirical' 
) -> Dict[str, Any]:
    """
    Certifies Asymptotic Stability using rigorous criteria from Grüne's NMPC theory.
    
    Parameters
    ----------
    dataset : MPCDataset
        Dataset of closed-loop MPC trajectories.
    Q : np.ndarray
        State cost weight matrix.
    R : np.ndarray
        Input cost weight matrix.
    mode : str
        Verification mode.
    
    Modes
    -----
    - 'empirical': Checks the Relaxed Lyapunov Inequality (alpha-descent) on data.
    - 'theoretical': Estimates the overshoot bound (gamma) to check horizon sufficiency (Grüne's condition).
    - 'hybrid': Performs both.
    """
    results = {"certified": False, "details": {}}

    # 1. Feasibility Check
    infeasible_ids = get_infeasible_ids(dataset)
    results["details"]["n_total"] = len(dataset)
    results["details"]["n_infeasible"] = len(infeasible_ids)
    results["details"]["infeasible_ids"] = infeasible_ids

    feasible_dataset = filter_feasible(dataset)
    results["details"]["n_feasible"] = len(feasible_dataset)
    results["details"]["feasibility"] = len(feasible_dataset) > 0
    if len(feasible_dataset) == 0:
        __logger__.error("❌ No feasible runs available for verification.")
        return results

    # 2. Relaxed Lyapunov Certification (Empirical Alpha Check)
    # Checks V(x+) <= V(x) - alpha * l(x,u) [cite: 2032]
    alpha_obs, is_stable_descent = verify_relaxed_lyapunov_descent(feasible_dataset, Q, R)
    results["details"]["empirical_alpha"] = alpha_obs
    results["details"]["lyapunov_stable"] = is_stable_descent

    # 3. Grüne's Horizon Sufficiency (Unconstrained NMPC Check)
    # Checks if Horizon N > N_stabilizing based on estimated overshoot gamma 
    grune_passed = False
    if feasible_dataset[0].config.N > 0:
        N = feasible_dataset[0].config.N
        gamma_est, N_req, grune_passed = verify_grune_unconstrained_condition(feasible_dataset, Q, N)
        results["details"]["gamma_estimate"] = gamma_est
        results["details"]["required_horizon"] = N_req
        results["details"]["grune_condition_met"] = grune_passed

    # Final Certification Logic
    if mode == 'empirical':
        # Asymptotic stability certified if alpha > 0 [cite: 2035]
        results["certified"] = is_stable_descent and (alpha_obs > 1e-4)
    elif mode == 'theoretical':
        results["certified"] = grune_passed
    else:
        results["certified"] = is_stable_descent and grune_passed and (alpha_obs > 1e-4)

    if results["certified"]:
        __logger__.info(f"🏆 ASYMPTOTIC STABILITY CERTIFIED (alpha={alpha_obs:.4f}).")
    else:
        __logger__.warning(f"⚠️ STABILITY NOT CERTIFIED. (Alpha={alpha_obs:.4f}, Grüne check={grune_passed})")
        
    return results

def verify_relaxed_lyapunov_descent(
    dataset: MPCDataset, 
    Q: np.ndarray, 
    R: np.ndarray, 
    tol: float = 1e-6
) -> Tuple[float, bool]:
    """
    Verifies the Relaxed Dynamic Programming Inequality[cite: 2032]:
        V(f(x,u)) <= V(x) - alpha * l(x,u)
        
    Parameters
    ----------
    dataset : MPCDataset
        Dataset of closed-loop MPC trajectories.
    Q : np.ndarray
        State cost weight matrix.
    R : np.ndarray
        Input cost weight matrix.
    tol : float
        Tolerance for numerical checks.
        
    Returns
    -------
    min_alpha : float
        The worst-case performance index observed. 
        If alpha > 0, the system is asymptotically stable.
    is_stable : bool
        True if alpha > 0 (strictly decaying).
    
    Note
    ----
    This empirical check relies on the quality of the dataset and may not guarantee stability.  
    Only for quadratic costs l(x,u) = x'Qx + u'Ru.
    """
    alphas = []
    
    for data in dataset:
        traj = data.trajectory
        T = len(traj.inputs)
        
        for k in range(T - 1):
            x_k = traj.states[k]
            u_k = traj.inputs[k]

            if not (np.all(np.isfinite(x_k)) and np.all(np.isfinite(u_k))):
                continue
            
            # Stage cost l(x,u)
            l_xu = x_k.T @ Q @ x_k + u_k.T @ R @ u_k
            
            # Optimal value function V(x) approx by cost-to-go
            V_curr = traj.cost[k]
            V_next = traj.cost[k+1]

            if not (np.isfinite(V_curr) and np.isfinite(V_next)):
                continue
            
            # Skip points effectively at the origin to avoid division by zero noise
            if l_xu < tol:
                continue
            
            # Calculate observed alpha: alpha <= (V_k - V_{k+1}) / l(x,u)
            # Rearranging V_{k+1} <= V_k - alpha * l(x,u)
            descent = V_curr - V_next
            current_alpha = descent / l_xu
            alphas.append(current_alpha)
            
            # If V increases (alpha < 0), strict Lyapunov descent is violated
            if current_alpha < -1e-3: 
                __logger__.debug(f"Lyapunov violation at step {k}: V_curr={V_curr:.4f}, V_next={V_next:.4f}")

    if not alphas:
        return 0.0, False

    min_alpha = np.min(alphas)
    
    # Asymptotic stability requires strictly positive alpha
    is_asymptotically_stable = min_alpha > 0.0
    
    return min_alpha, is_asymptotically_stable


def verify_grune_unconstrained_condition(
    dataset: MPCDataset,
    Q: np.ndarray,
    N: int,
    min_cost_threshold: float = 1e-3
) -> Tuple[float, float, bool]:
    """
    Verifies stability for Unconstrained NMPC using the controllability bound gamma.
    Based on Grüne/Pannek Theorem.
    
    Parameters
    ----------
    dataset : MPCDataset
        Dataset of closed-loop MPC trajectories.
    Q : np.ndarray
        State cost weight matrix.
    N : int
        MPC horizon length.
    min_cost_threshold : float
        Minimum stage cost to consider a state "far enough" from the origin.
        
    Returns
    -------
    gamma : float
        Estimated overshoot bound.
    N_stabilizing : float
        Required horizon length for stability.
    is_stable : bool
        True if N > N_stabilizing.
    
    Assumption
    ----------
    The MPC problem is unconstrained and uses quadratic costs l(x,u) = x'Qx + u'Ru.
        V_N(x) <= gamma * l*(x)
        l*(x) ≈ l(x, 0)
    
    If alpha_N > 0 (calculated from gamma and N), stability is guaranteed.
    """
    gamma_values = []
    
    for data in dataset:
        traj = data.trajectory
        # Using first step of each trajectory to estimate V_N(x)
        if len(traj.states) > 0:
            x_0 = traj.states[0]
            V_N = traj.cost[0]

            if not (np.all(np.isfinite(x_0)) and np.isfinite(V_N)):
                continue
            
            # Smallest possible approximation: l*(x) ≈ l(x, 0)
            l_star = x_0.T @ Q @ x_0 # TODO: Change if used with a Non quadratic cost or NN
            
            if l_star > min_cost_threshold:
                gamma_values.append(V_N / l_star)
                
    if not gamma_values:
        __logger__.warning("⚠️ No states far enough from origin to estimate Gamma reliably.")
        return 0.0, 0.0, False

    # Estimate gamma as the worst-case ratio observed
    gamma = np.max(gamma_values)
    
    # N > 2 + (ln(gamma-1) / ln(gamma/(gamma-1)))
    if gamma <= 1.0 + 1e-6:
        # V_N <= l*, implies N=1 is sufficient
        return gamma, 1.0, True
        
    numerator = np.log(gamma - 1)
    denominator = np.log(gamma) - numerator
    N_stabilizing = 2 + (numerator / denominator)
    is_stable = N > N_stabilizing
    
    __logger__.info(f"📈 GRÜNE CHECK: Est. Gamma={gamma:.2f}, Req. N > {N_stabilizing:.2f}, Actual N={N}")
    return gamma, N_stabilizing, is_stable


def get_infeasible_ids(dataset: MPCDataset) -> list:
    """Gets a list of IDs corresponding to infeasible runs in the dataset."""
    infeasible_ids = []
    for data in dataset:
        if data.meta.status_codes and any(c != 0 for c in data.meta.status_codes):
            infeasible_ids.append(data.meta.id)
            continue
        if np.isnan(data.trajectory.states).any() or np.isnan(data.trajectory.inputs).any() or np.isnan(data.trajectory.cost).any():
            infeasible_ids.append(data.meta.id)
    return infeasible_ids


def filter_feasible(dataset: MPCDataset) -> MPCDataset:
    """Return a subset of the given dataset containing only feasible runs."""
    feasible_entries = []
    for data in dataset:
        if data.meta.status_codes and any(c != 0 for c in data.meta.status_codes):
            continue
        if np.isnan(data.trajectory.states).any() or np.isnan(data.trajectory.inputs).any() or np.isnan(data.trajectory.cost).any():
            continue
        feasible_entries.append(data)
    return MPCDataset(data_buffer=feasible_entries)