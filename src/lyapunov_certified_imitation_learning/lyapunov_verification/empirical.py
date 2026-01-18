import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List

from ..data_generation.mpc_data import MPCData, MPCConfig, MPCDataset

class EmpiricalStabilityVerifier:
    """
    A tool to rigorously check stability and performance properties of NMPC trajectories
    based on the optimal value function V_N as a Lyapunov function.
    
    Theory based on:
    Lars Grüne, Nonlinear Model Predictive Control (2015)
    """

    def __init__(self, entry: MPCData, Q: Optional[np.ndarray] = None, R: Optional[np.ndarray] = None):
        """
        Initialize with a single dataset entry containing trajectory and config.
        """
        self.traj = entry.trajectory
        self.cfg = entry.config
        self.meta = entry.meta

        # Optional overrides (used by dataset-level certification helpers)
        self.Q = np.asarray(Q) if Q is not None else self.cfg.Q
        self.R = np.asarray(R) if R is not None else self.cfg.R
        
        # Ensure goal state is defined (default to zero origin)
        self.x_star = self.cfg.goal_state if self.cfg.goal_state is not None else np.zeros(self.traj.states.shape[1])
        
        # Pre-compute metrics for analysis
        self.valid = self._validate_data_integrity()

    def _validate_data_integrity(self) -> bool:
        """Check if OCP predictions (solved_states) are available for Lyapunov calculation."""
        if self.traj.solved_states is None or self.traj.solved_inputs is None:
            return False
        return True

    def _stage_cost(self, x: np.ndarray, u: np.ndarray) -> float:
        """
        Computes l(x, u). Assumes Quadratic Cost standard in linear MPC.
        l(x,u) = ||x - x_*||^2_Q + ||u||^2_R[cite: 1823].
        """
        dx = x - self.x_star
        # In the provided OCP setup, u is minimized directly (not u - u_ref), 
        # usually assuming u_ref=0 for regulation.
        du = u 
        
        # J = x'Qx + u'Ru
        cost = dx.T @ self.Q @ dx + du.T @ self.R @ du
        return float(cost)

    def _terminal_cost(self, x: np.ndarray) -> float:
        """
        Computes F(x) or V_f(x).
        Used for terminal cost calculation if Qf/P is provided[cite: 2210].
        """
        Qf = self._effective_terminal_cost_matrix(self.cfg.Qf)
        if Qf is None:
            return 0.0
        
        dx = x - self.x_star
        return float(dx.T @ Qf @ dx)

    @staticmethod
    def _effective_terminal_cost_matrix(Qf: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if Qf is None:
            return None
        Qf = np.asarray(Qf)
        if Qf.size == 0:
            return None
        if not np.any(np.abs(Qf) > 0.0):
            return None
        return Qf

    @staticmethod
    def _has_terminal_bounds(cfg: MPCConfig) -> bool:
        tb = cfg.terminal_state_bounds
        if tb is None:
            return False
        tb = np.asarray(tb)
        if tb.size == 0:
            return False
        return bool(np.any(np.isfinite(tb)))

    @staticmethod
    def _is_entry_feasible(entry: MPCData) -> bool:
        # Prefer explicit feasibility flag if present
        if hasattr(entry.trajectory, "feasible") and (entry.trajectory.feasible is False):
            return False

        # Non-zero solver status codes indicate failure
        if entry.meta is not None and getattr(entry.meta, "status_codes", None):
            try:
                if any(int(c) != 0 for c in entry.meta.status_codes):
                    return False
            except Exception:
                return False

        # NaNs in key arrays indicate an invalid run
        t = entry.trajectory
        if np.isnan(t.states).any() or np.isnan(t.inputs).any() or np.isnan(t.cost).any():
            return False
        return True

    def min_observed_alpha(self, tolerance: float = 1e-5) -> Optional[float]:
        """Return the minimum observed $\alpha$ along the closed loop.

        Computes
            \alpha_obs(n) = (V_N(x(n)) - V_N(x(n+1))) / l(x(n), u(n))
        over all timesteps with non-trivial stage cost.
        """
        if not self.valid:
            return None

        T_sim = int(self.traj.inputs.shape[0])
        min_alpha: float = float("inf")
        found = False

        for n in range(T_sim - 1):
            l_curr = self._stage_cost(self.traj.states[n], self.traj.inputs[n])
            if not np.isfinite(l_curr) or l_curr <= tolerance:
                continue

            V_curr = self.compute_optimal_value_function(n)
            V_next = self.compute_optimal_value_function(n + 1)
            if not (np.isfinite(V_curr) and np.isfinite(V_next)):
                continue

            alpha_obs = (V_curr - V_next) / l_curr
            if not np.isfinite(alpha_obs):
                continue
            found = True
            if alpha_obs < min_alpha:
                min_alpha = float(alpha_obs)

        return float(min_alpha) if found else None

    @classmethod
    def verify_mpc_asymptotic_stability(
        cls,
        dataset: MPCDataset,
        Q: np.ndarray,
        R: np.ndarray,
        mode: str = "empirical",
        alpha_threshold: float = 1e-4,
        tol: float = 1e-6,
        min_cost_threshold: float = 1e-3,
    ) -> Dict[str, Any]:
        """Dataset-level certification (no duplicate logic).

        This is the canonical implementation and is re-exported from
        `lyapunov_certified_imitation_learning.lyapunov_verification`.

        Returns a dict compatible with the old `verify_mpc_asymptotic_stability` output.
        """
        results: Dict[str, Any] = {"certified": False, "details": {}}

        # 1) Feasibility filtering
        infeasible_ids: List[int] = []
        feasible_entries: List[MPCData] = []
        for entry in dataset:
            entry_id = int(getattr(entry.meta, "id", -1))
            if cls._is_entry_feasible(entry):
                feasible_entries.append(entry)
            else:
                infeasible_ids.append(entry_id)

        results["details"]["n_total"] = int(len(dataset))
        results["details"]["n_infeasible"] = int(len(infeasible_ids))
        results["details"]["infeasible_ids"] = infeasible_ids
        results["details"]["n_feasible"] = int(len(feasible_entries))
        results["details"]["feasibility"] = bool(len(feasible_entries) > 0)
        if not feasible_entries:
            return results

        feasible_dataset = MPCDataset(data_buffer=feasible_entries)

        # 2) Relaxed DP / Lyapunov decrease (empirical)
        alpha_values: List[float] = []
        for entry in feasible_dataset:
            verifier = cls(entry, Q=Q, R=R)
            min_alpha = verifier.min_observed_alpha(tolerance=tol)
            if min_alpha is None:
                continue
            alpha_values.append(float(min_alpha))

        if not alpha_values:
            results["details"]["empirical_alpha"] = 0.0
            results["details"]["lyapunov_stable"] = False
            empirical_ok = False
            empirical_alpha = 0.0
        else:
            empirical_alpha = float(np.min(alpha_values))
            empirical_ok = bool(empirical_alpha > 0.0)
            results["details"]["empirical_alpha"] = empirical_alpha
            results["details"]["lyapunov_stable"] = empirical_ok

        # 3) Grüne horizon condition (only for the basic scheme w/o terminal ingredients)
        cfg0 = feasible_entries[0].config
        has_terminal_cost = cls._effective_terminal_cost_matrix(cfg0.Qf) is not None
        has_terminal_bounds = cls._has_terminal_bounds(cfg0)

        applicability = "applicable"
        if has_terminal_cost or has_terminal_bounds:
            applicability = "not_applicable: dataset includes terminal cost/bounds; no-terminal theorem does not directly apply"

        N = int(cfg0.N)

        def l_star(x: np.ndarray, x_star: np.ndarray) -> float:
            dx = x - x_star
            return float(dx.T @ Q @ dx)

        gamma_values: List[float] = []
        for entry in feasible_dataset:
            verifier = cls(entry, Q=Q, R=R)
            if not verifier.valid:
                continue

            x0 = np.asarray(entry.trajectory.states[0], dtype=float)
            if not np.all(np.isfinite(x0)):
                continue
            v0 = verifier.compute_optimal_value_function(0)
            if not np.isfinite(v0):
                continue

            l0 = l_star(x0, verifier.x_star)
            if not np.isfinite(l0) or l0 <= min_cost_threshold:
                continue
            gamma_values.append(float(v0 / l0))

        if not gamma_values:
            gamma = 0.0
            alpha_N = 0.0
            N_req = 0.0
            grune_passed = False
            grune_note = "insufficient_data"
        else:
            gamma = float(np.max(gamma_values))
            if N < 2:
                alpha_N = 0.0
                N_req = 0.0
                grune_passed = False
                grune_note = "not_applicable: N<2"
            elif gamma <= 1.0 + 1e-10:
                alpha_N = 1.0
                N_req = 1.0
                grune_passed = True
                grune_note = applicability
            else:
                denom = (gamma ** (N - 1)) - ((gamma - 1.0) ** (N - 1))
                if denom <= 0.0 or not np.isfinite(denom):
                    alpha_N = 0.0
                    N_req = float("inf")
                    grune_passed = False
                    grune_note = "numerical_failure"
                else:
                    alpha_N = float(1.0 - (((gamma - 1.0) ** N) / denom))

                    numerator = float(np.log(gamma - 1.0))
                    denominator = float(np.log(gamma) - np.log(gamma - 1.0))
                    N_req = float(2.0 + (numerator / denominator)) if denominator > 0 else float("inf")
                    grune_passed = bool(alpha_N > 0.0)
                    grune_note = applicability

        results["details"]["grune_applicability"] = grune_note
        results["details"]["grune_used"] = bool(grune_note == "applicable")
        results["details"]["gamma_estimate"] = float(gamma)
        results["details"]["alpha_N_estimate"] = float(alpha_N)
        results["details"]["required_horizon"] = float(N_req)
        results["details"]["grune_condition_met"] = bool(grune_passed)

        # Final certification logic
        grune_effective = bool(grune_passed or (grune_note != "applicable"))
        if mode == "empirical":
            results["certified"] = bool(empirical_ok and (empirical_alpha > alpha_threshold))
        elif mode == "theoretical":
            results["certified"] = bool(grune_passed and (grune_note == "applicable"))
        else:
            results["certified"] = bool(empirical_ok and grune_effective and (empirical_alpha > alpha_threshold))

        return results

    def compute_optimal_value_function(self, time_step: int) -> float:
        """
        Reconstructs V_N(x(n)) from the OCP predictions stored in the dataset.
        
        V_N(x_0) := inf sum(l(x_u(k), u(k)))[cite: 1956].
        
        Parameters
        ----------
        time_step : int
            The closed-loop simulation step 'n'.
            
        Returns
        -------
        float : The value of the cost function at this step.
        """
        if not self.valid:
            return np.nan

        # Extract predictions for the horizon N at time_step
        # solved_states shape: (T, N+1, nx)
        # solved_inputs shape: (T, N, nu)
        pred_x = self.traj.solved_states[time_step] 
        pred_u = self.traj.solved_inputs[time_step]
        
        # Calculate V_N manually to ensure consistency with Python Q/R matrices
        # vs potentially different scaling in Acados solver cost output.
        V_N = 0.0
        N = self.cfg.N
        
        for k in range(N):
            V_N += self._stage_cost(pred_x[k], pred_u[k])
            
        # Add terminal cost F(x(N)) [cite: 2210]
        V_N += self._terminal_cost(pred_x[N])
        
        return V_N

    def check_lyapunov_decrease(self, tolerance: float = 1e-5, alpha_min: float = 0.0) -> pd.DataFrame:
        """
        Verifies the Lyapunov decrease condition for the MPC closed loop.
        
        Grüne relaxed DP / Lyapunov inequality along the closed loop:
            V_N(x(n+1)) <= V_N(x(n)) - alpha * l(x(n), u(n))
        where alpha in (0,1].
        
        This implementation estimates the *observed* alpha at each step as
            alpha_obs(n) = (V_N(x(n)) - V_N(x(n+1))) / l(x(n), u(n))
        and checks alpha_obs(n) >= alpha_min up to numerical tolerance.
        
        Returns
        -------
        pd.DataFrame : Analysis per timestep.
        """
        if not self.valid:
            return pd.DataFrame()

        results = []
        T_sim = self.traj.inputs.shape[0]

        for n in range(T_sim - 1): # Can't calc V_N for the very last state if inputs missing
            
            # 1. Get Value Function at current step n
            V_curr = self.compute_optimal_value_function(n)
            
            # 2. Get Value Function at next step n+1 (closed loop)
            # This represents V_N(x^+) in the notation [cite: 1926]
            V_next = self.compute_optimal_value_function(n + 1)
            
            # 3. Calculate actual stage cost incurred l(x(n), u(n))
            l_curr = self._stage_cost(self.traj.states[n], self.traj.inputs[n])
            
            # 4. Compute observed alpha and check relaxed decrease
            if l_curr <= tolerance:
                alpha_obs = np.nan
                is_decaying = True  # near equilibrium / tiny stage cost
            else:
                alpha_obs = (V_curr - V_next) / l_curr
                # V_next - V_curr + alpha*l_curr <= 0  <=> alpha <= (V_curr - V_next)/l
                # We require the observed alpha to be >= alpha_min.
                is_decaying = (alpha_obs + 0.0) >= (alpha_min - 1e-12)
            
            # Check if we are essentially at the goal (Lyapunov decay not required at equilibrium)
            dist_to_goal = np.linalg.norm(self.traj.states[n] - self.x_star)
            at_equilibrium = dist_to_goal < 1e-3

            results.append({
                "time_step": n,
                "V_curr": V_curr,
                "V_next": V_next,
                "stage_cost": l_curr,
                "alpha_obs": float(alpha_obs) if np.isfinite(alpha_obs) else np.nan,
                "is_decaying": bool(is_decaying or at_equilibrium),
                "dist_to_goal": dist_to_goal
            })
            
        return pd.DataFrame(results)

    def check_infinite_horizon_performance(self) -> Dict[str, float]:
        """
        Estimates the Infinite Horizon Performance.
        
        According to Relaxed Dynamic Programming:
        J_inf_cl(x0, mu_N) <= V_N(x0) / alpha
        
        For alpha=1 we obtain J_inf_cl(x0, mu_N) <= V_N(x0).
        Here we approximate J_inf_cl by the *realized* sum of stage costs along the
        simulated closed loop (finite-time truncation).
        """
        if not self.valid:
            return {}

        # Initial predicted cost
        V_0 = self.compute_optimal_value_function(0)
        
        # Realized closed-loop cost (sum of stage costs over simulation)
        J_cl = 0.0
        for k in range(self.traj.inputs.shape[0]):
            x_k = self.traj.states[k]
            u_k = self.traj.inputs[k]
            if not (np.all(np.isfinite(x_k)) and np.all(np.isfinite(u_k))):
                break
            J_cl += self._stage_cost(x_k, u_k)
        
        # If simulation was cut short, J_cl is a lower bound on infinite horizon
        return {
            "V_N_initial": V_0,
            "J_closed_loop": J_cl,
            "performance_ratio": J_cl / V_0 if V_0 > 1e-6 else 1.0,
            "satisfied_bound": J_cl <= V_0 * 1.05 # 5% tolerance
        }

    def check_terminal_set_invariance(self, tolerance: float = 0.1) -> pd.DataFrame:
        """
        Checks if the end of the predicted horizon lands in the terminal set.
        For MPC with terminal constraints x(N) = x_*, or x(N) in X_0[cite: 2119, 2209].
        
        In this linear setup, we often assume a terminal set X_f (level set of P).
        We check if the predicted final state is close to x_star.
        """
        if not self.valid:
            return pd.DataFrame()
            
        N = self.cfg.N
        T_sim = self.traj.inputs.shape[0]
        results = []

        for n in range(T_sim):
            pred_terminal_x = self.traj.solved_states[n][N]
            dist_terminal = np.linalg.norm(pred_terminal_x - self.x_star)
            
            results.append({
                "time_step": n,
                "terminal_dist": dist_terminal,
                "in_terminal_set": dist_terminal < tolerance
            })
            
        return pd.DataFrame(results)

    def summary(self) -> Dict[str, Any]:
        """
        Aggregates all checks into a final stability report.
        """
        if not self.valid:
            return {"status": "INVALID_DATA"}
            
        lyap_df = self.check_lyapunov_decrease()
        perf = self.check_infinite_horizon_performance()
        term = self.check_terminal_set_invariance()
        
        # Decay violation rate (ignoring very small numbers close to zero)
        decay_violations = lyap_df[~lyap_df["is_decaying"]]
        
        is_practically_stable = (
            lyap_df["dist_to_goal"].iloc[-1] < 1e-1  # Final state near goal
            and 
            len(decay_violations) / len(lyap_df) < 0.1 # Less than 10% non-decreasing steps
        )

        return {
            "is_stable": is_practically_stable,
            "avg_alpha_obs": float(lyap_df["alpha_obs"].mean()) if (not lyap_df.empty and "alpha_obs" in lyap_df) else None,
            "min_alpha_obs": float(lyap_df["alpha_obs"].min()) if (not lyap_df.empty and "alpha_obs" in lyap_df) else None,
            "performance_bound_satisfied": perf.get("satisfied_bound", False),
            "performance_ratio": perf.get("performance_ratio", 0.0),
            "avg_terminal_error": term["terminal_dist"].mean(),
            "initial_V_N": perf.get("V_N_initial", 0.0),
            "final_dist": lyap_df["dist_to_goal"].iloc[-1] if not lyap_df.empty else None
        }

    @classmethod
    def summarize_dataset(
        cls,
        dataset: MPCDataset,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        only_feasible: bool = True,
        alpha_threshold: float = 1e-4,
        min_cost_threshold: float = 1e-3,
    ) -> Dict[str, Any]:
        """Run the empirical checker over an entire dataset and aggregate results.

        This is intended as a *diagnostic* companion to
        `EmpiricalStabilityVerifier.verify_mpc_asymptotic_stability`, providing
        richer statistics (rates/means/minima) across trajectories.
        """
        n_total = int(len(dataset))

        # Determine Grüne applicability (based on configuration of first considered entry)
        cfg0: Optional[MPCConfig] = None

        n_considered = 0
        n_invalid = 0
        n_stable = 0
        n_perf_bound_ok = 0

        avg_alpha_vals: List[float] = []
        min_alpha_vals: List[float] = []
        perf_ratio_vals: List[float] = []
        terminal_err_vals: List[float] = []
        final_dist_vals: List[float] = []

        infeasible_ids: List[int] = []

        for entry in dataset:
            if only_feasible and not cls._is_entry_feasible(entry):
                infeasible_ids.append(int(getattr(entry.meta, "id", -1)))
                continue

            if cfg0 is None:
                cfg0 = entry.config

            n_considered += 1
            verifier = cls(entry, Q=Q, R=R)
            rep = verifier.summary()
            if rep.get("status") == "INVALID_DATA":
                n_invalid += 1
                continue

            if bool(rep.get("is_stable", False)):
                n_stable += 1
            if bool(rep.get("performance_bound_satisfied", False)):
                n_perf_bound_ok += 1

            avg_alpha = rep.get("avg_alpha_obs")
            if avg_alpha is not None and np.isfinite(avg_alpha):
                avg_alpha_vals.append(float(avg_alpha))

            min_alpha = rep.get("min_alpha_obs")
            if min_alpha is not None and np.isfinite(min_alpha):
                min_alpha_vals.append(float(min_alpha))

            pr = rep.get("performance_ratio")
            if pr is not None and np.isfinite(pr):
                perf_ratio_vals.append(float(pr))

            te = rep.get("avg_terminal_error")
            if te is not None and np.isfinite(te):
                terminal_err_vals.append(float(te))

            fd = rep.get("final_dist")
            if fd is not None and np.isfinite(fd):
                final_dist_vals.append(float(fd))

        n_valid = n_considered - n_invalid

        def _safe_mean(values: List[float]) -> Optional[float]:
            return float(np.mean(values)) if values else None

        def _safe_min(values: List[float]) -> Optional[float]:
            return float(np.min(values)) if values else None

        def _safe_max(values: List[float]) -> Optional[float]:
            return float(np.max(values)) if values else None

        empirical_alpha_global = _safe_min(min_alpha_vals)
        empirical_certified = bool(empirical_alpha_global is not None and empirical_alpha_global > alpha_threshold)

        grune_applicability = "insufficient_data"
        grune_used = False
        gamma_estimate: Optional[float] = None
        alpha_N_estimate: Optional[float] = None
        grune_condition_met: Optional[bool] = None

        if cfg0 is not None:
            has_terminal_cost = cls._effective_terminal_cost_matrix(cfg0.Qf) is not None
            has_terminal_bounds = cls._has_terminal_bounds(cfg0)
            if has_terminal_cost or has_terminal_bounds:
                grune_applicability = (
                    "not_applicable: dataset includes terminal cost/bounds; no-terminal theorem does not directly apply"
                )
            else:
                grune_applicability = "applicable"
                grune_used = True

                # Estimate gamma via V_N(x0) / l*(x0) using stored predictions.
                # We only consider entries that have valid predictions.
                gamma_values: List[float] = []
                Q_use = np.asarray(Q) if Q is not None else np.asarray(cfg0.Q)

                for entry in dataset:
                    if only_feasible and not cls._is_entry_feasible(entry):
                        continue

                    verifier = cls(entry, Q=Q_use, R=R)
                    if not verifier.valid:
                        continue

                    x0 = np.asarray(entry.trajectory.states[0], dtype=float)
                    if not np.all(np.isfinite(x0)):
                        continue

                    v0 = verifier.compute_optimal_value_function(0)
                    if not np.isfinite(v0):
                        continue

                    dx0 = x0 - verifier.x_star
                    l0 = float(dx0.T @ Q_use @ dx0)
                    if not np.isfinite(l0) or l0 <= min_cost_threshold:
                        continue

                    gamma_values.append(float(v0 / l0))

                N = int(cfg0.N)
                if gamma_values and N >= 2:
                    gamma_estimate = float(np.max(gamma_values))
                    gamma = float(gamma_estimate)

                    if gamma <= 1.0 + 1e-10:
                        alpha_N_estimate = 1.0
                        grune_condition_met = True
                    else:
                        denom = (gamma ** (N - 1)) - ((gamma - 1.0) ** (N - 1))
                        if denom > 0.0 and np.isfinite(denom):
                            alpha_N_estimate = float(1.0 - (((gamma - 1.0) ** N) / denom))
                            grune_condition_met = bool(alpha_N_estimate > 0.0)
                        else:
                            alpha_N_estimate = None
                            grune_condition_met = False
                else:
                    gamma_estimate = None
                    alpha_N_estimate = None
                    grune_condition_met = False

        return {
            "n_total": n_total,
            "n_considered": int(n_considered),
            "n_valid": int(n_valid),
            "n_invalid": int(n_invalid),
            "n_skipped_infeasible": int(len(infeasible_ids)),
            "skipped_infeasible_ids": infeasible_ids,
            "stable_rate": (float(n_stable) / float(n_valid)) if n_valid > 0 else None,
            "performance_bound_rate": (float(n_perf_bound_ok) / float(n_valid)) if n_valid > 0 else None,
            "avg_alpha_obs_mean": _safe_mean(avg_alpha_vals),
            "min_alpha_obs_global": _safe_min(min_alpha_vals),
            "empirical_certified": bool(empirical_certified),
            "grune_used": bool(grune_used),
            "grune_applicability": grune_applicability,
            "gamma_estimate": gamma_estimate,
            "alpha_N_estimate": alpha_N_estimate,
            "grune_condition_met": grune_condition_met,
            "performance_ratio_mean": _safe_mean(perf_ratio_vals),
            "avg_terminal_error_mean": _safe_mean(terminal_err_vals),
            "avg_terminal_error_max": _safe_max(terminal_err_vals),
            "final_dist_mean": _safe_mean(final_dist_vals),
            "final_dist_max": _safe_max(final_dist_vals),
        }