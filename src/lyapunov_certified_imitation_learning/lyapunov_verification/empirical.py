import numpy as np
from typing import Dict, Any, Generator, Optional, List
from dataclasses import dataclass, field


from ..utils.package_logger import PackageLogger
from .stability_report import StabilityReport
from ..data_generation.mpc_data import MPCData, MPCConfig, MPCDataset


__logger__ = PackageLogger.get_logger(__name__)
    


@dataclass
class LyapunovDecreaseReport(StabilityReport):
    method: str = "Lyapunov Decrease"
    empirical_alpha: float = float("nan")
    applicability: bool = True # Always applicable

@dataclass
class GrüneHorizonReport(StabilityReport):
    method: str = "Grüne Horizon Condition"
    gamma_estimate: float = float("nan")
    alpha_N_estimate: float = float("nan")
    required_horizon: float = float("nan")

@dataclass
class InfiniteHorizonPerformanceReport:
    V_N_initial: float = float("nan")
    J_closed_loop: float = float("nan")
    performance_ratio: float = float("nan")
    satisfied_bound: bool = False

@dataclass
class TerminalSetInvarianceReport:
    terminal_dist: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    is_terminal_set: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=bool))

@dataclass
class DistributionSummary:
    n: int = 0
    min: float = float("nan")
    p05: float = float("nan")
    p25: float = float("nan")
    median: float = float("nan")
    p75: float = float("nan")
    p95: float = float("nan")
    max: float = float("nan")
    mean: float = float("nan")
    
    @classmethod
    def from_stamples(cls, values: List[float]) -> None:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return cls()

        q = np.percentile(arr, [0.0, 5.0, 25.0, 50.0, 75.0, 95.0, 100.0])
        return cls(
            n=int(arr.size),
            min=float(q[0]),
            p05=float(q[1]),
            p25=float(q[2]),
            median=float(q[3]),
            p75=float(q[4]),
            p95=float(q[5]),
            max=float(q[6]),
            mean=float(np.mean(arr))
        )

@dataclass
class EmpiricalDatasetDetails:
    # Counts
    n_total: int = 0
    n_feasible: int = 0
    n_with_predictions: int = 0
    n_used: int = 0
    n_missing_predictions: int = 0

    alpha_min: DistributionSummary = field(default_factory=DistributionSummary())
    perf_ratio: DistributionSummary = field(default_factory=DistributionSummary())
    terminal_last_dist: DistributionSummary = field(default_factory=DistributionSummary())
    terminal_hit_frac: DistributionSummary = field(default_factory=DistributionSummary())

    # Small diagnostic lists
    example_missing_prediction_ids: List[int] = field(default_factory=list)
    worst_alpha_ids: List[int] = field(default_factory=list)
    worst_alpha_values: List[float] = field(default_factory=list)


# ============================================
# ======= EMPIRICAL STABILITY VERIFIER =======
# ============================================
class EmpiricalStabilityVerifier:
    """
    A tool to rigorously check stability and performance properties of NMPC trajectories
    based on the optimal value function V_N as a Lyapunov function.
    
    Theory based on:
    Lars Grüne, Nonlinear Model Predictive Control (2015)
    """

    def __init__(self, dataset: MPCDataset, Q: np.ndarray, R: np.ndarray):
        """Create an empirical verifier over an entire dataset.

        Parameters
        ----------
        dataset : MPCDataset
            The dataset containing MPC trajectories and configurations.
        Q : np.ndarray
            State cost matrix used in the MPC cost function.
        R : np.ndarray
            Input cost matrix used in the MPC cost function.
        """
        self.dataset = dataset
        self.Q_global = np.asarray(Q)
        self.R_global = np.asarray(R)

        self._active_entry: Optional[MPCData] = None

        # These are set by _bind_entry before any per-trajectory computation.
        self.traj = None
        self.cfg = None
        self.meta = None
        self.Q = None
        self.R = None
        self.x_star = None
        self.valid = False

    def __getitem__(self, index: int) -> MPCData:
        return self.dataset[index]

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Generator[MPCData, None, None]:
        for entry in self.dataset:
            yield entry


    # --- INTERNAL HELPERS ---
    def _bind_entry(self, entry: MPCData) -> bool:
        """Bind internal state to a specific trajectory entry."""
        self._active_entry = entry

        self.traj = entry.trajectory
        self.cfg = entry.config
        self.meta = entry.meta

        cfg_Q = np.asarray(self.cfg.Q)
        cfg_R = np.asarray(self.cfg.R)
        if cfg_Q.shape != self.Q_global.shape or not np.allclose(cfg_Q, self.Q_global, rtol=0.0, atol=0.0):
            raise ValueError(
                "Entry Q does not match verifier Q_global. "
                "This verifier assumes a single (Q,R) pair for the entire dataset."
            )
        if cfg_R.shape != self.R_global.shape or not np.allclose(cfg_R, self.R_global, rtol=0.0, atol=0.0):
            raise ValueError(
                "Entry R does not match verifier R_global. "
                "This verifier assumes a single (Q,R) pair for the entire dataset."
            )

        # Use consistent Q/R across the dataset for empirical claims.
        self.Q = self.Q_global
        self.R = self.R_global

        # Ensure goal state is defined (default to zero origin)
        self.x_star = self.cfg.goal_state if self.cfg.goal_state is not None else np.zeros(self.traj.states.shape[1])

        self.valid = self._validate_data_integrity()
        
        return self.valid

    def _require_bound_entry(self) -> None:
        if self._active_entry is None or self.traj is None or self.cfg is None:
            raise ValueError(
                "No active entry is bound (internal error). Dataset-level methods must bind an entry before calling per-trajectory helpers."
            )

    def _validate_data_integrity(self) -> bool:
        """Check if OCP predictions (solved_states) are available for Lyapunov calculation."""
        self._require_bound_entry()
        if self.traj.solved_states is None or self.traj.solved_inputs is None:
            __logger__.warning(f"Entry ID {getattr(self.meta, 'id', 'unknown')} missing OCP predictions (solved_states/solved_inputs).")
            return False
        return True


    # --- COST CALCULATIONS ---
    def _stage_cost(self, x: np.ndarray, u: np.ndarray) -> float:
        """
        Computes l(x, u). Assumes Quadratic Cost standard in linear MPC.
        l(x,u) = ||x - x_*||^2_Q + ||u||^2_R[cite: 1823].
        """
        dx = x - self.x_star
        du = u 
        
        # J = x'Qx + u'Ru
        cost = dx.T @ self.Q @ dx + du.T @ self.R @ du
        return float(cost)

    def _terminal_cost(self, x: np.ndarray) -> float:
        """
        Computes V_f(x).
        Used for terminal cost calculation if Qf/P is provided[cite: 2210].
        """
        Qf = self._effective_terminal_cost_matrix(self.cfg.Qf)
        if Qf is None:
            return 0.0
        
        dx = x - self.x_star
        return float(dx.T @ Qf @ dx)
    
    def _l_star(self, x: np.ndarray, x_star: np.ndarray) -> float:
        """Stage cost with no input (u=0)"""
        dx = x - x_star
        return float(dx.T @ self.Q_global @ dx)

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
            if any(int(c) != 0 for c in entry.meta.status_codes):
                __logger__.warning(f"Entry ID {getattr(entry.meta, 'id', 'unknown')} indicates non-zero solver status codes. "
                                   f"Status Codes: {np.unique(entry.meta.status_codes).tolist()}")
                return False

        # NaNs in key arrays indicate an invalid run
        t = entry.trajectory
        if np.isnan(t.states).any() or np.isnan(t.inputs).any() or np.isnan(t.cost).any():
            return False
        return True

    def iter_binded_entries(self, require_feasibility: bool = True) -> Generator[MPCData, None, None]:
        """Iterator over dataset entries with optional filtering/binding.
        
        Parameters
        ----------
        require_feasibility : bool
            Whether to only yield feasible entries.
            
        Yields
        -------
        Generator[MPCData, None, None]
            The next entry in the dataset.
        """
        for entry in self:
            if require_feasibility and (not self._is_entry_feasible(entry)):
                continue
            
            self._bind_entry(entry)
            yield entry

    def get_feasible_dataset(self) -> MPCDataset:
        """Extract a feasible-only subset of the dataset."""
        feasible_entries: List[MPCData] = []
        for entry in self.dataset:
            if self._is_entry_feasible(entry):
                feasible_entries.append(entry)
        return MPCDataset(data_buffer=feasible_entries)

    def min_observed_alpha(self, tolerance: float = 1e-5) -> Optional[float]:
        """Verifies the Lyapunov decrease condition for the MPC closed loop.

        Grüne relaxed DP / Lyapunov inequality along the closed loop:
            V_N(x(n+1)) <= V_N(x(n)) - alpha * l(x(n), u(n))
        where alpha in (0,1].

        This implementation estimates the *observed* alpha at each step as
            alpha_obs(n) = (V_N(x(n)) - V_N(x(n+1))) / l(x(n), u(n))
        and checks alpha_obs(n) >= alpha_min up to numerical tolerance.

        Parameters
        ----------
        tolerance : float
            Numerical tolerance for stage cost to avoid division by zero.

        Returns
        -------
        Optional[float]
            The minimum observed alpha over the trajectory, or None if no valid 
            observations were found.
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

    def compute_optimal_value_function(self, time_step: int) -> float:
        """Reconstructs V_N(x(n)) from the OCP predictions stored in the dataset.
        V_N(x_0) := inf sum(l(x_u(k), u(k))).

        Parameters
        ----------
        time_step : int
            The closed-loop simulation step 'n'.

        Returns
        -------
        float
            The value of the cost function at this step.
        """
        if not self.valid:
            return np.nan

        pred_x = self.traj.solved_states[time_step] 
        pred_u = self.traj.solved_inputs[time_step]

        V_N = 0.0
        N = self.cfg.N

        for k in range(N):
            V_N += self._stage_cost(pred_x[k], pred_u[k])

        # Add terminal cost F(x(N)) [cite: 2210]
        V_N += self._terminal_cost(pred_x[N])

        return V_N

    def check_infinite_horizon_performance(self) -> InfiniteHorizonPerformanceReport:
        """
        Estimates the Infinite Horizon Performance.
        
        According to Relaxed Dynamic Programming:
        J_inf_cl(x0, mu_N) <= V_N(x0) / alpha
        
        For alpha=1 we obtain J_inf_cl(x0, mu_N) <= V_N(x0).
        Here we approximate J_inf_cl by the *realized* sum of stage costs along the
        simulated closed loop (finite-time truncation).
        
        Returns
        -------
        InfiniteHorizonPerformanceReport
            A report containing initial value function, closed-loop cost,
            performance ratio, and whether the bound is satisfied.
        """
        if not self.valid:
            return InfiniteHorizonPerformanceReport()

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
        return InfiniteHorizonPerformanceReport(
            V_N_initial=V_0,
            J_closed_loop=J_cl,
            performance_ratio=J_cl / V_0 if V_0 > 1e-6 else 1.0,
            satisfied_bound=J_cl <= V_0 * 1.05 # 5% tolerance
        )

    def check_terminal_set_invariance(self, tolerance: float = 0.1) -> TerminalSetInvarianceReport:
        """Check whether predicted terminal states lie in the terminal set.

        Parameters
        ----------
        tolerance : float
            Distance tolerance to consider the terminal state as being in the goal set.

        Returns
        -------
        TerminalSetInvarianceReport
            A report containing distances to the goal and boolean flags indicating terminal set membership.
        """
        if not self.valid:
            return TerminalSetInvarianceReport()

        N = self.cfg.N
        T_sim = self.cfg.T_sim
        terminal_dist = np.full((T_sim,), np.nan, dtype=float)
        is_terminal_set = np.zeros((T_sim,), dtype=bool)

        for n in range(T_sim):
            pred_terminal_x = self.traj.solved_states[n][N]
            dist_terminal = np.linalg.norm(pred_terminal_x - self.x_star)
            terminal_dist[n] = float(dist_terminal)
            is_terminal_set[n] = bool(dist_terminal < tolerance)

        return TerminalSetInvarianceReport(
            terminal_dist=terminal_dist,
            is_terminal_set=is_terminal_set
        )


    # -- Certification methods ---
    def lyapunov_decrease(self, tol: float = 1e-6) -> LyapunovDecreaseReport:
        """Dataset-level empirical Lyapunov decrease certification.
        Using the minimum observed alpha over all feasible trajectories.
        min(alpha_obs(n) = (V_N(x(n)) - V_N(x(n+1))) / l(x(n), u(n)))

        Parameters
        ----------
        feasible_dataset : MPCDataset
            A dataset containing only feasible trajectories.
        tol : float
            Numerical tolerance for stage cost to avoid division by zero.

        Returns
        -------
        LyapunovDecreaseReport
            A report indicating the empirical alpha and whether the decrease condition is satisfied.
        """
        alpha_values: List[float] = []
        for _ in self.iter_binded_entries():
            min_alpha = self.min_observed_alpha(tolerance=tol)
            if min_alpha is None:
                continue
            alpha_values.append(float(min_alpha))

        if not alpha_values:
            empirical_ok = False
            empirical_alpha = 0.0
        else:
            empirical_alpha = float(np.min(alpha_values))
            empirical_ok = bool(empirical_alpha > 0.0)

        return LyapunovDecreaseReport(
            empirical_alpha=empirical_alpha,
            is_stable=empirical_ok,
            message=f"{'Satisfied' if empirical_ok else 'Not satisfied'} with minimum observed alpha={empirical_alpha:.4e}."
        )

    def grüne_horizon_condition(self, min_cost_threshold: float = 0.1) -> GrüneHorizonReport:
        """Dataset-level Grüne horizon condition certification.

        Parameters
        ----------
        min_cost_threshold : float
            Minimum stage cost l*(x0) to consider a data point valid for gamma estimation.

        Returns
        -------
        GrüneHorizonReport
            A report indicating applicability and estimates of gamma, alpha_N, and required horizon.
        """
        cfg0 = self.dataset[0].config
        has_terminal_cost = self._effective_terminal_cost_matrix(cfg0.Qf) is not None
        has_terminal_bounds = self._has_terminal_bounds(cfg0)

        if has_terminal_cost or has_terminal_bounds:
            return GrüneHorizonReport(
                applicability=False,
                message="Not applicable: Dataset includes terminal cost/bounds; no-terminal theorem does not directly apply")

        N = int(cfg0.N)
        if N < 2:
            return GrüneHorizonReport(
                applicability=False,
                message="Not applicable: N<2")

        gamma_values: List[float] = []
        
        for entry in self.iter_binded_entries():
            x0 = np.asarray(entry.trajectory.states[0], dtype=float)
            if not np.all(np.isfinite(x0)):
                continue
            v0 = self.compute_optimal_value_function(0)
            if not np.isfinite(v0):
                continue

            l0 = self._l_star(x0, self.x_star)
            if not np.isfinite(l0) or l0 <= min_cost_threshold:
                continue
            gamma_values.append(float(v0 / l0))

        
        if not gamma_values:
            return GrüneHorizonReport(
                applicability=False,
                message="Not applicable: insufficient data")
        
        gamma = float(np.max(gamma_values))
        
        if gamma <= 1.0 + 1e-10:
            return GrüneHorizonReport(
                applicability=True,
                gamma_estimate=gamma,
                alpha_N_estimate=1.0,
                required_horizon=1.0,
                is_stable=True,
                message=f"Trivially satisfied Grüne condition with gamma={gamma:.4f} <= 1.0")

        denom = (gamma ** (N - 1)) - ((gamma - 1.0) ** (N - 1))
        
        if denom <= 0.0 or not np.isfinite(denom):
            return GrüneHorizonReport(
                applicability=False,
                gamma_estimate=gamma,
                alpha_N_estimate=0.0,
                required_horizon=float("inf"),
                is_stable=False,
                message="Not applicable: invalid denominator in Grüne condition calculation.")
           
           
        alpha_N = float(1.0 - (((gamma - 1.0) ** N) / denom))
        numerator = float(np.log(gamma - 1.0))
        denominator = float(np.log(gamma) - np.log(gamma - 1.0))
        
        return GrüneHorizonReport(
            applicability=True,
            gamma_estimate=gamma,
            alpha_N_estimate=alpha_N,
            required_horizon=float(2.0 + (numerator / denominator)) if denominator > 0 else float("inf"),
            is_stable=bool(alpha_N > 0.0),
            message=f"Grüne condition estimated with gamma={gamma:.4f}, alpha_N={alpha_N:.4f}.")


    # --- Certification Interface ---
    def verify(
        self,
        alpha_threshold: float = 1e-4,
        tol: float = 1e-6,
        min_cost_threshold: float = 1e-3,
        more_details: bool = False
    ) -> StabilityReport:
        """Dataset-level certification using the optimal value function as a Lyapunov candidate.
        
        Parameters
        ----------
        alpha_threshold : float
            Minimum empirical alpha required for certification.
        tol : float
            Numerical tolerance for stage cost to avoid division by zero.
        min_cost_threshold : float
            Minimum stage cost l*(x0) to consider a data point valid for gamma estimation.

        Returns
        -------
        StabilityReport
            `is_stable` is interpreted as "certified".
        """
        # Empirical tests
        lyap_report = self.lyapunov_decrease(tol=tol)
        grune_report = self.grüne_horizon_condition(min_cost_threshold=min_cost_threshold)

        certified = bool(lyap_report.is_stable and \
            (grune_report.is_stable or grune_report.applicability) \
            and (lyap_report.empirical_alpha > alpha_threshold))
        
        
        if certified:
            msg = (
                f"PASS. Certified with empirical_alpha={lyap_report.empirical_alpha:.3e} "
                f"and alpha_threshold={alpha_threshold:.3e}.")
        else:
            msg = (
                f"FAIL. Not certified with empirical_alpha={lyap_report.empirical_alpha:.3e}, "
                f"alpha_threshold={alpha_threshold:.3e}, grune_applicability='{grune_report.message}'.")

        details = {
            "lyapunov_decrease_report": lyap_report,
            "grune_report": grune_report
        }
        
        if more_details:
            details["more_details"] = self.details()
    
        return StabilityReport(
            method=f"Empirical Verification",
            is_stable=certified,
            details=details,
            message=msg)
        
        
    def details(self) -> EmpiricalDatasetDetails:
        """Return compressed, dataset-level diagnostics as a flat dict.

        Returns
        -------
        EmpiricalDatasetDetails
            A flat summary of empirical checks across the dataset.
        """
        max_examples = 10

        n_feasible = 0
        n_with_predictions = 0
        n_missing_predictions = 0

        alpha_min_values: List[float] = []
        alpha_min_by_id: List[tuple[int, float]] = []

        perf_ratio_values: List[float] = []

        terminal_last_dist_values: List[float] = []
        terminal_hit_frac_values: List[float] = []

        missing_prediction_ids: List[int] = []

        for idx, entry in enumerate(self):
            entry_id = int(getattr(getattr(entry, "meta", None), "id", f"unknown_{idx}"))

            if not self._is_entry_feasible(entry):
                continue
            n_feasible += 1

            # Bind and check predictions
            has_predictions = self._bind_entry(entry)
            if not has_predictions:
                n_missing_predictions += 1
                if len(missing_prediction_ids) < max_examples:
                    missing_prediction_ids.append(entry_id)
                continue

            n_with_predictions += 1

            # Lyapunov decrease margin (per-trajectory)
            min_alpha = self.min_observed_alpha(tolerance=1e-6)
            if min_alpha is not None and np.isfinite(min_alpha):
                alpha_min_values.append(float(min_alpha))
                alpha_min_by_id.append((entry_id, float(min_alpha)))

            # Infinite-horizon performance proxy
            perf = self.check_infinite_horizon_performance()
            if np.isfinite(perf.performance_ratio):
                perf_ratio_values.append(float(perf.performance_ratio))

            # Terminal set invariance proxy
            term = self.check_terminal_set_invariance(tolerance=0.1)
            td = np.asarray(term.terminal_dist, dtype=float)
            it = np.asarray(term.is_terminal_set, dtype=bool)
            if td.size > 0 and np.isfinite(td[-1]):
                terminal_last_dist_values.append(float(td[-1]))
            if it.size > 0:
                terminal_hit_frac_values.append(float(np.mean(it.astype(float))))

        worst = sorted(alpha_min_by_id, key=lambda t: t[1])[:max_examples]

        return EmpiricalDatasetDetails(
            n_total=int(len(self)),
            n_feasible=int(n_feasible),
            n_with_predictions=int(n_with_predictions),
            n_used=int(n_with_predictions),
            n_missing_predictions=int(n_missing_predictions),
            alpha_min=DistributionSummary.from_stamples(alpha_min_values),
            perf_ratio=DistributionSummary.from_stamples(perf_ratio_values),
            terminal_last_dist=DistributionSummary.from_stamples(terminal_last_dist_values),
            terminal_hit_frac=DistributionSummary.from_stamples(terminal_hit_frac_values),
            example_missing_prediction_ids=missing_prediction_ids,
            worst_alpha_ids=[int(i) for i, _ in worst],
            worst_alpha_values=[float(a) for _, a in worst]
        )
        