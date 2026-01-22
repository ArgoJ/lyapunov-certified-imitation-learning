import numpy as np
import casadi as ca

from typing import Dict, Any, Generator, Optional, List
from dataclasses import dataclass, field
from acados_template import AcadosOcpSolver


from .reports import *
from ..utils.package_logger import PackageLogger
from ..utils.linalg import discretize_and_linearize_rk4
from ..data_generation.mpc_data import MPCData, MPCDataset


__logger__ = PackageLogger.get_logger(__name__)
    

class StabilityVerifier:
    """
    A tool to rigorously check stability and performance properties of NMPC trajectories
    based on the optimal value function V_N as a Lyapunov function.
    
    Theory based on:
    Lars Grüne, Nonlinear Model Predictive Control (2015)
    """

    def __init__(self, dataset: MPCDataset, solver: AcadosOcpSolver):
        """Create a linear stability verifier over an entire dataset.

        Parameters
        ----------
        dataset : MPCDataset
            The dataset containing MPC trajectories and configurations.
        solver : AcadosOcpSolver
            The Acados OCP solver instance used for linear stability verification.
        """
        self.dataset = dataset
        self.solver = solver
        self.ocp = solver.acados_ocp
        
        # Extract Dimensions and Horizon
        self.N = self.ocp.solver_options.N_horizon
        self.nx = self.ocp.model.x.size()[0]
        self.nu = self.ocp.model.u.size()[0]
        self.dt = float(self.ocp.solver_options.tf) / float(self.N)

        self.x_star, self.u_star = self._extract_reference()
        self.A, self.B = self._extract_discretized_dynamics(self.x_star, self.u_star)
        self.Q, self.R, self.P = self._extract_cost_matrices()
        
        # Extract Constraints
        self.x_bounds = (self.ocp.constraints.lbx, self.ocp.constraints.ubx)
        self.u_bounds = (self.ocp.constraints.lbu, self.ocp.constraints.ubu)
        self.term_x_bounds = (self.ocp.constraints.lbx_e, self.ocp.constraints.ubx_e)

        self._active_entry: Optional[MPCData] = None

        # These are set by _bind_entry before any per-trajectory computation.
        self.traj = None
        self.cfg = None
        self.meta = None
        self.valid = False

    def __getitem__(self, index: int) -> MPCData:
        return self.dataset[index]

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Generator[MPCData, None, None]:
        for entry in self.dataset:
            yield entry

    
    # --- EXTRACTION HELPERS ---
    def _extract_reference(self) -> tuple[np.ndarray, np.ndarray]:
        """Best-effort extraction of (x*, u*) from yref/yref_e.

        For the regulation case used throughout Grüne's Part A, one typically has
        (x*, u*) = (0, 0) after shifting coordinates.
        """
        x_star = np.zeros(self.nx)
        u_star = np.zeros(self.nu)

        cost = self.ocp.cost
        if hasattr(cost, "yref") and cost.yref is not None:
            yref = np.asarray(cost.yref).reshape(-1)
            if yref.size >= (self.nx + self.nu):
                x_star = yref[: self.nx].copy()
                u_star = yref[self.nx : self.nx + self.nu].copy()

        if hasattr(cost, "yref_e") and cost.yref_e is not None:
            yref_e = np.asarray(cost.yref_e).reshape(-1)
            if yref_e.size >= self.nx:
                x_star = yref_e[: self.nx].copy()

        return x_star, u_star
        
    def _extract_discretized_dynamics(self, x_lin: np.ndarray, u_lin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Computes the discrete-time  (A, B)."""
        if self.ocp.solver_options.integrator_type != "ERK" or self.ocp.model.f_expl_expr is None:
            raise NotImplementedError("Only explicit ODE models are supported in this verifier.")

        if self.ocp.solver_options.sim_method_num_stages is not None and np.any(self.ocp.solver_options.sim_method_num_stages != 4):
            raise NotImplementedError("Only RK4 integration is supported in this verifier.")
        
        if self.ocp.solver_options.sim_method_num_steps is not None and np.any(self.ocp.solver_options.sim_method_num_steps < 1):
            raise NotImplementedError("Number of integration steps must be at least 1.")

        x = self.ocp.model.x
        u = self.ocp.model.u
        f_expr = self.ocp.model.f_expl_expr

        Ad, Bd, _ = discretize_and_linearize_rk4(
            x, u, f_expr, self.dt, x_lin, u_lin)
        return Ad, Bd

    def _extract_cost_matrices(self) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Extracts Q, R, and P from the cost configuration."""
        if self.ocp.cost.cost_type != 'LINEAR_LS' and self.ocp.cost.cost_type_e != 'LINEAR_LS':
            raise NotImplementedError("Only LINEAR_LS cost type is supported in this verifier.")
        
        # Stage cost
        W = np.asarray(self.ocp.cost.W)
        Vx = np.asarray(self.ocp.cost.Vx)
        Vu = np.asarray(self.ocp.cost.Vu)

        Q = Vx.T @ W @ Vx
        R = Vu.T @ W @ Vu
        
        # Terminal cost
        P = None
        if hasattr(self.ocp.cost, 'W_e') and self.ocp.cost.W_e is not None:
            W_e = np.asarray(self.ocp.cost.W_e)
            if hasattr(self.ocp.cost, 'Vx_e') and self.ocp.cost.Vx_e is not None:
                Vx_e = np.asarray(self.ocp.cost.Vx_e)
                P_candidate = Vx_e.T @ W_e @ Vx_e
            else:
                P_candidate = W_e

            if not np.allclose(P_candidate, 0.0, atol=0.0, rtol=0.0):
                P = P_candidate
            
        return Q, R, P


    # --- INTERNAL HELPERS ---
    def _bind_entry(self, entry: MPCData) -> bool:
        """Bind internal state to a specific trajectory entry."""
        self._active_entry = entry

        self.traj = entry.trajectory
        self.cfg = entry.config
        self.meta = entry.meta

        cfg_Q = np.asarray(self.cfg.Q)
        cfg_R = np.asarray(self.cfg.R)
        cfg_P = np.asarray(self.cfg.Qf) if self.cfg.has_terminal_cost() else None

        if cfg_Q.shape != self.Q.shape or not np.allclose(cfg_Q, self.Q, rtol=0.0, atol=0.0):
            raise ValueError(
                "Entry Q does not match solver Q. "
                "This verifier assumes a single (Q,R) pair for the entire dataset."
            )
        if cfg_R.shape != self.R.shape or not np.allclose(cfg_R, self.R, rtol=0.0, atol=0.0):
            raise ValueError(
                "Entry R does not match solver R. "
                "This verifier assumes a single (Q,R) pair for the entire dataset."
            )
        if self.P is not None and (cfg_P is None or cfg_P.shape != self.P.shape or not np.allclose(cfg_P, self.P, rtol=0.0, atol=0.0)):
            raise ValueError(
                "Entry P does not match solver P. "
                "This verifier assumes a single terminal cost P for the entire dataset."
            )

        # Ensure goal state is defined (default to zero origin)
        self.x_star = self.cfg.goal_state if self.cfg.goal_state is not None else np.zeros(self.traj.states.shape[1])
        # self.u_star = self.cfg.goal_input if self.cfg.goal_input is not None else np.zeros(self.traj.inputs.shape[1])
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
        
        if self.cfg.dt != self.dt:
            __logger__.warning(f"Entry ID {getattr(self.meta, 'id', 'unknown')} has mismatched dt (dataset dt={self.cfg.dt}, verifier dt={self.dt}).")
            return False
        return True


    # --- COST CALCULATIONS ---
    def _stage_cost(self, x: np.ndarray, u: np.ndarray) -> float:
        """
        Computes l(x, u). Assumes Quadratic Cost standard in linear MPC.
        l(x,u) = ||x - x_*||^2_Q + ||u - u_*||^2_R. 
        """
        dx = x - self.x_star
        du = u - self.u_star
        cost = dx.T @ self.Q @ dx + du.T @ self.R @ du
        return float(cost)

    def _terminal_cost(self, x: np.ndarray) -> float:
        """
        Computes V_f(x).
        Used for terminal cost calculation if Qf/P is provided.
        """
        if not self.cfg.has_terminal_cost():
            return 0.0
        
        dx = x - self.x_star
        Qf = np.asarray(self.cfg.Qf)
        return float(dx.T @ Qf @ dx)
    
    def _l_star(self, x: np.ndarray) -> float:
        """Optimal Stage cost"""
        u_opt = self.u_star

        u_min, u_max = self.u_bounds
        if u_min is not None and u_max is not None:
            u_opt = np.clip(self.u_star, u_min, u_max)

        return self._stage_cost(x, u_opt)


    # --- DATASET ITERATORS ---
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
            if require_feasibility and (not entry.is_feasible()):
                continue
            
            self._bind_entry(entry)
            yield entry

    def get_feasible_dataset(self) -> MPCDataset:
        """Extract a feasible-only subset of the dataset."""
        feasible_entries: List[MPCData] = []
        for entry in self.dataset:
            if entry.is_feasible():
                feasible_entries.append(entry)
        return MPCDataset(data_buffer=feasible_entries)


    # --- LYAPUNOV STABILITY CHECK ---
    def alpha_and_max_violation(self, alpha_required: float = 1e-3, min_cost_threshold: float = 1e-5) -> AlphaViolationStats:
        """
        This implementation estimates the *observed* alpha and maximum violation at each step as
            alpha_obs(n) = min((V_n - V_{n+1}) / l_n)
            viol(n)      = max(0, V_{n+1} - (V_n - alpha_required*l_n))

        Parameters
        ----------
        alpha_required : float
            Minimum empirical alpha required for certification.
        min_cost_threshold : float
            Minimum stage cost to consider for certification.

        Returns
        -------
        AlphaViolationStats
            The observed minimum alpha and maximum violation statistics.
        """
        if not self.valid:
            return AlphaViolationStats()

        min_alpha: float = float("inf")
        max_violation = 0.0
        min_residual = float("inf")
        n_used = 0

        T_sim = min(self.cfg.T_sim, len(self.traj.states), len(self.traj.cost))
        for n in range(T_sim - 1):
            V_curr = self.traj.cost[n]
            V_next = self.traj.cost[n+1]
            if not (np.isfinite(V_curr) and np.isfinite(V_next)):
                __logger__.debug(f"Skipping step {n} due to non-finite value function V_N(x): V_curr={V_curr:.4e}, V_next={V_next:.4e}")
                continue

            # acados stage cost is scaled with dt by default.
            l_unscaled = self._stage_cost(self.traj.states[n], self.traj.inputs[n])
            l_curr = float(self.dt) * float(l_unscaled)

            if not np.isfinite(l_curr) or l_curr <= min_cost_threshold:
                __logger__.debug(f"Skipping step {n} due to small stage cost l(x,u)={l_curr:.4e} <= tol={min_cost_threshold:.4e}")
                continue

            alpha_obs = (V_curr - V_next) / l_curr
            if not np.isfinite(alpha_obs):
                __logger__.debug(f"Skipping step {n} due to non-finite observed alpha_obs={alpha_obs:.4e}")
                continue

            min_alpha = min(min_alpha, float(alpha_obs))
            rhs = V_curr - float(alpha_required) * l_curr
            residual = float(V_next - rhs)               # <0 means satisfied with margin
            violation = max(0.0, residual)               # >=0 by definition
            max_violation = max(max_violation, violation)
            min_residual = min(min_residual, residual)

            n_used += 1

        if n_used == 0:
            return AlphaViolationStats()

        return AlphaViolationStats(
            min_alpha=float(min_alpha), 
            max_violation=float(max_violation), 
            min_residual=float(min_residual), 
            n_used=int(n_used))

    def lyapunov_decrease(self, alpha_required: float = 1e-3, min_cost_threshold: float = 1e-6) -> LyapunovDecreaseReport:
        """Dataset-level empirical Lyapunov decrease certification.
        Using the minimum observed alpha and maximum violation over all feasible trajectories.

        Parameters
        ----------
        alpha_required : float
            Minimum empirical alpha required for certification.
        min_cost_threshold : float
            Minimum stage cost to consider for certification.

        Returns
        -------
        LyapunovDecreaseReport
            A report indicating the empirical alpha and whether the decrease condition is satisfied.
        """
        alphas = []
        violations = []
        used = 0

        for _ in self.iter_binded_entries():
            stats = self.alpha_and_max_violation(alpha_required=alpha_required, min_cost_threshold=min_cost_threshold)
            if stats.min_alpha is None:
                continue
            alphas.append(float(stats.min_alpha))
            violations.append(float(stats.max_violation))
            used += 1

        if used == 0:
            min_alpha = 0.0
            max_violation = float("inf")
            empirical_ok = False
        else:
            min_alpha = float(np.min(alphas))
            max_violation = float(np.max(violations))

        empirical_ok = bool((min_alpha >= alpha_required) and (max_violation <= 0.0))

        return LyapunovDecreaseReport(
            is_stable=empirical_ok,
            message=(
                f"{'Satisfied' if empirical_ok else 'Not satisfied'}: "
                f"min_alpha={min_alpha:.4e}, "
                f"max_violation={max_violation:.4e}, "
                f"alpha_required={alpha_required:.4e}."
            ),
            min_alpha=min_alpha,
            max_violation=max_violation
        )


    # --- GRÜNE CONDITION CHECK ---
    def gamma_estimates(self, min_cost_threshold: float = 1e-6) -> List[float]:
        """Estimate the maximum gamma value over the dataset."""
        gamma_values: List[float] = []

        T_sim = min(self.cfg.T_sim, len(self.traj.states), len(self.traj.cost))
        for n in range(T_sim):
            x = np.asarray(self.traj.states[n], dtype=float)
            if not np.all(np.isfinite(x)):
                __logger__.debug(f"Skipping step {n} due to non-finite state x={x}")
                continue

            Vn = float(self.traj.cost[n])
            if not np.isfinite(Vn):
                __logger__.debug(f"Skipping step {n} due to non-finite cost Vn={Vn:.4e}")
                continue

            # l*(x) := min_u l(x,u)
            lstar_unscaled = float(self._l_star(x))
            lstar = float(self.cfg.dt) * lstar_unscaled  # match acados default scaling

            if (not np.isfinite(lstar)) or (lstar <= min_cost_threshold):
                __logger__.debug(f"Skipping step {n} due to small stage cost l(x,u)={lstar:.4e} <= tol={min_cost_threshold:.4e}")
                continue

            gamma_values.append(float(Vn / lstar))
        return gamma_values

    def grüne_horizon_condition(self, min_cost_threshold: float = 1e-6) -> GrüneHorizonReport:
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
        has_terminal_cost = cfg0.has_terminal_cost()
        has_terminal_bounds = cfg0.has_terminal_bounds()

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
        
        for _ in self.iter_binded_entries():
            gamma_values.extend(
                self.gamma_estimates(min_cost_threshold=min_cost_threshold))
                

        if not gamma_values:
            return GrüneHorizonReport(
                applicability=False,
                message="Not applicable: insufficient data")
        
        gamma = float(np.max(gamma_values))
        
        if gamma <= 1.0 + 1e-12:
            return GrüneHorizonReport(
                applicability=True,
                gamma_estimate=gamma,
                alpha_N_estimate=1.0,
                required_horizon=2,
                is_stable=True,
                message=f"Trivially satisfied Grüne condition with gamma={gamma:.4f} <= 1")

        eps = gamma - 1.0
        if (not np.isfinite(eps)) or (eps <= 0.0):
            return GrüneHorizonReport(
                applicability=False,
                gamma_estimate=gamma,
                message="Not applicable: gamma must be > 1.",
            )
        
        denom = np.log1p(1.0/eps)
        if (not np.isfinite(denom)) or (denom <= 0.0):
            return GrüneHorizonReport(
                gamma_estimate=gamma,
                message="Not applicable: invalid log denominator in N0(gamma) computation."
            )

        N0_real = 2.0 + (np.log(eps) / denom)
        N_required = max(2, int(np.ceil(N0_real)))

        # alpha_N = 1 - 1/( (gamma/(gamma-1))^N - 1 )
        ratio = gamma / eps
        log_ratio = np.log(ratio)

        den = np.expm1(float(N) * float(log_ratio))  # = ratio^N - 1
        if (not np.isfinite(den)) or (den <= 0.0):
            return GrüneHorizonReport(
                gamma_estimate=gamma,
                required_horizon=N_required,
                message="Not applicable: invalid denominator in alpha_N computation."
            )

        alpha_N = float(1.0 - 1.0 / den)
        stable_flag = bool((N >= N_required) and (alpha_N > 0.0))
        
        return GrüneHorizonReport(
            applicability=True,
            gamma_estimate=gamma,
            alpha_N_estimate=alpha_N,
            required_horizon=N_required,
            is_stable=stable_flag,
            message=f"Grüne condition estimated with gamma={gamma:.4f}, alpha_N={alpha_N:.4f}, required_horizon={N_required}.")


    # --- Certification Interface ---
    def verify(
        self,
        alpha_required: float = 1e-4,
        min_cost_threshold: float = 1e-3
    ) -> StabilityReport:
        """Dataset-level certification using the optimal value function as a Lyapunov candidate.
        
        Parameters
        ----------
        alpha_required : float
            Minimum empirical alpha required for certification.
        min_cost_threshold : float
            Minimum stage cost l*(x0) to consider a data point valid for gamma estimation.

        Returns
        -------
        StabilityReport
            `is_stable` is interpreted as "certified".
        """
        # Empirical tests
        lyap_report = self.lyapunov_decrease(alpha_required=alpha_required, min_cost_threshold=min_cost_threshold)
        grune_report = self.grüne_horizon_condition(min_cost_threshold=min_cost_threshold)

        certified = bool((lyap_report.is_stable and (lyap_report.min_alpha > alpha_required)) \
            or (grune_report.is_stable or grune_report.applicability))
        
        
        if certified:
            msg = (
                f"PASS. Certified with min_alpha={lyap_report.min_alpha:.3e} "
                f"and alpha_required={alpha_required:.3e}.")
        else:
            msg = (
                f"FAIL. Not certified with min_alpha={lyap_report.min_alpha:.3e}, "
                f"alpha_required={alpha_required:.3e}, grune_applicability='{grune_report.message}'.")

        details = {
            "lyapunov_decrease_report": lyap_report,
            "grune_report": grune_report
        }
    
        return StabilityReport(
            method=f"Empirical Verification",
            is_stable=certified,
            details=details,
            message=msg)
    
