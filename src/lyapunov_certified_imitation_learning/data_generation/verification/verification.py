import numpy as np
import casadi as ca

from typing import Dict, Any, Generator, Optional, List, Tuple

try:
    from acados_template import AcadosOcpSolver
except Exception:
    raise ImportError(
        "StabilityVerifier requires `acados_template` (acados Python interface). "
        "Install it and ensure it is importable."
    )


from .reports import *
from ..extractor import MPCConfigExtractor, LinearSystemExtractor
from ..mpc_data import MPCData, MPCDataset
from ...utils.package_logger import PackageLogger


__logger__ = PackageLogger.get_logger(__name__)
    

class StabilityVerifier:
    """
    A tool to rigorously check stability and performance properties of NMPC trajectories
    based on the optimal value function V_N as a Lyapunov function.
    
    Theory based on:
    Lars Grüne, Nonlinear Model Predictive Control (2015)
    """

    def __init__(self, dataset: MPCDataset, solver: Optional[AcadosOcpSolver] = None):
        """Create a linear stability verifier over an entire dataset.

        Parameters
        ----------
        dataset : MPCDataset
            The dataset containing MPC trajectories and configurations.
        solver : AcadosOcpSolver, optional
            The Acados OCP solver instance used for linear stability verification.
        """
        self.dataset = dataset
        
        # Extract Dimensions and Horizon
        self._use_recomputed_value: bool = False

        # Bindable entry 
        self._active_entry: Optional[MPCData] = None
        self.traj = None
        self.meta = None
        self.valid = False
        
        self.cfg = None
        self.sys = None
        self.ocp = None
        if isinstance(solver, AcadosOcpSolver):
            self.ocp = solver.acados_ocp
            self.cfg = MPCConfigExtractor.get_cfg(solver)
            self.cfg.T_sim = dataset[0].config.T_sim
            self.sys = LinearSystemExtractor.get_system(solver)
        else:
            raise NotImplementedError(
                "StabilityVerifier currently only supports verification with AcadosOcpSolver instances."
            )

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
        self.meta = entry.meta
        local_cfg = entry.config
        
        if self.cfg is None:
            self.cfg = local_cfg

        if local_cfg.dt != self.cfg.dt:
            raise ValueError(
                "Entry dt does not match solver dt. "
                "This verifier assumes a single dt for the entire dataset."
            )
        if local_cfg.N != self.cfg.N:
            raise ValueError(
                "Entry horizon N does not match solver N. "
                "This verifier assumes a single horizon N for the entire dataset."
            )
        if local_cfg.x_ref.shape != self.cfg.x_ref.shape or not np.allclose(local_cfg.x_ref, self.cfg.x_ref, rtol=0.0, atol=0.0):
            raise ValueError(
                "Entry x_ref does not match solver x_ref. "
                "This verifier assumes a single (x_ref, u_ref) pair for the entire dataset."
            )
        if local_cfg.u_ref.shape != self.cfg.u_ref.shape or not np.allclose(local_cfg.u_ref, self.cfg.u_ref, rtol=0.0, atol=0.0):
            raise ValueError(
                "Entry u_ref does not match solver u_ref. "
                "This verifier assumes a single (x_ref, u_ref) pair for the entire dataset."
            )
        if self.cfg.T_sim != local_cfg.T_sim:
            __logger__.warning(
                f"Entry T_sim ({local_cfg.T_sim}) does not match configuration T_sim ({self.cfg.T_sim}). "
                "Using dataset T_sim for verification."
            )
            self.cfg.T_sim = local_cfg.T_sim
            
        if self.cfg.T_sim > self.traj.length:
            raise ValueError(
                "Entry T_sim exceeds trajectory length. T_sim must be <= trajectory length."
                f"{self.cfg.T_sim} > {self.traj.length}"
            )
            
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
        # Empirical verification only requires stored value function and trajectories.
        if self.traj.cost is None or len(self.traj.cost) == 0:
            __logger__.warning(
                f"Entry ID {getattr(self.meta, 'id', 'unknown')} missing stored cost; cannot verify Lyapunov decrease."
            )
            return False

        if self.traj.solved_states is None or self.traj.solved_inputs is None:
            __logger__.warning(
                f"Entry ID {getattr(self.meta, 'id', 'unknown')} missing OCP predictions (solved_states/solved_inputs); "
                "using stored cost for verification."
            )
            self._use_recomputed_value = False
            return True

        self._use_recomputed_value = False
        self._audit_value_function()
        return True

    def _audit_value_function(self, max_steps: int = 3, rtol: float = 1e-3, atol: float = 1e-6) -> None:
        """Compare stored traj.cost with recomputed V_N from OCP predictions."""
        self._require_bound_entry()
        if self.traj.solved_states is None or self.traj.solved_inputs is None:
            return

        T_sim = min(self.cfg.T_sim, int(self.traj.cost.shape[0]))
        n_check = min(int(max_steps), max(0, T_sim))
        if n_check <= 0:
            return

        mismatch_count = 0
        error_sum = 0.0
        for n in range(n_check):
            stored = float(self.traj.cost[n])
            if not np.isfinite(stored):
                continue

            # Compare against the recomputation using the currently extracted scaling conventions.
            recomputed = float(self._V_from_predictions(n))
            if not np.isfinite(recomputed):
                continue
            if not np.isclose(stored, recomputed, rtol=rtol, atol=atol):
                mismatch_count += 1
            error_sum += float(abs(stored - recomputed))

        if mismatch_count > 0:
            self._use_recomputed_value = True
            __logger__.warning(
                f"Detected {mismatch_count} mismatches between stored cost and recomputed V_N from predictions, "
                f"avg_abs_error={error_sum/max(1, n_check):.3e}); switching to recomputed V_N for verification."
            )


    # --- COST CALCULATIONS ---
    def _stage_cost(self, x: np.ndarray, u: np.ndarray) -> float:
        """Raw LINEAR_LS stage cost without any global prefactors."""
        x = np.asarray(x, dtype=float).reshape(-1)
        u = np.asarray(u, dtype=float).reshape(-1)
        y = self.ocp.cost.Vx @ x + self.ocp.cost.Vu @ u
        e = (y - self.ocp.cost.yref).reshape(-1)
        return 0.5 * float(e.T @ self.ocp.cost.W @ e)

    def _terminal_cost(self, x: np.ndarray) -> float:
        """Raw LINEAR_LS terminal cost without any global prefactors."""
        if self.ocp.cost.W_e is None:
            return 0.0
        x = np.asarray(x, dtype=float).reshape(-1)

        Vx_e = self.ocp.cost.Vx_e if self.ocp.cost.Vx_e is not None else np.eye(x.size)
        yref_e = self.ocp.cost.yref_e if self.ocp.cost.yref_e is not None else np.zeros((Vx_e.shape[0],), dtype=float)
        y_e = Vx_e @ x
        e = (y_e - yref_e).reshape(-1)
        return 0.5 * float(e.T @ self.ocp.cost.W_e @ e)
    
    def _l_star(self, x: np.ndarray) -> float:
        """Lower bound on l*(x) := min_u l(x,u) via unconstrained minimization.

        This ignores input constraints, yielding a conservative (small) lower bound on l*(x)
        for constrained problems. In Grüne's gamma estimate, this makes V/l* larger (safer).
        """
        x = np.asarray(x, dtype=float).reshape(-1)

        a = (self.ocp.cost.Vx @ x - self.ocp.cost.yref).reshape(-1)  # y = a + Vu u
        H = self.ocp.cost.Vu.T @ self.ocp.cost.W @ self.ocp.cost.Vu
        H = 0.5 * (H + H.T)
        g = (self.ocp.cost.Vu.T @ self.ocp.cost.W @ a).reshape(-1)

        if H.size == 0:
            return float(a.T @ self.ocp.cost.W @ a)

        try:
            u_star = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            u_star = -np.linalg.lstsq(H, g, rcond=None)[0]

        return float(self._stage_cost(x, u_star))

    def _V_from_predictions(self, step_index: int) -> float:
        """Recompute V_N(x) from stored OCP predictions at a simulation step."""
        self._require_bound_entry()
        if self.traj.solved_states is None or self.traj.solved_inputs is None:
            raise ValueError("Missing OCP predictions for V_N recomputation.")

        x_pred = np.asarray(self.traj.solved_states[step_index], dtype=float)
        u_pred = np.asarray(self.traj.solved_inputs[step_index], dtype=float)
        if x_pred.shape[0] - 1 != u_pred.shape[0]:
            raise ValueError("Predicted trajectory shapes do not match horizon.")


        total = 0.0
        for k in range(self.cfg.N):
            # Stage cost as in acados with dt scaled
            total += self.cfg.dt * float(self._stage_cost(x_pred[k], u_pred[k]))
        total += float(self._terminal_cost(x_pred[self.cfg.N]))
        return float(total)

    def _V(self, step_index: int) -> float:
        """Get V_N(x_step) using the selected value-function source."""
        self._require_bound_entry()
        if self._use_recomputed_value:
            return float(self._V_from_predictions(step_index))
        return float(self.traj.cost[step_index])


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
            V_curr = self._V(n)
            V_next = self._V(n + 1)
            if not (np.isfinite(V_curr) and np.isfinite(V_next)):
                __logger__.debug(f"Skipping step {n} due to non-finite value function V_N(x): V_curr={V_curr:.4e}, V_next={V_next:.4e}")
                continue

            # Stage-cost scaling must match the value-function scaling used in V_N.
            l_unscaled = self._stage_cost(self.traj.states[n], self.traj.inputs[n])
            l_curr = self.cfg.dt * float(l_unscaled)

            if not np.isfinite(l_curr) or l_curr <= min_cost_threshold:
                # Near-equilibrium regime: alpha_obs is ill-conditioned, but we still want to
                # detect increases in V_N.
                eps = 1e-10 + 1e-8 * max(1.0, abs(V_curr))
                residual = float(V_next - V_curr)
                violation = max(0.0, residual - eps)
                max_violation = max(max_violation, violation)
                min_residual = min(min_residual, residual)
                n_used += 1
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

        if min_alpha == float("inf"):
            min_alpha = float("nan")

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
            if stats.n_used == 0:
                continue
            if np.isfinite(stats.min_alpha):
                alphas.append(float(stats.min_alpha))
            violations.append(float(stats.max_violation))
            used += 1

        if used == 0:
            min_alpha = 0.0
            max_violation = float("inf")
            empirical_ok = False
        else:
            min_alpha = float(np.min(alphas)) if alphas else 0.0
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

            Vn = float(self._V(n))
            if not np.isfinite(Vn):
                __logger__.debug(f"Skipping step {n} due to non-finite cost Vn={Vn:.4e}")
                continue

            # l*(x) := min_u l(x,u)
            lstar_unscaled = float(self._l_star(x))
            lstar = self.cfg.dt * lstar_unscaled

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
    @classmethod
    def verify(
        cls,
        dataset: MPCDataset,
        solver: AcadosOcpSolver,
        alpha_required: float = 1e-4,
        min_cost_threshold: float = 1e-3,
    ) -> StabilityReport:
        """Dataset-level verification using the optimal value function as a Lyapunov candidate.
        
        Parameters
        ----------
        dataset : MPCDataset
            The dataset containing MPC trajectories and configurations.
        solver : AcadosOcpSolver
            The Acados OCP solver instance used for linear stability verification.
        alpha_required : float
            Minimum empirical alpha required for verification.
        min_cost_threshold : float
            Minimum stage cost l*(x0) to consider a data point valid for gamma estimation.

        Returns
        -------
        StabilityReport
            Stability report indicating whether the dataset passes empirical checks.
        """
        verifier = StabilityVerifier(dataset, solver)
        lyap_report = verifier.lyapunov_decrease(alpha_required=alpha_required, min_cost_threshold=min_cost_threshold)
        grune_report = verifier.grüne_horizon_condition(min_cost_threshold=min_cost_threshold)

        gruene_pass = bool(grune_report.applicability and grune_report.is_stable)
        lyap_pass = bool(lyap_report.is_stable)

        if lyap_pass:
            msg = (
                f"PASS. Lyapunov decrease observed with min_alpha={lyap_report.min_alpha:.3e} "
                f"and alpha_required={alpha_required:.3e}.")
        elif gruene_pass:
            msg = (
                f"PASS. Grüne horizon condition estimated with gamma={grune_report.gamma_estimate:.3e} "
                f"and required_horizon={grune_report.required_horizon}.")
        else:
            msg = (
                f"FAIL. lyapunov='{lyap_report.message}', grune='{grune_report.message}'.")

        return StabilityReport(
            method="Empirical Verification",
            is_stable=bool(gruene_pass or lyap_pass),
            details={
                "lyapunov_decrease_report": lyap_report,
                "grune_report": grune_report,
            },
            message=msg,
        )