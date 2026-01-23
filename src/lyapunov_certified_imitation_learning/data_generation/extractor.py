import numpy as np
import casadi as ca

from typing import Any, Literal, Optional, Tuple
from dataclasses import dataclass, field, fields

from .mpc_data import MPCConfig, LinearSystem
from ..utils.linalg import discretize_and_linearize_rk4

try:
    from acados_template import AcadosOcpSolver
except Exception:
    raise ImportError(
        "StabilityVerifier requires `acados_template` (acados Python interface). "
        "Install it and ensure it is importable."
    )

# --- Helpers ---
def _ensure_linear_ls_cost_type(cost_type: Literal['LINEAR_LS', 'NONLINEAR_LS']) -> None:
    """Ensure that only LINEAR_LS cost type is used."""
    if cost_type != 'LINEAR_LS':
        raise NotImplementedError("Only LINEAR_LS cost type is supported in this verifier.")

def _is_none(*values: Optional[Any]) -> Any:
    """Ensure that a value is not None, otherwise raise an error."""
    for val in values:
        if val is None:
            return True
        
    return False

# --- Extracts ---
def extract_stage_reference(
    yref: Optional[np.ndarray],
    nx: int, 
    nu: int
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Extraction of (x*, u*) from yref."""
    if _is_none(yref, nx, nu):
        return None
    x_ref = np.zeros(nx)
    u_ref = np.zeros(nu)
    
    yref = np.asarray(yref).reshape(-1)
    if yref.size == (nx + nu):
        x_ref = yref[: nx].copy()
        u_ref = yref[nx : nx + nu].copy()
    else:
        raise NotImplementedError(
            "Cannot extract (x*, u*) from yref with unexpected size. Require size "
            f"nx + nu = {nx + nu}, got {yref.size}."
        )

    return x_ref, u_ref


def extract_terminal_reference(yref_e: Optional[np.ndarray], nx: int) -> Optional[np.ndarray]:
    """Extraction of x_e* from yref_e."""
    if _is_none(yref_e, nx):
        return None

    yref_e = np.asarray(yref_e).reshape(-1)
    if yref_e.shape[0] != nx:
        raise NotImplementedError(
            "Cannot extract x_e* from yref_e with unexpected size. Require size "
            f"nx = {nx}, got {yref_e.size}."
        )

    return yref_e.copy()


def indexed_bounds(
    lb: np.ndarray,
    ub: np.ndarray,
    idx: np.ndarray,
    dim: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Reconstruct full bounds vectors from acados indexed bounds."""
    if _is_none(lb, ub, idx):
        return None

    lb = np.asarray(lb, dtype=float).reshape(-1)
    ub = np.asarray(ub, dtype=float).reshape(-1)
    idx = np.asarray(idx, dtype=int).reshape(-1)

    if lb.size == 0 or ub.size == 0 or idx.size == 0:
        return None
    if lb.shape != ub.shape or lb.shape != idx.shape:
        return None
    if dim <= 0:
        return None

    full_lb = -np.inf * np.ones((dim,), dtype=float)
    full_ub = np.inf * np.ones((dim,), dtype=float)

    # Guard against out-of-range indices.
    if np.any(idx < 0) or np.any(idx >= dim):
        return None

    full_lb[idx] = lb
    full_ub[idx] = ub
    return full_lb, full_ub

def extract_stage_cost_matrices(
    W: np.ndarray, 
    Vx: np.ndarray, 
    Vu: np.ndarray, 
    cost_type: Literal['LINEAR_LS', 'NONLINEAR_LS']
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Extracts Q, R from the cost configuration."""
    if _is_none(W, Vx, Vu):
        return None

    _ensure_linear_ls_cost_type(cost_type)

    W = np.asarray(W)
    Vx = np.asarray(Vx)
    Vu = np.asarray(Vu)

    Q = Vx.T @ W @ Vx
    R = Vu.T @ W @ Vu

    return Q, R

def extract_terminal_cost_matrices(
    W_e: np.ndarray,
    Vx_e: np.ndarray,
    cost_type_e: Literal['LINEAR_LS', 'NONLINEAR_LS']
) -> Optional[np.ndarray]:
    """Extracts Qf from the terminal cost configuration."""
    if _is_none(W_e, Vx_e):
        return None

    _ensure_linear_ls_cost_type(cost_type_e)

    W_e = np.asarray(W_e)
    Vx_e = np.asarray(Vx_e)

    Qf = Vx_e.T @ W_e @ Vx_e
    if np.allclose(Qf, 0.0, atol=0.0, rtol=0.0):
        return None
    return Qf


# --- Extractor class for AcadosOcpSolver ---
class MPCConfigExtractor():
    """Extractor for AcadosOcpSolver objects.

    This class extracts relevant matrices and parameters from an `AcadosOcpSolver`
    instance for use in Lyapunov verification routines.
    """

    def __init__(self, solver: AcadosOcpSolver) -> None:
        self.ocp = solver.acados_ocp
        self.cfg = MPCConfig(
            T_sim=0,  # not extractable here
            N=self.ocp.dims.N,
            nx=self.ocp.dims.nx,
            nu=self.ocp.dims.nu,
            dt=float(self.ocp.solver_options.tf) / float(self.ocp.dims.N),
        )
        
        self.cfg.Q, self.cfg.R, self.cfg.Qf = self._extract_cost_matrices()

        references = self._extract_reference()
        self.cfg.x_ref, self.cfg.u_ref = references if references is not None else (None, None)

        state_bounds = self._extract_state_bounds()
        self.cfg.state_bounds = np.vstack(state_bounds) if state_bounds is not None else None

        input_bounds = self._extract_input_bounds()
        self.cfg.input_bounds = np.vstack(input_bounds) if input_bounds is not None else None

        self.cfg.terminal_state_bounds = self._extract_terminal_state_bounds()
        self.cfg.x0 = self._extract_initial_state()
    
    @classmethod
    def get_cfg(cls, solver: AcadosOcpSolver) -> MPCConfig:
        """Get the extracted MPCConfig.
        
        Note
        ----
        T_sim needs to be set separately, as it is not extractable from AcadosOcpSolver.
        """
        extractor = cls(solver)
        return extractor.cfg
    
    def _extract_initial_state(self) -> Optional[np.ndarray]:
        """Extract the initial state x0 from acados constraints."""
        x0 = self.ocp.constraints.x0
        if x0 is None:
            return None
        return np.asarray(x0).reshape(-1)

    def _extract_state_bounds(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Extract full state bounds from acados indexed bounds."""
        return indexed_bounds(
            self.ocp.constraints.lbx,
            self.ocp.constraints.ubx,
            self.ocp.constraints.idxbx,
            self.cfg.nx,
        )

    def _extract_input_bounds(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Extract full input bounds from acados indexed bounds."""
        return indexed_bounds(
            self.ocp.constraints.lbu,
            self.ocp.constraints.ubu,
            self.ocp.constraints.idxbu,
            self.cfg.nu,
        )
        
    def _extract_terminal_state_bounds(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Extract full terminal state bounds from acados indexed bounds."""
        return indexed_bounds(
            self.ocp.constraints.lbx_e,
            self.ocp.constraints.ubx_e,
            self.ocp.constraints.idxbx_e,
            self.cfg.nx,
        )
        
    def _extract_reference(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Extract (x*, u*) from acados yref."""
        yref = self.ocp.cost.yref

        x_u_ref = extract_stage_reference(yref, self.cfg.nx, self.cfg.nu)

        if x_u_ref is None:
            return None, None
        x_ref, u_ref = x_u_ref
        return x_ref, u_ref
        
    def _extract_cost_matrices(self) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Extract the exact LINEAR_LS cost structure used by the OCP.

        This is the authoritative definition for the stage and terminal costs:
            l(x,u) = || Vx x + Vu u - yref ||^2_W
            Vf(x)  = || Vx_e x - yref_e ||^2_{W_e}
        """
        cost = self.ocp.cost
        Q, R = extract_stage_cost_matrices(
            W=cost.W,
            Vx=cost.Vx,
            Vu=cost.Vu,
            cost_type=cost.cost_type,
        )
        Qf = extract_terminal_cost_matrices(
            W_e=cost.W_e,
            Vx_e=cost.Vx_e,
            cost_type_e=cost.cost_type_e,
        )

        return Q, R, Qf
    
    
class LinearSystemExtractor:
    """Extractor for AcadosOcpSolver model dynamics.
    
    This class extracts the nonlinear dynamics function from an `AcadosOcpSolver`
    instance for use in Lyapunov verification routines.
    """
    
    def __init__(self, solver: AcadosOcpSolver) -> None:
        self.ocp = solver.acados_ocp
        self.cfg = MPCConfigExtractor.get_cfg(solver)
        self.linear_system = self._extract_discretized_dynamics(self.cfg.x_ref, self.cfg.u_ref, self.cfg.dt)
        
    @classmethod
    def get_system(cls, solver: AcadosOcpSolver) -> LinearSystem:
        """Get the linearized system matrices from the solver."""
        extractor = cls(solver)
        return extractor.linear_system

    def _extract_discretized_dynamics(self, x_lin: np.ndarray, u_lin: np.ndarray, dt: float) -> LinearSystem:
        """Compute the discrete-time linearization (A, B, g).
        
        Parameters
        ----------
        x_lin, u_lin : np.ndarray
            Linearization points for state and input.
        dt : float
            Sampling time.

        Returns
        -------
        Ad, Bd : np.ndarray
            Discrete-time state and input matrices.
        gd : np.ndarray
            Affine offset term so that $x^+ \\approx Ad x + Bd u + gd$.
        """
        if self.ocp.solver_options.integrator_type != "ERK" or self.ocp.model.f_expl_expr is None:
            raise NotImplementedError("Only explicit ODE models are supported in this verifier.")

        if self.ocp.solver_options.sim_method_num_stages is not None and np.any(self.ocp.solver_options.sim_method_num_stages != 4):
            raise NotImplementedError("Only RK4 integration is supported in this verifier.")
        
        if self.ocp.solver_options.sim_method_num_steps is not None and np.any(self.ocp.solver_options.sim_method_num_steps < 1):
            raise NotImplementedError("Number of integration steps must be at least 1.")

        x = self.ocp.model.x
        u = self.ocp.model.u
        f_expr = self.ocp.model.f_expl_expr

        return LinearSystem(*discretize_and_linearize_rk4(
            x, u, f_expr, dt, x_lin, u_lin
        ))
