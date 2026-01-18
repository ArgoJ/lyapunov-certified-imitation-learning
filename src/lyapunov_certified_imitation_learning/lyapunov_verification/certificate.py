import numpy as np
import scipy.linalg

from dataclasses import dataclass


@dataclass
class NMPCFormalCertificate:
    P: np.ndarray
    K: np.ndarray
    alpha: float
    invariant_terminal_set: bool
    constraint_compatible: bool
    recursive_feasible: bool
    lyapunov_decrease: bool
    domain_of_attraction: float



class NMPCFormalCertificateGenerator:

    def __init__(self, A, B, Q, R, x_bounds, u_bounds):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.x_bounds = x_bounds
        self.u_bounds = u_bounds

    def compute_certificate(self, tol: float = 1e-8) -> NMPCFormalCertificate:
        """Compute a simple *linear-quadratic* terminal certificate.

        This implements the standard regional-terminal-cost idea from Grüne (Part A, Section 3b)
        for linear systems:
        - terminal cost F(x)=x^T P x with local controller u=-Kx
        - decrease condition: F(A_cl x) <= F(x) - l(x,-Kx)
        - terminal set chosen as an ellipsoid inside the box constraints.

        Notes
        -----
        - This is not a general nonlinear certificate.
        - It assumes the equilibrium is at the origin and constraints are symmetric boxes.
        """
        A = np.asarray(self.A, dtype=float)
        B = np.asarray(self.B, dtype=float)
        Q = np.asarray(self.Q, dtype=float)
        R = np.asarray(self.R, dtype=float)

        nx = A.shape[0]
        nu = B.shape[1]

        # LQR terminal cost and local controller u = -K x
        P = scipy.linalg.solve_discrete_are(A, B, Q, R)
        K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
        Acl = A - B @ K

        # Lyapunov decrease check: Acl^T P Acl - P <= -(Q + K^T R K)
        Qcl = Q + K.T @ R @ K
        lhs = Acl.T @ P @ Acl - P
        sym_check = 0.5 * ((lhs + Qcl) + (lhs + Qcl).T)
        eig_max = float(np.max(np.linalg.eigvalsh(sym_check)))
        lyap_ok = eig_max <= tol

        # Ellipsoidal terminal set: X_rho = {x | x^T P x <= rho}
        # Pick rho to satisfy |x_i| <= b_i and |(Kx)_j| <= d_j for all x in X_rho.
        lbx, ubx = self.x_bounds
        lbu, ubu = self.u_bounds
        lbx = np.asarray(lbx, dtype=float).reshape(nx)
        ubx = np.asarray(ubx, dtype=float).reshape(nx)
        lbu = np.asarray(lbu, dtype=float).reshape(nu)
        ubu = np.asarray(ubu, dtype=float).reshape(nu)

        b = np.minimum(np.abs(lbx), np.abs(ubx))
        d = np.minimum(np.abs(lbu), np.abs(ubu))

        P_inv = np.linalg.inv(P)

        rho_candidates = []
        # state constraints
        for i in range(nx):
            if not np.isfinite(b[i]) or b[i] <= 0:
                continue
            denom = float(P_inv[i, i])
            if denom <= 0:
                continue
            rho_candidates.append(float((b[i] ** 2) / denom))

        # input constraints
        for j in range(nu):
            if not np.isfinite(d[j]) or d[j] <= 0:
                continue
            kj = K[j, :].reshape(1, nx)
            denom = float(kj @ P_inv @ kj.T)
            if denom <= 0:
                continue
            rho_candidates.append(float((d[j] ** 2) / denom))

        rho = float(min(rho_candidates)) if rho_candidates else float("inf")
        constraint_compatible = bool(np.isfinite(rho) and rho > 0)

        # Invariance (sufficient): Acl^T P Acl <= P
        inv_mat = 0.5 * ((Acl.T @ P @ Acl - P) + (Acl.T @ P @ Acl - P).T)
        inv_eig_max = float(np.max(np.linalg.eigvalsh(inv_mat)))
        invariant_terminal_set = bool(inv_eig_max <= tol)

        recursive_feasible = bool(invariant_terminal_set and constraint_compatible)

        return NMPCFormalCertificate(
            P=P,
            K=K,
            alpha=rho,
            invariant_terminal_set=invariant_terminal_set,
            constraint_compatible=constraint_compatible,
            recursive_feasible=recursive_feasible,
            lyapunov_decrease=bool(lyap_ok),
            domain_of_attraction=rho,
        )
