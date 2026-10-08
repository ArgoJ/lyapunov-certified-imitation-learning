import numpy as np
import pytest
import torch as th

from lcil.lyapunov_learning import (
    compute_induced_1norm_gain,
    optimize_polyhedral_contraction_matrix,
)


def test_compute_induced_1norm_gain_diagonal():
    # Diagonal A with eigenvalues 0.5, 0.8
    A = th.tensor([[0.5, 0.0], [0.0, 0.8]], dtype=th.float64)
    P = th.eye(2, dtype=th.float64)
    gain = compute_induced_1norm_gain(P, A)
    assert th.allclose(gain, th.tensor([0.5, 0.8], dtype=th.float64))


def test_compute_induced_1norm_gain_scaling():
    # If P = diag(2, 1), coordinate change scales off-diagonals
    A = th.tensor([[0.5, 0.2], [0.1, 0.5]], dtype=th.float64)
    P = th.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=th.float64)
    # P A P^{-1} = [[2, 0], [0, 1]] @ [[0.5, 0.2], [0.1, 0.5]] @ [[0.5, 0], [0, 1]]
    #            = [[1, 0.4], [0.1, 0.5]] @ [[0.5, 0], [0, 1]]
    #            = [[0.5, 0.4], [0.05, 0.5]]
    # Column 0: |0.5| + |0.05| = 0.55
    # Column 1: |0.4| + |0.5| = 0.90
    gain = compute_induced_1norm_gain(P, A)
    expected = th.tensor([0.55, 0.90], dtype=th.float64)
    assert th.allclose(gain, expected)


def test_optimize_polyhedral_contraction_matrix_already_contractive():
    # Stable matrix with gain already < 0.9
    A = th.tensor([[0.6, 0.1], [0.0, 0.7]], dtype=th.float64)
    P_init = th.eye(2, dtype=th.float64)
    P_opt, max_gain = optimize_polyhedral_contraction_matrix(
        A, P_init, kappa=0.01, target_gain=0.95
    )
    assert max_gain < 0.95
    assert isinstance(P_opt, np.ndarray)


def test_optimize_polyhedral_contraction_matrix_coupled_system():
    # Stable matrix with spectral radius 0.8, but uncoordinated coupling:
    # A = [[0.8, 0.5], [0.0, 0.7]]
    # For P = I, column 1 norm is 0.5 + 0.7 = 1.20 > 1.0 (expansive in Euclidean frame)!
    A = th.tensor(
        [[0.8, 0.5],
         [0.0, 0.7]],
        dtype=th.float64,
    )
    gain_eye = compute_induced_1norm_gain(th.eye(2, dtype=th.float64), A).max().item()
    assert gain_eye > 1.0  # Expansive in standard 1-norm frame!

    # Optimize polyhedral contraction matrix
    P_opt, max_gain = optimize_polyhedral_contraction_matrix(
        A, th.eye(2, dtype=th.float64), kappa=0.01, steps=500
    )
    assert max_gain < 1.0 - 0.01


def test_scale_riccati_matrix():
    from lcil.lyapunov_learning import scale_riccati_matrix

    P = th.tensor([[4.0, 0.0], [0.0, 2.0]], dtype=th.float32)
    # Spectral scale divides by max eigenvalue = 4.0
    p_spec = scale_riccati_matrix(P, scale_mode="spectral")
    assert th.allclose(p_spec, th.tensor([[1.0, 0.0], [0.0, 0.5]], dtype=th.float32))

    # Numeric divisor
    p_num = scale_riccati_matrix(P, scale_mode=2.0)
    assert th.allclose(p_num, th.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=th.float32))


def test_compute_polyhedral_value_matrix():
    from lcil.lyapunov_learning import compute_polyhedral_value_matrix

    A = th.tensor([[0.8, 0.5], [0.0, 0.7]], dtype=th.float64)
    P_opt = compute_polyhedral_value_matrix(
        a_closed_loop=A,
        p_initial=th.eye(2, dtype=th.float64),
        kappa=0.01,
        scale_mode="spectral",
    )
    assert isinstance(P_opt, np.ndarray)
    gain = compute_induced_1norm_gain(th.as_tensor(P_opt), A).max().item()
    assert gain < 1.0 - 0.01


def test_calculate_r_factor_from_riccati():
    from lcil.lyapunov_learning import calculate_r_factor_from_riccati

    P = th.tensor([[3.0, 1.0], [1.0, 2.0]], dtype=th.float32)
    r, eps = calculate_r_factor_from_riccati(P, eps=1e-3)
    reconstructed = eps * th.eye(2) + r.T @ r
    assert th.allclose(P, reconstructed, atol=1e-5)

