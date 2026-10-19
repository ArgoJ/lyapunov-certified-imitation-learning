"""Unit tests for shared control and matrix utilities in lcil.utils.control."""

import numpy as np
import pytest
import torch as th
import torch.nn as nn

from lcil.imitation_learning.models import LQRController
from lcil.utils import (
    QuadraticLyapunov,
    RiccatiQuadraticLyapunov,
    cholesky_factor,
    compute_lqr_gain,
    extract_p_matrix,
    get_bounds_center_and_radius,
    is_positive_definite,
    quad_form,
    quadratic_sublevel_set_volume,
    solve_discrete_are,
    symmetrize,
)


def test_symmetrize_numpy_and_tensor():
    # 2D numpy
    arr = np.array([[1.0, 2.0], [4.0, 5.0]])
    sym_arr = symmetrize(arr)
    expected_arr = np.array([[1.0, 3.0], [3.0, 5.0]])
    np.testing.assert_allclose(sym_arr, expected_arr)

    # 2D torch
    t = th.tensor([[1.0, 2.0], [4.0, 5.0]])
    sym_t = symmetrize(t)
    assert th.allclose(sym_t, th.tensor([[1.0, 3.0], [3.0, 5.0]]))

    # Batched (B, n, n)
    batched_t = th.stack([t, t * 2.0], dim=0)
    sym_batched = symmetrize(batched_t)
    assert sym_batched.shape == (2, 2, 2)
    assert th.allclose(sym_batched[0], sym_t)
    assert th.allclose(sym_batched[1], sym_t * 2.0)

    # Batched numpy
    batched_arr = np.stack([arr, arr * 2.0], axis=0)
    sym_batched_arr = symmetrize(batched_arr)
    assert sym_batched_arr.shape == (2, 2, 2)
    np.testing.assert_allclose(sym_batched_arr[0], expected_arr)

    # Error cases
    with pytest.raises(ValueError):
        symmetrize(th.ones(3))  # 1D
    with pytest.raises(ValueError):
        symmetrize(np.ones((2, 3)))  # non-square
    with pytest.raises(TypeError):
        symmetrize("invalid")  # type: ignore[arg-type]


def test_is_positive_definite():
    # Identity is PD
    eye = np.eye(3)
    assert is_positive_definite(eye)
    assert is_positive_definite(th.eye(3))

    # Indefinite matrix
    indef = np.array([[1.0, 0.0], [0.0, -1.0]])
    assert not is_positive_definite(indef)
    assert not is_positive_definite(th.as_tensor(indef))

    # Singular / zero eigenvalue
    singular = np.array([[1.0, 0.0], [0.0, 0.0]])
    assert not is_positive_definite(singular, tol=1e-8)


def test_solve_discrete_are_and_compute_lqr_gain():
    # Double integrator discrete system
    dt = 0.1
    A = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
    B = np.array([[0.5 * dt * dt], [dt]], dtype=np.float64)
    Q = np.diag([1.0, 0.5]).astype(np.float64)
    R = np.array([[0.25]], dtype=np.float64)

    # Solve as numpy array
    P_np = solve_discrete_are(A, B, Q, R, as_tensor=False)
    assert isinstance(P_np, np.ndarray)
    assert P_np.shape == (2, 2)
    assert is_positive_definite(P_np)
    # Check DARE residual: A^T P A - P - (A^T P B)(R + B^T P B)^(-1)(B^T P A) + Q = 0
    M = R + B.T @ P_np @ B
    gain = np.linalg.solve(M, B.T @ P_np @ A)
    dare_res = A.T @ P_np @ A - P_np - A.T @ P_np @ B @ gain + Q
    np.testing.assert_allclose(dare_res, np.zeros((2, 2)), atol=1e-7)

    # Compute gain with compute_lqr_gain
    K_np = compute_lqr_gain(A, B, P_np, R, as_tensor=False)
    assert isinstance(K_np, np.ndarray)
    assert K_np.shape == (1, 2)
    np.testing.assert_allclose(K_np, gain, atol=1e-7)

    # Solve as torch tensor
    P_th = solve_discrete_are(A, B, Q, R, as_tensor=True, dtype=th.float32)
    assert isinstance(P_th, th.Tensor)
    assert P_th.dtype == th.float32
    assert P_th.shape == (2, 2)

    K_th = compute_lqr_gain(A, B, P_th, R, as_tensor=True, dtype=th.float32)
    assert isinstance(K_th, th.Tensor)
    assert K_th.dtype == th.float32
    assert K_th.shape == (1, 2)
    np.testing.assert_allclose(K_th.numpy(), K_np, atol=1e-5)


def test_cholesky_factor():
    P = np.array([[4.0, 2.0], [2.0, 5.0]], dtype=np.float64)
    # as tensor
    L_th = cholesky_factor(P, as_tensor=True, dtype=th.float64)
    assert isinstance(L_th, th.Tensor)
    P_reconstructed = L_th @ L_th.T
    np.testing.assert_allclose(P_reconstructed.numpy(), P, atol=1e-7)

    # as array
    L_np = cholesky_factor(P, as_tensor=False)
    assert isinstance(L_np, np.ndarray)
    np.testing.assert_allclose(L_np @ L_np.T, P, atol=1e-7)

    # with jitter on near-singular matrix
    singular_P = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    L_jitter = cholesky_factor(singular_P, jitter=1e-4, as_tensor=False)
    assert np.all(np.isfinite(L_jitter))


def test_quad_form():
    x = th.tensor([[1.0, 2.0], [-1.0, 3.0]], dtype=th.float32)
    P = th.tensor([[2.0, 0.0], [0.0, 3.0]], dtype=th.float32)
    # x0: 2*1^2 + 3*2^2 = 2 + 12 = 14
    # x1: 2*(-1)^2 + 3*3^2 = 2 + 27 = 29
    res = quad_form(x, P)
    assert res.shape == (2, 1)
    assert th.allclose(res, th.tensor([[14.0], [29.0]]))


def test_get_bounds_center_and_radius():
    bounds = np.array([[-2.0, 1.0], [4.0, 5.0]])
    center, radius = get_bounds_center_and_radius(bounds)
    np.testing.assert_allclose(center, [1.0, 3.0])
    np.testing.assert_allclose(radius, [3.0, 2.0])

    bounds_th = th.as_tensor(bounds)
    center_th, radius_th = get_bounds_center_and_radius(bounds_th)
    assert th.allclose(center_th, th.tensor([1.0, 3.0], dtype=th.float64))
    assert th.allclose(radius_th, th.tensor([3.0, 2.0], dtype=th.float64))

    with pytest.raises(ValueError):
        get_bounds_center_and_radius(np.ones((3, 2)))


def test_extract_p_matrix():
    raw_P = np.array([[2.0, 0.5], [0.5, 3.0]])
    # Direct matrix
    P_th = extract_p_matrix(raw_P, as_tensor=True, dtype=th.float64)
    assert isinstance(P_th, th.Tensor)
    np.testing.assert_allclose(P_th.numpy(), raw_P)

    P_np = extract_p_matrix(raw_P, as_tensor=False)
    assert isinstance(P_np, np.ndarray)
    np.testing.assert_allclose(P_np, raw_P)

    # Object with get_pd_matrix
    class DummyPD:
        def get_pd_matrix(self):
            return raw_P

    np.testing.assert_allclose(extract_p_matrix(DummyPD(), as_tensor=False), raw_P)

    # Object with p_matrix attribute
    class DummyAttr:
        def __init__(self):
            self.p_matrix = raw_P

    np.testing.assert_allclose(extract_p_matrix(DummyAttr(), as_tensor=False), raw_P)

    # LQRController instance
    A = th.tensor([[1.0, 0.1], [0.0, 1.0]])
    B = th.tensor([[0.0], [0.1]])
    Q = th.diag(th.tensor([1.0, 0.5]))
    R = th.tensor([[0.25]])
    controller = LQRController(A=A, B=B, Q=Q, R=R)
    extracted = extract_p_matrix(controller, as_tensor=True)
    assert isinstance(extracted, th.Tensor)
    assert extracted.shape == (2, 2)
    assert is_positive_definite(extracted)

    # Invalid object
    with pytest.raises(ValueError):
        extract_p_matrix("not_a_matrix")


def test_quadratic_sublevel_set_volume():
    # Unit 2D ball: P = I_2, rho = 1 => Area = pi
    P_unit = np.eye(2)
    vol = quadratic_sublevel_set_volume(1.0, P_unit)
    np.testing.assert_allclose(vol, np.pi, rtol=1e-5)

    # Ellipsoid x^2 / a^2 + y^2 / b^2 <= 1 => Area = pi * a * b
    # Here x^T diag(1/a^2, 1/b^2) x <= 1
    a, b = 2.0, 3.0
    P_ell = np.diag([1.0 / (a * a), 1.0 / (b * b)])
    vol_ell = quadratic_sublevel_set_volume(1.0, P_ell)
    np.testing.assert_allclose(vol_ell, np.pi * a * b, rtol=1e-5)

    # Error cases
    with pytest.raises(ValueError):
        quadratic_sublevel_set_volume(-1.0, P_unit)
    with pytest.raises(ValueError):
        quadratic_sublevel_set_volume(1.0, np.array([[1.0, 0.0], [0.0, -1.0]]))


def test_quadratic_lyapunov_module():
    P = np.array([[3.0, 0.5], [0.5, 2.0]], dtype=np.float32)
    model = QuadraticLyapunov(P)
    assert isinstance(model, nn.Module)
    assert hasattr(model, "P")
    assert hasattr(model, "p_matrix")
    assert th.allclose(model.get_pd_matrix(), th.as_tensor(P))

    x = th.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=th.float32)
    val = model(x)
    assert val.shape == (3, 1)
    # x0: 3*1 = 3
    # x1: 2*1 = 2
    # x2: 3 + 2 + 2*0.5 = 6
    expected = th.tensor([[3.0], [2.0], [6.0]])
    assert th.allclose(val, expected)

    # Alias check
    assert RiccatiQuadraticLyapunov is QuadraticLyapunov
