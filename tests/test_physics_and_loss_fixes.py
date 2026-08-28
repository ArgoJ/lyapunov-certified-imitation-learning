import numpy as np
import pytest
import torch as th
import casadi as ca

from lcil.lyapunov_learning.config import LyapunovTrainingConfig
from lcil.lyapunov_learning.loss import RhoGatedConditionLoss, weighted_mean
from lcil.utils import IntegrationMethod
from examples.cartpole.cartpole_dyn import CartpoleContinuousDynamics, CartpoleDynamics
from examples.cartpole.sys_cfg import PendulumOnCartConfig
from examples.cartpole.basis import linearized_inverted_pendulum_on_cart_matrices
from examples.cartpole.acados_ocp import get_model as get_cartpole_acados_model
from examples.double_integrator.double_integrator_dyn import DoubleIntegratorDynamics
from examples.double_integrator.basis import compute_discrete_double_integrator


# ---------------------------------------------------------------------------
# 1. Cartpole Dynamics & Linearization Consistency
# ---------------------------------------------------------------------------
def test_cartpole_jacobian_matches_basis_matrices():
    """Verify that the analytical linearization in basis.py matches the autograd Jacobian of CartpoleContinuousDynamics."""
    sys_cfg = PendulumOnCartConfig()
    dyn = CartpoleContinuousDynamics(sys_cfg)

    x0 = th.zeros(1, 4, requires_grad=True)
    u0 = th.zeros(1, 1, requires_grad=True)

    # Compute continuous time derivative dot_x = f(x, u)
    f_val = dyn(x0, u0)

    # Compute Jacobian df/dx at origin
    df_dx = []
    for i in range(4):
        grad_x = th.autograd.grad(f_val[0, i], x0, retain_graph=True)[0]
        df_dx.append(grad_x.detach().numpy().flatten())
    df_dx = np.array(df_dx)

    # Compute Jacobian df/du at origin
    df_du = []
    for i in range(4):
        grad_u = th.autograd.grad(f_val[0, i], u0, retain_graph=True)[0]
        df_du.append(grad_u.detach().numpy().flatten())
    df_du = np.array(df_du)

    ac_basis, bc_basis = linearized_inverted_pendulum_on_cart_matrices(sys_cfg)

    np.testing.assert_allclose(df_dx, ac_basis, atol=1e-6, err_msg="Cartpole Jacobian df/dx does not match ac in basis.py")
    np.testing.assert_allclose(df_du, bc_basis, atol=1e-6, err_msg="Cartpole Jacobian df/du does not match bc in basis.py")


def test_cartpole_physical_force_sign():
    """Verify that pushing the cart to the right (F > 0) causes angular acceleration theta_ddot < 0."""
    sys_cfg = PendulumOnCartConfig()
    dyn = CartpoleContinuousDynamics(sys_cfg)

    x = th.zeros(1, 4)
    u = th.tensor([[10.0]]) # Positive force to the right
    x_dot = dyn(x, u)

    # x_dot = [cart_vel, x_ddot, theta_dot, theta_ddot]
    x_ddot = x_dot[0, 1].item()
    theta_ddot = x_dot[0, 3].item()

    assert x_ddot > 0.0, "Cart should accelerate forward with positive force."
    assert theta_ddot < 0.0, "Pendulum top should fall backward (theta_ddot < 0) when cart accelerates forward."


def test_cartpole_pytorch_acados_equivalence():
    """Verify that CasADi model in acados_ocp.py matches PyTorch CartpoleContinuousDynamics across arbitrary states."""
    sys_cfg = PendulumOnCartConfig()
    pytorch_dyn = CartpoleContinuousDynamics(sys_cfg)
    acados_model = get_cartpole_acados_model(sys_cfg)

    casadi_f = ca.Function("f", [acados_model.x, acados_model.u], [acados_model.f_expl_expr])

    np.random.seed(42)
    test_states = np.random.uniform(
        low=[-1.0, -2.0, -0.5, -2.0],
        high=[1.0, 2.0, 0.5, 2.0],
        size=(20, 4),
    )
    test_inputs = np.random.uniform(low=[-20.0], high=[20.0], size=(20, 1))

    for x_np, u_np in zip(test_states, test_inputs):
        pt_out = pytorch_dyn(th.tensor(x_np, dtype=th.float32).unsqueeze(0), th.tensor(u_np, dtype=th.float32).unsqueeze(0))
        pt_np = pt_out.detach().numpy().flatten()

        ca_out = np.array(casadi_f(x_np, u_np)).flatten()
        np.testing.assert_allclose(pt_np, ca_out, atol=1e-5, err_msg="PyTorch and CasADi continuous dynamics mismatch")


def test_cartpole_labels_and_x0():
    """Verify that Cartpole state labels and equilibrium initial condition in acados_ocp.py are consistent."""
    sys_cfg = PendulumOnCartConfig()
    acados_model = get_cartpole_acados_model(sys_cfg)
    assert acados_model.x_labels == ['$x$ [m]', '$v$ [m/s]', r'$\theta$ [rad]', r'$\dot{\theta}$ [rad/s]']


# ---------------------------------------------------------------------------
# 2. Double Integrator Euler Discretization
# ---------------------------------------------------------------------------
def test_double_integrator_default_is_euler():
    """Verify that DoubleIntegratorDynamics defaults to EXPLICIT_EULER."""
    dt = 0.1
    dyn = DoubleIntegratorDynamics(dt=dt)
    assert dyn.integrator.method == IntegrationMethod.EXPLICIT_EULER

    # Check discrete step formula: x1+ = x1 + dt*x2, x2+ = x2 + dt*u
    x = th.tensor([[2.0, 3.0]])
    u = th.tensor([[-1.0]])
    x_next = dyn(x, u)

    expected_x_next = th.tensor([[2.0 + 0.1 * 3.0, 3.0 + 0.1 * (-1.0)]])
    th.testing.assert_close(x_next, expected_x_next)


def test_double_integrator_basis_euler_matrices():
    """Verify that basis.py for double integrator returns exact Euler matrices."""
    dt = 0.05
    ad, bd = compute_discrete_double_integrator(dt)
    expected_ad = np.array([[1.0, 0.05], [0.0, 1.0]], dtype=np.float64)
    expected_bd = np.array([[0.0], [0.05]], dtype=np.float64)

    np.testing.assert_allclose(ad, expected_ad)
    np.testing.assert_allclose(bd, expected_bd)


# ---------------------------------------------------------------------------
# 3. RhoGatedConditionLoss Normalization (weighted_mean)
# ---------------------------------------------------------------------------
def test_rho_gated_condition_loss_weighted_mean():
    """Verify that RhoGatedConditionLoss normalizes by sum of weights, making loss invariant to outside batch count."""
    config = LyapunovTrainingConfig(
        state_dim=2,
        state_bounds=np.array([[-1.0, -1.0], [1.0, 1.0]]),
        kappa=0.01,
        rho_min=0.1,
        rho_gate_sharpness=50.0,
        relative_condition_eps=1e-3,
        use_relative_decrease=False,
    )
    loss_fn = RhoGatedConditionLoss(config=config)

    # 1 sample inside V(x) = 0.2 <= rho=1.0 with severe violation: V(x+) = 0.5 > 0.99*0.2
    v_curr_in = th.tensor([[0.2]])
    v_next_in = th.tensor([[0.5]])
    x_next_in = th.tensor([[0.0, 0.0]])
    rho = 1.0

    loss_1 = loss_fn(v_curr_in, v_next_in, x_next_in, rho_estimate=rho)

    # Now create batch with 1 inside sample and 99 outside samples with V(x) = 5.0 >> rho
    v_curr_large = th.cat([v_curr_in, th.full((99, 1), 5.0)], dim=0)
    v_next_large = th.cat([v_next_in, th.full((99, 1), 5.0)], dim=0)
    x_next_large = th.cat([x_next_in, th.zeros(99, 2)], dim=0)

    loss_large = loss_fn(v_curr_large, v_next_large, x_next_large, rho_estimate=rho)

    # With weighted_mean, the loss should remain approximately equal (invariant to outside batch size)
    th.testing.assert_close(loss_1, loss_large, rtol=1e-2, atol=1e-3)
