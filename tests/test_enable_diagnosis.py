import pytest
import torch as th
import torch.nn as nn
import numpy as np

from lcil.lyapunov_learning.config import LyapunovTrainingConfig
from lcil.lyapunov_learning.loss import LyapunovTrainingLoss
from lcil.lyapunov_learning.counterexample import estimate_rho_from_boundary, BoundaryTermDiagnostics
from lcil.utils import MLP


class DummyPolicy(nn.Module):
    def forward(self, x: th.Tensor) -> th.Tensor:
        return th.zeros((x.shape[0], 1), device=x.device)


class DummyDynamics(nn.Module):
    def forward(self, x: th.Tensor, u: th.Tensor) -> th.Tensor:
        return x * 0.95


class DummyLyapunov(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = MLP([2, 16, 1], ["relu", "identity"])

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.net(x).pow(2).sum(dim=-1, keepdim=True) + 0.01


@pytest.fixture
def base_config():
    return LyapunovTrainingConfig(
        state_dim=2,
        state_bounds=np.array([[-1.0, -1.0], [1.0, 1.0]]),
        equilibrium_weight=0.0,
        enable_diagnosis=True,
    )


def test_loss_zero_weight_with_diagnosis(base_config):
    policy = DummyPolicy()
    lyap = DummyLyapunov()
    dyn = DummyDynamics()

    cfg_diag = base_config
    loss_mod = LyapunovTrainingLoss(policy, lyap, dyn, cfg_diag)

    x_batch = th.randn(10, 2, requires_grad=True)
    roa_candidates = th.randn(10, 2)

    parts = loss_mod.compute_loss_parts(x_batch, roa_candidates, rho_estimate=0.5)

    # Equilibrium weight is 0.0, but enable_diagnosis=True
    # equilibrium_raw should be computed without gradients (requires_grad=False)
    assert parts.equilibrium_raw.requires_grad is False
    assert parts.equilibrium_weight == 0.0
    assert parts.equilibrium.item() == 0.0


def test_loss_zero_weight_without_diagnosis(base_config):
    policy = DummyPolicy()
    lyap = DummyLyapunov()
    dyn = DummyDynamics()

    cfg_no_diag = LyapunovTrainingConfig(
        state_dim=2,
        state_bounds=np.array([[-1.0, -1.0], [1.0, 1.0]]),
        equilibrium_weight=0.0,
        enable_diagnosis=False,
    )
    loss_mod = LyapunovTrainingLoss(policy, lyap, dyn, cfg_no_diag)

    x_batch = th.randn(10, 2, requires_grad=True)
    roa_candidates = th.randn(10, 2)

    parts = loss_mod.compute_loss_parts(x_batch, roa_candidates, rho_estimate=0.5)

    # When enable_diagnosis=False and weight=0.0, raw value is zero tensor directly
    assert parts.equilibrium_raw.item() == 0.0
    assert parts.equilibrium_raw.requires_grad is False


def test_rho_diagnostics_enable_diagnosis():
    lyap = DummyLyapunov()

    cfg_diag = LyapunovTrainingConfig(
        state_dim=2,
        state_bounds=np.array([[-1.0, -1.0], [1.0, 1.0]]),
        enable_diagnosis=True,
    )
    eval_diag, _ = estimate_rho_from_boundary(lyap, cfg_diag)

    cfg_no_diag = LyapunovTrainingConfig(
        state_dim=2,
        state_bounds=np.array([[-1.0, -1.0], [1.0, 1.0]]),
        enable_diagnosis=False,
    )
    eval_no_diag, _ = estimate_rho_from_boundary(lyap, cfg_no_diag)

    # When enable_diagnosis=False, term diagnostics returns NaN without computing
    assert np.isnan(eval_no_diag.terms.feature_term_quantile)
    assert np.isnan(eval_no_diag.terms.linear_term_quantile)
