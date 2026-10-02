"""Analytic reference cases for the numerical Cartpole inspector."""

import numpy as np
import pytest
import torch as th
from torch import nn
import matplotlib.pyplot as plt

from examples.cartpole.inspect_lyapunov import (
    _plot_slice,
    _summary,
    evaluate_states,
    polyhedral_expansion,
)


class L1Candidate(nn.Module):
    def __init__(self, dimensions=2):
        super().__init__()
        self.register_buffer("origin", th.zeros(dimensions, dtype=th.float64))

    def forward(self, x):
        return (x - self.origin).abs().sum(-1, keepdim=True)


class LinearDynamics(nn.Module):
    def __init__(self, matrix):
        super().__init__()
        self.register_buffer("matrix", th.as_tensor(matrix, dtype=th.float64))

    def forward(self, x, u):
        return x @ self.matrix.T


@pytest.mark.parametrize("matrix,expected", [
    (np.array([[0.5, 1.0], [0.0, 0.5]]), 1.5),
    (0.5 * np.eye(2), 0.5),
    (np.zeros((2, 2)), 0.0),
])
def test_polyhedral_gain_including_stable_expansive_map(matrix, expected):
    # A stable Jacobian can expand a Lyapunov norm; zero dynamics still need a valid ray.
    direction, gain = polyhedral_expansion(matrix, np.eye(2))
    assert np.linalg.norm(direction) == pytest.approx(1.0)
    assert gain == pytest.approx(expected)
    assert np.abs(matrix @ direction).sum() / np.abs(direction).sum() == pytest.approx(gain)


def test_conditions_include_origin_hole_and_clip_domain():
    states = np.array([[0.0, 0.0], [0.0, 0.001], [0.0, 0.5], [0.0, 2.0]])
    bounds = np.array([[-1.0, -1.0], [1.0, 1.0]])
    values = evaluate_states(L1Candidate(), nn.Identity(),
                             LinearDynamics([[0.5, 1.0], [0.0, 0.5]]),
                             states, bounds, kappa=0.001, batch_size=2)
    assert np.isnan(values["ratio"][0])
    np.testing.assert_allclose(values["ratio"][1:], 1.5)
    assert values["decrease"][0] == 0.0
    assert not values["invariant"][-1]
    summary = _summary(states, values, rho=0.6, origin=np.zeros(2),
                       exclusion=np.full(2, 0.01))
    assert summary["inside_sublevel"] == 3
    assert summary["decrease_violations"] == 2
    assert summary["decrease_violations_inside_origin_exclusion"] == 1
    assert summary["outside_bounds"] == 1
    assert summary["invariance_violations"] == 0
    # Domain membership must also apply when no rho is available.
    unfiltered = _summary(states, values, rho=None, origin=np.zeros(2), exclusion=np.zeros(2))
    assert unfiltered["inside_sublevel"] == 3
    assert unfiltered["decrease_violations"] == 2


def test_slice_restores_exact_equilibrium_midpoint():
    bounds = np.array([[-1.0] * 4, [1.0] * 4])

    def evaluate(states):
        return evaluate_states(L1Candidate(4), nn.Identity(), LinearDynamics(0.5 * np.eye(4)),
                               states, bounds, kappa=0.001)

    figure, axis = plt.subplots()
    try:
        result = _plot_slice(axis, (1, 2), np.array([[-0.3, -0.3], [0.3, 0.3]]),
                             np.zeros(4), evaluate, rho=None, grid_size=301, kappa=0.001)
        center = result["states"].reshape(301, 301, 4)[150, 150]
        np.testing.assert_array_equal(center, np.zeros(4))
        assert not (result["values"]["decrease"] > 0.0).any()
    finally:
        plt.close(figure)
