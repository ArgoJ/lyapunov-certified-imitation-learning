import unittest

import torch as th
import torch.nn as nn

from lcil.utils import ERKIntegrator, IntegrationMethod


class ExponentialDynamics(nn.Module):
    def forward(self, x: th.Tensor, u: th.Tensor) -> th.Tensor:
        del u
        return x


class TestERKIntegrator(unittest.TestCase):
    def test_contains_expected_butcher_tableaus(self) -> None:
        expected_tableaus = {
            IntegrationMethod.EXPLICIT_EULER: (
                [[0.0]],
                [1.0],
                [0.0],
            ),
            IntegrationMethod.HEUN2: (
                [[0.0, 0.0], [1.0, 0.0]],
                [0.5, 0.5],
                [0.0, 1.0],
            ),
            IntegrationMethod.MIDPOINT_RK2: (
                [[0.0, 0.0], [0.5, 0.0]],
                [0.0, 1.0],
                [0.0, 0.5],
            ),
            IntegrationMethod.KUTTA3: (
                [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [-1.0, 2.0, 0.0]],
                [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
                [0.0, 0.5, 1.0],
            ),
            IntegrationMethod.HEUN3: (
                [[0.0, 0.0, 0.0], [1.0 / 3.0, 0.0, 0.0], [0.0, 2.0 / 3.0, 0.0]],
                [1.0 / 4.0, 0.0, 3.0 / 4.0],
                [0.0, 1.0 / 3.0, 2.0 / 3.0],
            ),
            IntegrationMethod.CLASSICAL_RK4: (
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.5, 0.0, 0.0, 0.0],
                    [0.0, 0.5, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ],
                [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0],
                [0.0, 0.5, 0.5, 1.0],
            ),
            IntegrationMethod.KUTTA_38_RK4: (
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [1.0 / 3.0, 0.0, 0.0, 0.0],
                    [-1.0 / 3.0, 1.0, 0.0, 0.0],
                    [1.0, -1.0, 1.0, 0.0],
                ],
                [1.0 / 8.0, 3.0 / 8.0, 3.0 / 8.0, 1.0 / 8.0],
                [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
            ),
        }

        for method, (a_expected, b_expected, c_expected) in expected_tableaus.items():
            with self.subTest(method=method.value):
                integrator = ERKIntegrator(ExponentialDynamics(), dt=0.1, method=method, dtype=th.float64)
                self.assertTrue(th.allclose(integrator.a, th.tensor(a_expected, dtype=th.float64)))
                self.assertTrue(th.allclose(integrator.b, th.tensor(b_expected, dtype=th.float64)))
                self.assertTrue(th.allclose(integrator.c, th.tensor(c_expected, dtype=th.float64)))

    def test_supports_requested_methods(self) -> None:
        dynamics = ExponentialDynamics()
        x = th.tensor([[1.0]], dtype=th.float64)
        u = th.zeros((1, 1), dtype=th.float64)
        expected = {
            IntegrationMethod.EXPLICIT_EULER: 1.1,
            IntegrationMethod.HEUN2: 1.105,
            IntegrationMethod.MIDPOINT_RK2: 1.105,
            IntegrationMethod.KUTTA3: 1.1051666666666666,
            IntegrationMethod.HEUN3: 1.1051666666666666,
            IntegrationMethod.CLASSICAL_RK4: 1.1051708333333334,
            IntegrationMethod.KUTTA_38_RK4: 1.1051708333333334,
        }

        for method, expected_value in expected.items():
            with self.subTest(method=method.value):
                integrator = ERKIntegrator(dynamics, dt=0.1, method=method, dtype=th.float64)
                result = integrator(x, u)
                target = th.tensor([[expected_value]], dtype=th.float64)
                self.assertTrue(th.allclose(result, target, atol=1e-10, rtol=0.0))

    def test_rejects_unknown_method(self) -> None:
        with self.assertRaises(ValueError):
            ERKIntegrator(ExponentialDynamics(), dt=0.1, method="rk5", dtype=th.float64)


if __name__ == "__main__":
    unittest.main(verbosity=2)