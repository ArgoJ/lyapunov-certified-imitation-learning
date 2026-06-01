import unittest

import numpy as np
import torch as th
import torch.nn as nn
from shared_utils import _DoubleDynamics, _IdentityLyapunov, _ZeroLyapunov, _ZeroPolicy

from lcil.certification.empirical_certification_tester import CertificationResultTester
from lcil.certification.config import LyapunovCertificationConfig


class _FixedCandidateTester(CertificationResultTester):
    def __init__(
        self,
        *,
        candidate_batches: list[np.ndarray],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if len(candidate_batches) == 0:
            raise ValueError("candidate_batches must not be empty.")
        self._candidate_batches = [
            th.as_tensor(batch, dtype=th.float32, device=self.device).reshape(-1, self.config.state_dim)
            for batch in candidate_batches
        ]
        self._candidate_index = 0

    def _sample_uniform_states(self, sample_size: int) -> th.Tensor:
        del sample_size
        batch_index = min(self._candidate_index, len(self._candidate_batches) - 1)
        self._candidate_index += 1
        return self._candidate_batches[batch_index]


class _DeterministicBoundaryTester(CertificationResultTester):
    def __init__(
        self,
        *,
        sampled_states: np.ndarray,
        sampled_values: np.ndarray,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._sampled_states = th.as_tensor(
            sampled_states,
            dtype=th.float32,
            device=self.device,
        ).reshape(-1, self.config.state_dim)
        self._sampled_values = th.as_tensor(
            sampled_values,
            dtype=th.float32,
            device=self.device,
        ).reshape(-1)

    def _sample_near_rho_within_sublevel(
        self,
        rho: float,
        sample_size: int,
    ) -> tuple[th.Tensor, th.Tensor]:
        del rho, sample_size
        return self._sampled_states, self._sampled_values


class _MappedVerifier(nn.Module):
    def __init__(self, transitions: dict[float, tuple[float, float]]) -> None:
        super().__init__()
        self._transitions = transitions

    def forward(self, x: th.Tensor) -> th.Tensor:
        outputs = []
        for value in x.reshape(-1).tolist():
            violation, x_next = self._transitions[float(value)]
            outputs.append((violation, 0.0, x_next))
        return th.as_tensor(outputs, dtype=x.dtype, device=x.device)


class _MappedViolationTester(_DeterministicBoundaryTester):
    def __init__(self, *, transitions: dict[float, tuple[float, float]], **kwargs) -> None:
        super().__init__(**kwargs)
        self.verifier = _MappedVerifier(transitions).to(self.device).eval()

    def _hard_condition_violation(self, verifier_output: th.Tensor) -> th.Tensor:
        return verifier_output[:, :1]


class TestCertificationResultTester(unittest.TestCase):
    @staticmethod
    def _make_config(
        *,
        bounds: list[list[float]] | None = None,
        kappa: float = 0.1,
        origin_exclusion: float | tuple[float, ...] = 0.0,
    ) -> LyapunovCertificationConfig:
        cert_bounds = np.array([[0.0], [10.0]], dtype=np.float32) if bounds is None else np.array(bounds, dtype=np.float32)
        return LyapunovCertificationConfig(
            state_dim=1,
            cert_bounds=cert_bounds,
            kappa=kappa,
            rho_min=1e-6,
            bins_per_dim=1,
            origin_exclusion=origin_exclusion,
            sublevel_tolerance=1e-6,
            condition_tolerance=1e-6,
        )

    @classmethod
    def _make_tester(
        cls,
        tester_cls: type[CertificationResultTester] = CertificationResultTester,
        *,
        bounds: list[list[float]] | None = None,
        kappa: float = 0.1,
        origin_exclusion: float | tuple[float, ...] = 0.0,
        policy_model: nn.Module | None = None,
        lyap_model: nn.Module | None = None,
        dyn_model: nn.Module | None = None,
        **tester_kwargs,
    ) -> CertificationResultTester:
        return tester_cls(
            policy_model=_ZeroPolicy() if policy_model is None else policy_model,
            lyap_model=_IdentityLyapunov() if lyap_model is None else lyap_model,
            dyn_model=_DoubleDynamics() if dyn_model is None else dyn_model,
            config=cls._make_config(bounds=bounds, kappa=kappa, origin_exclusion=origin_exclusion),
            device=th.device("cpu"),
            **tester_kwargs,
        )

    def test_evaluate_rho_returns_empty_summary_when_no_inside_samples_exist(self) -> None:
        tester = self._make_tester(bounds=[[1.0], [2.0]])

        result = tester._evaluate_rho(
            rho=0.5,
            sample_size=4,
            tolerance=1e-6,
            rollout_steps=1,
        )

        self.assertEqual(result.num_samples, 0)
        self.assertEqual(result.violation_rate, 0.0)
        self.assertEqual(result.max_violation, 0.0)
        self.assertIsNone(result.sampled_states)
        self.assertIsNone(result.sampled_values)
        self.assertIsNone(result.violations_per_step)
        self.assertIsNone(result.rollout_states)

    def test_sample_near_rho_prefers_states_closest_to_rho_from_below(self) -> None:
        tester = self._make_tester(
            _FixedCandidateTester,
            bounds=[[0.0], [2.0]],
            candidate_batches=[np.array([[0.2], [0.9], [1.2], [0.8]], dtype=np.float32)],
        )

        sampled_states, sampled_values = tester._sample_near_rho_within_sublevel(
            rho=1.0,
            sample_size=2,
        )

        np.testing.assert_allclose(
            sampled_states.cpu().numpy(),
            np.array([[0.9], [0.8]], dtype=np.float32),
        )
        np.testing.assert_allclose(
            sampled_values.cpu().numpy(),
            np.array([0.9, 0.8], dtype=np.float32),
        )

    def test_inside_origin_exclusion_mask_marks_only_states_in_exclusion_box(self) -> None:
        tester = self._make_tester(origin_exclusion=0.1)

        mask = tester._inside_origin_exclusion_mask(
            th.as_tensor([[0.0], [0.05], [0.1], [0.1001], [0.2]], dtype=th.float32)
        )

        self.assertListEqual(mask.cpu().tolist(), [True, True, True, False, False])

    def test_inside_origin_exclusion_mask_supports_rollout_state_tensor(self) -> None:
        tester = self._make_tester(origin_exclusion=0.1)

        mask = tester._inside_origin_exclusion_mask(
            th.as_tensor(
                [
                    [[0.2], [0.05], [0.01]],
                    [[0.2], [0.15], [0.1]],
                ],
                dtype=th.float32,
            )
        )

        self.assertListEqual(mask.cpu().tolist(), [[False, True, True], [False, False, True]])

    def test_inside_origin_exclusion_mask_supports_per_dimension_sequence(self) -> None:
        config = LyapunovCertificationConfig(
            state_dim=2,
            cert_bounds=np.array([[-1.0, -1.0], [1.0, 1.0]], dtype=np.float32),
            kappa=0.1,
            rho_min=1e-6,
            bins_per_dim=1,
            origin_exclusion=(0.1, 0.2),
            sublevel_tolerance=1e-6,
            condition_tolerance=1e-6,
        )
        tester = CertificationResultTester(
            policy_model=_ZeroPolicy(),
            lyap_model=_ZeroLyapunov(),
            dyn_model=_DoubleDynamics(),
            config=config,
            device=th.device("cpu"),
        )

        mask = tester._inside_origin_exclusion_mask(
            th.as_tensor(
                [[0.05, 0.15], [0.05, 0.25], [0.15, 0.15], [0.15, 0.25]],
                dtype=th.float32,
            )
        )

        self.assertListEqual(mask.cpu().tolist(), [True, False, False, False])

    def test_sample_near_rho_excludes_states_inside_origin_exclusion(self) -> None:
        tester = self._make_tester(
            _FixedCandidateTester,
            bounds=[[-1.0], [1.0]],
            origin_exclusion=0.1,
            candidate_batches=[np.array([[0.0], [0.05], [0.2], [0.9]], dtype=np.float32)],
        )

        sampled_states, sampled_values = tester._sample_near_rho_within_sublevel(
            rho=1.0,
            sample_size=2,
        )

        np.testing.assert_allclose(
            sampled_states.cpu().numpy(),
            np.array([[0.9], [0.2]], dtype=np.float32),
        )
        np.testing.assert_allclose(
            sampled_values.cpu().numpy(),
            np.array([0.9, 0.2], dtype=np.float32),
        )

    def test_evaluate_rho_rolls_out_sampled_boundary_states(self) -> None:
        tester = self._make_tester(
            _DeterministicBoundaryTester,
            sampled_states=np.array([[1.5]], dtype=np.float32),
            sampled_values=np.array([1.5], dtype=np.float32),
        )

        result = tester._evaluate_rho(
            rho=2.0,
            sample_size=1,
            tolerance=1e-6,
            rollout_steps=1,
        )

        self.assertEqual(result.num_samples, 1)
        self.assertEqual(result.violation_rate, 1.0)
        self.assertAlmostEqual(result.max_violation, 1.65, places=6)
        np.testing.assert_allclose(result.sampled_states, np.array([[1.5]], dtype=np.float32))
        np.testing.assert_allclose(result.sampled_values, np.array([1.5], dtype=np.float32))
        np.testing.assert_allclose(result.violations_per_step, np.array([[1.65]], dtype=np.float32))
        np.testing.assert_allclose(
            result.rollout_states,
            np.array([[[1.5], [3.0]]], dtype=np.float32),
        )

    def test_test_result_wraps_rho_boundary_summary(self) -> None:
        tester = self._make_tester(
            _DeterministicBoundaryTester,
            sampled_states=np.array([[0.9], [0.8]], dtype=np.float32),
            sampled_values=np.array([0.9, 0.8], dtype=np.float32),
        )

        result = tester.test_result(rho=1.0, sample_size=8, rollout_steps=1)

        self.assertEqual(result.sample_size, 8)
        self.assertEqual(result.rollout_steps, 1)
        self.assertEqual(result.rho, 1.0)
        self.assertEqual(result.rho_boundary.num_samples, 2)
        np.testing.assert_allclose(
            result.rho_boundary.sampled_values,
            np.array([0.9, 0.8], dtype=np.float32),
        )
        np.testing.assert_allclose(
            result.rho_boundary.rollout_states,
            np.array([[[0.9], [1.8]], [[0.8], [1.6]]], dtype=np.float32),
        )

    def test_evaluate_rho_captures_state_invariance_violations_from_verifier_output(self) -> None:
        tester = self._make_tester(
            _DeterministicBoundaryTester,
            bounds=[[0.0], [3.0]],
            lyap_model=_ZeroLyapunov(),
            sampled_states=np.array([[2.0]], dtype=np.float32),
            sampled_values=np.array([0.0], dtype=np.float32),
        )

        result = tester._evaluate_rho(
            rho=0.1,
            sample_size=1,
            tolerance=1e-6,
            rollout_steps=1,
        )

        self.assertEqual(result.num_samples, 1)
        self.assertEqual(result.violation_rate, 1.0)
        self.assertAlmostEqual(result.max_violation, 1.0, places=6)
        np.testing.assert_allclose(result.violations_per_step, np.array([[1.0]], dtype=np.float32))
        np.testing.assert_allclose(
            result.rollout_states,
            np.array([[[2.0], [4.0]]], dtype=np.float32),
        )

    def test_evaluate_rho_records_all_rollout_states(self) -> None:
        tester = self._make_tester(
            _DeterministicBoundaryTester,
            sampled_states=np.array([[1.5]], dtype=np.float32),
            sampled_values=np.array([1.5], dtype=np.float32),
        )

        result = tester._evaluate_rho(
            rho=2.0,
            sample_size=1,
            tolerance=1e-6,
            rollout_steps=3,
        )

        np.testing.assert_allclose(
            result.rollout_states,
            np.array([[[1.5], [3.0], [6.0], [12.0]]], dtype=np.float32),
        )

    def test_evaluate_rho_computes_violation_stats_per_trajectory(self) -> None:
        tester = self._make_tester(
            _MappedViolationTester,
            sampled_states=np.array([[0.0], [1.0]], dtype=np.float32),
            sampled_values=np.array([0.0, 1.0], dtype=np.float32),
            transitions={
                0.0: (0.0, 10.0),
                10.0: (2.0, 20.0),
                20.0: (0.0, 30.0),
                1.0: (0.0, 11.0),
                11.0: (0.0, 21.0),
                21.0: (0.0, 31.0),
            },
        )

        result = tester._evaluate_rho(
            rho=2.0,
            sample_size=2,
            tolerance=1e-6,
            rollout_steps=3,
        )

        self.assertEqual(result.num_samples, 2)
        self.assertEqual(result.violation_rate, 0.5)
        self.assertAlmostEqual(result.max_violation, 2.0, places=6)
        np.testing.assert_allclose(
            result.violations_per_step,
            np.array([[0.0, 2.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32),
        )

    def test_evaluate_rho_ignores_rollout_states_inside_origin_exclusion(self) -> None:
        tester = self._make_tester(
            _MappedViolationTester,
            origin_exclusion=0.1,
            sampled_states=np.array([[0.1875], [0.5]], dtype=np.float32),
            sampled_values=np.array([0.1875, 0.5], dtype=np.float32),
            transitions={
                0.1875: (0.0, 0.0625),
                0.0625: (5.0, 0.03125),
                0.03125: (6.0, 0.015625),
                0.5: (0.0, 0.375),
                0.375: (2.0, 0.25),
                0.25: (0.0, 0.125),
            },
        )

        result = tester._evaluate_rho(
            rho=1.0,
            sample_size=2,
            tolerance=1e-6,
            rollout_steps=3,
        )

        self.assertEqual(result.num_samples, 2)
        self.assertEqual(result.violation_rate, 0.5)
        self.assertAlmostEqual(result.max_violation, 2.0, places=6)
        np.testing.assert_allclose(
            result.violations_per_step,
            np.array([[0.0, np.nan, np.nan], [0.0, 2.0, 0.0]], dtype=np.float32),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            result.rollout_states,
            np.array(
                [
                    [[0.1875], [np.nan], [np.nan], [np.nan]],
                    [[0.5], [0.375], [0.25], [0.125]],
                ],
                dtype=np.float32,
            ),
            equal_nan=True,
        )

    def test_evaluate_rho_handles_empty_origin_exclusion_hits(self) -> None:
        tester = self._make_tester(
            _MappedViolationTester,
            origin_exclusion=0.1,
            sampled_states=np.array([[0.5]], dtype=np.float32),
            sampled_values=np.array([0.5], dtype=np.float32),
            transitions={
                0.5: (0.0, 0.4),
                0.4: (0.0, 0.3),
            },
        )

        result = tester._evaluate_rho(
            rho=1.0,
            sample_size=1,
            tolerance=1e-6,
            rollout_steps=2,
        )

        self.assertEqual(result.num_samples, 1)
        self.assertEqual(result.violation_rate, 0.0)
        self.assertAlmostEqual(result.max_violation, 0.0, places=6)
        np.testing.assert_allclose(
            result.violations_per_step,
            np.array([[0.0, 0.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(
            result.rollout_states,
            np.array([[[0.5], [0.4], [0.3]]], dtype=np.float32),
        )

    def test_rollout_steps_and_sample_size_must_be_positive(self) -> None:
        tester = self._make_tester(
            _DeterministicBoundaryTester,
            sampled_states=np.array([[1.0]], dtype=np.float32),
            sampled_values=np.array([1.0], dtype=np.float32),
        )

        with self.assertRaisesRegex(ValueError, "rollout_steps must be positive"):
            tester._evaluate_rho(rho=1.0, sample_size=1, tolerance=1e-6, rollout_steps=0)

        with self.assertRaisesRegex(ValueError, "sample_size must be positive"):
            tester._evaluate_rho(rho=1.0, sample_size=0, tolerance=1e-6, rollout_steps=1)

        with self.assertRaisesRegex(ValueError, "rollout_steps must be positive"):
            tester.test_result(rho=1.0, sample_size=1, rollout_steps=0)

        with self.assertRaisesRegex(ValueError, "sample_size must be positive"):
            tester.test_result(rho=1.0, sample_size=0, rollout_steps=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
