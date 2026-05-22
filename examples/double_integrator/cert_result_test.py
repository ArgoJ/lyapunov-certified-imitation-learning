from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch as th
from mpc_datagen import MPCDataset, mdg_plt

from lcil.certification import (
    CertificationResultTester,
    LyapunovCertificationConfig,
    estimate_level_set_measure,
)
from lcil.imitation_learning.policy_rollout import PolicyRolloutConfig, PolicyRolloutGenerator
from lcil.lyapunov_learning import LyapunovRollout
from lcil.certification import RegionCertificationResult, CertificationTesterResult
from lcil.utils import lcil_plt
from lcil.utils.base_config import ArgumentParserConfig, config_field

from . import (
    DoubleIntegratorDynamics,
    load_lyapunov_model,
    load_policy_model,
)
from ..example_utils import discover_latest_cert_result_path, require_file, require_dir

__logger__ = logging.getLogger("lcil.examples.double_integrator.cert_result_test")

_DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results" / "double_integrator"


@dataclass(frozen=True)
class RetestCertResultScriptConfig(ArgumentParserConfig):
    cert_result_path: str | None = config_field(
        help="Optional path to certification_details.npz. Defaults to the latest saved certification result under results/double_integrator.",
    )
    device: str = config_field(
        default="cpu", 
        help="Torch device string (for example cpu or cuda).")
    rollout_steps: int = config_field(
        default=50,
        help="Closed-loop rollout steps per region center during empirical result testing.",
    )
    test_sample_size: int = config_field(
        default=256,
        help="Number of states sampled near the rho-sublevel boundary for empirical testing.",
    )


def _build_script_defaults() -> RetestCertResultScriptConfig:
    default_cert_result_path = discover_latest_cert_result_path(_DEFAULT_RESULTS_ROOT)
    return RetestCertResultScriptConfig(
        cert_result_path=str(default_cert_result_path),
    )


def parse_args() -> RetestCertResultScriptConfig:
    script_defaults = _build_script_defaults()
    parser = argparse.ArgumentParser(
        description="Load a saved certification result and rerun empirical center-rollout testing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    script_defaults.add_to_argparse(parser)
    args = parser.parse_args()
    return script_defaults.from_namespace(args)


def _build_lyapunov_func(
    lyap_model,
    device: th.device,
) -> Callable[[np.ndarray], np.ndarray]:
    def lyapunov_func(states: np.ndarray) -> np.ndarray:
        x = th.as_tensor(states, dtype=th.float32, device=device)
        with th.no_grad():
            values = lyap_model(x)
        return values.detach().cpu().numpy().reshape(-1)

    return lyapunov_func


def _build_rollout_dataset(
    *,
    test_results: CertificationTesterResult,
    policy_model,
    dyn_model,
    lyap_model,
    rollout_steps: int,
    device: th.device,
) -> MPCDataset | None:
    initial_states = test_results.rho_boundary.sampled_states
    if initial_states.shape[0] == 0:
        __logger__.warning("No certification-region centers available for rollout plotting.")
        return None

    rollout_config = PolicyRolloutConfig.from_mpc_config(
        policy_model.global_config,
        t_sim=int(rollout_steps),
    )
    policy_rollout_generator = PolicyRolloutGenerator(
        policy=policy_model,
        simulator=dyn_model,
        cfg=rollout_config,
        sampler=None,
        device=device,
    )
    rollout_dataset = policy_rollout_generator.generate_from_states(initial_states)
    LyapunovRollout(
        mpc_dataset=rollout_dataset,
        lyap_model=lyap_model,
        device=device,
    ).rollout()
    return rollout_dataset


def _save_lyapunov_plot(
    *,
    cert_dir: Path,
    certification_config: LyapunovCertificationConfig,
    lyapunov_func: Callable[[np.ndarray], np.ndarray],
    rollout_dataset: MPCDataset | None,
) -> None:
    cert_bounds = np.asarray(certification_config.cert_bounds, dtype=float)
    plot_path = cert_dir / "certification_tester_lyapunov_plot.html"
    mdg_plt.lyapunov(
        lyapunov_func=lyapunov_func,
        dataset=rollout_dataset,
        state_indices=[0, 1],
        state_labels=["$x$", "$v$"],
        limits=cert_bounds.T.tolist(),
        plot_3d=False,
        html_path=plot_path,
    )
    __logger__.info("Saved certification tester Lyapunov plot to %s", plot_path)


def _save_level_set_metrics(
    *,
    cert_dir: Path,
    cert_result: RegionCertificationResult,
    lyapunov_fn: Callable[[th.Tensor], th.Tensor],
    state_dim: int,
    device: th.device,
) -> None:
    metrics_path = cert_dir / "certification_tester_metrics.json"
    level_set_estimate = estimate_level_set_measure(
        lyapunov_fn=lyapunov_fn,
        rho=float(cert_result.rho),
        num_states=int(state_dim),
        device=device,
    )
    level_set_estimate.save(metrics_path)
    __logger__.info("Saved certification tester metrics to %s", metrics_path)


def main() -> None:
    script_config = parse_args()
    device = th.device(script_config.device)

    cert_result_path = require_file(Path(script_config.cert_result_path), name="Certification result")
    cert_dir = cert_result_path.parent
    lyapunov_dir = require_dir(cert_dir.parent, name="Lyapunov run directory")
    certification_config_path = require_file(cert_dir / "certification_config.json", name="Certification config")
    lyapunov_path = require_file(lyapunov_dir / "lyapunov_model.pt", name="Lyapunov checkpoint")
    policy_path = require_file(lyapunov_dir / "policy_model.pt", name="Policy checkpoint")
    results_path = cert_dir / "certification_tester_results.json"

    certification_config = LyapunovCertificationConfig.load(certification_config_path)
    cert_result = RegionCertificationResult.load(cert_result_path)

    policy_model = load_policy_model(policy_path, device)
    lyap_model = load_lyapunov_model(lyapunov_path, device)
    lyapunov_func = _build_lyapunov_func(lyap_model, device)

    dyn_model = DoubleIntegratorDynamics(
        dt=policy_model.global_config.dt,
        abcrown_compatible_ops=True,
    ).to(device)
    dyn_model.eval()

    __logger__.info("Loaded certification result from %s", cert_result_path)
    __logger__.info("Loaded certification config from %s", certification_config_path)

    cert_tester = CertificationResultTester(
        policy_model=policy_model,
        lyap_model=lyap_model,
        dyn_model=dyn_model,
        config=certification_config,
        device=device,
    )
    test_results = cert_tester.test_result(
        rho=float(cert_result.rho),
        sample_size=int(script_config.test_sample_size),
        rollout_steps=int(script_config.rollout_steps),
    )
    test_results.save(results_path)
    __logger__.info("Saved certification tester results to %s", results_path)
    rollout_dataset = _build_rollout_dataset(
        test_results=test_results,
        policy_model=policy_model,
        dyn_model=dyn_model,
        lyap_model=lyap_model,
        rollout_steps=int(script_config.rollout_steps),
        device=device,
    )
    _save_level_set_metrics(
        cert_dir=cert_dir,
        cert_result=cert_result,
        lyapunov_fn=lyap_model,
        state_dim=int(policy_model.global_config.nx),
        device=device,
    )
    _save_lyapunov_plot(
        cert_dir=cert_dir,
        certification_config=certification_config,
        lyapunov_func=lyapunov_func,
        rollout_dataset=rollout_dataset,
    )


if __name__ == "__main__":
    main()