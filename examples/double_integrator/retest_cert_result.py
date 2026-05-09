from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch as th
from mpc_datagen import MPCConfig, MPCData, MPCDataset, MPCMeta, MPCTrajectory

from lcil.certification import CertificationResultTester, LyapunovCertificationConfig
from lcil.certification.bisect_certifier import RegionCertificationResult
from lcil.utils import lcil_plt
from lcil.utils.base_config import ArgumentParserConfig, config_field

from . import DoubleIntegratorDynamics, load_lyapunov_model, load_policy_model

__logger__ = logging.getLogger("lcil.examples.double_integrator.retest_cert_result")

_DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results" / "double_integrator"


def _discover_latest_cert_result_path(results_root: Path) -> Path:
    candidates = sorted(results_root.rglob("certification_details.npz"))
    if not candidates:
        raise FileNotFoundError(
            f"No certification_details.npz found under '{results_root}'. "
            "Provide --cert-result-path explicitly."
        )
    return candidates[-1]


def _require_file(path: Path, *, name: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{name} not found: '{resolved}'.")
    return resolved


def _require_dir(path: Path, *, name: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{name} not found: '{resolved}'.")
    return resolved


def _infer_lyapunov_dir(cert_result_path: Path) -> Path:
    lyapunov_dir = cert_result_path.parent.parent
    return _require_dir(lyapunov_dir, name="Lyapunov run directory")


def _infer_policy_dir(lyapunov_dir: Path) -> Path:
    for candidate in lyapunov_dir.parents:
        if (candidate / "model.pt").is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not infer the policy run directory from '{lyapunov_dir}'. "
        "Provide --policy-dir explicitly."
    )


@dataclass(frozen=True)
class RetestCertResultScriptConfig(ArgumentParserConfig):
    cert_result_path: str | None = config_field(
        default=None,
        help="Optional path to certification_details.npz. Defaults to the latest saved certification result under results/double_integrator.",
    )
    certification_config_path: str | None = config_field(
        default=None,
        help="Optional path to certification_config.json. Defaults to the file next to cert_result_path.",
    )
    policy_dir: str | None = config_field(
        default=None,
        help="Optional policy run directory containing model.pt. Defaults to the policy run inferred from cert_result_path.",
    )
    lyapunov_dir: str | None = config_field(
        default=None,
        help="Optional Lyapunov run directory containing lyapunov_model.pt. Defaults to the run inferred from cert_result_path.",
    )
    device: str = config_field(default="cpu", help="Torch device string (for example cpu or cuda).")
    rollout_steps: int = config_field(
        default=50,
        help="Closed-loop rollout steps per region center during empirical result testing.",
    )

    def resolve_paths(self, results_root: Path) -> "RetestCertResultScriptConfig":
        cert_result_path = _require_file(
            Path(self.cert_result_path)
            if self.cert_result_path is not None
            else _discover_latest_cert_result_path(results_root),
            name="Certification result",
        )

        certification_config_path = _require_file(
            Path(self.certification_config_path)
            if self.certification_config_path is not None
            else cert_result_path.parent / "certification_config.json",
            name="Certification config",
        )

        lyapunov_dir = _require_dir(
            Path(self.lyapunov_dir) if self.lyapunov_dir is not None else _infer_lyapunov_dir(cert_result_path),
            name="Lyapunov run directory",
        )
        _require_file(lyapunov_dir / "lyapunov_model.pt", name="Lyapunov checkpoint")

        policy_dir = _require_dir(
            Path(self.policy_dir) if self.policy_dir is not None else _infer_policy_dir(lyapunov_dir),
            name="Policy run directory",
        )
        _require_file(policy_dir / "model.pt", name="Policy checkpoint")

        return replace(
            self,
            cert_result_path=str(cert_result_path),
            certification_config_path=str(certification_config_path),
            lyapunov_dir=str(lyapunov_dir),
            policy_dir=str(policy_dir),
        )


def parse_args() -> RetestCertResultScriptConfig:
    script_defaults = RetestCertResultScriptConfig()
    parser = argparse.ArgumentParser(
        description="Load a saved certification result and rerun empirical center-rollout testing."
    )
    script_defaults.add_to_argparse(parser)
    args = parser.parse_args()
    return script_defaults.from_namespace(args).resolve_paths(_DEFAULT_RESULTS_ROOT)


def _iter_rollout_groups(test_results):
    yield test_results.certified.rollout_states
    yield test_results.failed.rollout_states
    yield test_results.outside_sublevel.rollout_states


def _build_rollout_dataset(
    test_results,
    *,
    mpc_config: MPCConfig,
    lyapunov_func,
) -> MPCDataset:
    dataset = MPCDataset()
    traj_id = 0

    for rollout_group in _iter_rollout_groups(test_results):
        if rollout_group is None or len(rollout_group) == 0:
            continue

        for rollout_states in np.asarray(rollout_group, dtype=np.float32):
            traj = MPCTrajectory.empty_from_cfg(mpc_config)
            traj.states[:, :] = rollout_states
            traj.inputs[:, :] = 0.0
            traj.V_N = np.asarray(lyapunov_func(rollout_states[:-1]), dtype=np.float32).reshape(-1)

            meta = MPCMeta(
                id=traj_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                steps_simulated=int(mpc_config.T_sim),
                status_codes=[0] * int(mpc_config.T_sim),
                feasible=True,
            )
            dataset.add(MPCData(trajectory=traj, meta=meta, config=mpc_config))
            traj_id += 1

    return dataset


def _save_lyapunov_plot(
    *,
    cert_dir: Path,
    certification_config: LyapunovCertificationConfig,
    policy_model,
    lyap_model,
    test_results,
    device: th.device,
) -> None:
    rollout_mpc_config = MPCConfig(
        T_sim=int(test_results.rollout_steps),
        nx=int(policy_model.global_config.nx),
        nu=int(policy_model.global_config.nu),
        dt=float(policy_model.global_config.dt),
    )

    def lyapunov_func(states: np.ndarray) -> np.ndarray:
        x = th.as_tensor(states, dtype=th.float32, device=device)
        with th.no_grad():
            values = lyap_model(x)
        return values.detach().cpu().numpy().reshape(-1)

    rollout_dataset = _build_rollout_dataset(
        test_results,
        mpc_config=rollout_mpc_config,
        lyapunov_func=lyapunov_func,
    )
    if len(rollout_dataset) == 0:
        __logger__.warning("No tester rollouts available for Lyapunov plotting.")
        return

    cert_bounds = np.asarray(certification_config.cert_bounds, dtype=float)
    plot_path = cert_dir / "certification_tester_lyapunov_plot.html"
    lcil_plt.lyapunov(
        lyapunov_func=lyapunov_func,
        dataset=rollout_dataset,
        state_indices=[0, 1],
        state_labels=["$x$", "$v$"],
        limits=[
            (float(cert_bounds[0, 0]), float(cert_bounds[1, 0])),
            (float(cert_bounds[0, 1]), float(cert_bounds[1, 1])),
        ],
        plot_3d=True,
        use_dataset_v=True,
        html_path=plot_path,
    )
    __logger__.info("Saved certification tester Lyapunov plot to %s", plot_path)


def main() -> None:
    script_config = parse_args()
    device = th.device(script_config.device)

    cert_result_path = Path(script_config.cert_result_path).resolve()
    certification_config_path = Path(script_config.certification_config_path).resolve()
    policy_dir = Path(script_config.policy_dir).resolve()
    lyapunov_dir = Path(script_config.lyapunov_dir).resolve()
    cert_dir = cert_result_path.parent
    results_path = cert_dir / "certification_tester_results.json"

    certification_config = LyapunovCertificationConfig.load(certification_config_path)
    cert_result = RegionCertificationResult.load(cert_result_path)

    policy_model = load_policy_model(policy_dir, device)
    lyap_model = load_lyapunov_model(lyapunov_dir, device)

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
        cert_result=cert_result,
        rollout_steps=int(script_config.rollout_steps),
    )
    test_results.save(results_path)
    __logger__.info("Saved certification tester results to %s", results_path)
    _save_lyapunov_plot(
        cert_dir=cert_dir,
        certification_config=certification_config,
        policy_model=policy_model,
        lyap_model=lyap_model,
        test_results=test_results,
        device=device,
    )


if __name__ == "__main__":
    main()