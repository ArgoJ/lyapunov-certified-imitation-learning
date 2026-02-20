from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch as th

from torch import nn
from numpy.typing import ArrayLike, NDArray

from mpc_datagen import MPCConfig, MPCData, MPCDataset, MPCMeta, MPCTrajectory

from .dataset import StateActionDataset
from ..utils.package_logger import get_package_logger

__logger__ = get_package_logger(__name__)


def _normalize_bounds(
    bounds: ArrayLike | None,
    expected_dim: int | None,
    name: str,
) -> NDArray | None:
    if bounds is None:
        return None

    bounds_array = np.asarray(bounds, dtype=float)
    if bounds_array.ndim != 2 or bounds_array.shape[0] != 2:
        raise ValueError(f"{name} must have shape (2, dim), got {bounds_array.shape}.")
    if expected_dim is not None and bounds_array.shape[1] != expected_dim:
        raise ValueError(f"{name} must have shape (2, {expected_dim}), got {bounds_array.shape}.")
    if np.any(bounds_array[0] > bounds_array[1]):
        raise ValueError(f"{name} lower bounds must be <= upper bounds.")

    return bounds_array



# --- CONFIG ---
@dataclass(slots=True)
class PolicyRolloutConfig:
    """Configuration for policy rollout data generation."""

    T_sim: int
    dt: float
    nx: int
    nu: int
    state_bounds: ArrayLike | None = None
    input_bounds: ArrayLike | None = None

    @staticmethod
    def _extract_bounds(lower: ArrayLike, upper: ArrayLike) -> NDArray | None:
        """Stack lower/upper bounds into shape ``(2, dim)`` when available."""
        lower_arr = np.asarray(lower, dtype=np.float32).reshape(-1)
        upper_arr = np.asarray(upper, dtype=np.float32).reshape(-1)

        if lower_arr.size == 0 and upper_arr.size == 0:
            return None
        if lower_arr.size == 0 or upper_arr.size == 0:
            raise ValueError("Incomplete bounds in MPCConfig constraints.")
        if lower_arr.size != upper_arr.size:
            raise ValueError(
                f"Mismatched bounds size: lower={lower_arr.size}, upper={upper_arr.size}."
            )

        return np.vstack((lower_arr, upper_arr))

    @classmethod
    def from_mpc_config(
        cls,
        mpc_config: MPCConfig,
        t_sim: int | None = None,
    ) -> "PolicyRolloutConfig":
        """Build rollout config from an ``MPCConfig``.

        Parameters
        ----------
        mpc_config : MPCConfig
            Source MPC configuration.
        t_sim : int, optional
            Optional override for rollout simulation horizon.
        """
        if not isinstance(mpc_config, MPCConfig):
            raise ValueError(f"mpc_config must be an instance of MPCConfig, got type {type(mpc_config)}.")
            
        state_bounds = cls._extract_bounds(
            mpc_config.constraints.lbx,
            mpc_config.constraints.ubx,
        )
        input_bounds = cls._extract_bounds(
            mpc_config.constraints.lbu,
            mpc_config.constraints.ubu,
        )

        return cls(
            T_sim=int(mpc_config.T_sim) if t_sim is None else int(t_sim),
            dt=float(mpc_config.dt),
            nx=int(mpc_config.nx),
            nu=int(mpc_config.nu),
            state_bounds=state_bounds,
            input_bounds=input_bounds,
        )

    def __post_init__(self) -> None:
        if self.T_sim <= 0:
            raise ValueError("T_sim must be positive.")
        if self.dt <= 0:
            raise ValueError("dt must be positive.")
        if self.nx <= 0 or self.nu <= 0:
            raise ValueError("nx and nu must be positive.")

        self.state_bounds = _normalize_bounds(self.state_bounds, self.nx, "state_bounds")
        self.input_bounds = _normalize_bounds(self.input_bounds, self.nu, "input_bounds")

    def to_mpc_config(self) -> MPCConfig:
        """Build an MPCConfig object from rollout parameters."""
        mpc_config = MPCConfig(
            T_sim=int(self.T_sim),
            nx=int(self.nx),
            nu=int(self.nu),
            dt=float(self.dt),
        )

        if self.state_bounds is not None:
            mpc_config.constraints.lbx = self.state_bounds[0].astype(float)
            mpc_config.constraints.ubx = self.state_bounds[1].astype(float)
        if self.input_bounds is not None:
            mpc_config.constraints.lbu = self.input_bounds[0].astype(float)
            mpc_config.constraints.ubu = self.input_bounds[1].astype(float)

        return mpc_config



# --- SAMPLER ---
class StateSampler(Protocol):
    """Protocol for initial-state sampling used by PolicyRolloutGenerator."""

    def sample_x0(self) -> NDArray:
        """Sample one initial state."""


class RandomBoundsSampler(StateSampler):
    """Uniform random sampler over a fixed state-bounds box."""

    def __init__(self, bounds: ArrayLike, seed: int | None = None) -> None:
        self.bounds = _normalize_bounds(bounds, expected_dim=None, name="bounds")
        assert self.bounds is not None
        self.rng = np.random.default_rng(seed)

    def sample_x0(self) -> NDArray:
        low = self.bounds[0]
        high = self.bounds[1]
        return self.rng.uniform(low=low, high=high).astype(np.float32)


class FeasibleSetSampler(StateSampler):
    def __init__(self, dataset: StateActionDataset, seed: int | None = None) -> None:
        self.dataset = dataset
        self.rng = np.random.default_rng(seed)

        if len(self.dataset) <= 0:
            raise ValueError("FeasibleSetSampler requires a non-empty dataset.")

    def sample_x0(self) -> NDArray:
        idx = int(self.rng.integers(low=0, high=len(self.dataset)))
        x0, _ = self.dataset[idx]
        return np.asarray(x0.detach().cpu().numpy(), dtype=np.float32)



# --- GENERATOR ---
class PolicyRolloutGenerator:
    """Generator for closed-loop policy rollout datasets."""

    def __init__(
        self,
        policy: nn.Module,
        simulator: nn.Module,
        rollout_config: PolicyRolloutConfig | None = None,
        sampler: StateSampler | None = None,
        device: th.device | str = "cpu",
    ) -> None:
        """
        Initialize the generator with the given policy and rollout configuration.

        Parameters
        ----------
        policy : torch.nn.Module
            The policy to be rolled out. Should take state tensors as input and output action tensors.
        simulator : torch.nn.Module
            The simulator to be used for simulating the system dynamics. Should take (state, action) tensors as input and output next-state tensors.
        rollout_config : PolicyRolloutConfig, optional
            Rollout configuration independent from MPCConfig. Preferred interface.
        sampler : StateSampler, optional
            Sampler for generating initial states. If None, `RandomBoundsSampler` is used when
            state bounds are available.
        device : torch.device or str, optional
            Device to run the policy on (e.g., "cpu" or "cuda"). Default is "cpu".
        """
        self.policy = policy
        self.simulator = simulator
        self.device = th.device(device)
        
        self.rollout_config = rollout_config
        self.mpc_config = rollout_config.to_mpc_config()
        
        self.t_sim = int(self.rollout_config.T_sim)
        self.dt = float(self.rollout_config.dt)

        if sampler is None:
            if self.rollout_config.state_bounds is None:
                raise ValueError(
                    "No sampler was provided and state_bounds are not set. "
                    "Provide a sampler or set state_bounds in PolicyRolloutConfig."
                )
            sampler = RandomBoundsSampler(bounds=self.rollout_config.state_bounds)
        self.sampler = sampler

    def _rollout_single(self, x0: NDArray, traj_id: int) -> MPCData:
        """
        Roll out one trajectory from `x0` and return an `MPCData` entry.
        
        Parameters
        ----------
        x0 : NDArray
            Initial state for the rollout. Should have shape (nx,).
        traj_id : int
            Unique identifier for the trajectory, used in MPCMeta.
            
        Returns
        -------
        data : MPCData
            An MPCData object containing the rolled-out trajectory and metadata.
        """
        traj = MPCTrajectory.empty_from_cfg(self.mpc_config)
        traj.states[0] = np.asarray(x0, dtype=np.float32)

        with th.no_grad():
            for k in range(self.t_sim):
                x_k = traj.states[k]
                x_tensor = th.as_tensor(x_k, dtype=th.float32, device=self.device).unsqueeze(0)
                u_tensor = self.policy(x_tensor)
                u_vec = np.asarray(u_tensor.squeeze(0).detach().cpu().numpy(), dtype=np.float32).reshape(-1)
                if u_vec.size != self.mpc_config.nu:
                    raise ValueError(
                        f"Policy output dimension mismatch: expected {self.mpc_config.nu}, got {u_vec.size}."
                    )

                traj.inputs[k, :] = u_vec
                u_sim_tensor = th.as_tensor(u_vec, dtype=th.float32, device=self.device).unsqueeze(0)
                x_next_tensor = self.simulator(x_tensor, u_sim_tensor)
                x_next_vec = np.asarray(
                    x_next_tensor.squeeze(0).detach().cpu().numpy(), dtype=np.float32
                ).reshape(-1)
                if x_next_vec.size != self.mpc_config.nx:
                    raise ValueError(
                        f"Simulator output dimension mismatch: expected {self.mpc_config.nx}, got {x_next_vec.size}."
                    )
                traj.states[k + 1, :] = x_next_vec

                traj.V_solver[k] = float(np.dot(x_k, x_k) + 0.1 * float(np.dot(u_vec, u_vec)))

        meta = MPCMeta(
            id=int(traj_id),
            steps_simulated=self.t_sim,
            status_codes=[0] * self.t_sim,
        )

        return MPCData(trajectory=traj, meta=meta, config=self.mpc_config)


    def generate(self, n_samples: int) -> MPCDataset:
        """Generate a dataset of `n_samples` policy rollouts using the configured sampler."""
        self.policy.to(self.device)
        self.policy.eval()
        self.simulator.to(self.device)
        self.simulator.eval()

        dataset = MPCDataset()

        with __logger__.tqdm(range(n_samples), desc="Generating Policy Rollouts") as pbar:
            for idx in pbar:
                x0 = self.sampler.sample_x0()
                mpc_data = self._rollout_single(x0=x0, traj_id=idx)
                dataset.add(mpc_data)

        __logger__.info(f"Generated {len(dataset)} policy rollouts.")
        return dataset
