import numpy as np
import torch as th

from mpc_datagen import MPCConfig, MPCData, MPCDataset, MPCMeta, MPCTrajectory, Sampler, SamplerBase

from ..utils.package_logger import get_package_logger

__logger__ = get_package_logger(__name__)


class PolicyRolloutGenerator:
    """Generator for closed-loop policy rollout datasets."""

    def __init__(
        self,
        policy: th.nn.Module,
        t_sim: int,
        dt: float,
        state_bounds: np.ndarray | None = None,
        input_bounds: np.ndarray | None = None,
        sampler: SamplerBase | None = None,
        device: th.device | str = "cpu",
    ) -> None:
        """
        Initialize the generator with the given policy and rollout configuration.
        
        Parameters
        ----------
        policy : torch.nn.Module
            The policy to be rolled out. Should take state tensors as input and output action tensors.
        t_sim : int
            Number of simulation steps for each rollout.
        dt : float
            Time step for the discrete-time dynamics.
        state_bounds : np.ndarray, optional
            State bounds with shape ``(2, nx)`` as ``[lbx; ubx]``.
            If ``None``, no state bounds are written to the rollout config.
        input_bounds : np.ndarray, optional
            Input bounds with shape ``(2, nu)`` as ``[lbu; ubu]``.
            If ``None``, policy outputs are not clipped.
        sampler : SamplerBase, optional
            Sampler for generating initial states. If None, a default Sampler will be used.
        device : torch.device or str, optional
            Device to run the policy on (e.g., "cpu" or "cuda"). Default is "cpu".
        """
        self.policy = policy
        self.t_sim = int(t_sim)
        self.dt = float(dt)
        self.device = th.device(device)

        self.mpc_config = MPCConfig(
            T_sim=self.t_sim,
            N=1,
            nx=2,
            nu=1,
            dt=self.dt,
        )

        self.state_bounds = None if state_bounds is None else np.asarray(state_bounds, dtype=float)
        self.input_bounds = None if input_bounds is None else np.asarray(input_bounds, dtype=float)

        if self.state_bounds is not None:
            if self.state_bounds.shape != (2, self.mpc_config.nx):
                raise ValueError(
                    f"state_bounds must have shape (2, {self.mpc_config.nx}), got {self.state_bounds.shape}."
                )
            self.mpc_config.constraints.lbx = self.state_bounds[0].astype(float)
            self.mpc_config.constraints.ubx = self.state_bounds[1].astype(float)

        if self.input_bounds is not None:
            if self.input_bounds.shape != (2, self.mpc_config.nu):
                raise ValueError(
                    f"input_bounds must have shape (2, {self.mpc_config.nu}), got {self.input_bounds.shape}."
                )
            self.mpc_config.constraints.lbu = self.input_bounds[0].astype(float)
            self.mpc_config.constraints.ubu = self.input_bounds[1].astype(float)

        if sampler is None:
            sampler = Sampler(bounds=self.state_bounds) if self.state_bounds is not None else Sampler()
        self.sampler = sampler
        self.sampler.post_init_cfg(self.mpc_config)

        self.policy.to(self.device)
        self.policy.eval()

    def _rollout_single(self, x0: np.ndarray, traj_id: int) -> MPCData:
        """Roll out one trajectory from `x0` and return an `MPCData` entry."""
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
                if self.input_bounds is not None:
                    u_vec = np.clip(u_vec, self.mpc_config.constraints.lbu, self.mpc_config.constraints.ubu)
                u_k = float(u_vec[0])

                traj.inputs[k, :] = u_vec
                traj.states[k + 1, 0] = x_k[0] + self.dt * x_k[1]
                traj.states[k + 1, 1] = x_k[1] + self.dt * u_k

                traj.V_solver[k] = float(np.dot(x_k, x_k) + 0.1 * float(np.dot(u_vec, u_vec)))

        meta = MPCMeta(
            id=int(traj_id),
            steps_simulated=self.t_sim,
            status_codes=[0] * self.t_sim,
        )

        return MPCData(trajectory=traj, meta=meta, config=self.mpc_config)

    def generate(self, n_samples: int) -> MPCDataset:
        """Generate `n_samples` policy rollouts using the configured sampler."""
        dataset = MPCDataset()
        accepted_x0: list[np.ndarray] = []

        with __logger__.tqdm(range(n_samples), desc="Generating Policy Rollouts") as pbar:
            for idx in pbar:
                x0 = self.sampler.sample_x0(accepted_x0)
                mpc_data = self._rollout_single(x0=x0, traj_id=idx)
                dataset.add(mpc_data)
                accepted_x0.append(x0)

        __logger__.info(f"Generated {len(dataset)} policy rollouts.")
        return dataset