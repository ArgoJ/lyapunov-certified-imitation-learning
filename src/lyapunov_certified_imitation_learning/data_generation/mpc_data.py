import json
import h5py
import numpy as np
import pandas as pd

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Iterator
from pathlib import Path

from ..utils.package_logger import PackageLogger

__logger__ = PackageLogger.get_logger(__name__)

@dataclass
class MPCMeta:
    """Metadata regarding the MPC execution."""
    id: int = -1
    timestamp: str = ""
    solve_time_mean: float = 0.0
    solve_time_max: float = 0.0
    solve_time_total: float = 0.0
    sim_duration_wall: float = 0.0
    steps_simulated: int = 0
    status_codes: List[int] = field(default_factory=list)
     

@dataclass
class MPCConfig:
    """Configuration used for the MPC problem."""
    T_sim: int = 0                  # Simulation steps
    N: int = 10                     # Prediction horizon
    dt: float = 0.1                 # Sampling time
    Q: np.ndarray = field(default_factory=lambda: np.eye(2))  # State cost
    R: np.ndarray = field(default_factory=lambda: np.eye(1))  # Input cost
    Qf: Optional[np.ndarray] = None  # Terminal state cost
    x0: Optional[np.ndarray] = None  # Initial state
    goal_state: Optional[np.ndarray] = None  # Target state
    state_bounds: Optional[np.ndarray] = None  # Shape (2, nx)
    terminal_state_bounds: Optional[np.ndarray] = None  # Shape (2, nx)
    input_bounds: Optional[np.ndarray] = None  # Shape (2, nu)

    @classmethod
    def from_hdf5(cls, grp: h5py.Group) -> "MPCConfig":
        """
        Load config from a trajectory group (expects a `config` subgroup).
        """
        cfg_grp = grp.get("config", None)
        if cfg_grp is None:
            raise ValueError("No 'config' group found in the provided HDF5 group.")

        t_sim = int(cfg_grp.attrs.get("T_sim", 0))
        n_horizon = int(cfg_grp.attrs.get("N", 10))
        dt = float(cfg_grp.attrs.get("dt", 0.1))

        return cls(
            T_sim=t_sim,
            N=n_horizon,
            dt=dt,
            Q=cfg_grp["Q"][:],
            R=cfg_grp["R"][:],
            Qf=cfg_grp["Qf"][:] if "Qf" in cfg_grp else None,
            x0=cfg_grp["x0"][:] if "x0" in cfg_grp else None,
            goal_state=cfg_grp["goal_state"][:] if "goal_state" in cfg_grp else None,
            state_bounds=cfg_grp["state_bounds"][:] if "state_bounds" in cfg_grp else None,
            terminal_state_bounds=cfg_grp["terminal_state_bounds"][:] if "terminal_state_bounds" in cfg_grp else None,
            input_bounds=cfg_grp["input_bounds"][:] if "input_bounds" in cfg_grp else None,
        )


@dataclass
class MPCTrajectory:
    """The actual data resulting from a run."""
    states: np.ndarray          # (T, nx)
    inputs: np.ndarray          # (T, nu)
    time: np.ndarray            # (T,)
    cost: np.ndarray            # (T,)
    solved_states: Optional[np.ndarray] = None   # (T, N+1, nx) - OCP predictions at each step
    solved_inputs: Optional[np.ndarray] = None   # (T, N, nu)   - OCP predictions at each step
    feasible: bool = True

    @classmethod
    def from_hdf5(cls, grp: h5py.Group) -> "MPCTrajectory":
        """Load trajectory arrays from a trajectory group."""
        return cls(
            states=grp["states"][:, :],
            inputs=grp["inputs"][:, :],
            time=grp["time"][:],
            cost=grp["cost"][:],
            solved_states=grp["solved_states"][:, :, :] if "solved_states" in grp else None,
            solved_inputs=grp["solved_inputs"][:, :, :] if "solved_inputs" in grp else None,
            feasible=bool(grp.attrs.get("feasible", True)),
        )

    @classmethod
    def init(cls, T_sim: int, N: int, nx: int, nu: int, dt: float = 0.1) -> 'MPCTrajectory':
        """
        Initialize the trajectory with NaNs.
        
        Parameters
        ----------
        T_sim : int
            Number of simulation steps (trajectory will have length T_sim + 1 for states).
        N : int
            Prediction horizon length (steps).
        nx : int
            State dimension.
        nu : int
            Input dimension.
        dt : float
            Sampling time.
        """
        states = np.full((T_sim + 1, nx), np.nan)
        inputs = np.full((T_sim, nu), np.nan)
        time = np.arange(T_sim + 1) * dt
        cost = np.full((T_sim,), np.nan)
        solved_states = np.full((T_sim, N + 1, nx), np.nan)
        solved_inputs = np.full((T_sim, N, nu), np.nan)
        
        return cls(
            states=states,
            inputs=inputs,
            time=time,
            cost=cost,
            solved_states=solved_states,
            solved_inputs=solved_inputs,
            feasible=True
        )


@dataclass
class MPCData:
    """A single dataset entry combining config and result."""
    trajectory: MPCTrajectory
    meta: MPCMeta = field(default_factory=MPCMeta)
    config: MPCConfig = field(default_factory=MPCConfig)


class MPCDataset:
    """
    True Lazy-Loading Dataset.
    - Holds a file handle (_h5_file) instead of a list of data.
    - Reads arrays from disk only when __getitem__ is called.
    """
    def __init__(self, file_path: Optional[str] = None, data_buffer: List[MPCData] = None):
        self.file_path = Path(file_path) if file_path else None
        self.memory_buffer = data_buffer if data_buffer else []
        
        self._h5_file = None
        self._indices = [] # List of keys ['traj_0', 'traj_1', ...] in the file
        
        # Open file in read mode if it exists
        if self.file_path and self.file_path.exists():
            self._h5_file = h5py.File(self.file_path, 'r')
            self._indices = sorted(list(self._h5_file.keys()), key=lambda x: int(x.split('_')[1]))

    def add(self, entry: MPCData):
        """Add to temporary memory buffer (for generation phase)."""
        entry.meta.id = len(self)  # Assign ID based on current length
        self.memory_buffer.append(entry)

    def save(self, path: str = None, mode: str = 'w', save_ocp_trajs: bool = True) -> None:
        """Flushes memory buffer to HDF5."""
        target_path = Path(path) if path else self.file_path
        if not target_path: 
            raise ValueError("No path provided")
        target_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(target_path, mode) as f:
            start_idx = len(f.keys())
            for i, entry in enumerate(self.memory_buffer):
                group_name = f"traj_{start_idx + i}"
                grp = f.create_group(group_name)

                # Config (store as datasets, not JSON)
                cfg = entry.config
                cfg_grp = grp.create_group("config")
                cfg_grp.attrs["T_sim"] = int(cfg.T_sim)
                cfg_grp.attrs["N"] = int(cfg.N)
                cfg_grp.attrs["dt"] = float(cfg.dt)
                cfg_grp.create_dataset("Q", data=np.asarray(cfg.Q), compression="gzip")
                cfg_grp.create_dataset("R", data=np.asarray(cfg.R), compression="gzip")
                if cfg.Qf is not None:
                    cfg_grp.create_dataset("Qf", data=np.asarray(cfg.Qf), compression="gzip")
                if cfg.x0 is not None:
                    cfg_grp.create_dataset("x0", data=np.asarray(cfg.x0), compression="gzip")
                if cfg.goal_state is not None:
                    cfg_grp.create_dataset("goal_state", data=np.asarray(cfg.goal_state), compression="gzip")
                if cfg.state_bounds is not None:
                    cfg_grp.create_dataset("state_bounds", data=np.asarray(cfg.state_bounds), compression="gzip")
                if cfg.terminal_state_bounds is not None:
                    cfg_grp.create_dataset("terminal_state_bounds", data=np.asarray(cfg.terminal_state_bounds), compression="gzip")
                if cfg.input_bounds is not None:
                    cfg_grp.create_dataset("input_bounds", data=np.asarray(cfg.input_bounds), compression="gzip")

                # Trajectory Data
                t = entry.trajectory
                grp.create_dataset("states", data=t.states, compression="gzip")
                grp.create_dataset("inputs", data=t.inputs, compression="gzip")
                grp.create_dataset("time", data=t.time, compression="gzip")
                grp.create_dataset("cost", data=t.cost, compression="gzip")

                if save_ocp_trajs:
                    grp.create_dataset("solved_states", data=t.solved_states, compression="gzip")
                    grp.create_dataset("solved_inputs", data=t.solved_inputs, compression="gzip")

                # Metadata & Config as Attributes
                grp.attrs["meta_json"] = json.dumps(asdict(entry.meta))
                grp.attrs["feasible"] = t.feasible
        
        # Clear buffer and reload file to refresh indices
        self.memory_buffer = []
        self.file_path = target_path
        if self._h5_file: self._h5_file.close()
        self._h5_file = h5py.File(self.file_path, 'r')
        self._indices = sorted(list(self._h5_file.keys()), key=lambda x: int(x.split('_')[1]))

    @classmethod
    def load(cls, path: str) -> 'MPCDataset':
        """
        Lazy Load: Just opens the file, does NOT read data.
        """
        path = Path(path)
        if not path.exists():
            __logger__.warning(f"File {path} not found.")
            return cls()
        return cls(file_path=path)

    def __len__(self) -> int:
        return len(self.memory_buffer) + len(self._indices)

    def __getitem__(self, idx) -> MPCData:
        """
        Reads from disk on-demand.
        """
        # Check memory buffer first
        if idx < len(self.memory_buffer):
            return self.memory_buffer[idx]
        
        # Check File
        # Calculate index relative to the file content
        file_idx = idx - len(self.memory_buffer)
        key = self._indices[file_idx]
        grp = self._h5_file[key]

        # Read Arrays (This reads binary data from disk into RAM)
        traj = MPCTrajectory.from_hdf5(grp)

        # Read Metadata
        meta = MPCMeta(**json.loads(grp.attrs["meta_json"]))

        # Read Config
        config = MPCConfig.from_hdf5(grp)

        return MPCData(trajectory=traj, meta=meta, config=config)

    def __iter__(self) -> Iterator[MPCData]:
        """
        Explicit iterator to help static analysis tools infer the type of elements.
        """
        for i in range(len(self)):
            yield self[i]

    def to_dataframe(self) -> pd.DataFrame:
        """
        Fast Filtering: Reads ONLY the metadata attributes (tiny), ignores arrays (huge).
        """
        rows = []
        # From Memory
        for i, entry in enumerate(self.memory_buffer):
            row = {
                "T_sim": int(entry.config.T_sim),
                "N": int(entry.config.N),
                "dt": float(entry.config.dt),
            }
            row.update(asdict(entry.meta))
            row['original_index'] = i
            row['source'] = 'mem'
            rows.append(row)

        # From File
        for i, key in enumerate(self._indices):
            grp = self._h5_file[key]
            meta = json.loads(grp.attrs["meta_json"])
            row = {}
            cfg_grp = grp.get("config", None)
            if cfg_grp is not None:
                row.update({
                    "T_sim": int(cfg_grp.attrs.get("T_sim", 0)),
                    "N": int(cfg_grp.attrs.get("N", 10)),
                    "dt": float(cfg_grp.attrs.get("dt", 0.1)),
                })
                
            row.update(meta)
            row['original_index'] = len(self.memory_buffer) + i
            row['source'] = 'file'
            rows.append(row)

        return pd.DataFrame(rows)

    def filter(self, query: str) -> 'MPCDataset':
        """
        Returns a NEW dataset instance containing only the indices that match.
        """
        df = self.to_dataframe()
        filtered_indices = df.query(query)['original_index'].values.astype(int)
        subset_data = [self[i] for i in filtered_indices]
        return MPCDataset(data_buffer=subset_data)
    
    def validate(
        self,
        x_bounds: Optional[np.ndarray] = None,
        u_bounds: Optional[np.ndarray] = None,
        goal_state: Optional[np.ndarray] = None,
        tol_constraints: float = 1e-4,
        tol_stability: float = 1e-2
    ) -> pd.DataFrame:
        """
        Validates the generated dataset for consistency using Lazy Loading.
        Iterates over the dataset one-by-one to keep RAM usage low.
        """
        results = []
        
        for idx, entry in enumerate(self):
            traj = entry.trajectory
            meta = entry.meta
            cfg = entry.config

            traj_has_nan = (
                np.isnan(traj.states).any()
                or np.isnan(traj.inputs).any()
                or np.isnan(traj.cost).any()
            )

            # Determine effective constraints (Priority: function arg > dataset entry > None)
            eff_x_bounds = x_bounds if x_bounds is not None else cfg.state_bounds
            eff_u_bounds = u_bounds if u_bounds is not None else cfg.input_bounds
            eff_goal = goal_state if goal_state is not None else cfg.goal_state

            eff_x_terminal_bounds = cfg.terminal_state_bounds
            if eff_x_terminal_bounds is None:
                eff_x_terminal_bounds = eff_x_bounds
            
            # Check Solver Feasibility
            solver_errors = [code for code in meta.status_codes if code != 0]
            solver_success = len(solver_errors) == 0
            
            # Check State Constraints
            state_violations = np.zeros(traj.states.shape[1], dtype=bool)
            if eff_x_bounds is not None:
                # Row 0: Lower bounds, Row 1: Upper bounds
                lower_vio = np.any(traj.states[:-1] < (eff_x_bounds[0] - tol_constraints), axis=0)
                upper_vio = np.any(traj.states[:-1] > (eff_x_bounds[1] + tol_constraints), axis=0)
                state_violations = lower_vio | upper_vio
            
            # Check Terminal State Constraints
            terminal_state_violations = np.zeros(traj.states.shape[1], dtype=bool)
            if eff_x_terminal_bounds is not None:
                # Row 0: Lower bounds, Row 1: Upper bounds
                lower_vio = traj.states[-1] < (eff_x_terminal_bounds[0] - tol_constraints)
                upper_vio = traj.states[-1] > (eff_x_terminal_bounds[1] + tol_constraints)
                terminal_state_violations = lower_vio | upper_vio
                
            # Check Input Constraints
            input_violations = np.zeros(traj.inputs.shape[1], dtype=bool)
            if eff_u_bounds is not None:
                lower_vio_u = np.any(traj.inputs < (eff_u_bounds[0] - tol_constraints), axis=0)
                upper_vio_u = np.any(traj.inputs > (eff_u_bounds[1] + tol_constraints), axis=0)
                input_violations = lower_vio_u | upper_vio_u

            all_constraints_met = not (
                np.any(state_violations) or \
                np.any(terminal_state_violations) or \
                np.any(input_violations))

            # Any NaNs mean the rollout is incomplete/invalid; treat as failing checks.
            if traj_has_nan:
                all_constraints_met = False

            # Check Stability (Convergence to Goal)
            if eff_goal is None:
                eff_goal = np.zeros(traj.states.shape[1])
                
            final_state = traj.states[-1]
            dist_to_goal = float(np.linalg.norm(final_state - eff_goal)) if np.all(np.isfinite(final_state)) else float("inf")
            is_stable = dist_to_goal <= tol_stability

            # Compile Report
            results.append({
                "id": idx,
                "feasible": bool(traj.feasible and solver_success and (not traj_has_nan)),
                "constraints_met": all_constraints_met,
                "stable": is_stable,
                "final_dist": dist_to_goal,
                # Convert list to set to avoid storing 100 identical error codes
                "solver_codes": list(set(meta.status_codes)),
                "violated_state_dims": np.where(state_violations)[0].tolist(),
                "violated_terminal_state_dims": np.where(terminal_state_violations)[0].tolist(),
                "violated_input_dims": np.where(input_violations)[0].tolist()
            })

        df = pd.DataFrame(results)
        
        # Summary Logging
        if not df.empty:
            n_feas = df['feasible'].sum()
            n_stab = df['stable'].sum()
            n_cons = df['constraints_met'].sum()
            total = len(df)
            __logger__.info(f"Validation Results ({total} trajectories):")
            __logger__.info(f"  Feasible:        {n_feas}/{total} ({n_feas/total:.1%})")
            __logger__.info(f"  Stable:          {n_stab}/{total} ({n_stab/total:.1%})")
            __logger__.info(f"  Constraints Met: {n_cons}/{total} ({n_cons/total:.1%})")
        
        return df
    
    def close(self):
        if self._h5_file: self._h5_file.close()

    def __del__(self):
        self.close()