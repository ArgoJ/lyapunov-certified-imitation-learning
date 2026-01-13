import json
import zipfile
import io
import h5py
import numpy as np
import pandas as pd

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..utils.package_logger import PackageLogger

__logger__ = PackageLogger.get_logger(__name__)

@dataclass
class MPCMeta:
    """Metadata regarding the MPC execution."""
    timestamp: str = ""
    solve_time_mean: float = 0.0
    solve_time_max: float = 0.0
    solve_time_total: float = 0.0
    sim_duration_wall: float = 0.0
    steps_simulated: int = 0
    status_codes: List[int] = field(default_factory=list)


@dataclass
class MPCTrajectory:
    """The actual data resulting from a run."""
    states: np.ndarray          # (T, nx)
    inputs: np.ndarray          # (T, nu)
    time: np.ndarray            # (T,)
    solved_states: np.ndarray   # (T, N+1, nx) - OCP predictions at each step
    solved_inputs: np.ndarray   # (T, N, nu)   - OCP predictions at each step
    feasible: bool = True

    @classmethod
    def initialize(cls, T_sim: int, N: int, nx: int, nu: int, dt: float = 0.1) -> 'MPCTrajectory':
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
        solved_states = np.full((T_sim, N + 1, nx), np.nan)
        solved_inputs = np.full((T_sim, N, nu), np.nan)
        
        return cls(
            states=states,
            inputs=inputs,
            time=time,
            solved_states=solved_states,
            solved_inputs=solved_inputs,
            feasible=True
        )


@dataclass
class MPCConstraints:
    """Constraints and goals for the MPC problem."""
    state_bounds: Optional[np.ndarray] = None  # Shape (2, nx): row 0 is lower, 1 is upper
    input_bounds: Optional[np.ndarray] = None  # Shape (2, nu): row 0 is lower, 1 is upper
    goal_state: Optional[np.ndarray] = None    # Shape (nx,)


@dataclass
class MPCData:
    """A single dataset entry combining config and result."""
    trajectory: MPCTrajectory
    meta: MPCMeta = field(default_factory=MPCMeta)
    config: Dict[str, Any] = field(default_factory=dict)
    constraints: Optional[MPCConstraints] = None


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
            # Sort keys to ensure deterministic ordering
            self._indices = sorted(list(self._h5_file.keys()), key=lambda x: int(x.split('_')[1]))

    def add(self, entry: MPCData):
        """Add to temporary memory buffer (for generation phase)."""
        self.memory_buffer.append(entry)

    def save(self, path: str = None, mode: str = 'a'):
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

                # Trajectory Data
                t = entry.trajectory
                grp.create_dataset("states", data=t.states, compression="gzip")
                grp.create_dataset("inputs", data=t.inputs, compression="gzip")
                grp.create_dataset("time", data=t.time, compression="gzip")

                # Optional: Only save solved predictions if needed to save space
                grp.create_dataset("solved_states", data=t.solved_states, compression="gzip")
                grp.create_dataset("solved_inputs", data=t.solved_inputs, compression="gzip")

                # Constraints
                if entry.constraints:
                    c = entry.constraints
                    c_grp = grp.create_group("constraints")
                    if c.state_bounds is not None: c_grp.create_dataset("state_bounds", data=c.state_bounds)
                    if c.input_bounds is not None: c_grp.create_dataset("input_bounds", data=c.input_bounds)
                    if c.goal_state is not None:   c_grp.create_dataset("goal_state", data=c.goal_state)

                # Metadata & Config as Attributes
                grp.attrs["meta_json"] = json.dumps(asdict(entry.meta))
                grp.attrs["config_json"] = json.dumps(entry.config)
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
        The Magic Method: Reads from disk on-demand.
        """
        # 1. Check memory buffer first (unsaved data)
        if idx < len(self.memory_buffer):
            return self.memory_buffer[idx]
        
        # 2. Check File
        # Calculate index relative to the file content
        file_idx = idx - len(self.memory_buffer)
        key = self._indices[file_idx]
        grp = self._h5_file[key]

        # Read Arrays (This reads binary data from disk into RAM)
        traj = MPCTrajectory(
            states=grp["states"][:],
            inputs=grp["inputs"][:],
            time=grp["time"][:],
            solved_states=grp["solved_states"][:] if "solved_states" in grp else None,
            solved_inputs=grp["solved_inputs"][:] if "solved_inputs" in grp else None,
            feasible=bool(grp.attrs["feasible"])
        )

        # Read Metadata (Fast JSON decode)
        meta = MPCMeta(**json.loads(grp.attrs["meta_json"]))
        config = json.loads(grp.attrs["config_json"])
        
        # Read Constraints
        constraints = None
        if "constraints" in grp:
            c_grp = grp["constraints"]
            constraints = MPCConstraints(
                state_bounds=c_grp["state_bounds"][:] if "state_bounds" in c_grp else None,
                input_bounds=c_grp["input_bounds"][:] if "input_bounds" in c_grp else None,
                goal_state=c_grp["goal_state"][:] if "goal_state" in c_grp else None,
            )

        return MPCData(trajectory=traj, meta=meta, config=config, constraints=constraints)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Fast Filtering: Reads ONLY the metadata attributes (tiny), ignores arrays (huge).
        """
        rows = []
        # 1. From Memory
        for i, entry in enumerate(self.memory_buffer):
            row = entry.config.copy()
            row.update(asdict(entry.meta))
            row['original_index'] = i
            row['source'] = 'mem'
            rows.append(row)

        # 2. From File (Fast Scan)
        for i, key in enumerate(self._indices):
            grp = self._h5_file[key]
            # ONLY reading attributes, extremely fast
            meta = json.loads(grp.attrs["meta_json"])
            config = json.loads(grp.attrs["config_json"])
            row = config.copy()
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
            cons = entry.constraints

            # Determine effective constraints (Priority: function arg > dataset entry > None)
            eff_x_bounds = x_bounds if x_bounds is not None else (cons.state_bounds if cons else None)
            eff_u_bounds = u_bounds if u_bounds is not None else (cons.input_bounds if cons else None)
            eff_goal = goal_state if goal_state is not None else (cons.goal_state if cons else None)
            
            # Check Solver Feasibility
            solver_errors = [code for code in meta.status_codes if code != 0]
            solver_success = len(solver_errors) == 0
            
            # Check State Constraints
            state_violations = np.zeros(traj.states.shape[1], dtype=bool)
            if eff_x_bounds is not None:
                # Row 0: Lower bounds, Row 1: Upper bounds
                lower_vio = np.any(traj.states < (eff_x_bounds[0] - tol_constraints), axis=0)
                upper_vio = np.any(traj.states > (eff_x_bounds[1] + tol_constraints), axis=0)
                state_violations = lower_vio | upper_vio
                
            # Check Input Constraints
            input_violations = np.zeros(traj.inputs.shape[1], dtype=bool)
            if eff_u_bounds is not None:
                lower_vio_u = np.any(traj.inputs < (eff_u_bounds[0] - tol_constraints), axis=0)
                upper_vio_u = np.any(traj.inputs > (eff_u_bounds[1] + tol_constraints), axis=0)
                input_violations = lower_vio_u | upper_vio_u

            all_constraints_met = not (np.any(state_violations) or np.any(input_violations))

            # Check Stability (Convergence to Goal)
            if eff_goal is None:
                eff_goal = np.zeros(traj.states.shape[1])
                
            final_state = traj.states[-1]
            dist_to_goal = np.linalg.norm(final_state - eff_goal)
            is_stable = dist_to_goal <= tol_stability

            # Compile Report
            results.append({
                "id": idx,
                "feasible": traj.feasible and solver_success,
                "constraints_met": all_constraints_met,
                "stable": is_stable,
                "final_dist": dist_to_goal,
                # Convert list to set to avoid storing 100 identical error codes
                "solver_codes": list(set(meta.status_codes)),
                "violated_state_dims": np.where(state_violations)[0].tolist(),
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