import pickle
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path

import numpy as np
import pandas as pd



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
class MPCData:
    """A single dataset entry combining config and result."""
    trajectory: MPCTrajectory
    meta: MPCMeta = field(default_factory=MPCMeta)
    config: Dict[str, Any] = field(default_factory=dict)


class MPCDataset:
    """
    Helper to manage a list of MPCData entries.
    Allows saving/loading and filtering via Pandas.
    """
    def __init__(self, data: List[MPCData] = None):
        self.data = data if data is not None else []

    def add(self, entry: MPCData):
        self.data.append(entry)

    def save(self, path: str):
        """Save the entire dataset to a pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.data, f)
        print(f"Saved {len(self.data)} entries to {path}")

    @classmethod
    def load(cls, path: str) -> 'MPCDataset':
        """Load a dataset from a pickle file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return cls(data)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert configurations to a Pandas DataFrame for easy sorting/filtering.
        Returns a DataFrame where the index corresponds to the list index in self.data.
        """
        rows = []
        for i, entry in enumerate(self.data):
            row = entry.config.copy()
            row.update(asdict(entry.meta))
            row['feasible'] = entry.trajectory.feasible
            row['original_index'] = i
            rows.append(row)
        return pd.DataFrame(rows)

    def filter(self, query: str) -> 'MPCDataset':
        """
        Filter dataset using pandas query string syntax.
        Example: dataset.filter("horizon > 10 and feasible == True")
        """
        df = self.to_dataframe()
        filtered_df = df.query(query)
        indices = filtered_df['original_index'].values.astype(int)
        
        subset = [self.data[i] for i in indices]
        return MPCDataset(subset)

    def __getitem__(self, idx) -> MPCData:
        return self.data[idx]

    def __len__(self) -> int:
        return len(self.data)