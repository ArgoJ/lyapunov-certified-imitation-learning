from __future__ import annotations

import os

import torch as th
import torch.nn as nn

from pathlib import Path

from mpc_datagen import MPCDataset

from ..imitation_learning.dataset import SequenceStateActionDataset, load_imitation_dataset


class PolicyWrapper(nn.Module):
    """Base wrapper that exposes a state-feedback interface for wrapped policies."""

    def __init__(self, policy: nn.Module) -> None:
        super().__init__()
        self.policy = policy

    @staticmethod
    def _reduce_policy_output(policy_output: th.Tensor) -> th.Tensor:
        """Reduce sequence outputs to one control vector per sample when needed."""
        if policy_output.ndim == 2:
            return policy_output
        if policy_output.ndim == 3:
            return policy_output[:, -1, :]
        raise ValueError(
            "Policy output must have shape (batch, nu) or (batch, seq_len, nu), "
            f"got {tuple(policy_output.shape)}."
        )

    def _call_policy(self, x: th.Tensor, use_raw: bool = False) -> th.Tensor:
        """Call the wrapped policy and normalize its output shape."""
        if use_raw:
            raw_forward = getattr(self.policy, "forward_raw", None)
            if not callable(raw_forward):
                raise NotImplementedError(
                    "The wrapped policy does not implement 'forward_raw'."
                )
            policy_output = raw_forward(x)
        else:
            policy_output = self.policy(x)

        return self._reduce_policy_output(policy_output)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self._call_policy(x, use_raw=False)

    def forward_raw(self, x: th.Tensor) -> th.Tensor:
        return self._call_policy(x, use_raw=True)


def _resolve_sequence_length(policy: nn.Module, sequence_length: int | None = None) -> int:
    """Infer the effective sequence length for sequence-aware policy wrappers."""
    resolved_sequence_length = getattr(policy, "max_seq_len", 1) if sequence_length is None else sequence_length
    resolved_sequence_length = int(resolved_sequence_length)
    if resolved_sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")
    return resolved_sequence_length


class RepeatCurrentPolicyWrapper(PolicyWrapper):
    """Wrap a sequence policy by repeating the current state to fill the history window."""

    def __init__(self, policy: nn.Module, sequence_length: int | None = None) -> None:
        super().__init__(policy)
        self.sequence_length = _resolve_sequence_length(policy, sequence_length)

    def _prepare_inputs(self, x: th.Tensor) -> th.Tensor:
        if x.ndim == 2 and self.sequence_length > 1:
            return x.unsqueeze(1).repeat(1, self.sequence_length, 1)
        if x.ndim in {2, 3}:
            return x
        raise ValueError(
            "RepeatCurrentPolicyWrapper expects inputs with shape (batch, nx) or "
            f"(batch, seq_len, nx), got {tuple(x.shape)}."
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self._call_policy(self._prepare_inputs(x), use_raw=False)

    def forward_raw(self, x: th.Tensor) -> th.Tensor:
        return self._call_policy(self._prepare_inputs(x), use_raw=True)


class FromRolloutsPolicyWrapper(PolicyWrapper):
    """Wrap a sequence policy with history reconstructed from rollout state windows."""

    def __init__(
        self,
        policy: nn.Module,
        rollout_dataset: SequenceStateActionDataset,
        sequence_length: int | None = None,
    ) -> None:
        super().__init__(policy)
        if len(rollout_dataset) <= 0:
            raise ValueError("rollout_dataset must contain at least one sequence window.")

        self.sequence_length = _resolve_sequence_length(policy, sequence_length)
        if rollout_dataset.sequence_length != self.sequence_length:
            raise ValueError(
                "rollout_dataset sequence length mismatch: "
                f"expected {self.sequence_length}, got {rollout_dataset.sequence_length}."
            )

        history_windows = rollout_dataset._states.detach().cpu()
        self.register_buffer("_history_windows", history_windows, persistent=False)
        self.register_buffer("_history_last_states", history_windows[:, -1, :], persistent=False)

    @classmethod
    def from_rollouts(
        cls,
        policy: nn.Module,
        rollout_source: SequenceStateActionDataset | MPCDataset | os.PathLike,
        sequence_length: int | None = None,
        stride: int = 1,
        dtype: th.dtype = th.float32,
    ) -> "FromRolloutsPolicyWrapper":
        """Construct the wrapper from a sequence dataset, rollout HDF5, or serialized dataset file."""
        resolved_sequence_length = _resolve_sequence_length(policy, sequence_length)

        if isinstance(rollout_source, SequenceStateActionDataset):
            rollout_dataset = rollout_source
        elif isinstance(rollout_source, MPCDataset):
            rollout_dataset = SequenceStateActionDataset.from_mpc_dataset(
                rollout_source,
                sequence_length=resolved_sequence_length,
                stride=stride,
                dtype=dtype,
                target_mode="last",
            )
        elif isinstance(rollout_source, (str, os.PathLike)):
            source_path = Path(rollout_source)
            if source_path.suffix == ".pt":
                loaded_dataset = load_imitation_dataset(source_path)
                if not isinstance(loaded_dataset, SequenceStateActionDataset):
                    raise ValueError("from_rollouts requires a sequence dataset, got a flat state-action dataset.")
                rollout_dataset = loaded_dataset
            else:
                rollout_dataset = SequenceStateActionDataset.from_mpc_dataset(
                    source_path,
                    sequence_length=resolved_sequence_length,
                    stride=stride,
                    dtype=dtype,
                    target_mode="last",
                )
        else:
            raise TypeError(
                "rollout_source must be a SequenceStateActionDataset, MPCDataset, or path-like object."
            )

        return cls(policy=policy, rollout_dataset=rollout_dataset, sequence_length=resolved_sequence_length)

    def _prepare_inputs(self, x: th.Tensor) -> th.Tensor:
        if x.ndim == 3:
            return x
        if x.ndim != 2:
            raise ValueError(
                "FromRolloutsPolicyWrapper expects inputs with shape (batch, nx) or "
                f"(batch, seq_len, nx), got {tuple(x.shape)}."
            )
        if self.sequence_length <= 1:
            return x
        if x.shape[1] != self._history_last_states.shape[1]:
            raise ValueError(
                "Input state dimension mismatch: "
                f"expected {self._history_last_states.shape[1]}, got {x.shape[1]}."
            )

        x_query = x.to(device=self._history_last_states.device, dtype=self._history_last_states.dtype)
        distances = th.cdist(x_query, self._history_last_states)
        nearest_indices = distances.argmin(dim=1)
        matched_windows = self._history_windows.index_select(0, nearest_indices)
        history_prefix = matched_windows[:, :-1, :]
        return th.cat((history_prefix, x_query.unsqueeze(1)), dim=1)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self._call_policy(self._prepare_inputs(x), use_raw=False)

    def forward_raw(self, x: th.Tensor) -> th.Tensor:
        return self._call_policy(self._prepare_inputs(x), use_raw=True)