from __future__ import annotations

import logging
import torch as th
from dataclasses import dataclass
from typing import Callable, Sequence
from numpy.typing import NDArray

from .config import LyapunovTrainingConfig
from .sampling import (
    _bounds_tensor,
    project_to_box,
)

__logger__ = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CounterexampleMiningConfig:
    """Configuration for adversarial counterexample mining."""

    train_bounds: NDArray | Sequence[float] | th.Tensor
    step_size: float = 0.05
    descent_steps: int = 10
    origin_exclusion: float | Sequence[float] | NDArray | None = None

    @classmethod
    def from_training_config(
        cls,
        config: LyapunovTrainingConfig,
        **overrides,
    ) -> "CounterexampleMiningConfig":
        """Build a CounterexampleMiningConfig from a LyapunovTrainingConfig."""
        bounds = config.train_bounds if config.train_bounds is not None else config.state_bounds
        data = {
            "train_bounds": bounds,
            "step_size": config.cex_step_size,
            "descent_steps": config.cex_descent_steps,
            "origin_exclusion": config.origin_exclusion,
        }
        data.update(overrides)
        return cls(**data)


def find_counter_examples(
    objective: Callable[[th.Tensor], th.Tensor],
    condition_evaluator: Callable[[th.Tensor], tuple[th.Tensor, th.Tensor]],
    config: CounterexampleMiningConfig,
    initial_states: th.Tensor,
    device: th.device | int | str | None = None,
) -> tuple[th.Tensor, th.Tensor]:
    """Find rho-gated training counterexamples via PGD.

    Performs adversarial mining using the provided objective,
    then filters the results using the condition evaluator to return 
    only true counterexamples strictly inside the current rho-sublevel set.
    """
    bounds = _bounds_tensor(config.train_bounds, device)
    lbx, ubx = bounds[0], bounds[1]
    adv_states = initial_states.clone().to(device=device)
    step = config.step_size * (ubx - lbx).unsqueeze(0)

    with th.no_grad():
        best_states = adv_states.clone()
        best_violations = th.zeros(
            adv_states.shape[0],
            dtype=adv_states.dtype,
            device=device,
        )
        init_violations, init_mask = condition_evaluator(adv_states)
        init_violations = init_violations.flatten()
        best_violations[init_mask] = init_violations[init_mask]

    for _ in range(config.descent_steps):
        adv_states.requires_grad_(True)
        raw_objective = objective(adv_states)

        grad = th.autograd.grad(
            raw_objective.mean(),
            adv_states,
            retain_graph=False,
            create_graph=False,
        )[0]

        with th.no_grad():
            candidate_states = adv_states - step * grad.sign()
            candidate_states = project_to_box(candidate_states, lbx, ubx)
            candidate_violations, candidate_mask = condition_evaluator(candidate_states)
            candidate_violations = candidate_violations.flatten()

            improved = candidate_mask & (candidate_violations > best_violations)

            best_states[improved] = candidate_states[improved]
            best_violations[improved] = candidate_violations[improved]

            adv_states = candidate_states

    with th.no_grad():
        if config.origin_exclusion is not None:
            exclusion = th.as_tensor(config.origin_exclusion, dtype=best_states.dtype, device=device)
            inside_exclusion = th.all(th.abs(best_states) <= exclusion, dim=-1)
            counter_mask = (best_violations > 0.0) & (~inside_exclusion)
        else:
            counter_mask = best_violations > 0.0

    cex_states = best_states[counter_mask].clone().detach()
    cex_violations = best_violations[counter_mask].clone().detach()
    return cex_states, cex_violations