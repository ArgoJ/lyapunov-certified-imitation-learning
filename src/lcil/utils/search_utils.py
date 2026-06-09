import logging

from functools import partial
from typing import Callable
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from .helpers import none_to_float

__logger__ = logging.getLogger(__name__)


def iterative_rho_search(
    total: int,
    desc: str,
    initial_values: tuple[float, float],
    step_fn: Callable,
    eval_fn: Callable[[float], bool],
    progress: Progress | None = None,
) -> tuple[float | None, float]:
    """Run iterative value updates until stopping criterion is met.

    Parameters
    ----------
    total : int
        Maximum number of iterations.
    desc : str
        Progress-bar description.
    initial_values : tuple[float, float]
        Initial ``(val_lo, val_up)`` values.
    step_fn : Callable
        Function mapping ``(val_lo, val_up, eval_fn)`` to ``(stop, val_lo, val_up)``.
    eval_fn : Callable[[float], bool]
        The actual evaluation function to determine if a value is certified.
    transient : bool, optional
        Whether to use a transient progress bar.

    Returns
    -------
    tuple[float | None, float]
        Final ``(val_lo, val_up)`` pair.
    """
    val_lo, val_up = initial_values
    is_progress = isinstance(progress, Progress)
    task = progress.add_task(desc, total=total, detail="") if is_progress else None

    def update_progress(trial: float, advance: int = 0) -> None:
        if is_progress and task is not None:
            detail_str = (
                f" lo: {none_to_float(val_lo):.4f} "
                f"up: {none_to_float(val_up):.4f} "
                f"[magenta]test: {none_to_float(trial):.4f}[/magenta] "
            )
            progress.update(task, advance=advance, detail=detail_str)
            progress.refresh()

    def tracked_eval(trial: float) -> bool:
        update_progress(trial)
        return eval_fn(trial)

    try:
        update_progress(0.0)
        for _ in range(total):
            try:
                stop, val_lo, val_up = step_fn(val_lo, val_up, tracked_eval)
                if stop:
                    break
            finally:
                update_progress(val_up, advance=1)
    finally:
        if is_progress and task is not None:
            progress.remove_task(task)

    return val_lo, val_up


def scale_rho_up(
    lo: float, 
    up: float, 
    eval_fn: Callable[[float], bool], 
    scaling: float
) -> tuple[bool, float, float]:
    """Scales up ``lo`` until the new trial is not certified."""
    trial = up * scaling
    if eval_fn(trial):
        return False, trial, trial
    return True, lo, trial


def scale_rho_down(
    lo: float | None, 
    up: float, 
    eval_fn: Callable[[float], bool], 
    min_val: float, 
    scaling: float
) -> tuple[bool, float | None, float]:
    """Scales down ``up`` until the new trial is either certified 
    or it is below min_val, in which case we stop and return the last certified val as lower bound.
    """
    trial = max(min_val, up / scaling)
    if eval_fn(trial):
        return True, trial, up

    if trial <= min_val:
        return True, None, trial
    return False, lo, trial


def bisect_rho(
    lo: float, 
    up: float, 
    eval_fn: Callable[[float], bool], 
    bisection_tol: float
) -> tuple[bool, float, float]:
    """Uses the midpoint of upper and lower for bisection."""
    mid = 0.5 * (lo + up)

    if eval_fn(mid):
        lo = mid
    else:
        up = mid

    if up - lo <= bisection_tol:
        return True, lo, up
    return False, lo, up


def search_and_bisect_value(
    initial_estimate: float,
    eval_fn: Callable[[float], bool],
    min_val: float,
    scaling_factor: float,
    bisection_tol: float,
    max_scale_steps: int,
    max_bisection_steps: int,
    value_name: str = "val",
    progress: Progress | None = None,
) -> float:
    """Orchestrates scaling and bisection to find the maximum certifiable value."""
    if initial_estimate <= 0:
        raise ValueError("initial_estimate must be positive.")

    if min_val > initial_estimate:
        __logger__.warning(
            "Provided initial estimate (%.4f) is below min_val (%.4f). Starting search from min_%s.",
            initial_estimate,
            min_val,
            value_name,
        )
        initial_val = min_val
    else:
        initial_val = float(initial_estimate)

    initial_ok = eval_fn(initial_val)

    # Scale up
    if initial_ok:
        val_lo, val_up = iterative_rho_search(
            total=max_scale_steps,
            desc=f"Scale up {value_name}",
            initial_values=(initial_val, initial_val),
            step_fn=partial(scale_rho_up, scaling=scaling_factor),
            eval_fn=eval_fn,
            progress=progress,
        )
        if val_lo == val_up:
            __logger__.warning(
                "Maximum scaling steps (%d) reached without finding an upper bound. " \
                "Consider increasing max_scale_steps or initial_estimate.",
                max_scale_steps,
            )

    # Scale down
    else:
        val_lo, val_up = iterative_rho_search(
            total=max_scale_steps,
            desc=f"Scale down {value_name}",
            initial_values=(None, initial_val),
            step_fn=partial(scale_rho_down, min_val=min_val, scaling=scaling_factor),
            eval_fn=eval_fn,
            progress=progress,
        )

        # Fallback
        if val_lo is None:
            __logger__.warning(
                "Could not find any certified value >= min_%s (%.0e).",
                value_name,
                min_val,
            )
            return 0.0

    # Bisect
    if val_up - val_lo >= bisection_tol:
        val_lo, val_up = iterative_rho_search(
            total=max_bisection_steps,
            desc=f"Bisect {value_name}",
            initial_values=(val_lo, val_up),
            step_fn=partial(bisect_rho, bisection_tol=bisection_tol),
            eval_fn=eval_fn,
            progress=progress,
        )

    return val_lo