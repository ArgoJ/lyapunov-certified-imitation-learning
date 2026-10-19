"""Utility functions for certification and control boundaries."""

from __future__ import annotations

import itertools
import torch as th

from collections.abc import Sequence
from typing import Any



def compute_blended_lqr_c0_c1(
    exclusion_box: Sequence[float],
    P: Any,
    factor: float = 4.0,
) -> tuple[float, float]:
    """Compute Lyapunov sublevel thresholds c0 and c1 for BlendedLQRPolicy.

    Takes a normalized exclusion box sequence (matching state_dim) and matrix P,
    and returns (c0, c1) where:
    - c0 is the circumscribed ellipsoid around the exclusion box (max_{x in B} x^T P x).
    - c1 is the outer ellipsoid threshold (c1 = factor * c0).

    Parameters
    ----------
    exclusion_box : Sequence[float]
        Per-dimension half-widths of the origin exclusion box, as produced
        by configuration normalization (length must equal state_dim).
    P : Tensor, ndarray, LQRController, or BlendedLQRPolicy
        Symmetric positive definite matrix P for V(x) = x^T P x, or an
        object containing P.
    factor : float, optional
        Factor by which c1 is larger than c0 (c1 = factor * c0), by default 4.0
        (corresponding to a 2x box expansion since (2x)^T P (2x) = 4 x^T P x).
        Must be strictly greater than 1.0.

    Returns
    -------
    tuple[float, float]
        (c0, c1) for BlendedLQRPolicy.
    """
    if factor <= 1.0:
        raise ValueError(
            f"factor must be greater than 1.0 so that c1 > c0, got {factor}."
        )

    from ..utils import extract_p_matrix

    p_sym = extract_p_matrix(P, as_tensor=True, dtype=th.float64)
    n = p_sym.shape[0]

    if len(exclusion_box) != n:
        raise ValueError(
            f"exclusion_box length ({len(exclusion_box)}) must match state_dim ({n})."
        )

    r = th.tensor(exclusion_box, dtype=th.float64, device=p_sym.device)

    # Evaluate extreme vertices of the hyperrectangle
    if n == 1:
        signs = th.tensor([[1.0]], dtype=th.float64, device=p_sym.device)
    else:
        combos = list(itertools.product([-1.0, 1.0], repeat=n - 1))
        signs = th.tensor([[1.0] + list(c) for c in combos], dtype=th.float64, device=p_sym.device)

    vertices = signs * r.unsqueeze(0)
    c0 = float((vertices @ p_sym * vertices).sum(dim=-1).max().item())
    c1 = float(factor * c0)

    return c0, c1
