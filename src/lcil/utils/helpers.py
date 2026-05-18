import numpy as np
from numpy.typing import ArrayLike

def none_to_float(value: float | None) -> float:
    return float(value) if value is not None else float("nan")


def shoelace(points: ArrayLike) -> float:
    """Calculate the area of a 2D polygon defined by `points` using the shoelace formula."""
    points = np.asarray(points)
    if points.shape[1] != 2:
        raise ValueError("Input points must be of shape (N, 2) for 2D polygons.")
    x = points[:, 0]
    y = points[:, 1]
    if x.shape != y.shape:
        raise ValueError("Input arrays x and y must have the same shape.")
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) 