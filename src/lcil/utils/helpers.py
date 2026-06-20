import numpy as np
import time
import logging

from pathlib import Path
from numpy.typing import ArrayLike
from typing import Callable
from functools import wraps

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


def add_entry(entry: str, output_root: str | Path, summary_name: str) -> None:
    """Appends a string to a file in the specified directory."""
    summary_path = Path(output_root) / summary_name
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "a", encoding="utf-8") as f:
        if not entry.endswith("\n"):
            entry += "\n"
        f.write(entry)


def timeit(logger: logging.Logger) -> Callable:
    """Decorator factory to measure execution time of a function."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"{func.__name__} executed in {elapsed:.4f} seconds")
            return result
            
        return wrapper
        
    return decorator
