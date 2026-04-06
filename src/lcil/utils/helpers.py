def none_to_float(value: float | None) -> float:
    return float(value) if value is not None else float("nan")
