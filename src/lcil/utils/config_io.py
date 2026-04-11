from __future__ import annotations

import json
from dataclasses import asdict
from os import PathLike
from pathlib import Path
from typing import Any, ClassVar

import numpy as np


def _to_json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_json_compatible(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_compatible(val) for val in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class JsonConfigMixin:
    """Reusable JSON I/O mixin for dataclass-based config objects.

    Notes
    -----
    ``save`` and ``load`` accept either a JSON file path or a directory path.
    When a directory is provided, ``<ClassName>.json`` is used.
    """

    NP_ARRAY_FIELDS: ClassVar[tuple[str, ...]] = ()
    DEFAULT_FILE_NAME: ClassVar[str | None] = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary representation."""
        return _to_json_compatible(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JsonConfigMixin:
        """Build config from a dictionary and restore numpy fields."""
        values = dict(data)
        for field_name in cls.NP_ARRAY_FIELDS:
            if field_name in values and values[field_name] is not None:
                values[field_name] = np.asarray(values[field_name])
        return cls(**values)

    @classmethod
    def _default_file_name(cls) -> str:
        return cls.DEFAULT_FILE_NAME or f"{cls.__name__}.json"

    @classmethod
    def _resolve_path(cls, path: PathLike[str] | str) -> Path:
        resolved = Path(path)
        if resolved.suffix.lower() == ".json":
            return resolved
        return resolved / cls._default_file_name()

    def save(self, path: PathLike[str] | str) -> Path:
        """Persist configuration to human-readable JSON."""
        output_path = self._resolve_path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output_path

    @classmethod
    def load(cls, path: PathLike[str] | str) -> JsonConfigMixin:
        """Load configuration from JSON file or directory."""
        input_path = cls._resolve_path(path)
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        return cls.from_dict(payload)