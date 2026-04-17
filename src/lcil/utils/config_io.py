from __future__ import annotations

import json
from dataclasses import asdict, fields, is_dataclass
from os import PathLike
from pathlib import Path
from types import UnionType
from typing import Any, ClassVar, Union, get_args, get_origin, get_type_hints

import numpy as np


def _to_json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_json_compatible(val) for key, val in value.items()}
    elif isinstance(value, (list, tuple)):
        return [_to_json_compatible(val) for val in value]
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, np.generic):
        return value.item()
    elif isinstance(value, (PathLike, Path)):
        return str(value)
    elif hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
        return value.to_dict()
    return value

def resolve_path(p: str | Path | None) -> str | None:
    return str(Path(p).resolve()) if p is not None else None


def _restore_typed_value(value: Any, annotation: Any) -> Any:
    """Restore nested values from JSON-compatible payloads using type hints."""
    if value is None:
        return None

    origin = get_origin(annotation)

    if origin in (Union, UnionType):
        for option in get_args(annotation):
            if option is type(None):
                continue
            try:
                return _restore_typed_value(value, option)
            except (TypeError, ValueError, KeyError):
                continue
        return value

    if origin is list:
        item_type = get_args(annotation)[0] if get_args(annotation) else Any
        if isinstance(value, list):
            return [_restore_typed_value(item, item_type) for item in value]
        return value

    if origin is tuple:
        item_types = get_args(annotation)
        if isinstance(value, list):
            if len(item_types) == 2 and item_types[1] is Ellipsis:
                return tuple(_restore_typed_value(item, item_types[0]) for item in value)
            if len(item_types) == len(value):
                return tuple(_restore_typed_value(item, item_type) for item, item_type in zip(value, item_types))
            return tuple(value)
        return value

    if isinstance(value, dict) and hasattr(annotation, "from_dict") and callable(getattr(annotation, "from_dict")):
        return annotation.from_dict(value)

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

        try:
            type_hints = get_type_hints(cls)
        except (NameError, TypeError):
            type_hints = {}

        for field_name, annotation in type_hints.items():
            if field_name in values:
                values[field_name] = _restore_typed_value(values[field_name], annotation)

        for field_name in cls.NP_ARRAY_FIELDS:
            if field_name in values and values[field_name] is not None:
                values[field_name] = np.asarray(values[field_name])

        if is_dataclass(cls):
            init_field_names = {
                field.name for field in fields(cls) if field.init
            }
            values = {
                key: value for key, value in values.items() if key in init_field_names
            }

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