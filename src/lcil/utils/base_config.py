from __future__ import annotations

import itertools
import json
import numpy as np

from argparse import ArgumentParser, BooleanOptionalAction
from collections.abc import Sequence
from dataclasses import MISSING, asdict, field as dataclass_field, fields, is_dataclass, replace
from os import PathLike
from pathlib import Path
from types import UnionType
from typing import Any, ClassVar, Literal, Union, Self, get_args, get_origin, get_type_hints


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


def config_field(
    *,
    help: str | None = None,
    description: str | None = None,
    cli: bool = True,
    argparse_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Create a dataclass field with CLI metadata for ``ArgumentParserConfig``."""
    metadata = dict(kwargs.pop("metadata", {}))

    if help is not None:
        metadata["help"] = help
    if description is not None:
        metadata["description"] = description

    metadata.setdefault("cli", cli)

    if argparse_kwargs is not None:
        merged_argparse_kwargs = dict(metadata.get("argparse", {}))
        merged_argparse_kwargs.update(argparse_kwargs)
        metadata["argparse"] = merged_argparse_kwargs

    return dataclass_field(metadata=metadata, **kwargs)


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


def _infer_argparse_kwargs(annotation: Any) -> dict[str, Any]:
    """Infer common ``argparse`` kwargs from a dataclass field annotation."""
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        options = tuple(option for option in get_args(annotation) if option is not type(None))
        sequence_options = tuple(option for option in options if _annotation_is_sequence(option))
        if sequence_options:
            return _infer_argparse_kwargs(sequence_options[0])
        if len(options) == 1:
            return _infer_argparse_kwargs(options[0])
        if str in options or any(option in (Path, PathLike) for option in options):
            return {"type": str}
        if int in options:
            return {"type": int}
        if float in options:
            return {"type": float}
        return {}

    if annotation is bool:
        return {"action": BooleanOptionalAction}
    if annotation in (str, int, float):
        return {"type": annotation}
    if annotation in (Path, PathLike):
        return {"type": str}

    if origin is Literal:
        choices = get_args(annotation)
        if choices:
            return {"type": type(choices[0]), "choices": choices}
        return {}

    if origin in (list, tuple, Sequence):
        item_types = get_args(annotation)
        if not item_types:
            return {"nargs": "+"}
        if len(item_types) == 2 and item_types[1] is Ellipsis:
            item_annotation = item_types[0]
        elif all(item_type == item_types[0] for item_type in item_types):
            item_annotation = item_types[0]
        else:
            return {"nargs": "+"}

        arg_kwargs = _infer_argparse_kwargs(item_annotation)
        arg_kwargs.pop("action", None)
        arg_kwargs["nargs"] = "+"
        return arg_kwargs

    return {}


def _annotation_is_sequence(annotation: Any) -> bool:
    """Return whether an annotation represents a sequence-like config value."""
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        options = tuple(option for option in get_args(annotation) if option is not type(None))
        return any(_annotation_is_sequence(option) for option in options)
    return origin in (list, tuple, Sequence)


def _namespace_dest(prefix: str, field_name: str) -> str:
    """Return the argparse namespace destination for a prefixed field name."""
    return f"{prefix}{field_name}".replace("-", "_")


class JsonDataclass:
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
    def from_dict(cls, data: dict[str, Any]) -> JsonDataclass:
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
    def load(cls, path: PathLike[str] | str) -> JsonDataclass:
        """Load configuration from JSON file or directory."""
        input_path = cls._resolve_path(path)
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        return cls.from_dict(payload)


class ArgumentParserConfig:
    """"Mixin for dataclass-based configs that can be constructed from argparse arguments."""

    def add_to_argparse(
        self,
        parser: ArgumentParser,
        *,
        prefix: str = "",
        include_fields: set[str] | None = None,
        exclude_fields: set[str] | None = None,
        nargs_fields: set[str] | None = None,
    ) -> None:
        """Add configuration fields to an argparse.ArgumentParser.

        Parameters
        ----------
        parser : ArgumentParser
            Parser receiving the generated CLI arguments.
        prefix : str, optional
            Prefix prepended to argument names.
        include_fields : set[str] | None, optional
            Restrict registration to this field subset.
        exclude_fields : set[str] | None, optional
            Skip registration for this field subset.
        nargs_fields : set[str] | None, optional
            Scalar fields that should be registered with ``nargs='+'`` so they
            can drive grid/sweep expansion through ``iter_from_namespace``.
        """
        try:
            type_hints = get_type_hints(type(self))
        except (NameError, TypeError):
            type_hints = {}

        for field_info in fields(self):
            if include_fields is not None and field_info.name not in include_fields:
                continue
            if exclude_fields is not None and field_info.name in exclude_fields:
                continue

            metadata = dict(field_info.metadata)
            if metadata.get("cli", True) is False or not field_info.init:
                continue

            arg_name = f"--{prefix}{field_info.name.replace('_', '-')}"
            annotation = type_hints.get(field_info.name, field_info.type)
            arg_kwargs = _infer_argparse_kwargs(annotation)
            arg_kwargs.update(metadata.get("argparse", {}))

            if nargs_fields is not None and field_info.name in nargs_fields:
                if _annotation_is_sequence(annotation):
                    raise ValueError(
                        f"nargs_fields only supports scalar fields, got sequence field '{field_info.name}'."
                    )
                if "action" in arg_kwargs:
                    raise ValueError(
                        f"nargs_fields is incompatible with action-based field '{field_info.name}'."
                    )
                arg_kwargs["nargs"] = "+"

            help_text = metadata.get("help") or metadata.get("description")
            if help_text is not None and "help" not in arg_kwargs:
                arg_kwargs["help"] = help_text

            default_value = getattr(self, field_info.name, MISSING)
            if default_value is not MISSING and "default" not in arg_kwargs:
                arg_kwargs["default"] = default_value

            parser.add_argument(arg_name, **arg_kwargs)

    def iter_from_namespace(
        self,
        args: Any,
        *,
        prefix: str = "",
    ) -> Sequence[Self]:
        """Build one or more concrete config instances from parsed argparse values.

        Scalar config fields that were parsed via ``nargs`` are expanded using
        ``itertools.product``. Sequence-typed config fields remain sequence-valued.
        Only fields that are present in the parsed namespace are considered.
        """
        try:
            type_hints = get_type_hints(type(self))
        except (NameError, TypeError):
            type_hints = {}

        fixed_values: dict[str, Any] = {}
        sweep_field_names: list[str] = []
        sweep_field_values: list[list[Any]] = []

        for field_info in fields(self):
            metadata = dict(field_info.metadata)
            if metadata.get("cli", True) is False or not field_info.init:
                continue

            dest = _namespace_dest(prefix, field_info.name)
            if not hasattr(args, dest):
                continue

            annotation = type_hints.get(field_info.name, field_info.type)
            raw_value = getattr(args, dest)

            if isinstance(raw_value, list) and not _annotation_is_sequence(annotation):
                sweep_field_names.append(field_info.name)
                sweep_field_values.append([
                    _restore_typed_value(item, annotation) for item in raw_value
                ])
                continue

            fixed_values[field_info.name] = _restore_typed_value(raw_value, annotation)

        if not sweep_field_names:
            return [replace(self, **fixed_values)]

        configs: list[Self] = []
        for combination in itertools.product(*sweep_field_values):
            values = dict(fixed_values)
            values.update(dict(zip(sweep_field_names, combination)))
            configs.append(replace(self, **values))
        return configs

    def from_namespace(
        self,
        args: Any,
        *,
        prefix: str = "",
    ) -> Self:
        """Build a single concrete config instance from parsed argparse values."""
        configs = self.iter_from_namespace(
            args,
            prefix=prefix,
        )
        if len(configs) != 1:
            raise ValueError(
                f"Expected exactly one config, got {len(configs)}. Use iter_from_namespace for sweep arguments."
            )
        return configs[0]
