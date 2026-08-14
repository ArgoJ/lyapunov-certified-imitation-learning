from __future__ import annotations

import importlib
import itertools
import logging
import json
import numpy as np

from argparse import ArgumentParser, BooleanOptionalAction
from collections.abc import Callable, Sequence
from dataclasses import MISSING, asdict, field as dataclass_field, fields, is_dataclass, replace
from os import PathLike
from pathlib import Path
from types import UnionType
from typing import Any, ClassVar, Literal, Union, Self, get_args, get_origin, get_type_hints, TypeVar

__logger__ = logging.getLogger(__name__)


T = TypeVar("T")


def positive_validator(value: int | float, name: str) -> None:
    fvalue = float(value)
    if fvalue <= 0.0:
        raise ValueError(f"{name} must be a positive number.")
    
def non_negative_validator(value: int | float, name: str) -> None:
    fvalue = float(value)
    if fvalue < 0.0:
        raise ValueError(f"{name} must be a non-negative number.")

def fraction_validator(value: float, name: str) -> None:
    fvalue = float(value)
    if fvalue < 0.0 or fvalue > 1.0:
        raise ValueError(f"{name} must be in the range [0, 1].")

def growth_rate_validator(value: float, name: str) -> None:
    fvalue = float(value)
    if fvalue <= 1.0:
        raise ValueError(f"{name} must be greater than 1.")

def pathlike_validator(value: str | PathLike, name: str) -> None:
    try:
        Path(value)
    except Exception as e:
        raise ValueError(f"{name} must be a valid file system path.") from e


def ndarray_validator(value: Any, name: str) -> None:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be a numpy.ndarray.")


def bounds_include_origin_validator(value: Any, name: str) -> None:
    ndarray_validator(value, name)
    if value.ndim != 2 or value.shape[0] != 2:
        raise ValueError(
            f"{name} must have shape (2, n), got {value.shape}."
        )

    lbx = value[0]
    ubx = value[1]
    if (lbx > ubx).any():
        raise ValueError(f"{name} has invalid bounds: lbx must be <= ubx.")

    if (lbx > 0).any() or (ubx < 0).any():
        raise ValueError(
            f"{name} must include the origin in every dimension (lbx <= 0 <= ubx)."
        )

def array_shape_validator(expected_shape: tuple[int, ...]) -> Callable[[Any, str], None]:
    def validator(value: Any, name: str) -> None:
        if not isinstance(value, np.ndarray):
            raise ValueError(f"{name} must be a numpy.ndarray.")
        if value.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {value.shape}."
            )
    return validator

def literal_validator(allowed_values: Sequence[Any]) -> Callable[[Any, str], None]:
    def validator(value: Any, name: str) -> None:
        if value not in allowed_values:
            raise ValueError(f"{name} must be one of {allowed_values}, got {value}.")
    return validator

def optional_validator(*base_validators: Callable[[Any, str], None]) -> Callable[[Any, str], None]:
    def wrapper(value: Any, name: str) -> None:
        if value is not None:
            for base_validator in base_validators:
                base_validator(value, name)
    return wrapper

def normalize_scalar_or_sequence(
    value: Any,
    *,
    state_dim: int,
    name: str,
    caster: Callable[[Any], T],
) -> T | tuple[T, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        normalized = tuple(caster(item) for item in value)
        if len(normalized) != state_dim:
            raise ValueError(f"{name} must be scalar or match state_dim.")
        return normalized

    scalar = caster(value)
    return (scalar,) * state_dim

def sequence_validator(
    base_validator: Callable[[Any, str], None],
) -> Callable[[Any, str], None]:
    """Apply a scalar validator element-wise to scalar-or-sequence config values."""

    def wrapper(value: Any, name: str) -> None:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for index, item in enumerate(value):
                base_validator(item, f"{name}[{index}]")
            return
        base_validator(value, name)

    return wrapper

def run_field_validators(instance: Any) -> None:
    for field_info in fields(instance):
        for validator in field_info.metadata.get("validators", ()):
            validator(getattr(instance, field_info.name), field_info.name)

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


def _safe_get_type_hints(obj: Any) -> dict[str, Any]:
    """Return resolved type hints, re-importing the defining module if needed."""
    try:
        return get_type_hints(obj)
    except (NameError, TypeError):
        module_name = getattr(obj, "__module__", None)
        if isinstance(module_name, str):
            try:
                importlib.import_module(module_name)
                return get_type_hints(obj)
            except (ImportError, NameError, TypeError):
                pass
        return {}


def config_field(
    *,
    help: str | None = None,
    description: str | None = None,
    display_alias: str | None = None,
    cli: bool = True,
    validators: Sequence[Callable[[Any, str], None]] | None = None,
    argparse_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Create a dataclass field with CLI metadata for ``ArgumentParserConfig``."""
    metadata = dict(kwargs.pop("metadata", {}))

    if help is not None:
        metadata["help"] = help
    if description is not None:
        metadata["description"] = description
    if display_alias is not None:
        metadata["display_alias"] = display_alias

    metadata.setdefault("cli", cli)

    if validators is not None:
        metadata["validators"] = tuple(validators)

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
        options = tuple(option for option in get_args(annotation) if option is not type(None))

        if isinstance(value, list) and len(value) == 1:
            scalar_options = tuple(
                option for option in options if not _annotation_is_sequence(option)
            )
            for option in scalar_options:
                try:
                    return _restore_typed_value(value[0], option)
                except (TypeError, ValueError, KeyError):
                    continue

        for option in options:
            try:
                return _restore_typed_value(value, option)
            except (TypeError, ValueError, KeyError):
                continue
        return value

    if annotation in (str, int, float):
        if isinstance(value, (list, tuple, dict)):
            raise TypeError(f"Cannot restore {annotation} from {type(value).__name__}.")
        return annotation(value)

    if annotation in (Path, PathLike):
        if isinstance(value, (list, tuple, dict)):
            raise TypeError(f"Cannot restore {annotation} from {type(value).__name__}.")
        return Path(value)

    if origin is list:
        item_type = get_args(annotation)[0] if get_args(annotation) else Any
        if isinstance(value, list):
            return [_restore_typed_value(item, item_type) for item in value]
        return value

    if origin is Sequence:
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


def extend_fields(arg_fields: set[str] | None, _fields: Sequence[Any]) -> set[str]:
    expanded_args_fields: set[str] = set()
    
    if arg_fields is not None:
        for entry in arg_fields:
            # *substring*
            if entry.startswith("*") and entry.endswith("*") and len(entry) > 2:
                substring = entry[1:-1]
                for field_info in _fields:
                    if substring in field_info.name:
                        expanded_args_fields.add(field_info.name)
            
            # *suffix
            elif entry.startswith("*") and len(entry) > 1:
                suffix = entry[1:]
                for field_info in _fields:
                    if field_info.name.endswith(suffix):
                        expanded_args_fields.add(field_info.name)
            
            # prefix*
            elif entry.endswith("*") and len(entry) > 1:
                prefix = entry[:-1]
                for field_info in _fields:
                    if field_info.name.startswith(prefix):
                        expanded_args_fields.add(field_info.name)
            
            # exact match
            elif any(field_info.name == entry for field_info in _fields):
                expanded_args_fields.add(entry)
            
            # no match
            else:
                __logger__.warning(f"Argument field '{entry}' does not match any config fields")
    
    return expanded_args_fields


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
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Build config from a dictionary and restore numpy fields."""
        values = dict(data)

        type_hints = _safe_get_type_hints(cls)

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
    def load(cls, path: PathLike[str] | str) -> Self:
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
        suppress_defaults: bool = False,
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

        Notes
        -----
        All fields can be used with a * pre- or/and suffix to match multiple fields based on substring or prefix/suffix patterns.
        For example, *learning_rate would match all fields ending with learning_rate, and weight*
        """
        type_hints = _safe_get_type_hints(type(self))

        dataclass_fields = fields(self)
        if include_fields is not None:
            include_fields = extend_fields(include_fields, dataclass_fields)
        if exclude_fields is not None:
            exclude_fields = extend_fields(exclude_fields, dataclass_fields)
        if nargs_fields is not None:
            nargs_fields = extend_fields(nargs_fields, dataclass_fields)

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
            if "help" in arg_kwargs:
                help_text = arg_kwargs["help"]

            default_value = getattr(self, field_info.name, MISSING)
            if default_value is not MISSING and "default" not in arg_kwargs:
                if suppress_defaults:
                    import argparse
                    arg_kwargs["default"] = argparse.SUPPRESS
                    if help_text is not None:
                        help_text = f"{help_text} (default: {default_value})"
                    else:
                        help_text = f"(default: {default_value})"
                else:
                    arg_kwargs["default"] = default_value

            if help_text is not None:
                arg_kwargs["help"] = help_text

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
        type_hints = _safe_get_type_hints(type(self))

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
