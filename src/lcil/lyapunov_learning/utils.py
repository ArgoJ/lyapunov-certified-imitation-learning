from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Any, Generic, TypeVar

import numpy as np

from ..utils.base_config import ArgumentParserConfig

ConfigT = TypeVar("ConfigT")
NamespaceConfigT = TypeVar("NamespaceConfigT", bound=ArgumentParserConfig)

_SANITIZE_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


class TrainingAbortedError(RuntimeError):
	"""Raised when Lyapunov training aborts before producing a valid result."""


@dataclass
class ThresholdMonitor:
	"""Track values and detect sustained low-value runs.

	Parameters
	----------
	threshold : float
		Values below this threshold count toward the stopping streak.
	patience : int
		Number of consecutive below-threshold values required to trigger.
	"""

	threshold: float = 1.0
	patience: int = 10
	value_history: list[float] = field(default_factory=list)
	consecutive_low: int = 0

	def __post_init__(self) -> None:
		if self.patience <= 0:
			raise ValueError("patience must be positive.")
		if self.threshold <= 0.0:
			raise ValueError("threshold must be positive.")

	def update(self, value: float) -> bool:
		"""Register one value and return whether training should stop."""
		value = float(value)
		self.value_history.append(value)
		if value < self.threshold:
			self.consecutive_low += 1
		else:
			self.consecutive_low = 0
		return self.should_stop

	@property
	def should_stop(self) -> bool:
		"""Return whether the low-value stopping criterion is active."""
		return self.consecutive_low >= self.patience


@dataclass(frozen=True)
class GridSearchRun(Generic[ConfigT]):
	"""Concrete metadata for one run in a hyperparameter sweep.

	Parameters
	----------
	index : int
		Zero-based index of the run inside the sweep.
	total : int
		Total number of runs in the sweep.
	config : ConfigT
		Concrete configuration for this run.
	run_name : str
		Directory-safe identifier derived from the sweep parameters.
	output_dir : Path
		Output directory created for this run.
	description : str
		Human-readable description of the swept parameters.
	"""

	index: int
	total: int
	config: ConfigT
	run_name: str
	output_dir: Path
	description: str

	@property
	def display_index(self) -> int:
		"""Return the one-based index used in progress output."""
		return self.index + 1

	def progress_message(self) -> str:
		"""Format a concise progress message for this run."""
		if self.description:
			return f"[{self.display_index}/{self.total}] {self.description}"
		return f"[{self.display_index}/{self.total}] {self.run_name}"


class GridSearchHelper(Generic[ConfigT]):
	"""Create concrete sweep runs with stable names and directories.

	The helper infers which config fields vary across a sweep, builds
	directory-safe run names from those fields, and materializes per-run output
	directories inside a timestamped sweep directory.
	"""

	def __init__(
		self,
		configs: Sequence[ConfigT],
		*,
		output_root: str | Path,
		sweep_id: str | None = None,
		run_name_fields: Sequence[str] | None = None,
		exclude_fields: Sequence[str] | None = None,
		field_aliases: Mapping[str, str] | None = None,
		extra_name_parts: Mapping[str, Any] | None = None,
	) -> None:
		if not configs:
			raise ValueError("Grid search requires at least one configuration.")

		self._configs = tuple(configs)
		self.sweep_id = sweep_id or datetime.now().strftime("%Y%m%d_%H%M%S")
		self.sweep_base_path = Path(output_root) / self.sweep_id
		self.sweep_base_path.mkdir(parents=True, exist_ok=True)

		self.field_aliases = dict(field_aliases or {})
		self.extra_name_parts = dict(extra_name_parts or {})

		if run_name_fields is None:
			run_name_fields = infer_varying_fields(
				self._configs,
				exclude_fields=exclude_fields,
			)
		self.run_name_fields = tuple(run_name_fields)

	@classmethod
	def from_namespace(
		cls,
		config_defaults: NamespaceConfigT,
		args: Any,
		*,
		output_root: str | Path,
		prefix: str = "",
		sweep_id: str | None = None,
		run_name_fields: Sequence[str] | None = None,
		exclude_fields: Sequence[str] | None = None,
		field_aliases: Mapping[str, str] | None = None,
		extra_name_parts: Mapping[str, Any] | None = None,
	) -> GridSearchHelper[NamespaceConfigT]:
		"""Build a grid-search helper directly from parsed CLI arguments."""
		configs = config_defaults.iter_from_namespace(args, prefix=prefix)
		return cls(
			configs,
			output_root=output_root,
			sweep_id=sweep_id,
			run_name_fields=run_name_fields,
			exclude_fields=exclude_fields,
			field_aliases=field_aliases,
			extra_name_parts=extra_name_parts,
		)

	def __len__(self) -> int:
		return len(self._configs)

	def __iter__(self) -> Iterator[GridSearchRun[ConfigT]]:
		return self.iter_runs()

	@property
	def configs(self) -> tuple[ConfigT, ...]:
		"""Return all concrete configs in sweep order."""
		return self._configs

	def iter_runs(self) -> Iterator[GridSearchRun[ConfigT]]:
		"""Yield concrete runs with names, descriptions, and output folders."""
		used_names: set[str] = set()

		for index, config in enumerate(self._configs):
			base_name = build_grid_search_run_name(
				config,
				include_fields=self.run_name_fields,
				field_aliases=self.field_aliases,
				extra_parts=self.extra_name_parts,
			)
			run_name = base_name or f"run_{index + 1:03d}"
			if run_name in used_names:
				run_name = f"{run_name}__run_{index + 1:03d}"
			used_names.add(run_name)

			output_dir = self.sweep_base_path / run_name
			output_dir.mkdir(parents=True, exist_ok=True)

			yield GridSearchRun(
				index=index,
				total=len(self._configs),
				config=config,
				run_name=run_name,
				output_dir=output_dir,
				description=describe_grid_search_run(
					config,
					include_fields=self.run_name_fields,
					field_aliases=self.field_aliases,
					extra_parts=self.extra_name_parts,
				),
			)


def infer_varying_fields(
	configs: Sequence[ConfigT],
	*,
	exclude_fields: Sequence[str] | None = None,
) -> tuple[str, ...]:
	"""Infer scalar-like dataclass fields that vary across a sweep."""
	if not configs:
		return ()

	excluded = set(exclude_fields or ())
	reference_items = _iter_config_items(configs[0])
	varying_fields: list[str] = []

	for field_name, reference_value in reference_items:
		if field_name in excluded or not _supports_run_name_value(reference_value):
			continue

		if any(
			not _values_equal(reference_value, current_value)
			for current_value in (_lookup_config_value(config, field_name) for config in configs[1:])
		):
			varying_fields.append(field_name)

	return tuple(varying_fields)


def build_grid_search_run_name(
	config: Any,
	*,
	include_fields: Sequence[str],
	field_aliases: Mapping[str, str] | None = None,
	extra_parts: Mapping[str, Any] | None = None,
) -> str:
	"""Build a directory-safe run name from selected config fields."""
	aliases = dict(field_aliases or {})
	segments: list[str] = []

	for field_name in include_fields:
		value = _lookup_config_value(config, field_name)
		segments.append(
			f"{_sanitize_segment(aliases.get(field_name, field_name))}_{_format_name_value(value)}"
		)

	for field_name, value in (extra_parts or {}).items():
		segments.append(
			f"{_sanitize_segment(aliases.get(field_name, field_name))}_{_format_name_value(value)}"
		)

	return "__".join(segment for segment in segments if segment)


def describe_grid_search_run(
	config: Any,
	*,
	include_fields: Sequence[str],
	field_aliases: Mapping[str, str] | None = None,
	extra_parts: Mapping[str, Any] | None = None,
) -> str:
	"""Build a readable one-line description for a concrete sweep run."""
	aliases = dict(field_aliases or {})
	parts: list[str] = []

	for field_name in include_fields:
		value = _lookup_config_value(config, field_name)
		parts.append(f"{aliases.get(field_name, field_name)}: {_format_display_value(value)}")

	for field_name, value in (extra_parts or {}).items():
		parts.append(f"{aliases.get(field_name, field_name)}: {_format_display_value(value)}")

	return ", ".join(parts)


def _iter_config_items(config: Any) -> list[tuple[str, Any]]:
	if is_dataclass(config):
		return [(field_info.name, getattr(config, field_info.name)) for field_info in fields(config)]
	return list(vars(config).items())


def _lookup_config_value(config: Any, field_name: str) -> Any:
	if not hasattr(config, field_name):
		raise AttributeError(f"Config object has no field '{field_name}'.")
	return getattr(config, field_name)


def _supports_run_name_value(value: Any) -> bool:
	value = _normalize_value(value)
	if _is_scalar_value(value):
		return True
	if isinstance(value, (list, tuple)):
		return all(_is_scalar_value(_normalize_value(item)) for item in value)
	return False


def _is_scalar_value(value: Any) -> bool:
	return value is None or isinstance(value, (str, bool, int, float))


def _normalize_value(value: Any) -> Any:
	if isinstance(value, np.generic):
		return value.item()
	if isinstance(value, np.ndarray):
		return value.tolist()
	if isinstance(value, Path):
		return value.as_posix()
	return value


def _values_equal(left: Any, right: Any) -> bool:
	left_value = _normalize_value(left)
	right_value = _normalize_value(right)

	if isinstance(left_value, list) or isinstance(right_value, list):
		return left_value == right_value
	return left_value == right_value


def _format_name_value(value: Any) -> str:
	normalized = _normalize_value(value)

	if normalized is None:
		return "none"
	if isinstance(normalized, bool):
		return str(normalized).lower()
	if isinstance(normalized, float):
		return _sanitize_segment(f"{normalized:g}")
	if isinstance(normalized, (int, str)):
		return _sanitize_segment(str(normalized))
	if isinstance(normalized, list):
		return "-".join(_format_name_value(item) for item in normalized)
	if isinstance(normalized, tuple):
		return "-".join(_format_name_value(item) for item in normalized)
	raise TypeError(f"Unsupported run-name value type: {type(value)!r}")


def _format_display_value(value: Any) -> str:
	normalized = _normalize_value(value)

	if isinstance(normalized, float):
		return f"{normalized:g}"
	if isinstance(normalized, list):
		return "[" + ", ".join(_format_display_value(item) for item in normalized) + "]"
	if isinstance(normalized, tuple):
		return "[" + ", ".join(_format_display_value(item) for item in normalized) + "]"
	return str(normalized)


def _sanitize_segment(value: str) -> str:
	sanitized = _SANITIZE_PATTERN.sub("-", value).strip("-_")
	return sanitized or "value"