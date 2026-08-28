#!/usr/bin/env python3
"""Inspect and read keys, configurations, and datasets from HDF5 data files.

Usage examples:
    # Read P matrix (terminal cost W_e) from the latest dataset
    python examples/read_data.py config/P
    python examples/read_data.py global_config/cost/W_e

    # Inspect the full HDF5 tree structure
    python examples/read_data.py --tree

    # Read constraints or specific trajectory states
    python examples/read_data.py config/constraints
    python examples/read_data.py traj_0/trajectory/states

    # Specify a custom HDF5 file
    python examples/read_data.py config/P -f path/to/dataset.hdf5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .constants import RESULTS_DIR


def discover_latest_hdf5_file(root_dir: Path | str | None = None) -> Path:
    """Find the most recently modified .hdf5 file under the given root directory."""
    root = Path(root_dir) if root_dir is not None else RESULTS_DIR
    if not root.exists():
        raise FileNotFoundError(f"Root directory '{root}' does not exist.")

    hdf5_files = sorted(
        root.rglob("*.hdf5"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not hdf5_files:
        raise FileNotFoundError(f"No .hdf5 files found under '{root}'.")
    return hdf5_files[0]


def format_matrix(mat: np.ndarray, name: str = "") -> str:
    """Format a 1D or 2D numpy array cleanly."""
    prefix = f"{name} " if name else ""
    if mat.ndim == 0:
        return f"{prefix}{mat.item()}"
    if mat.ndim == 1:
        formatted = ", ".join(f"{v:10.4f}" if isinstance(v, (float, np.floating)) else str(v) for v in mat)
        return f"{prefix}[{formatted}]"
    if mat.ndim == 2:
        lines = [f"{prefix}shape: {mat.shape}, dtype: {mat.dtype}"]
        for row in mat:
            row_str = ", ".join(f"{v:12.6f}" if isinstance(v, (float, np.floating)) else f"{v!s:>8}" for v in row)
            lines.append(f"  [{row_str}]")
        return "\n".join(lines)
    return f"{prefix}shape: {mat.shape}, dtype: {mat.dtype}\n{mat}"


def print_tree(h5_group: h5py.Group, prefix: str = "", max_depth: int = 4, current_depth: int = 0) -> None:
    """Print a clean tree representation of an HDF5 group or file."""
    if current_depth >= max_depth:
        print(f"{prefix}...")
        return

    keys = list(h5_group.keys())
    # Truncate large lists of similar trajectory keys for readability
    traj_keys = [k for k in keys if k.startswith("traj_")]
    other_keys = [k for k in keys if not k.startswith("traj_")]

    display_keys = other_keys.copy()
    if traj_keys:
        display_keys.append(traj_keys[0])
        if len(traj_keys) > 1:
            display_keys.append(f"... ({len(traj_keys)} trajectories: traj_00000 .. {traj_keys[-1]})")

    for i, key in enumerate(display_keys):
        is_last = (i == len(display_keys) - 1)
        branch = "└── " if is_last else "├── "
        next_prefix = prefix + ("    " if is_last else "│   ")

        if key.startswith("..."):
            print(f"{prefix}{branch}{key}")
            continue

        item = h5_group[key]
        if isinstance(item, h5py.Dataset):
            print(f"{prefix}{branch}{key} [Dataset: shape={item.shape}, dtype={item.dtype}]")
        elif isinstance(item, h5py.Group):
            attr_count = len(item.attrs)
            attr_str = f" ({attr_count} attrs)" if attr_count > 0 else ""
            print(f"{prefix}{branch}{key}/{attr_str}")
            print_tree(item, prefix=next_prefix, max_depth=max_depth, current_depth=current_depth + 1)


def resolve_path_alias(query: str) -> list[str]:
    """Map common user queries and aliases (e.g. config/P) to actual HDF5 paths."""
    clean = query.strip("/ ")
    candidates = [clean]

    # Standardize aliases
    alias_map = {
        "p": ["global_config/cost/W_e", "cost/W_e", "W_e"],
        "config/p": ["global_config/cost/W_e", "global_config/cost/P", "config/cost/W_e"],
        "cost/p": ["global_config/cost/W_e", "cost/W_e"],
        "global_config/p": ["global_config/cost/W_e"],
        "global_config/cost/p": ["global_config/cost/W_e"],
        "q": ["global_config/cost/W", "cost/W", "W"],
        "config/q": ["global_config/cost/W", "config/cost/W"],
        "cost/q": ["global_config/cost/W", "cost/W"],
        "global_config/cost/q": ["global_config/cost/W"],
        "config": ["global_config"],
        "constraints": ["global_config/constraints"],
        "config/constraints": ["global_config/constraints"],
        "cost": ["global_config/cost"],
        "config/cost": ["global_config/cost"],
        "linear_system": ["global_config/linear_system"],
        "config/linear_system": ["global_config/linear_system"],
        "states": ["traj_00000/trajectory/states", "traj_0/trajectory/states"],
        "inputs": ["traj_00000/trajectory/inputs", "traj_0/trajectory/inputs"],
        "times": ["traj_00000/trajectory/times", "traj_0/trajectory/times"],
        "v_n": ["traj_00000/trajectory/V_N", "traj_0/trajectory/V_N"],
    }

    lower_query = clean.lower()
    if lower_query in alias_map:
        candidates = alias_map[lower_query] + candidates

    # Also handle 'config/...' -> 'global_config/...'
    if clean.startswith("config/"):
        candidates.append("global_config/" + clean[len("config/"):])

    return candidates


def find_item_in_h5(h5_file: h5py.File, candidate_paths: list[str]) -> tuple[str, Any, str | None]:
    """Search for the requested item or attribute in the HDF5 file."""
    # 1. Exact path check
    for p in candidate_paths:
        if p in h5_file:
            return p, h5_file[p], None

    # 2. Check if it refers to a group attribute: e.g. "global_config/dt" or "global_config.attrs['dt']"
    for p in candidate_paths:
        if "/" in p:
            grp_path, attr_name = p.rsplit("/", 1)
            if grp_path in h5_file and hasattr(h5_file[grp_path], "attrs") and attr_name in h5_file[grp_path].attrs:
                return grp_path, h5_file[grp_path].attrs[attr_name], attr_name

    # 3. Fuzzy search in all keys
    all_keys: list[str] = []
    h5_file.visit(lambda k: all_keys.append(k))

    target = candidate_paths[0].lower()
    for k in all_keys:
        if k.lower() == target or k.lower().endswith("/" + target):
            return k, h5_file[k], None

    for k in all_keys:
        if target in k.lower():
            return k, h5_file[k], None

    raise KeyError(f"Could not find path '{candidate_paths[0]}' in HDF5 file.")


def display_item(name: str, item: Any, attr_name: str | None = None) -> None:
    """Pretty-print the resolved HDF5 item (Dataset, Group, or Attribute)."""
    print(f"\nTarget Path: {name}" + (f" (Attribute: {attr_name})" if attr_name else ""))
    print("=" * 60)

    if attr_name is not None:
        print(f"Attribute '{attr_name}': {item}")
        return

    if isinstance(item, h5py.Dataset):
        data = item[()]
        if isinstance(data, np.ndarray):
            print(format_matrix(data))
            if data.ndim == 2 and data.shape[0] == data.shape[1] and np.issubdtype(data.dtype, np.number):
                try:
                    eigs = np.linalg.eigvals(data)
                    sym_eigs = np.linalg.eigvalsh(0.5 * (data + data.T))
                    print("\nMatrix Properties:")
                    print(f"  Symmetric Eigenvalues: {sym_eigs}")
                    cond = sym_eigs[-1] / sym_eigs[0] if sym_eigs[0] > 0 else float("inf")
                    print(f"  Condition Number:      {cond:.4f}")
                except Exception:
                    pass
        else:
            print(f"Value: {data}")

    elif isinstance(item, h5py.Group):
        print(f"Group: '{name}' containing {len(item.keys())} child items and {len(item.attrs)} attributes:")
        if item.attrs:
            print("\n  Attributes:")
            for k, v in item.attrs.items():
                print(f"    {k}: {v}")

        if item.keys():
            print("\n  Datasets / Subgroups:")
            for k in item.keys():
                child = item[k]
                if isinstance(child, h5py.Dataset):
                    val = child[()]
                    if isinstance(val, np.ndarray) and val.size <= 8:
                        val_str = f" = {val.tolist()}"
                    else:
                        val_str = ""
                    print(f"    {k:<15} [Dataset: shape={child.shape}, dtype={child.dtype}]{val_str}")
                elif isinstance(child, h5py.Group):
                    print(f"    {k:<15} [Group: {len(child)} items]")

    print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect and read datasets, groups, or attributes from an HDF5 data file.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Path or key inside the HDF5 file (e.g. 'config/P', 'global_config/cost/W_e', 'config/constraints').\n"
             "If omitted, displays the tree structure.",
    )
    parser.add_argument(
        "-f", "--file",
        dest="file_path",
        type=str,
        default=None,
        help="Path to .hdf5 file. Defaults to latest dataset in results/.",
    )
    parser.add_argument(
        "-t", "--tree",
        action="store_true",
        help="Display the full group/dataset tree structure of the file.",
    )
    parser.add_argument(
        "-d", "--depth",
        type=int,
        default=4,
        help="Maximum depth to display when printing the tree (default: 4).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Locate HDF5 file
    if args.file_path:
        file_path = Path(args.file_path).resolve()
        if not file_path.is_file():
            print(f"Error: File '{file_path}' does not exist.", file=sys.stderr)
            sys.exit(1)
    else:
        try:
            file_path = discover_latest_hdf5_file()
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    print(f"HDF5 File: {file_path}")

    with h5py.File(file_path, "r") as h5_file:
        # If no query is provided or tree flag is passed, show tree
        if args.query is None or args.tree:
            print("\nFile Structure Overview:")
            print("=" * 60)
            print_tree(h5_file, max_depth=args.depth)
            print("=" * 60)
            if args.query is None:
                print("\nTip: Provide a path argument to read specific data, e.g.:")
                print("  python examples/read_data.py config/P")
                print("  python examples/read_data.py global_config/cost/W_e")
                print("  python examples/read_data.py global_config/constraints")
                return

        # Resolve candidate paths
        candidate_paths = resolve_path_alias(args.query)

        try:
            resolved_name, item, attr_name = find_item_in_h5(h5_file, candidate_paths)
            display_item(resolved_name, item, attr_name)
        except KeyError as e:
            print(f"\nError: {e}", file=sys.stderr)
            print("Use 'python examples/read_data.py --tree' to see all available keys.", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
