"""Inspect Cartpole checkpoints using state slices and local expansion rays.

Run from the repository root with ``python -m examples.cartpole.inspect_lyapunov
--model-path PATH``. The sibling policy, training configuration, MPC configuration
and training result supply defaults. All calculations use float64. This is
numerical falsification, not a stability certificate.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from matplotlib.axes import Axes
from numpy.typing import NDArray
from scipy.optimize import linprog
from torch import nn

from lcil.lyapunov_learning.config import LyapunovTrainingConfig
from lcil.utils import IntegrationMethod, load_mpc_config_json

from . import CartpoleDynamics, load_lyapunov_model, load_policy_model

_LABELS = ("Position x [m]", "Wagengeschwindigkeit v [m/s]", "Winkel θ [rad]",
           "Winkelgeschwindigkeit ω [rad/s]")


def evaluate_states(
    lyapunov: nn.Module,
    policy: nn.Module,
    dynamics: nn.Module,
    states: NDArray[np.float64],
    bounds: NDArray[np.float64],
    kappa: float,
    batch_size: int = 8192,
) -> dict[str, NDArray]:
    """Evaluate the unmodified discrete decrease and invariance conditions.

    Parameters
    ----------
    lyapunov, policy, dynamics : torch.nn.Module
        Models in evaluation mode with float64 parameters and buffers.
    states : ndarray of shape (N, nx)
        States to inspect, including states near the equilibrium.
    bounds : ndarray of shape (2, nx)
        Bounds used for the next-state invariance check.
    kappa : float
        Required discrete contraction factor: V(next) <= (1-kappa) V(x).
    batch_size : int, optional
        Maximum number of states evaluated at once.

    Returns
    -------
    dict
        Current/next values, raw decrease residuals, ratios and next states.
        Ratios are NaN wherever V(x) <= 0; the equilibrium is not a violation.
    """
    if batch_size < 1 or states.ndim != 2 or len(states) == 0:
        raise ValueError("Expected nonempty (N, nx) states and a positive batch size.")
    if not np.isfinite(states).all():
        raise ValueError("States must be finite.")
    reference = next(itertools.chain(lyapunov.parameters(), lyapunov.buffers()))
    chunks: dict[str, list[NDArray]] = {key: [] for key in ("v", "v_next", "next_states")}
    terms = {name: getattr(lyapunov, method, None) for name, method in
             (("feature", "get_feature_term"), ("linear", "get_linear_term"))}
    terms = {name: method for name, method in terms.items() if callable(method)}
    for name in terms:
        chunks[name] = []
        chunks[f"{name}_next"] = []
    with th.no_grad():
        for start in range(0, len(states), batch_size):
            x = th.as_tensor(states[start:start + batch_size], device=reference.device,
                             dtype=th.float64)
            x_next = dynamics(x, policy(x))
            chunks["v"].append(lyapunov(x).reshape(-1).cpu().numpy())
            chunks["v_next"].append(lyapunov(x_next).reshape(-1).cpu().numpy())
            chunks["next_states"].append(x_next.cpu().numpy())
            for name, method in terms.items():
                chunks[name].append(method(x).reshape(-1).cpu().numpy())
                chunks[f"{name}_next"].append(method(x_next).reshape(-1).cpu().numpy())
    values = {key: np.concatenate(parts) for key, parts in chunks.items()}
    if not all(np.isfinite(value).all() for value in values.values()):
        raise ValueError("Model evaluation produced nonfinite values.")
    values["decrease"] = values["v_next"] - (1.0 - kappa) * values["v"]
    values["ratio"] = np.divide(values["v_next"], values["v"],
                                out=np.full_like(values["v"], np.nan),
                                where=values["v"] > 0.0)
    values["invariant"] = ((values["next_states"] >= bounds[0]) &
                           (values["next_states"] <= bounds[1])).all(axis=1)
    values["inside_bounds"] = ((states >= bounds[0]) & (states <= bounds[1])).all(axis=1)
    return values


def polyhedral_expansion(
    closed_loop: NDArray[np.float64],
    norm_matrix: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    """Find the largest gain of a linear map in the norm ||B x||_1.

    Parameters
    ----------
    closed_loop : ndarray of shape (nx, nx)
        Jacobian of the discrete closed-loop map.
    norm_matrix : ndarray of shape (m, nx)
        Full-column-rank matrix B defining the local polyhedral norm.
        At most ten rows are supported to bound sign enumeration cost.

    Returns
    -------
    direction : ndarray of shape (nx,)
        A maximizing direction normalized in the Euclidean norm.
    gain : float
        ||B A direction||_1 / ||B direction||_1. This concerns the linear
        surrogate, not a verified neighborhood of the nonlinear models.
    """
    rows, dimensions = norm_matrix.shape
    if closed_loop.shape != (dimensions, dimensions):
        raise ValueError("The closed-loop matrix and norm dimensions must match.")
    if rows > 10 or np.linalg.matrix_rank(norm_matrix) != dimensions:
        raise ValueError("The local norm needs full column rank and at most ten rows.")
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=rows)))
    constraints = signs @ norm_matrix
    objectives = constraints @ closed_loop
    best_value = -np.inf
    best_direction = None
    for objective in objectives:
        result = linprog(-objective, A_ub=constraints, b_ub=np.ones(len(signs)),
                         bounds=[(None, None)] * dimensions, method="highs")
        if not result.success:
            raise RuntimeError(f"Local expansion LP failed: {result.message}")
        if -result.fun > best_value:
            best_value, best_direction = -result.fun, result.x
    assert best_direction is not None
    if np.linalg.norm(best_direction) == 0.0:
        best_direction = np.eye(dimensions)[0]
    best_direction = best_direction / np.linalg.norm(best_direction)
    gain = np.abs(norm_matrix @ closed_loop @ best_direction).sum()
    gain /= np.abs(norm_matrix @ best_direction).sum()
    return best_direction, float(gain)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-path", type=Path, required=True,
                        help="Lyapunov checkpoint (.pt), or its run directory.")
    parser.add_argument("--policy-path", type=Path, help="Default: sibling policy_model.pt.")
    parser.add_argument("--training-config", type=Path,
                        help="Default: sibling training_config.json.")
    parser.add_argument("--mpc-config", type=Path, help="Default: sibling mpc_config.json.")
    parser.add_argument("--output-dir", type=Path, help="Default: RUN/inspection.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dt", type=float, help="Override the MPC time step.")
    parser.add_argument("--kappa", type=float, help="Override the training decay factor.")
    parser.add_argument("--rho", type=float, help="Override rho_estimate from training_result.json.")
    parser.add_argument("--integration-method", choices=[method.value for method in IntegrationMethod],
                        default=IntegrationMethod.EXPLICIT_EULER.value)
    parser.add_argument("--grid-size", type=int, default=301)
    parser.add_argument("--slice-scale", type=float, default=0.25,
                        help="Scale training bounds around x* for both default slice windows.")
    parser.add_argument("--v-theta-window", type=float, nargs=4,
                        metavar=("V_MIN", "V_MAX", "THETA_MIN", "THETA_MAX"))
    parser.add_argument("--theta-omega-window", type=float, nargs=4,
                        metavar=("THETA_MIN", "THETA_MAX", "OMEGA_MIN", "OMEGA_MAX"))
    parser.add_argument("--radii", type=float, nargs=2, default=(1e-8, 1e-2),
                        metavar=("MIN", "MAX"))
    parser.add_argument("--radius-points", type=int, default=61)
    parser.add_argument("--samples", type=int, default=16384,
                        help="Uniform global samples plus the same number of local ray samples.")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _resolve_checkpoint(path: Path, filename: str) -> Path:
    path = path.expanduser().resolve()
    if path.is_dir():
        path = path / filename
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def _summary(states: NDArray, values: dict[str, NDArray], rho: float | None,
             origin: NDArray, exclusion: NDArray) -> dict:
    relevant = values["inside_bounds"] & (True if rho is None else values["v"] <= rho)
    nonzero = np.any(states != origin, axis=1)
    violating = relevant & nonzero & (values["decrease"] > 0.0)
    inside_hole = (np.abs(states - origin) <= exclusion).all(axis=1)
    eligible = np.flatnonzero(relevant & nonzero & np.isfinite(values["ratio"]))
    witness = None
    if len(eligible):
        index = eligible[np.argmax(values["ratio"][eligible])]
        witness = {"state": states[index].tolist(),
                   "next_state": values["next_states"][index].tolist(),
                   "v": float(values["v"][index]), "v_next": float(values["v_next"][index]),
                   "ratio": float(values["ratio"][index]),
                   "decrease_residual": float(values["decrease"][index]),
                   "inside_origin_exclusion": bool(inside_hole[index])}
        witness.update({name: float(values[name][index]) for name in
                        ("feature", "feature_next", "linear", "linear_next") if name in values})
    return {"sample_count": len(states), "inside_sublevel": int(relevant.sum()),
            "outside_bounds": int((~values["inside_bounds"]).sum()),
            "decrease_violations": int(violating.sum()),
            "decrease_violations_inside_origin_exclusion": int((violating & inside_hole).sum()),
            "invariance_violations": int((relevant & ~values["invariant"]).sum()),
            "nonpositive_off_equilibrium": int((nonzero & (values["v"] <= 0.0)).sum()),
            "worst_ratio_state": witness}


def _plot_slice(axis: Axes, dimensions: tuple[int, int], window: NDArray, origin: NDArray,
                evaluate: Callable[[NDArray], dict[str, NDArray]], rho: float | None,
                grid_size: int, kappa: float) -> dict:
    if not np.isfinite(window).all() or np.any(window[1] <= window[0]):
        raise ValueError("Slice windows must have finite, strictly increasing limits.")
    horizontal = np.linspace(window[0, 0], window[1, 0], grid_size)
    vertical = np.linspace(window[0, 1], window[1, 1], grid_size)
    # linspace can represent its zero midpoint as ~1e-17. Ratios there are
    # dominated by cancellation of phi(x)-phi(x*), so restore the intended x*.
    for coordinates, dimension in zip((horizontal, vertical), dimensions):
        tolerance = 4 * np.finfo(float).eps * max(1.0, np.abs(coordinates).max())
        coordinates[np.abs(coordinates - origin[dimension]) <= tolerance] = origin[dimension]
    xx, yy = np.meshgrid(horizontal, vertical)
    states = np.tile(origin, (xx.size, 1))
    states[:, dimensions[0]], states[:, dimensions[1]] = xx.ravel(), yy.ravel()
    values = evaluate(states)
    shape = xx.shape
    v = values["v"].reshape(shape)
    relevant = values["inside_bounds"] & (True if rho is None else values["v"] <= rho)
    violating = relevant & (values["decrease"] > 0.0)
    axis.set_facecolor("#edf4fa")
    axis.contourf(xx, yy, violating.reshape(shape).astype(float), levels=[0.5, 1.5],
                  colors=["#e96666"], alpha=0.95)
    axis.contourf(xx, yy, (~relevant).reshape(shape).astype(float), levels=[0.5, 1.5],
                  colors=["#aaaaaa"], alpha=0.18)
    if np.ptp(v) > 0.0:
        # Logarithmic levels expose the narrow low-V valleys near violations.
        vmax = v.max()
        lower = min(max(v[v > 0].min(), vmax * 1e-4), vmax * 0.5)
        contours = axis.contour(xx, yy, v, levels=np.geomspace(lower, vmax, 9),
                                colors="#39698c", linewidths=0.9)
        axis.clabel(contours, inline=True, fontsize=8, fmt="%.3g")
        if rho is not None and v.min() < rho < v.max():
            axis.contour(xx, yy, v, levels=[rho], colors="black", linestyles="--")
    candidates = np.flatnonzero(violating & np.isfinite(values["ratio"]))
    if len(candidates):
        index = candidates[np.argmax(values["ratio"][candidates])]
        axis.plot(states[index, dimensions[0]], states[index, dimensions[1]], "k+", ms=11)
    fixed = [index for index in range(4) if index not in dimensions]
    axis.set_title(", ".join(f"{('x', 'v', 'θ', 'ω')[index]} = {origin[index]:g}" for index in fixed))
    axis.set_xlabel(_LABELS[dimensions[0]])
    axis.set_ylabel(_LABELS[dimensions[1]])
    axis.text(0.02, 0.02, f"Rot: V(next) > {1 - kappa:g} V(x)\nKonturen: V(x)\n"
              "Grau: außerhalb des Prüfgebiets",
              transform=axis.transAxes, fontsize=8, va="bottom",
              bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    return {"states": states, "values": values, "window": window.tolist()}


def main(argv: list[str] | None = None) -> None:
    """Load a run, inspect its discrete conditions and save plots and raw diagnostics.

    Parameters
    ----------
    argv : list[str] or None, optional
        Command-line arguments; None reads the process arguments.
    """
    parser = _parser()
    args = parser.parse_args(argv)
    if (args.grid_size < 3 or args.radius_points < 2 or args.samples < 1
            or args.batch_size < 1 or not 0 < args.slice_scale <= 1
            or not np.isfinite(args.radii).all() or not 0 < args.radii[0] < args.radii[1]):
        parser.error("Invalid grid, sample count, slice scale, batch size or radius range.")
    model_path = _resolve_checkpoint(args.model_path, "lyapunov_model.pt")
    run_dir = model_path.parent
    policy_path = _resolve_checkpoint(args.policy_path or run_dir, "policy_model.pt")
    training_path = (args.training_config or run_dir / "training_config.json").expanduser().resolve()
    training = LyapunovTrainingConfig.load(training_path)
    kappa = training.kappa if args.kappa is None else args.kappa
    mpc_path = (args.mpc_config or run_dir / "mpc_config.json").expanduser().resolve()
    dt = load_mpc_config_json(mpc_path).dt if args.dt is None else args.dt
    result_path = run_dir / "training_result.json"
    rho = args.rho
    if rho is None and result_path.is_file():
        rho = json.loads(result_path.read_text(encoding="utf-8")).get("rho_estimate")
    if (not np.isfinite(dt) or dt <= 0 or not np.isfinite(kappa) or not 0 <= kappa < 1
            or (rho is not None and (not np.isfinite(rho) or rho <= 0))):
        parser.error("dt/rho must be positive and finite; kappa must be in [0, 1).")
    device = th.device(args.device)
    lyapunov = load_lyapunov_model(model_path, device).to(dtype=th.float64).eval()
    policy = load_policy_model(policy_path, device).to(dtype=th.float64).eval()
    dynamics = CartpoleDynamics(dt=dt, method=IntegrationMethod(args.integration_method)).to(device)
    for model in (lyapunov, policy, dynamics):
        model.requires_grad_(False)
    origin_tensor = lyapunov.x_star.reshape(-1)
    origin = origin_tensor.cpu().numpy()
    if origin.shape != (4,):
        raise ValueError("This inspector expects Cartpole states [x, v, theta, omega].")
    bounds = np.asarray(training.train_bounds if training.train_bounds is not None
                        else training.state_bounds, dtype=np.float64)
    exclusion = np.broadcast_to(np.asarray(training.origin_exclusion, dtype=float), (4,))

    def evaluate(states: NDArray) -> dict[str, NDArray]:
        return evaluate_states(lyapunov, policy, dynamics, states, bounds, kappa, args.batch_size)

    def closed_loop(x: th.Tensor) -> th.Tensor:
        return dynamics(x.unsqueeze(0), policy(x.unsqueeze(0))).squeeze(0)

    jacobian = th.autograd.functional.jacobian(closed_loop, origin_tensor).cpu().numpy()
    feature_jacobian = th.autograd.functional.jacobian(
        lambda x: lyapunov.feature_net(x.unsqueeze(0)).reshape(-1), origin_tensor).cpu().numpy()
    norm_matrix = np.vstack((lyapunov._pd_matrix().cpu().numpy(), feature_jacobian))
    direction, gain = polyhedral_expansion(jacobian, norm_matrix)
    radii = np.geomspace(*args.radii, args.radius_points)
    ray_states = origin + radii[:, None] * direction
    ray_values = evaluate(ray_states)
    rng = np.random.default_rng(args.seed)
    global_states = rng.uniform(bounds[0], bounds[1], size=(args.samples, 4))
    directions = rng.normal(size=(args.samples, 4))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    local_radii = np.exp(rng.uniform(np.log(args.radii[0]), np.log(args.radii[1]), args.samples))
    local_states = origin + directions * local_radii[:, None]

    figure, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    slices = []
    for axis, dimensions, custom in zip(axes[:2], ((1, 2), (2, 3)),
                                        (args.v_theta_window, args.theta_omega_window)):
        window = origin[list(dimensions)] + args.slice_scale * (
            bounds[:, dimensions] - origin[list(dimensions)])
        if custom is not None:
            window = np.asarray(custom, dtype=float).reshape(2, 2).T
        slices.append(_plot_slice(axis, dimensions, window, origin, evaluate, rho,
                                  args.grid_size, kappa))
    axes[2].semilogx(radii, ray_values["ratio"], label="V(F(x)) / V(x)")
    axes[2].axhline(1.0 - kappa, color="black", linestyle="--", label=f"Gefordert: ≤ {1-kappa:g}")
    axes[2].axhline(gain, color="grey", linestyle=":", label=f"Linearisiert: {gain:.5g}")
    axes[2].set_title("Lokale Expansionsrichtung")
    axes[2].set_xlabel("Abstand zum Gleichgewicht ‖x − x*‖₂")
    axes[2].set_ylabel("V(next) / V(x)")
    axes[2].grid(alpha=0.2)
    axes[2].legend(fontsize=8)
    rho_label = f"{rho:.5g}" if rho is not None else "ungefiltert"
    figure.suptitle(f"{run_dir.parent.name}/{run_dir.name} · dt={dt:g} · κ={kappa:g} · "
                   f"ρ={rho_label} · Float64")

    eigenvalues = np.linalg.eigvals(jacobian)
    equilibrium_values = evaluate(origin[None, :])
    report = {
        "model_path": str(model_path), "policy_path": str(policy_path),
        "training_config": str(training_path), "mpc_config": str(mpc_path),
        "dt": dt, "integration_method": args.integration_method, "kappa": kappa,
        "rho": rho, "bounds": bounds.tolist(), "seed": args.seed, "dtype": "float64",
        "origin_exclusion": exclusion.tolist(),
        "note": "Numerical inspection includes the origin exclusion; it is not a certificate. "
                "The LP gain applies to the autograd linear surrogate. At activation kinks "
                "a single Jacobian does not represent every local branch.",
        "equilibrium": {"state": origin.tolist(), "v": float(equilibrium_values["v"][0]),
                        "next_state_error": float(np.linalg.norm(
                            equilibrium_values["next_states"][0] - origin))},
        "local": {"closed_loop_jacobian": jacobian.tolist(),
                  "eigenvalues": [[float(value.real), float(value.imag)] for value in eigenvalues],
                  "spectral_radius": float(np.abs(eigenvalues).max()),
                  "linear_surrogate_gain": gain, "direction": direction.tolist(),
                  "radii": radii.tolist(),
                  "ratios": [float(value) if np.isfinite(value) else None
                             for value in ray_values["ratio"]],
                  "summary": _summary(ray_states, ray_values, rho, origin, exclusion)},
        "global_samples": _summary(global_states, evaluate(global_states), rho, origin, exclusion),
        "local_samples": _summary(local_states, evaluate(local_states), rho, origin, exclusion),
        "slices": [{"dimensions": list(dimensions), "window": item["window"],
                    **_summary(item["states"], item["values"], rho, origin, exclusion)}
                   for dimensions, item in zip(((1, 2), (2, 3)), slices)],
    }
    output_dir = (args.output_dir or run_dir / "inspection").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "inspection.png", dpi=180)
    figure.savefig(output_dir / "inspection.pdf")
    plt.close(figure)
    # Store raw slices and the expansion ray so the figure can be reproduced.
    arrays = {"radii": radii, "direction": direction, "ray_states": ray_states,
              **{f"ray_{key}": value for key, value in ray_values.items()}}
    for index, item in enumerate(slices):
        arrays[f"slice_{index}_states"] = item["states"]
        arrays.update({f"slice_{index}_{key}": value for key, value in item["values"].items()})
    np.savez_compressed(output_dir / "inspection.npz", **arrays)
    (output_dir / "inspection.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                                encoding="utf-8")
    print(f"Model: {model_path}\nDiscrete map: {args.integration_method}, dt={dt:g}, kappa={kappa:g}")
    print(f"Local spectral radius: {report['local']['spectral_radius']:.6g}; "
          f"linear surrogate V gain: {gain:.6g} (required <= {1-kappa:g})")
    for label, item in (("expansion_ray", report["local"]["summary"]),
                        ("global_samples", report["global_samples"]),
                        ("local_samples", report["local_samples"])):
        print(f"{label}: {item['decrease_violations']} decrease violations / "
              f"{item['inside_sublevel']} sublevel samples; "
              f"{item['invariance_violations']} invariance violations")
    print(f"Saved inspection.png, inspection.pdf, inspection.json, inspection.npz in {output_dir}")


if __name__ == "__main__":
    main()
