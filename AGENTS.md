
# Copilot / Coding-Agent Instructions

This repo is about **MPC-generated imitation learning** plus **Lyapunov-style verification**.
Act like a control + DL engineer: keep changes small, numerically careful, and consistent with existing data formats.

## Must Follow (Workspace / Env)

- Work from the repo root: `/home/josua/programming_stuff/projects/lyapunov-certified-imitation-learning`.
- Before running anything that touches Python/acados, activate:
  - `source ~/.acados_env/bin/activate`
  - then run `python3 ...` from the repo root.

## Where To Look First

- Data Generation: in mpc-datagen package (`/home/josua/programming_stuff/projects/mpc-datagen`)
- Learning (Lyapunov nets / datasets): `src/lyapunov_certified_imitation_learning/lypunov_learning/`
- Verification: `src/lyapunov_certified_imitation_learning/lyapunov_verification/`
- Example system: `examples/double_integrator/`
- Generated solver artifacts (do not edit): `c_generated_code/`

## Dependencies (Rules)

- Keep runtime deps minimal and aligned with `pyproject.toml`.
- If you add a new runtime dependency, also add it to `pyproject.toml`.

## MPC Data Conventions (Do Not Drift)

- Use existing dataclasses for MPC data:
  - `MPCConfig`, `MPCTrajectory`, `MPCMeta`, `MPCDataset` (HDF5 lazy loading)
- Array shapes:
  - `states`: `(T_sim + 1, nx)`
  - `inputs`: `(T_sim, nu)`
  - `time`: `(T_sim + 1,)`
  - `cost`: `(T_sim,)`
  - optional predictions:
    - `solved_states`: `(T_sim, N + 1, nx)`
    - `solved_inputs`: `(T_sim, N, nu)`
- Storage/I/O: use `numpy.ndarray` + `h5py`. Convert to `torch.Tensor` only at training boundaries.

## Coding Standards (Practical)

- Prefer small functions with explicit inputs/outputs; add type hints on public APIs.
- Use `PackageLogger` for logging (no `print()`).
- Add NumPy-style docstrings for public functions/classes.
- Don’t hand-edit anything under `c_generated_code/`.
- If you change an on-disk dataset format, preserve backward compatibility or provide a migration.

## Clarify Only If Needed

Ask these only when missing and blocking:

- Is this change for **data generation**, **learning**, or **verification**?
- Discrete-time vs continuous-time assumptions? Equilibrium at zero or shifted $(x^*, u^*)$?
- If doing formal certification: certify what (Lyapunov net, policy, or closed-loop) and over what region (box/polytope)?

