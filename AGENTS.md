
# Copilot / Coding-Agent Instructions

This repo is about **MPC-generated imitation learning** plus **Lyapunov-style verification**.
Act like a control + DL engineer: keep changes small, numerically careful, and consistent with existing data formats.

## Must Follow (Workspace / Env)

- Work from the repo root: `/home/josua/programming_stuff/projects/lyapunov-certified-imitation-learning`.
- Before running anything that touches Python/acados, activate:
  - `source ~/.ma_env/bin/activate`
  - then run `python3 ...` from the repo root.

## Where To Look First

- Data Generation: in mpc-datagen package (`/home/josua/programming_stuff/projects/mpc-datagen`)
- Imitation Learning: `src/lcil/imitation_learning`
- Lyapunov Learning: `src/lcil/lypunov_learning/`
- Certification: `src/lcil/certification/`
- Example system: `examples/double_integrator/`
- Generated solver artifacts (do not edit): `c_generated_code/`

All folders in the `lcil` package do have somewhat a similar structure with:
- `models.py`
- `config.py`
- `trainer.py`
- `rollout.py`
- etc.

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