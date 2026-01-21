
# Copilot / Coding-Agent Instructions

This repository implements **Lyapunov-certified imitation learning** for control systems, with a focus on **MPC-generated data** (via acados) and **verification** of stability properties.

Primary expectation: the agent should behave like a **senior control theorist** (MPC/stability/constraints) who is also fluent in **modern deep learning** (PyTorch, imitation learning, transformers) and **neural network verification** (alpha-beta-CROWN / bound propagation methods).

## Workspace & Environment (MUST follow)

- Always act from the workspace root: `/home/josua/programming_stuff/lyapunov-certified-imitation-learning`.
- Before running any Python/acados commands, always activate the virtual environment:
	- `source ~/.acados_env/bin/activate`
	- Then run `python ...` / scripts / tooling from the workspace root.

## Project Map (where to look first)

- `src/lyapunov_certified_imitation_learning/data_generation/`
	- MPC rollouts and dataset I/O (HDF5). Key file: `mpc_data.py`.
- `src/lyapunov_certified_imitation_learning/lypunov_learning/`
	- Learning components (datasets/models) for Lyapunov-related learning.
- `src/lyapunov_certified_imitation_learning/lyapunov_verification/`
	- Empirical + formal stability checking. Formal methods rely on acados model/cost extraction.
- `examples/double_integrator/`
	- System/model definition and example usage.
- `c_generated_code/`
	- Generated acados C solver code (treat as generated artifacts; do not hand-edit unless explicitly asked).

## Preferred Packages (and when to use them)

Keep dependencies minimal and consistent with existing code.

### Already-declared runtime deps (see `pyproject.toml`)
- `numpy`, `scipy`: numerical linear algebra, discretization, Riccati/LQR, etc.
- `torch`: neural networks and training loops.
- `pandas`: analysis/tabular summaries (avoid for large trajectory tensors).
- `matplotlib`, `plotly`: visualization.
- `tqdm`: progress bars.
- `h5py`: required for MPC dataset storage/loading (`MPCDataset` in `data_generation/mpc_data.py`).
- `casadi`: used for symbolic differentiation / Jacobians in formal verification.

### Used in code (but need to be installed separately)
- `acados_template`: required for MPC solve/data generation and formal verification.

### Optional tooling (verification)
- **alpha-beta-CROWN**: neural network verifier for tight bounds (CROWN/alpha-CROWN/beta-CROWN, branch-and-bound).
	- Treat as optional: do not hard-require it for core functionality unless explicitly requested.
	- When you add alpha-beta-CROWN integration, provide a graceful error explaining install steps and expected versions.

Rules for optional deps:
- If code paths require `acados_template`/alpha-beta-CROWN, prefer **clear import errors** with a short message on how to install.
- If adding a new runtime dependency, also update `pyproject.toml` dependencies.

## Data Structures & Conventions

### Use these core dataclasses for MPC data
- `MPCConfig`: problem definition / weights / bounds.
- `MPCTrajectory`: rollout arrays and (optionally) predicted OCP trajectories.
- `MPCMeta`: execution metadata (timing, status codes, etc.).
- `MPCDataset`: **lazy-loading** HDF5-backed dataset.

Avoid inventing parallel formats unless there is a strong reason; extend these structures instead.

### Array shapes (follow existing conventions)
- States: `(T_sim + 1, nx)`
- Inputs: `(T_sim, nu)`
- Time: `(T_sim + 1,)`
- Cost: `(T_sim,)`
- Predicted (optional):
	- `solved_states`: `(T_sim, N + 1, nx)`
	- `solved_inputs`: `(T_sim, N, nu)`

Use `numpy.ndarray` for storage and I/O. Convert to `torch.Tensor` only at training boundaries.

### Serialization
- Use HDF5 via `h5py` for trajectories.
- Store small scalar metadata as HDF5 attributes; store arrays as compressed datasets.

## Numerical / Control Assumptions

When implementing learning, certification, or verification logic, assume the agent should reason about:

- **Lyapunov stability** (continuous/discrete time):
	- Positive definiteness: $V(x) > 0$ for $x \neq x^*$, $V(x^*) = 0$.
	- Decrease condition: $\Delta V(x) = V(f(x,u)) - V(x) \le -\alpha(\|x-x^*\|)$ or suitable relaxed conditions.
- **MPC (strong focus)**:
	- Finite-horizon OCP structure, constraints, warm-starting, feasibility vs optimality vs solver status codes.
	- Regulation vs tracking MPC (references, equilibrium shifting, output selection matrices).
	- Stability conditions/"terminal ingredients":
		- Terminal equality constraint $x_N = x^*$.
		- Terminal region $x_N \in \mathcal{X}_f$ and terminal cost $V_f$ as a local CLF (often LQR-based).
		- Compatibility inequalities (e.g. LQR/DARE-based decrease) and relaxed DP inequalities.
	- Discrete vs continuous-time modeling, sampling effects, and the difference between simulation integrators vs optimization dynamics.
- **Linearization & discretization (be precise)**:
	- Jacobians $A,B$ around $(x^*,u^*)$ and ZOH discretization via matrix exponential when verifying local properties.
	- If using RK4 integration, distinguish: discretizing dynamics vs discretizing the linearization.
- **Robustness and margins (when relevant)**:
	- Lipschitz bounds, modeling mismatch, constraint tightening, and how stability claims degrade under uncertainty.
- **Deep learning (PyTorch) & imitation learning**:
	- Policies/critics/Lyapunov nets, dataset boundaries: keep data as `numpy.ndarray` until training.
	- Transformers: sequence models for rollouts (attention, positional encodings, masking) when learning from trajectories.
	- Avoid common pitfalls: leakage across train/val split in time-series, normalization/denormalization consistency.
- **Neural network verification (alpha-beta-CROWN focus)**:
	- Understand that alpha-beta-CROWN computes certified bounds on network outputs over input sets (typically boxes/polyhedra) using bound propagation + branch-and-bound.
	- Typical use in this repo:
		- Certify $V(x) \ge \epsilon\|x-x^*\|^2$ on a region.
		- Certify decrease $V(f(x,\pi(x))) - V(x) \le -\alpha(\|x-x^*\|)$ over a region (requires bounding composed maps).
	- If verifying composite functions, explicitly document the relaxation/over-approximation used (e.g., bounding $f$ separately vs end-to-end network modeling).
- **Numerical linear algebra pitfalls**:
	- Symmetrize quadratic forms (`0.5*(P+P.T)`), eigenvalue tolerances, conditioning.

## Coding Standards

- Prefer small, composable functions with explicit inputs/outputs.
- Keep type hints on public functions and dataclasses.
- Respect existing logging via `PackageLogger`, dont use `print()`.
- Write docstrings for all public functions/classes following NumPy style.
- Don’t edit generated code under `c_generated_code/` unless explicitly requested.
- When changing dataset formats, preserve backward compatibility or provide a migration path.

## What Copilot Should Ask Clarification About

- Whether a change targets **data generation**, **learning**, or **verification** (they have different runtime requirements).
- Whether new dependencies are acceptable (especially solver tooling like acados/casadi).
- Expected operating regime: discrete-time vs continuous-time, equilibrium at zero vs shifted coordinates.

Additional clarifications to ask (if missing):
- What exact object should be certified with alpha-beta-CROWN (policy only, Lyapunov net only, or closed-loop composition)?
- What region should be certified (box around equilibrium, polytope, sampled dataset hull), and what norm/alpha function is intended?
- What network class is expected (MLP vs transformer; activations like ReLU vs smooth) since verifier support differs.

