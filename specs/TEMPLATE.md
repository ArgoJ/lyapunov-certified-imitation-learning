# Spec Template (Control Expert → Worker/Tests/Docu)

**Topic:** <short name>

**Owner:** Control Expert Agent

**Date:** <YYYY-MM-DD>

**Status:** Draft | Ready for implementation | Implemented | Verified

## 0) One-paragraph summary
Describe what will be built/changed and why (2–4 sentences). Include the user-facing outcome.

## 1) Scope
### In scope
- <bullet>

### Out of scope
- <bullet>

## 2) Assumptions & conventions
- **Time model:** Discrete-time / Continuous-time (and discretization method if continuous)
- **Equilibrium:** $x^*$, $u^*$ (state whether shifted coordinates are used)
- **State/input shapes:** $x\in\mathbb{R}^{n_x}$, $u\in\mathbb{R}^{n_u}$
- **Array shapes (repo convention):**
  - states: `(T_sim + 1, nx)`
  - inputs: `(T_sim, nu)`
  - time: `(T_sim + 1,)`
  - cost: `(T_sim,)`
  - predicted states: `(T_sim, N + 1, nx)` (optional)
  - predicted inputs: `(T_sim, N, nu)` (optional)
- **Numerical tolerances:** e.g. `atol=1e-8`, `rtol=1e-6` (state explicitly)

## 3) Mathematical specification
### 3.1 Dynamics
Define dynamics and all symbols:

- Discrete time:
  $$x_{k+1} = f(x_k, u_k)$$

- Continuous time (if applicable):
  $$\dot x = f(x,u),\quad x_{k+1}=\Phi(x_k,u_k)$$
  Specify discretization/integration and whether optimization dynamics match simulation dynamics.

### 3.2 MPC problem (if applicable)
- Horizon: $N$
- Stage cost: $\ell(x,u)$
- Terminal cost: $V_f(x)$
- Constraints: $x\in\mathcal{X}$, $u\in\mathcal{U}$, terminal set $x_N\in\mathcal{X}_f$ (if used)
- Notes on feasibility/optimality and expected solver status codes.

### 3.3 Policy / model components
- Policy: $u=\pi_\theta(x)$ (architecture class, activation, input normalization)
- Lyapunov candidate: $V_\phi(x)$ (architecture/parametrization)

### 3.4 Certification / verification targets
Specify exactly what is being certified and where:

- **Region:** $\mathcal{R}$ (box/polytope). For a box:
  $$\mathcal{R} = \{x: \ell \le x-x^* \le u\}$$

- **Positive definiteness:**
  $$V(x^*)=0,\quad V(x)\ge \alpha\|x-x^*\|^2\ \forall x\in\mathcal{R}$$

- **Decrease condition (closed-loop):**
  $$\Delta V(x) = V\big(f(x,\pi(x))\big) - V(x) \le -\beta\|x-x^*\|^2\ \forall x\in\mathcal{R}$$
  If relaxed (e.g., slack, margin, empirical-only), define it precisely.

- If using linearization/discretization locally, define $A,B$ and how they are computed.

- If using neural verification (optional):
  - Is it end-to-end on the composed map or decomposed bounds?
  - What bound method is assumed (e.g., alpha-beta-CROWN) and expected limitations.

## 4) Dataflow & interfaces (implementation contract)
### 4.1 Reuse/extend existing dataclasses
Explicitly state whether to use/extend:
- `MPCConfig`
- `MPCTrajectory`
- `MPCMeta`
- `MPCDataset`

### 4.2 New/changed public APIs
List required code artifacts with precise signatures.

Example:
- Module: `src/lyapunov_certified_imitation_learning/lyapunov_verification/verification.py`
  - `def check_lyapunov_decrease_on_box(...)-> VerificationResult:`

For each API specify:
- Inputs (types, shapes)
- Outputs (types, shapes)
- Error behavior (what exceptions/messages if optional deps missing)

### 4.3 Serialization/I/O (if applicable)
- HDF5 dataset names, attributes, compression.
- Backward compatibility plan if format changes.

## 5) Algorithm sketch (step-by-step)
Provide a numbered algorithm with explicit intermediate quantities.

Example:
1. Sample points $x^{(i)}$ from $\mathcal{R}$.
2. Compute $u^{(i)}=\pi(x^{(i)})$.
3. Propagate $x_+^{(i)}=f(x^{(i)},u^{(i)})$.
4. Compute $\Delta V^{(i)} = V(x_+^{(i)})-V(x^{(i)})$.
5. Check $\Delta V^{(i)} \le -\beta\|x^{(i)}-x^*\|^2 + \epsilon$.

## 6) Acceptance criteria (what tests must verify)
Write deterministic, checkable criteria.

- **Shapes & types:** <criteria>
- **Numerical properties:** <criteria + tolerances>
- **Error handling:** <criteria>
- **Reproducibility:** seeded RNG behavior.

Include explicit tolerances and seeds.

## 7) Testing plan (for Tests Worker)
- Unit tests to add (file names under `tests/`)
- Fixtures (toy linear system recommended)
- Skip conditions if optional deps not installed

## 8) Documentation requirements (for Docu Agent)
For each new/changed public API, list formulas to include in docstrings and any key assumptions to mention.

## 9) Risks & open questions
- <risk>
- <open decision + recommended default>
