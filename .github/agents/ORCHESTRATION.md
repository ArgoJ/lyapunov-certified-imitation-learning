# Multi-Agent Orchestration (Control Expert → Worker/Tester → Reviewer → Documeter)

This repo uses five specialized Copilot agents that can run **in parallel** with clear handoffs.

## Agents
- **Control Expert**: head/project manager; writes specs, acceptance criteria, and final change summary.
- **Worker**: implements production code.
- **Tester**: writes unit tests and approves behavior against acceptance criteria.
- **Reviewer**: writes a review report and flags issues/risks.
- **Documeter**: adds NumPy-style docstrings (with formulas) to changed public APIs.

## Core artifacts (files)
- Spec: `specs/<topic>.md` (start from `specs/TEMPLATE.md`)
- Review: `reviews/REVIEW_<YYYY-MM-DD>_<topic>.md`
- Optional tracking: `specs/<topic>_notes.md` (decisions, tolerances, open questions)

## Workflow (3 iterations)

### Phase 0 — Intake (Control Expert)
1. Create `specs/<topic>.md` from `specs/TEMPLATE.md`.
2. Define:
   - exact math (symbols, regions, norms, tolerances),
   - required public APIs (signatures + shapes),
   - acceptance criteria (deterministic checks).
3. Create/update a task list (use the `todo` tool) that includes:
   - Worker tasks,
   - Tester tasks,
   - what constitutes “approval”.

### Phase 1 — Implement + Test (Worker + Tester in parallel)
Run Worker and Tester simultaneously.

**Worker does**
- Implement only what’s in the spec (plus minimal glue).
- Keep changes minimal, typed, and dataclass-based when appropriate.
- Provide a short “implementation note” with:
  - new/changed symbols (modules, functions, classes),
  - any numerical tolerances used,
  - any intentional limitations.

**Tester does**
- Write unit tests under `tests/` that encode acceptance criteria.
- Prefer deterministic toy systems for control/verification logic.
- If optional deps are missing, add skips with clear reasons.

**Approval gate (end of Phase 1)**
- Worker: code builds/imports and meets spec.
- Tester: tests pass and cover acceptance criteria.
- If either fails, Worker/Tester iterate until both approve.

### Phase 2 — Review (Reviewer)
Control Expert hands:
- spec link `specs/<topic>.md`,
- list of changed files/symbols from Worker/Tester,
- any known issues/tolerances,

to the Reviewer.

Reviewer outputs `reviews/REVIEW_<YYYY-MM-DD>_<topic>.md` with:
- spec adherence,
- correctness & numerics,
- API & types,
- test quality,
- prioritized action items.

### Phase 3 — Manage Review (Control Expert)
Control Expert:
- triages review items,
- updates the `todo` list,
- resolves ambiguities (and updates the spec if needed),
- decides what changes must happen in the next round.

### Repeat
Repeat Phases 1–3 **three times** (Round 1 → Round 2 → Round 3), or until:
- acceptance criteria are satisfied,
- review has no blockers.

### Phase 4 — Documentation (Documeter)
After Round 3 (or when implementation is stable):
- Documeter adds NumPy-style docstrings with relevant formulas to all new/changed public APIs.
- Documeter must not change behavior.

### Phase 5 — Final summary (Control Expert)
Control Expert writes a final summary of:
- what changed (modules/APIs),
- why it changed (link to spec intent),
- how to run tests / reproduce results,
- any known limitations and next steps.

## Suggested prompt handoffs (copy/paste)

### Control Expert → Worker
"Implement `specs/<topic>.md` exactly. Keep APIs typed and minimal. Report changed files + new public symbols."

### Control Expert → Tester
"Write tests for acceptance criteria in `specs/<topic>.md`. Put them under `tests/`. Prefer deterministic toy systems and explicit tolerances."

### Control Expert → Reviewer
"Review changes against `specs/<topic>.md`. Write `reviews/REVIEW_<date>_<topic>.md` with prioritized action items."

### Control Expert → Documeter
"Add NumPy-style docstrings (with formulas) to all changed public APIs from this topic. No behavior changes."
