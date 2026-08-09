# Global Rules

Think before acting.

If requirements are ambiguous:
- State assumptions.
- Ask if ambiguity affects correctness.

Prefer:
- Existing code
- Existing patterns
- Minimal changes
- Existing libraries

Do:
- Read existing files before writing.
- Thorough in reasoning, concise in output.
- Skip files over 100KB unless required.

Do not:
- Overengineer
- Refactor unrelated code
- Rewrite entire files unnecessarily
- Re-read unless changed
- Use sycophantic openers or closing fluff, emojis, or em-dashes.
- Guess APIs, versions, flags, commit SHAs, or package names. Verify by reading code or docs before asserting.

Verify outcomes before declaring completion.

Be concise unless detail is requested.

## What this repo is

A BB84 polarization-encoded QKD photonic integrated circuit, designed in PhotonForge with Tidy3D
as the FDTD backend, on the SiEPIC EBeam PDK. Alice (transmitter) and Bob (receiver) top-level
circuits live in `main/`; component and variant experiments live in `tests/`.

Two constraints shape all current work:

1. **Tidy3D is metered.** A single S-matrix costs several cloud credits, so parameter sweeps are
   unaffordable. Migrating the solver to [fdtdx](https://github.com/ymahlau/fdtdx) (JAX, local GPU,
   autodiff) is the enabling task, not a nice-to-have.
2. **The hybrid plasmonic polarization rotator does not reach 90 degrees.** It gates both Alice and
   Bob, and it is the pilot device for the solver migration.

Work is split into two decoupled tracks: **Track A** reproduces and benchmarks the Bunandar
metropolitan-QKD architecture (PRX 8, 021009); **Track B** improves the TE to TM converter.

## Plan & state (PKOS / Second Brain)

This repo's plan, tasks, and knowledge live in the sibling vault `../Second Brain` (a Personal
Knowledge OS). Read from there; do not duplicate plan state in this repo.

- Board (tasks / subtasks / milestones): `../Second Brain/Projects/QKD PIC_tasks/` (Obsidian
  "Project Manager" plugin notes, Markdown + frontmatter). projectId `vzpq3uatmslkhdps`.
- Working memory / current state: read `../Second Brain/CURRENT.md` first, then the manifest
  `../Second Brain/100 indexes/llms.txt` (resolve ids via `100 indexes/INDEX.json`). `MAP.md` is
  human navigation, not a bulk-load target.
- Source-of-truth notes for this project (immutable, never edit):
  `../Second Brain/200 raw/Projects/1000 Hardware/1400 QKD PIC BB84/` — `Journal/! GOAL.md`,
  `Bottleneck.md`, `Flexcompute Guide.md`, `Notebook References.md`, `Journal/2026-08-0{7,8}.md`.
  These are in Indonesian; the board carries the English translation.
- Read the plan from this repo:
  - `python "../Second Brain/script/flow/pm.py" --project qkd-pic --list`
  - `python "../Second Brain/script/flow/pm.py" --sync`  (what changed since the AI last looked)
- Work loop: pick a `Claude`-assigned `todo`, set it `in-progress`, do the work here (code), then
  set the task to `review` on the board for James. Move a task with
  `python "../Second Brain/script/flow/pm.py" --set-status <task id> --to review`; it refuses `done`
  and `cancelled`, because only James graduates a task. If a task produces a knowledge draft under
  `../Second Brain/300 digest/`, add an inline `deliverable:: [[<id>]]` line to the task body so the
  vault's `make gate` promotes it into `400 wiki` when James accepts.
- Status vocab: `todo`, `in-progress`, `blocked`, `review`, `done`, `cancelled`. Assignees: `Claude`
  (AI worker), `James` (human reviewer).

## Repository layout

- `main/` — top-level circuits: `alice_top_MZI_*`, `bob_top_v*`, plus component notebooks
  (`hybrid_rotator`, `psr`, `miniPSR`, `tunable_mzi`).
- `tests/` — variant and component experiments. Despite the name these are **notebooks, not a test
  suite**; `bob_top_v3_psr_mmi.ipynb` is the most current Bob and the reference for Phase 0.

## Known facts worth not rediscovering

- **Models are pluggable, and that is the migration seam.** `ThermalModel(pf.Model)` in
  `tests/bob_top_v3_psr_mmi.ipynb` subclasses `pf.Model`, implements
  `start(component, frequencies, **kwargs)` under `@pf.cache_s_matrix`, provides `as_bytes` /
  `from_bytes`, and registers with `pf.register_model_class`. An `FDTDXModel` at the same seam makes
  the solver swap a single assignment: `component.active_model = FDTDXModel(...)`.
- **`rotator_0` and `rotator_90` are not rotators.** In `create_bob_top` they are
  `pf.parametric.transition(port_spec1="TE_1550_500", port_spec2="TE-TM_1550_500", length=10)`, a
  port-spec adapter that does not rotate polarization. Only the plus/minus 45 degree states route
  through the real plasmonic `create_hybrid_rotator`. Do not read the four BB84 state plots as
  physically equivalent until this is resolved.
- **Mode ordering on `TE-TM_1550_500` (`num_modes = 2`) is unverified.** Modes are ordered by
  effective index, so a swap is indistinguishable from a broken rotator.
- **The `updates` mechanism is unverified.** Keys like
  `("ps", 1, "ps", 0): {"model_updates": {"voltage": 2}}` address nested netlist instances; a
  notebook comment records doubt that they land. Prove it with a sweep before trusting any result
  that depends on it.

## Development

The importable package `src/bb84qkd/` arrives through the plan loop (task 0.0.1), which extracts the
component factories currently copy-pasted across the notebooks. Until then the notebooks are the
source of truth.

Before opening a PR:

- Run `pytest` locally once the harness exists (task 0.0.3). CI runs it again on every push and PR.
- CI must be green before review.
- Any simulated result needs its mesh-convergence evidence alongside it. A cross-solver disagreement
  gets root-caused (dispersion model, mode conventions, port reference planes, PML), never tuned away.
