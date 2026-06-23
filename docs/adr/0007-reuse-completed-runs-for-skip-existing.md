# ADR 0007: Reuse completed runs as pseudo run dirs for skip_existing

Date: 2026-06-22

Status: Proposed (deferred)

Recorded version: Compatibility version 7

## Context

`skip_existing` currently lives in `train.py`. At run time, `filter_pending_runs`
(`modules/train_results.py`) scans `runs/<experiment>`, matches each vmapped run's
params against completed run dirs, and drops matches from the vmap batch. This
already works with vmap, but it has three limitations:

1. **Search scope.** It only looks within the *same* experiment, so a parameter
   combination already trained under a different experiment name is not reused.
2. **No pseudo run dirs.** A skipped run simply vanishes. No run dir is created
   in the target experiment, so that parameter combination has no representation
   in the new run set, and downstream tooling (analysis, plotting) sees a gap.
3. **No compat filtering.** `find_completed_run_dir` ignores `compat_version`.
   Older runs carry `compat_version: null` and even renamed args (e.g.
   `activation_protects_memory`), so cross-experiment reuse must filter on the
   current `COMPAT_VERSION`.

The goal is: search **all** experiments at the current compatibility version, skip
training for parameter combinations already run, and represent reused runs as
"pseudo" run dirs that behave identically to freshly trained ones — most
importantly, carrying the saved weights (`net_jax.p`). Simulations can be reused
or regenerated.

## Decision

Not yet implemented; this ADR records the intended design so it can be picked up
later.

### Match scope

A completed run is reusable when its **runtime params** match the requested run.
Runtime params are env + training + network keys (`ENV_STATIC_PARAM_KEYS`,
`ENV_DYNAMIC_PARAM_KEYS`, `MODEL_SHAPE_PARAM_KEYS`, `TRAIN_SWEEP_KEYS`) — the
inputs that determine the trained weights. Meta fields (`label`, `experiment`,
`eval_episodes`, etc.) are ignored for matching. Defaults fill any key absent from
a candidate's `metadata.json` `args`.

A candidate is eligible only when:

- `metadata.json` records `compat_version == COMPAT_VERSION`, and
- `net_jax.p` is present (weights are the thing being reused).

`data_training_jax.p` is copied when present but is not required for eligibility;
it is not needed to re-simulate. (Note: some existing run dirs, e.g. under
`0622_both_forget`, have `net_jax.p` and `data_simulation.json` but no
`data_training_jax.p`. The reuse rule must not require training data.)

### Pseudo run dirs

For each reused combination, create a fresh run dir in the target experiment via
the normal `create_run_dir` / `run_prefix` path, then:

- copy `net_jax.p` (required) and `data_training_jax.p` (if present),
- write **fresh** metadata for the target experiment (new args, label, config
  path, varied keys, condition index), recording provenance such as
  `reused_from = <source run dir>`.

Do **not** copy simulation outputs. Pseudo run dirs are re-simulated by the
normal simulate job over the full target run set, so reused and freshly trained
runs are indistinguishable downstream.

### Where the logic lives

Shared helpers (proposed `modules/run_reuse.py`):

- `runtime_args(args)` — project a `metadata.json` `args` dict or a `run` dict
  onto the runtime param keys, filling defaults; the canonical match key.
- `index_completed_runs(results_root)` — scan all experiments under
  `results_root/runs/*`, keep eligible candidates, map runtime-arg key → newest
  matching run dir.
- `materialize_pseudo_run_dir(source_dir, run, ...)` — create the pseudo run dir
  and copy weights/training data as above.

Two integration options were considered; the choice is deferred:

**Option A — upgrade train.py's hook (smaller change).** Generators keep passing
`--skip-existing`. `train.py` filters the vmap batch as today, but uses the shared
index (all experiments, compat-filtered) and creates pseudo run dirs for skipped
combos instead of dropping them. The rectangular array/vmap grid is unchanged;
robust to races; reuse happens at run time on the cluster.

**Option B — resolve in the generators (matches "not train.py" literally).**
`generate_sbatch` / `generate_local` copy pseudo run dirs at generate time and
emit only the residual pending combinations. This removes reuse from `train.py`
entirely, but collapses the rectangular axes × vmap product into a flat per-combo
task list (partial completion makes the grid non-rectangular), changes array
sizing, and must handle an empty residual.

The stated preference is to keep this out of `train.py` (Option B), at the cost of
a more invasive change to grid enumeration in the generators.

## Consequences

- Cross-experiment reuse depends on `compat_version` being recorded correctly.
  Runs predating compat tagging (`null`) are never reused — acceptable, since
  their dynamics may differ.
- Matching on runtime params means two runs that differ only in meta (e.g.
  `label`) are treated as the same trained model. This is intended.
- Re-simulating reused runs costs simulation time but guarantees simulation
  outputs are consistent with the target experiment's settings.
- This is a tooling/workflow change only. It does not alter environment or
  network behavior and therefore does not require a `COMPAT_VERSION` bump.

## Non-goals

- Not reusing simulation outputs (always re-simulate pseudo run dirs).
- Not reusing runs across compatibility versions.
- Not matching on meta fields.
