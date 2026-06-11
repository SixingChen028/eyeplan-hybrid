# ADR 0002: Replace `wm_only` with explicit persistence flags

Date: 2026-06-05

Status: Superseded by `disable_persistence`

Recorded version: Compatibility version 1

## Context

`wm_only` was a broad configuration flag that made inactive nodes retain no node-specific information. That name was too coarse for experiment records because it hid which kinds of information persist outside working memory, which are cleared immediately, and which are merely hidden from the policy/value model.

The goal is not to search over every combination of these flags. The goal is to make each config file precisely document the cognitive architecture being fit or simulated. A mega-flag such as `wm_only` is poor for this because two runs can differ in several persistence assumptions while the config only records one boolean.

## Decision

The proposed decision was to replace `wm_only` with independent persistence flags for learned value estimates, visit counts, recency, terminal memory, and known path values. That proposal was not implemented.

## Current implementation

The code first preserved the old behavior in `9d93d0d`, then renamed `wm_only` to `disable_persistence` in `b13b075` and `ca0c535`.

Current configs use:

```toml
disable_persistence = false
```

or:

```toml
disable_persistence = true
```

When `disable_persistence = true`, inactive nodes clear `q_values`, `n_visits`, `fixation_recency`, and terminal memory. The environment also requires the default activation behavior for backup sink/source gating, memory protection, and observation masking.

The proposed independent persistence fields are not recognized config keys.

## Relationship to activation touch points

The proposed persistence flags would have complemented ADR 0001 rather than replacing it.

In the current implementation, `disable_corruption` controls whether stored memory skips ongoing decay, drift, and stochastic forgetting outside WM. `disable_persistence` defines whether inactive node-specific memory can exist outside WM at all.

`activation_masks_observation` controls whether inactive-node information is available to the policy/value model. Persistence and observation masking are different. A value may persist in internal memory but be hidden from the policy/value model when inactive, or it may be cleared from memory entirely.

`activation_gates_backup_sink` and `activation_gates_backup_source` remain separate. Persistence says whether inactive-node data exists; backup flags say whether that data can be used by the backup process.

## Non-goals

This ADR does not propose sweeping all combinations of persistence flags.

This ADR is retained as design history only. It does not describe currently accepted config fields beyond the implemented `disable_persistence` replacement.
