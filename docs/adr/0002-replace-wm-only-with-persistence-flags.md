# ADR 0002: Replace `wm_only` with explicit persistence flags

Date: 2026-06-05

Status: Proposed

## Context

`wm_only` is a broad configuration flag that makes inactive nodes retain no node-specific information. That name is too coarse for experiment records. It hides which kinds of information persist outside working memory, which are cleared immediately, and which are merely hidden from the policy/value model.

The goal is not to search over every combination of these flags. The goal is to make each config file precisely document the cognitive architecture being fit or simulated. A mega-flag such as `wm_only` is poor for this because two runs can differ in several persistence assumptions while the config only records one boolean.

## Decision

Replace `wm_only` with independent flags for each dimension of information that can persist outside working memory. Keep the flags explicit even when most experiments use a standard preset.

The initial persistence dimensions are:

- `q_values_persist`: inactive nodes retain learned value estimates.
- `n_visits_persist`: inactive nodes retain visit-count memory.
- `recency_persist`: inactive nodes retain fixation-recency memory, subject to ordinary `recency_decay`.
- `terminal_persist`: inactive terminal-node flags remain known after leaving working memory. This replaces `persist_terminal`.
- `known_path_values_persist`: known path values remain observable after the corresponding node leaves working memory.

These are persistence flags, not sweep axes. They exist so the model definition is explicit.

## Current behavior mapping

Current `wm_only = false` with default `persist_terminal = false` is approximately:

```toml
q_values_persist = true
n_visits_persist = true
recency_persist = true
terminal_persist = false
known_path_values_persist = true
```

Current `wm_only = false` with `persist_terminal = true` is:

```toml
q_values_persist = true
n_visits_persist = true
recency_persist = true
terminal_persist = true
known_path_values_persist = true
```

Current `wm_only = true` is approximately:

```toml
q_values_persist = false
n_visits_persist = false
recency_persist = false
terminal_persist = false
known_path_values_persist = false
```

This mapping makes clear that `wm_only` bundles at least five persistence assumptions.

## Field semantics

`q_values_persist` controls whether learned value estimates survive when the corresponding node is inactive. When false, inactive `q_values` are cleared rather than merely protected from corruption.

`n_visits_persist` controls whether remembered visit counts survive when the corresponding node is inactive. This is separate from `q_values_persist` because visit-count memory is a different record of experience.

`recency_persist` controls whether fixation-recency memory survives when the corresponding node is inactive. This is separate from ordinary `recency_decay`: recency can decay continuously while still persisting outside working memory, or it can be cleared immediately on WM loss.

`terminal_persist` controls whether known terminal status survives outside working memory. This should replace the current one-off `persist_terminal` flag.

`known_path_values_persist` controls whether discovered path-value information remains available after the node leaves working memory. In current code this is the `known_mask` versus `active_mask` distinction for `g_values` and `best_open_value`.

Because `known_path_values_persist` is conceptually separate from `n_visits_persist`, the implementation should not continue to derive known path-value visibility only from `n_visits > 0` if visit-count memory can be cleared independently. The migration may need an explicit discovered/known-path mask so "the path value remains known" and "the visit count remains remembered" can vary independently.

## Relationship to activation touch points

These persistence flags complement ADR 0001 rather than replacing it.

`activation_protects_memory` controls whether activation protects stored memory from ongoing decay, drift, forgetting, and clearing processes. The persistence flags define which stored quantities are allowed to exist outside working memory at all.

`activation_masks_observation` controls whether inactive-node information is available to the policy/value model. Persistence and observation masking are different. A value may persist in internal memory but be hidden from the policy/value model when inactive, or it may be cleared from memory entirely.

`activation_gates_backup_sink` and `activation_gates_backup_source` remain separate. Persistence flags say whether inactive-node data exists; backup flags say whether that data can be used by the backup process.

Static tree structure and the currently fixated node's immediate context are not part of this persistence decomposition. Current observations always expose the current fixation, its point value, its parent, its children, and the root. If those structural/context fields should become memory-limited later, that should be a separate representation decision rather than part of replacing `wm_only`.

## Rationale

The separated flags keep experiment records readable. A config should say whether terminal memory persists, whether value estimates persist, and whether known path values persist. Readers should not have to remember the hidden bundle implied by `wm_only`.

The flags also make odd model variants explicit. For example, a model where `q_values_persist = false` but `activation_gates_backup_sink = false` is not impossible, but the config would expose the unusual assumption: inactive ancestors can be updated even though their inactive value memory is normally cleared.

## Non-goals

This ADR does not propose sweeping all combinations of persistence flags.

This ADR does not implement the migration.

This ADR does not decide the final naming of every flag. The important decision is to replace `wm_only` with explicit persistence dimensions, not another bundled mode.
