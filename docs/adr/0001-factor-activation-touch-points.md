# ADR 0001: Factor activation touch points into explicit parameters

Date: 2026-06-05

Status: Proposed

## Context

Working-memory activation currently controls several conceptually distinct mechanisms in the decision-tree cognitive architecture. Some of these mechanisms are bundled behind broad modes such as `backup_mode` and `wm_only`. This makes comparison models difficult to interpret because a named condition can change more than one activation touch point at a time.

The recent `unbounded_masking` comparison exposed this problem. In the environment, the intended manipulation was close to "activation only changes action availability." With the `node_shared` architecture, however, the action mask is also a network input. The condition therefore changed both legal actions and the policy/value representation.

We want future ablations to be defined by explicit, mostly binary switches over the places where activation matters.

## Decision

Replace bundled activation modes with explicit activation touch-point parameters. The initial target set is:

- `activation_masks_actions`: activation determines which fixation actions are legal.
- `activation_gates_backup_sink`: activation determines whether an ancestor can receive a backup update.
- `activation_gates_backup_source`: activation determines whether child values are available when computing backup targets.
- `activation_protects_memory`: activation protects node-specific memory from corruption, forgetting, and terminal-flag clearing.
- `activation_masks_network_input`: activation-derived legality is exposed to the `node_shared` network as a feature and pooling mask.

Use `excluded_child_value` to refine backup-source gating. The default is `None`; TOML configs should represent this default by omitting the field unless the implementation later introduces an explicit string sentinel.

- `excluded_child_value = None` means inactive children are excluded from the backup target policy support. This matches the current `wm_partial` source behavior.
- `excluded_child_value = 0.0` means inactive children remain in the backup target calculation with value zero. This matches the current `wm_zero` source behavior.
- Other numeric values are allowed only if a future experiment intentionally gives unavailable child values a different default.
- `excluded_child_value` is ignored when `activation_gates_backup_source = false`.

This preserves `activation_gates_backup_source` as a binary touch-point flag while still representing the non-binary distinction between exclusion and zero filling.

## Current-mode mapping

The existing `backup_mode` values can be represented as:

| current mode | `activation_gates_backup_sink` | `activation_gates_backup_source` | `excluded_child_value` |
| --- | --- | --- | --- |
| `full` | `false` | `false` | ignored |
| `wm_both` | `true` | `false` | ignored |
| `wm_partial` | `true` | `true` | `None` |
| `wm_zero` | `true` | `true` | `0.0` |

This makes the confusing part of `wm_both` explicit: it gates backup sinks but does not gate backup sources.

## Backup-source semantics

When `activation_gates_backup_source = false`, all child values are used in the backup target, regardless of activation.

When `activation_gates_backup_source = true` and `excluded_child_value = None`, inactive children are removed from both the softmax support and the epsilon-random support. Probabilities are renormalized over active children.

When `activation_gates_backup_source = true` and `excluded_child_value` is numeric, inactive children remain in the child set with that substituted value. The target policy is then computed over the full child set using the substituted values.

## Relationship to `wm_only`

Do not preserve `wm_only` as a separate conceptual mode long term. It bundles several effects:

- inactive nodes retain no `q_values`, `n_visits`, `fixation_recency`, or terminal memory;
- inactive known path values are hidden from observations;
- even `full` backup cannot bypass active-sink gating because inactive node information is not available.

For the first migration, assume `wm_only = false` and factor the five touch points above. If `wm_only` is needed again, replace it with explicit representation-level parameters rather than reintroducing a broad mode. Candidate parameters include:

- whether inactive nodes retain node-specific memory;
- whether inactive known path values are observable;
- whether backup can use stored values for inactive nodes when memory retention is disabled.

## Relationship to network architecture

`activation_masks_actions` and `activation_masks_network_input` must remain separate. For `node_shared`, the activation-derived action mask is consumed as `legal_feature`, `legal_mean`, and `legal_max`, so it changes the policy/value input representation. For a clean "action availability only" ablation, legal-action masking should be separable from the mask passed into the network.

The MLP architecture is currently disabled until information masking is addressed, because it receives flattened observations for all emitted nodes and ignores the action mask as a representation input.

## Consequences

Conditions can be described by which activation touch points are enabled rather than by opaque model names.

The main model becomes interpretable as a vector of enabled mechanisms, not a special case. For example, the current main model is approximately:

```toml
activation_masks_actions = true
activation_gates_backup_sink = true
activation_gates_backup_source = true
activation_protects_memory = true
activation_masks_network_input = true
# excluded_child_value omitted: default None
```

An unbounded model is approximately:

```toml
activation_masks_actions = false
activation_gates_backup_sink = false
activation_gates_backup_source = false
activation_protects_memory = false
activation_masks_network_input = false
```

An "actions only" ablation is explicit:

```toml
activation_masks_actions = true
activation_gates_backup_sink = false
activation_gates_backup_source = false
activation_protects_memory = false
activation_masks_network_input = false
```

## Non-goals

This ADR does not implement the migration.

This ADR does not decide whether activation should be exposed as a direct observation feature. The current issue is that the action mask is implicitly reused as a network representation mask.

This ADR does not introduce edge-binding activation or other non-node working-memory objects.
