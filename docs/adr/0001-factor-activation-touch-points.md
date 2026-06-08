# ADR 0001: Factor activation touch points into explicit parameters

Date: 2026-06-05

Status: Partially implemented

Recorded version: Pipeline compatibility version 1

## Context

Working-memory activation used to control several conceptually distinct mechanisms in the decision-tree cognitive architecture through broad modes such as `backup_mode` and `wm_only`. That made comparison models difficult to interpret because a named condition could change more than one activation touch point at a time.

The recent `unbounded_masking` comparison exposed this problem. In the environment, the intended manipulation was close to "activation only changes action availability." With the `node_shared` architecture, however, the action mask is also a network input. The condition therefore changed both legal actions and the policy/value information available to the model.

We want future ablations to be defined by explicit, mostly binary switches over the places where activation matters.

## Decision

Replace bundled activation modes with explicit activation touch-point parameters:

- `activation_masks_actions`: activation determines which fixation actions are legal.
- `activation_gates_backup_sink`: activation determines whether an ancestor can receive a backup update.
- `activation_gates_backup_source`: activation determines whether child values are available when computing backup targets.
- `activation_protects_memory`: activation protects node-specific memory from corruption, forgetting, and terminal-flag clearing.
- `activation_masks_observation`: activation determines which node-specific information is available to the policy/value model.

`activation_masks_observation` is implemented differently by different network architectures, but represents one cognitive mechanism. For flat observations, inactive node fields should be masked in the observation itself. For `node_shared`, inactive nodes can be excluded from shared pooling and may receive an explicit mask feature; this is the architecture's way of ignoring unavailable information rather than consuming zero-filled placeholders.

Use `excluded_child_value` to refine backup-source gating. The default is `None`; TOML configs represent this default by omitting the field.

- `excluded_child_value = None` means inactive children are excluded from the backup target policy support. This matches the current `wm_partial` source behavior.
- `excluded_child_value = 0.0` means inactive children remain in the backup target calculation with value zero. This matches the current `wm_zero` source behavior.
- Other numeric values are allowed only if a future experiment intentionally gives unavailable child values a different default.
- `excluded_child_value` is ignored when `activation_gates_backup_source = false`.

This preserves `activation_gates_backup_source` as a binary touch-point flag while still representing the non-binary distinction between exclusion and zero filling.

## Current implementation

Implemented in `11d174c` with later cleanup in `876d1d5` and the `wm_only` rename in `b13b075`/`ca0c535`.

Current environment parameters include:

- `activation_masks_actions`
- `activation_gates_backup_sink`
- `activation_gates_backup_source`
- `activation_protects_memory`
- `activation_masks_observation`
- `excluded_child_value`
- `disable_persistence`

The implementation does not support every combination proposed here. In particular, `JaxDecisionTreeEnv` currently asserts that `activation_masks_actions` is true, so the original "actions only" and unbounded examples are not valid current configs. `disable_persistence` also remains a broad representation flag rather than the independent persistence flags proposed in ADR 0002.

## Legacy mode mapping

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

## Relationship to `disable_persistence`

`wm_only` was not kept under that name. The implemented successor is `disable_persistence`, which makes inactive nodes retain no node-specific information. This is still a broad flag and is not the fully factored representation proposed in ADR 0002.

## Relationship to network architecture

`activation_masks_actions` and `activation_masks_observation` must remain separate. For a clean "action availability only" ablation, legal-action masking should not also hide information from the policy/value model.

For `node_shared`, observation masking can be implemented by passing an observation mask to the network and using it for `legal_feature`, `legal_mean`, and `legal_max`. Under this interpretation, the shared-network mask is not a sixth touch point. It is the architecture-specific implementation of observation availability.

The MLP architecture receives flattened observations. `node_shared` and `global_shared` receive an explicit observation mask and implement observation availability through masked pooling and a per-node observation feature.

## Consequences

Conditions can be described by activation touch-point parameters rather than by opaque backup modes. The current default cognitive architecture is:

```toml
activation_masks_actions = true
activation_gates_backup_sink = true
activation_gates_backup_source = true
activation_protects_memory = true
activation_masks_observation = true
# excluded_child_value omitted: default None
disable_persistence = false
```

## Non-goals

This ADR does not decide whether activation should be exposed as a direct observation feature separate from masking. The current issue is that the action mask is implicitly reused as an information-availability mask.

This ADR does not introduce edge-binding activation or other non-node working-memory objects.
