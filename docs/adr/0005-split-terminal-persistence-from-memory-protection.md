# ADR 0005: Split terminal persistence from memory protection

Date: 2026-06-09

Status: Implemented

Recorded version: Compatibility version 2

## Context

`activation_protects_memory` was intended to say whether working-memory activation protects node-specific memory from corruption and forgetting. The previous implementation overloaded this with terminal persistence: setting `activation_protects_memory = false` skipped corruption entirely and kept inactive terminal flags forever. That made the flag read as if memory were unprotected while actually making memory persistent.

The comparison configs also use an `actions_only` condition where activation gates legal fixation actions but not backup, observation, or memory. That condition should keep its current behavior: ordinary value-memory corruption is disabled by parameters, and discovered terminal states remain known.

## Decision

Change `activation_protects_memory` to control the corruption mask only:

- `true`: decay, drift, and stochastic forgetting apply only to inactive node-specific memory.
- `false`: decay, drift, and stochastic forgetting apply to node-specific memory regardless of activation.

Add `persist_terminal` as a separate environment flag:

- `false`: seen-terminal indicators only survive while protected by the current memory regime.
- `true`: seen-terminal indicators persist outside working memory.

Keep `disable_persistence` as the stronger mode where inactive node-specific memory cannot exist. `persist_terminal = true` is invalid when `disable_persistence = true`.

Do not change `activation_masks_actions` semantics in this ADR. Arbitrary node fixation remains unsupported because it would bypass the current path-availability assumptions in the cognitive architecture.

## Current implementation

`JaxDecisionTreeEnv` has a new static config field, `persist_terminal`, defaulting to `false`.

Memory corruption now runs whenever persistence is enabled. The corruption mask is inactive nodes when `activation_protects_memory = true`, and all nodes when `activation_protects_memory = false`.

Terminal flags are cleared through `_clear_inactive_memory` unless `persist_terminal = true`. When `activation_protects_memory = false` and `persist_terminal = false`, terminal memory is cleared regardless of activation.

The old `actions_only` behavior is preserved by setting:

```toml
activation_gates_backup_sink = false
activation_gates_backup_source = false
activation_protects_memory = false
activation_masks_observation = false
persist_terminal = true
forget_rate = 0.0
q_drift = 0.0
q_decay = 1.0
```

With those corruption parameters, value-memory corruption is a no-op, while `persist_terminal = true` keeps discovered terminal states known.

## Consequences

The flag names now match the cognitive architecture assumptions they encode: memory protection is separate from whether terminal facts can persist outside working memory.

Historical configs that used `activation_protects_memory = false` to imply persistent terminal memory should set `persist_terminal = true` explicitly.

This does not change checkpoint parameter shapes, so the compatibility version remains 2.
