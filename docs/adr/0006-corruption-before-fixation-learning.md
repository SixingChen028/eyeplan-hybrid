# ADR 0006: Apply memory corruption before fixation learning

Date: 2026-06-16

Status: Proposed

Recorded version: Not implemented

## Context

We want to test a model where working-memory activation controls legal fixation
actions, while node-specific memory can be corrupted regardless of activation.
This separates action availability from memory protection.

The current implementation applies memory corruption at the end of `_look`, after
the current fixation refreshes activation, updates visit counts and recency, marks
terminal status, and updates value memory. If active nodes can be corrupted or
forgotten, end-of-look corruption can remove information that was just learned by
the same fixation before the policy observes the resulting state.

This timing is especially confusing for terminal observations, visit counts,
fixation recency, and incremental remembered path values. Those fields should be
available immediately after the fixation that learned them.

Explicit movement sampling is different. Movement is represented as repeated
internal looks along the chosen path, and those looks should still allow memory
corruption between movement steps.

## Decision

Move memory corruption to the beginning of `_look`, before the current fixation's
new learning and refresh operations.

Add a `skip_corruption` option to `_look`. `reset` should use this option for the
initial root look, because there is no preceding action interval before the first
observation.

The transition order for a look should be:

1. Apply ordinary memory corruption for previously stored node-specific memory,
   unless corruption is disabled, persistence is disabled, or `skip_corruption`
   is true.
2. Set the new fixation.
3. Update visit count, fixation recency, discovered children, terminal status,
   activation, Q-values, and incremental G-values.
4. Return the observation for the state after those fixation updates.

This keeps corruption within `_look`, so explicit movement sampling still applies
corruption between internal movement looks. It also ensures that a just-fixated
node's newly learned information survives into the immediate observation.

The intended event semantics are that corruption happens between actions. In an
ordinary fixation action, the corruption event belongs to the interval after the
previous action and before the new fixation learns anything. In a termination
action, corruption still occurs because termination is also an action. Explicit
movement sampling then unrolls additional implicit move actions, each with its
own between-action corruption event.

Unrolled around a terminal choice, starting with the second-to-last explicit look.
Here, a move means the moved-to fixation and learning event; the corruption event
belongs immediately before that event.

1. Corruption for explicit look `L[n-1]` has already happened.
2. `L[n-1]` refreshes activation, updates visit count and recency, discovers
   children, records terminal status, and updates Q/G values.
3. The policy observes the post-`L[n-1]` state and chooses explicit look `L[n]`.
4. Corruption for explicit look `L[n]` occurs.
5. `L[n]` refreshes activation, updates visit count and recency, discovers
   children, records terminal status, and performs the last explicit Q/G update.
6. The policy observes the post-`L[n]` state and chooses termination.
7. Corruption for the termination action occurs.
8. The movement sampler starts at the root without an explicit Q update.
9. Corruption for implicit move `M[1]` occurs.
10. `M[1]` chooses and fixates the first moved-to node.
11. Corruption for implicit move `M[2]` occurs.
12. `M[2]` chooses and fixates the second moved-to node.
13. Corruption for implicit move `M[3]` occurs.
14. The state is now immediately before the third moved-to node is learned.

This means there are two corruption events between the last explicit Q/G update
and the first moved-to node being learned: one for termination and one for the
first implicit move.

## Consequences

This timing decision does not by itself decide the corruption scope. If the
chosen scope includes active discovered nodes, those nodes may be corrupted before
they are refreshed by the current look. This is intentional under that scope: the
corruption event belongs to the interval before the new fixation operation, not
after it.

The observation returned by `reset` should still describe the post-root-look
state. `reset` should call `_look(..., skip_corruption=True)` so that initial
state generation does not consume a corruption event before any action has
occurred. Tests should make this explicit when the implementation changes.

The model's detailed simulation traces should record the same post-look state that
the policy observes at each decision point. If incremental G-values become
forgettable, traces of `gs` should reflect that remembered state rather than static
tree path values.

`is_discovered` remains persistent under this ADR. A node can be forgotten in the
sense that its node-specific memory is cleared while still remaining discovered.
This is a known open issue and should be revisited separately.

The implementation should bump `COMPAT_VERSION` when this timing change lands,
because it changes environment dynamics and therefore training, evaluation, and
simulation results.

## Non-goals

This ADR does not decide the exact parameter name for selecting corruption scope.
A string-valued scope such as `inactive_discovered` versus `all_discovered` may be
clearer than restoring the old `activation_protects_memory` boolean.

This ADR does not make undiscovered nodes corruptible. Unless a future design
explicitly chooses otherwise, corruption should still apply only to discovered
node-specific memory.
