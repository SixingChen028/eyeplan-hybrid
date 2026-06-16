# ADR 0006: Apply memory corruption after activation and before fixation learning

Date: 2026-06-16

Status: Implemented

Recorded version: Compatibility version 5

## Context

We want to test a model where working-memory activation controls legal fixation
actions, while node-specific memory can be corrupted regardless of activation.
This separates action availability from memory protection.

The previous implementation applied memory corruption at the end of `_look`, after
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

Move memory corruption within `_look` to after working-memory activation has been
updated for the new fixation, but before the current fixation's new learning
operations.

Add a `skip_corruption` option to `_look`. `reset` should use this option for the
initial root look, because there is no preceding action interval before the first
observation.

The transition order for a look should be:

1. Set the new fixation and update working-memory activation.
2. Apply ordinary memory corruption for previously stored node-specific memory,
   unless corruption is disabled, persistence is disabled, or `skip_corruption`
   is true.
3. Update visit count, fixation recency, discovered children, terminal status,
   Q-values, and incremental G-values.
4. Return the observation for the state after those fixation updates.

The corruption scope is controlled by `activation_prevents_corruption`, which
defaults to true. When true, corruption applies only to discovered nodes outside
working memory. When false, corruption applies to all discovered nodes, including
nodes currently active in working memory.

This keeps corruption within `_look`, so explicit movement sampling still applies
corruption between internal movement looks. It also ensures that a just-fixated
node's newly learned information survives into the immediate observation.

The intended event semantics are that corruption happens during the transition
between actions, after working memory has decayed and refreshed around the next
fixation target but before that fixation learns anything. In a termination
action, corruption still occurs because termination is also an action. Explicit
movement sampling then unrolls additional implicit move actions, each with its
own transition corruption event.

Unrolled around a terminal choice, starting with the second-to-last explicit look.
Here, a move means the moved-to fixation and learning event; the corruption event
belongs immediately before that event.

1. Transition corruption for explicit look `L[n-1]` has already happened.
2. `L[n-1]` refreshes activation, updates visit count and recency, discovers
   children, records terminal status, and updates Q/G values.
3. The policy observes the post-`L[n-1]` state and chooses explicit look `L[n]`.
4. `L[n]` refreshes activation.
5. Transition corruption for explicit look `L[n]` occurs.
6. `L[n]` updates visit count and recency, discovers
   children, records terminal status, and performs the last explicit Q/G update.
7. The policy observes the post-`L[n]` state and chooses termination.
8. The movement sampler refreshes activation around the root.
9. Transition corruption for the termination action occurs.
10. The movement sampler starts at the root without an explicit Q update.
11. The movement sampler chooses the first moved-to node from the root.
12. `M[1]` refreshes activation around the first moved-to node.
13. Transition corruption for implicit move `M[1]` occurs.
14. `M[1]` learns the first moved-to node without an explicit Q update.
15. The movement sampler chooses the second moved-to node.
16. `M[2]` refreshes activation around the second moved-to node.
17. Transition corruption for implicit move `M[2]` occurs.
18. `M[2]` learns the second moved-to node without an explicit Q update.
19. The movement sampler chooses the third moved-to node.
20. `M[3]` refreshes activation around the third moved-to node.
21. Transition corruption for implicit move `M[3]` occurs.
22. The state is now immediately before the third moved-to node is learned.

This means there are two transition corruption events between the last explicit
Q/G update and the first moved-to node being learned: one for termination and one
for the first implicit move.

## Consequences

With the default inactive-discovered scope, a node that drops out of working
memory during the activation update can be corrupted before the current fixation's
learning and before the policy observes the resulting state. If
`activation_prevents_corruption = false`, active discovered nodes can also be
corrupted before the current fixation's learning.

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

This ADR does not make undiscovered nodes corruptible. Unless a future design
explicitly chooses otherwise, corruption should still apply only to discovered
node-specific memory.
