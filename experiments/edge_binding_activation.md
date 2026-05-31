# Edge-Binding Activation

## Goal

Test whether forward consolidation emerges when the working-memory-limited object is a parent-child binding rather than only a node. The hypothesis is that the optimized model currently consolidates backward because active ancestor nodes are enough for automatic backup; requiring active forward bindings should make root-to-terminal rehearsal useful for both learning and movement.

## Conceptual Proposal

Keep node activation as the representation of which nodes are available in working memory. Add a separate edge-binding activation state representing whether the agent currently has access to the relation "from this parent, choose this child." Because each node has at most one parent, the conceptual binding can be identified by the child node, but the important distinction is representational: node activation says the node is available; edge activation says the incoming parent-child relation is available.

Edge activation should matter wherever the parent-child relation is needed. For backup, value should propagate through a child only when the relevant incoming binding is active. For movement after termination, a child value should influence the choice from its parent only when the corresponding binding is active, or should be weakened when the binding is weak.

The main intended manipulation is not a new value representation. The current child-indexed `q_values` can already be interpreted as incoming action values because each child has one parent. The new ingredient is the working-memory availability of the binding that lets the model use that value in the right parent context.

## Expected Effect

Backward parent looks should become less useful because they refresh ancestor nodes without necessarily refreshing the forward bindings needed for backup and movement. A sequence such as leaf -> parent -> grandparent may preserve scalar node values, but it should not reconstruct the executable relation root -> child -> grandchild unless reverse traversal is explicitly allowed to refresh edge bindings.

Forward consolidation should become useful because it refreshes the same relations that future backup and post-termination movement require. If the agent has found a good terminal path, rehearsing that path in root-to-terminal order should maintain the bindings needed to update and execute it.

## Decision Points

### Edge Refresh Rule

**Decision needed:** What counts as refreshing a binding?

The default conceptual rule should be strict: a binding is refreshed by representing the forward relation from parent to child, especially when the previous fixation is the parent and the current fixation is the child. This preserves the distinction between node availability and edge availability.

Open alternatives:

- allow any child look to refresh its incoming edge if the parent is active;
- allow looking at a parent to refresh bindings to active or recently selected children;
- allow backward traversal to refresh an edge because the child-parent relation is available.

The strict rule is most likely to create a forward-consolidation incentive. The looser rules may collapse edge activation back toward node activation.

### Edge Decay Rule

**Decision needed:** Should edge activation decay with the same dynamics as node activation, or have its own decay/reliability process?

Using the same working-memory decay makes the edge mechanism easier to interpret as another working-memory representation. A separate decay parameter would test whether bindings are more fragile than node availability, but it adds a degree of freedom that may be hard to identify.

### Backup Gating Rule

**Decision needed:** How should inactive edges affect value backup?

The cleanest conceptual rule is hard eligibility: value cannot propagate through an inactive edge. A softer alternative is graded weighting, where weak edge activation reduces the backup weight. Hard eligibility gives the clearest test of whether active forward bindings replace active ancestor nodes as the effective trace.

### Movement Gating Rule

**Decision needed:** How should inactive edges affect post-termination movement?

The main proposal is that movement should depend on edge activation because choosing a child from a parent requires the parent-child binding. The exact conceptual rule still needs to be chosen:

- inactive edges are unavailable choices;
- inactive edges remain legal but their values are treated as unknown or weakened;
- inactive edges use a default exploratory policy.

Masking inactive edges is the strongest manipulation but may make movement failure too abrupt. Value weakening may better match a noisy memory interpretation.

### Interaction With Node Activation

**Decision needed:** What happens when node activation and edge activation disagree?

Conceptually, node active / edge inactive means "the node is available, but the relation needed to use it from this parent is not." Edge active / node inactive is less clear. The conservative conceptual stance is that both node and edge availability are required when using a child value from a parent.

### Interaction With Existing Backup Modes

**Decision needed:** Does edge activation replace the current active-ancestor eligibility condition, or combine with it?

To test the hypothesis cleanly, edge activation should replace active-ancestor eligibility as the key working-memory condition for parent-chain backup. If both conditions remain independently sufficient, backward parent looks may stay useful and the experiment may not test the intended mechanism.

### Observation Exposure

**Decision needed:** Should the policy observe edge activation?

If edge activation is hidden, the agent must infer binding availability from fixation history and outcomes. If it is observed, the policy can deliberately rehearse weak bindings. Observing it may make the mechanism easier to learn but less cognitively minimal.

### Diagnostic Ablations

**Decision needed:** Which restricted variants are worth running?

The main condition should apply edge activation to both backup and movement. Backup-only and movement-only variants are still useful diagnostics: backup-only asks whether learning incentives are sufficient, while movement-only asks whether execution reliability alone produces forward consolidation.

## Success Criteria

The experiment should be considered conceptually promising if it produces more forward root-to-terminal rehearsal before termination without simply lowering performance or increasing arbitrary revisits. The stronger signature is selective forward consolidation after a good terminal path has been discovered, especially when working-memory decay is strong enough that bindings are unreliable.

The experiment should be considered weak if backward parent looks remain the dominant consolidation strategy under the strict edge-refresh rule, or if forward looks increase only because movement becomes generally unreliable.

## Non-Goals

This experiment should not introduce candidate-plan memory. It should not replace child-indexed values with a full parent-child value table unless a separate conceptual need emerges. It should also avoid treating edge activation as just a deterministic function of node activation, because that would remove the key distinction being tested.
