# Forward Consolidation

## Goal

Explain why people often consolidate a discovered plan in the forward direction before termination, while the optimized model often consolidates backward through parent looks.

The behavioral pattern suggests that people may not be preparing for termination only by refreshing scalar ancestor values. They may instead be stabilizing a plan representation: an ordered sequence from the root to a terminal node, or the parent-child choices along that sequence. In that interpretation, forward looks are useful because they refresh the information in the same order it will be used during post-termination movement.

The current model has a different incentive. A look first refreshes working memory for the fixated node, its parent, its children, and the root. It then updates `q_values` for the fixated node and automatically backs up through active ancestors, with backup depth controlled by `backup_steps` and backup weight controlled by `lamda_backup`. In `backup_mode = "full"` this parent-chain backup can ignore working-memory activation; in the working-memory backup modes, an inactive parent stops the backup. This means the model already has an eligibility-trace-like mechanism: active ancestors of the current fixation are eligible for automatic backup.

Termination also matters. The current environment does not store a chosen plan before termination. On termination, `_sample_move_path` starts at the root and samples each child from `q_values[children]`, applying fixation-memory updates without `q_values` updates during the movement sequence. The executed `choice_path` is returned in `info`, but it is not stored in the environment state for later use.

Therefore, a proposal for forward consolidation should not simply say "add eligibility traces" or "store action values." The current code already has active-ancestor eligibility, and the current node-indexed values already behave like child-action preferences at movement time. A coherent proposal must specify which currently missing representation is added, and which current incentive for backward parent looks is removed or weakened.

The aim is not just to penalize repeated parent looks. A penalty could suppress the symptom while leaving the model's cognitive architecture unchanged. The better goal is to change what information must be available for reward propagation or termination, so that forward consolidation is useful for the same reason it appears useful for people.

## Option 1: Replace Ancestor Eligibility With Path-Edge Eligibility

This option should be understood as a replacement or restriction of the current ancestor-backup eligibility, not as the addition of eligibility traces in general. The current trace-like mechanism is: after looking at node `n`, update `n`, then walk up `parent_nodes` while ancestors remain active and `backup_steps` allows it. That mechanism makes backward parent looks useful because they refresh the exact nodes that can receive backup.

The proposed change is to make the eligible unit an ordered path edge rather than an active ancestor node. Add a decaying path-edge trace, stored compactly as one value per non-root node because each node has a unique parent. The trace entry for child `c` means: "the incoming edge `parent_nodes[c] -> c` is currently represented as part of the traced path." A trace entry is refreshed only when the agent actually constructs that forward relation, for example by looking from parent `p` to child `c` when `parent_nodes[c] == p`. If this option is combined with candidate-plan memory, replaying the stored plan in root-to-terminal order would also refresh the corresponding trace entries.

Backups would then be gated by edge-trace activation rather than by ancestor-node activation. For example, after looking at a terminal or high-value descendant, the model could update only the contiguous traced edges on the currently represented path. If the edge `p -> c` has decayed out of the path trace, then `q_values[c]` or the corresponding edge value is not eligible for update through that path, even if `p` happens to be active as a node. Conversely, a forward rehearsal that refreshes `root -> child -> grandchild` keeps those edge entries eligible.

This option does not require eliminating Bellman-like learning. The update target can remain TD-like. The difference is the working-memory condition for update:

- current code: an ancestor node can be backed up when it is active, or always in `backup_mode = "full"`;
- proposed path-edge version: a parent-child edge can be backed up only when that forward edge is active in the path trace.

### Why this should solve the problem

The current model benefits from looking at parents because parent looks refresh ancestor nodes and make them eligible for automatic backup. Path-edge eligibility removes that particular benefit. A backward sequence like `leaf -> parent -> grandparent` refreshes nodes, but it does not reconstruct the forward edge sequence `root -> child -> grandchild` unless the implementation explicitly treats backward traversal as edge rehearsal. Under this option, it should not.

Forward consolidation becomes useful because it refreshes the exact relations that gate later value propagation. The behavioral interpretation is that people may be preserving a forward path trace long enough to commit to it or learn from it, rather than recomputing scalar values at each active ancestor.

This option is only coherent if the current parent-chain backup is changed. If the existing automatic ancestor backup remains fully available, then adding a path-edge trace simply adds another learning path and may not reduce the optimized model's incentive to look backward through parents.

## Option 2: Add Edge-Binding Memory, Not Just Action Values

The phrase "store `Q(parent, child)`" is ambiguous in this codebase. Because each node has at most one parent, a value stored on child `c` is already enough to represent the value of the incoming action `parent_nodes[c] -> c`. This is also how termination currently uses the values: at a parent node, `_sample_move_path` chooses among `state.q_values[children]`. In that practical sense, the current `q_values[child]` representation already behaves like an action value for choosing that child from its unique parent.

Therefore, merely reinterpreting existing node-indexed values as `Q(parent, child)` is not a substantive model change. It would not change the action probabilities in `_sample_move_path`, the automatic backup path in `_update_q`, or the working-memory corruption rules. It should not be expected to change the optimized consolidation direction.

For this option to have consequences, the model needs a distinct parent-child binding memory in addition to the scalar value. A compact implementation could still index this binding by child node:

- `q_values[c]`: the learned value of taking the unique incoming edge into child `c`;
- `edge_activation[c]` or `edge_recency[c]`: whether the binding `parent_nodes[c] -> c` is currently available in working memory.

The practical rule would be: a value can influence backup or movement only when the relevant edge binding is available, or its influence is reduced when that binding is weak. Looking at a parent alone would not refresh the binding to the selected child. Looking at a child alone would not necessarily refresh the binding unless the previous fixation or candidate-plan state supplies the parent context. A forward look from parent to child would refresh the binding directly.

### Why this should solve the problem

This option changes the useful working-memory object from "a node has a scalar value" to "this parent has this child as an available valued action." It aligns with termination because movement from each branch point requires the next-child relation, not just the existence of a valuable descendant somewhere in memory.

However, the important part is the binding memory, not the value table shape. If `q_values[c]` remains available whenever node `c` is active, and movement continues to choose from `q_values[children]` without checking edge bindings, then this option collapses back to the current model.

This option is closely related to Option 1. Option 1 uses edge activation to gate learning backups. Option 2 uses edge activation to gate or weight action selection and value availability. They could share the same `edge_activation[c]` state.

## Option 3: Candidate-Plan Memory

Add an explicit memory for the best candidate plan found so far. This is a different representation from both node activation and edge activation. It could include:

- the ordered path from root to terminal, stored as node ids or as next-child choices;
- an estimated return or confidence value for that path;
- a reliability or activation value for the plan representation.

The key current-code contrast is that `choice_path` exists only as a termination result. It is returned in `info` after `_sample_move_path` has executed, and the environment state does not retain a pre-termination plan. Candidate-plan memory would add such a state object before termination.

A minimal concrete rule is:

- when a terminal or otherwise valuable path is identified, reconstruct the root-to-terminal path from `parent_nodes` and store it as the current candidate plan;
- decay the candidate plan's reliability with working-memory decay or a plan-specific decay;
- refresh reliability when looks match the stored plan in root-to-terminal order;
- on termination, execute the candidate plan if reliability is high enough, or use it to bias child choices during `_sample_move_path`.

### Why this should solve the problem

The current model turns consolidation into a value-maintenance problem: make sure enough `q_values` survive and are backed up before termination. Candidate-plan memory changes termination into a plan-availability problem: make sure the intended sequence is active and reliable enough to execute.

Once termination depends on an ordered path, forward consolidation has a direct purpose. It rehearses the sequence that will be used during movement. Backward consolidation is less useful because it does not refresh the plan in execution order and may not preserve the next-child bindings needed at each branch.

This option is also behaviorally transparent. It predicts that forward consolidation should be strongest after a good path has been found and before termination, especially when working-memory decay makes the plan unreliable. That matches the qualitative human pattern more directly than an account based only on scalar value backup.

The cost is that candidate-plan memory adds a new representational object. That may be justified if the empirical target is specifically pre-termination plan rehearsal rather than general reward propagation.

## Comparison

The path-edge eligibility option is the smallest conceptual change to reward propagation, but only if it replaces or restricts the current active-ancestor backup. It keeps value learning central while changing the working-memory condition for update from "ancestor node is active" to "forward edge on the candidate path is active."

The edge-binding option is not a pure value-representation change. Purely reinterpreting `q_values[child]` as `Q(parent, child)` is already consistent with the current movement policy and has no practical consequence by itself. This option only becomes meaningful if the parent-child binding has its own working-memory availability and that availability gates backup, movement, or both.

The candidate-plan option changes what termination depends on. It directly models forward rehearsal as plan maintenance rather than value backup. It is the most explicit account of the human behavior, but also introduces the most new machinery.

These options are compatible, but they should be tested in a clean order. A useful first test is edge activation shared by Option 1 and Option 2: add `edge_activation[c]`, refresh it on parent-to-child looks, and use it to gate the existing automatic backup or movement choices. If that does not produce forward consolidation, candidate-plan memory is the more direct hypothesis because it changes the termination computation rather than only the learning computation.
