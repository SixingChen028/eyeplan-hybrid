# Forward Consolidation

## Goal

Explain why people often consolidate a discovered plan in the forward direction before termination, while the optimized model often consolidates backward through parent looks.

The behavioral pattern suggests that people may not be preparing for termination only by refreshing scalar ancestor values. They may instead be stabilizing a plan representation: an ordered sequence from the root to a terminal node, or the parent-child choices along that sequence. In that interpretation, forward looks are useful because they refresh the information in the same order it will be used during post-termination movement.

The current model has a different incentive. A look first refreshes working memory for the fixated node, its parent, its children, and the root. It then updates `q_values` for the fixated node and automatically backs up through active ancestors, with backup depth controlled by `backup_steps` and backup weight controlled by `lamda_backup`. In `backup_mode = "full"` this parent-chain backup can ignore working-memory activation; in the working-memory backup modes, an inactive parent stops the backup. This means the model already has an eligibility-trace-like mechanism: active ancestors of the current fixation are eligible for automatic backup.

Termination also matters. The current environment does not store a chosen plan before termination. On termination, `_sample_move_path` starts at the root and samples each child from `q_values[children]`, applying fixation-memory updates without `q_values` updates during the movement sequence. The executed `choice_path` is returned in `info`, but it is not stored in the environment state for later use.

Therefore, a proposal for forward consolidation should not simply say "add eligibility traces" or "store action values." The current code already has active-ancestor eligibility, and the current node-indexed values already behave like child-action preferences at movement time. A coherent proposal must specify which currently missing representation is added, and which current incentive for backward parent looks is removed or weakened.

The aim is not just to penalize repeated parent looks. A penalty could suppress the symptom while leaving the model's cognitive architecture unchanged. The better goal is to change what information must be available for reward propagation or termination, so that forward consolidation is useful for the same reason it appears useful for people.

## Option 1: Edge-Binding Activation

Add a working-memory activation state for parent-child bindings, separate from node activation. Because each node has at most one parent, this can be stored compactly as `edge_activation[c]`, meaning the binding `parent_nodes[c] -> c` is currently available.

### Concrete Proposal

Keep the existing node activation for node availability: legal looks, observations, node-specific memory, and corruption. Add `edge_activation`, which decays separately and is refreshed only when the model represents a forward relation, such as a look from parent `p` to child `c` where `parent_nodes[c] == p`. A parent-only look may refresh the parent node, and a child-only look may refresh the child node, but neither should automatically refresh the edge binding unless the implementation has explicit parent context available.

Apply edge activation wherever the parent-child relation is needed. For learning, `q_values[c]` or the corresponding edge value should be eligible for backup only when `edge_activation[c]` is active. For movement after termination, a child value should influence choice from its parent only when the relevant edge binding is active, or should be downweighted when the binding is weak. This does not require a full `Q(parent, child)` table: the scalar value can still be indexed by child because the parent is unique.

The main ablation is whether edge activation gates both backup and movement, or only one of them. The fully coherent version gates both because both operations require the same relation. A restricted backup-only or movement-only version could still be useful diagnostically, but it is a narrower mechanism rather than the main proposal.

### Justification

This targets the specific mismatch in the current model. Current active-ancestor backups already act like eligibility traces, and current `q_values[child]` already behave like action values during movement. The missing object is not another value table; it is the working-memory availability of the binding between a parent and the child to choose from it.

Backward parent looks are useful today because they refresh ancestor nodes and make them eligible for automatic backup. Edge activation changes the eligible unit from "active ancestor node" to "active forward relation." A backward sequence like `leaf -> parent -> grandparent` refreshes nodes, but it does not reconstruct the forward bindings `root -> child -> grandchild` unless the implementation explicitly treats backward traversal as edge rehearsal. Under this proposal, it should not.

Forward consolidation becomes useful because it refreshes the relations needed for both value propagation and post-termination movement. It prepares the same parent-to-child bindings that the agent will need when executing the plan.

## Option 2: Candidate-Plan Memory

Add an explicit memory for the best candidate plan found so far. This is a stronger proposal than edge-binding activation: it stores an ordered route or next-child sequence, not just currently available local bindings.

### Concrete Proposal

Extend the environment state with a candidate plan representation, such as a root-to-terminal path, next-child choices, an estimated return, and a reliability or activation value. This would be new state: currently `choice_path` exists only as an `info` output after `_sample_move_path` has already executed, and the environment does not retain a pre-termination plan.

When a terminal or otherwise valuable path is identified, reconstruct the root-to-terminal path from `parent_nodes` and store it as the current candidate plan. Decay the plan's reliability with working-memory decay or a plan-specific decay. Refresh reliability when looks match the stored plan in root-to-terminal order. On termination, execute the candidate plan if reliability is high enough, or use it to bias choices during `_sample_move_path`.

### Justification

The current model turns consolidation into a value-maintenance problem: make sure enough `q_values` survive and are backed up before termination. Candidate-plan memory changes termination into a plan-availability problem: make sure the intended sequence is active and reliable enough to execute.

Once termination depends on an ordered path, forward consolidation has a direct purpose. It rehearses the sequence that will be used during movement, while backward consolidation does not refresh the plan in execution order. This option is behaviorally direct, but it also introduces more machinery than edge-binding activation.

## Comparison

Edge-binding activation is the smaller change. It preserves the current value-learning architecture but changes the working-memory unit that matters for backup and movement from active nodes to active parent-child relations.

Candidate-plan memory is the more explicit account of pre-termination rehearsal. It should be preferred if the empirical target is specifically an ordered plan held for execution, rather than local relation availability affecting learning and movement.

These options are compatible. Edge activation could be tested first because it addresses the current model's backward-backup incentive directly. Candidate-plan memory is the next step if local edge bindings are not enough to produce forward consolidation.
