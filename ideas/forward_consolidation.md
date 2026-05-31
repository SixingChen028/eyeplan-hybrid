# Forward Consolidation

## Goal

Explain why people often consolidate a discovered plan in the forward direction before termination, while the optimized model often consolidates backward through parent looks.

The behavioral pattern suggests that people may not be preparing for termination by refreshing scalar ancestor values. They may instead be stabilizing an executable plan: an ordered sequence from the root to a terminal node. In that interpretation, forward looks are useful because they preserve the path in the same order it will be executed.

The current model has a different incentive. Value propagation is tied to node-value maintenance over the structural parent chain. Once a good path has been found, looking backward through parents can improve or refresh the ancestor `q_values` that determine the expected movement reward after termination. This makes backward consolidation computationally useful, even though it does not match the human eye-movement pattern.

The aim is not just to penalize repeated parent looks. A penalty could suppress the symptom while leaving the model's cognitive architecture unchanged. The better goal is to change what information must be available for reward propagation and termination, so that forward consolidation is useful for the same reason it appears useful for people.

## Option 1: Eligibility Trace Over the Discovered Path

Add a decaying trace of the recently constructed path. The trace could be over nodes, transitions, or parent-child bindings. When a valuable terminal path is discovered, value updates are applied through the active trace rather than requiring explicit parent looks.

In a simple version, each forward look refreshes the trace entry for the corresponding transition. When reward or downstream value becomes available, still-active trace entries receive a TD-style or return-style update. If a transition has decayed out of working memory, it is no longer eligible for update.

This does not require eliminating Bellman-like learning. The update target can still be TD-like, but the cognitive availability condition changes: the model can update a parent-child relation only if that relation is currently represented in the path trace.

### Why this should solve the problem

The current model benefits from looking at parents because parent looks help make ancestor values available for backup. A trace-based update removes that particular benefit. If value can propagate through an active path trace, the agent does not need to overtly walk backward through parents after finding a good terminal.

Forward consolidation becomes useful because it refreshes the exact information needed for later value use: the ordered path or transition sequence. Looking root -> child -> grandchild keeps the executable trace active. Looking leaf -> parent -> grandparent may refresh individual ancestors, but it does not refresh the forward transition sequence in execution order. Under low or moderate working-memory capacity, the best way to keep the discovered plan eligible for value use should therefore be to rehearse it forward.

This option also gives a principled interpretation of human consolidation. People may be preserving a path trace long enough to commit to it, not recomputing scalar values at each ancestor.

## Option 2: Store Transition or Action Values

Represent learned value as a parent-child relation rather than as a scalar value attached to the parent node. Conceptually, store `Q(parent, child)`: the value of choosing a particular child from a particular parent. Because the tree structure gives each child a unique parent, this could still be implemented compactly as a value stored on the child node, but the interpretation would be action value rather than parent state value.

Under this representation, the useful memory unit is not "the parent has value X." It is "from this parent, choose this child." A backup or refresh operation should require the parent context and the child/action context to be jointly available.

### Why this should solve the problem

Backward parent looks are useful in the current model because they can refresh or improve the scalar value of an ancestor. If the model instead needs a bound parent-child action value, looking at a parent alone is not enough. The relevant object is the forward choice relation.

A forward sequence naturally reconstructs those relations: root with selected child, then that child with its selected child, and so on. This is exactly the structure needed to execute the plan after termination. A backward sequence visits nodes in the opposite order and is less effective at maintaining the parent-to-child bindings that specify what to do next.

This change also aligns the model more closely with the decision problem induced by termination. After termination, the agent does not need a generic estimate of each ancestor. It needs to choose an action at each branch point. If the stored value is attached to the action relation, then forward consolidation prepares the same representation that termination will use.

The main implementation choice is whether to fully replace parent-state `q_values` with action values, or to reinterpret the existing node-indexed values as values of the incoming action from the parent. The second option may be simpler because the tree has unique parent-child edges.

## Option 3: Candidate-Plan Memory

Add an explicit memory for the best candidate plan found so far. The representation could include:

- the ordered path from root to terminal;
- an estimated return or confidence value;
- the current reliability of the plan in working memory.

Termination would then evaluate or execute this candidate plan, rather than relying only on recursively backed-up ancestor `q_values`. Value learning could still exist, but successful termination would depend on whether the model has an available plan representation.

Consolidation would mean stabilizing the candidate plan before termination. A forward look sequence could refresh each element of the plan in the order it will be used. If the plan representation decays, the agent may need to rehearse the route before committing.

### Why this should solve the problem

The current model turns consolidation into a value-maintenance problem: make sure the root and ancestors have the right scalar values before terminating. That makes backward backup attractive. Candidate-plan memory changes termination into a plan-availability problem: make sure the intended sequence is active and reliable enough to execute.

Once termination depends on an ordered path, forward consolidation has a direct purpose. It rehearses "what I will do" from root to terminal. Backward consolidation is less useful because it does not refresh the plan in execution order and may not preserve the next-action bindings needed at each branch.

This option is also behaviorally transparent. It predicts that forward consolidation should be strongest after a good path has been found and before termination, especially when working-memory decay makes the plan unreliable. That matches the qualitative human pattern more directly than an account based only on scalar value backup.

The cost is that candidate-plan memory adds a new representational object. That may be justified if the empirical target is specifically pre-termination plan rehearsal rather than general reward propagation.

## Comparison

The eligibility-trace option is the smallest conceptual change to reward propagation. It keeps value learning central, but changes the working-memory condition for update. It is a good first test if the main hypothesis is that reward can propagate through active forward traces without overt parent looks.

The transition-value option changes the value representation. It should reduce the usefulness of isolated parent refreshes because the useful object is a forward action relation. It is a good fit if the current scalar parent-value representation is the suspected source of the mismatch.

The candidate-plan option changes what termination depends on. It directly models forward rehearsal as plan maintenance rather than value backup. It is the most explicit account of the human behavior, but also introduces the most new machinery.

These options are compatible. A useful path might be to start with trace-gated transition updates, then add candidate-plan memory only if value-representation changes are not enough to produce forward consolidation.
