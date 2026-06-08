# ADR 0003: Add global context to shared-node fixation policy

Date: 2026-06-08

Status: Implemented

Recorded version: Pipeline compatibility version 1; checkpoint shape compatibility is distinguished by `network_type`

## Context

The `node_shared` policy/value network applies a shared encoder to each node and maps each node embedding directly to a fixation logit with a shared linear head. The terminate logit and value estimate are different: they use global features built from pooled observable-node embeddings plus scalar state features such as the current fixation reward and elapsed time.

This means fixation actions and termination do not receive the same context. The model can use global trial state to decide whether to terminate, but the relative logits among fixation targets are determined only by each target's local node features. For example, elapsed time can affect the terminate logit but cannot change which node is preferred if the model continues.

A simple concatenation of global features to each node embedding is not enough if the fixation policy head remains linear. With a linear head:

```text
logit_i = w_node * h_i + w_global * g + b
```

the global term is identical for every node. It shifts all fixation logits by the same amount, which can change the total probability assigned to fixation actions relative to termination, but it cannot change the ranking among fixation targets. To make global context affect which node is fixated, the policy head needs a nonlinearity after local and global features are combined.

## Decision

Add a global-conditioned shared-node fixation policy as a new architecture variant rather than changing the existing `node_shared` architecture in place.

The new policy path should:

1. Compute per-node embeddings as in `node_shared`.
2. Compute `global_features` and `global_hidden` using the same pooled observable-node summaries and scalar state features currently used for terminate/value.
3. Broadcast `global_hidden` to every node.
4. Concatenate each node embedding with the broadcast global hidden state.
5. Pass the concatenated vector through a shared nonlinear policy head before producing the node fixation logit.

Conceptually:

```text
h_i = node_encoder(node_features_i)
g = global_encoder(pool_observable_nodes(h), scalar_global_features)
z_i = relu(W_policy [h_i, g] + b_policy)
fixation_logit_i = v_policy z_i + c_policy
```

The terminate and value heads should continue to use `global_hidden`.

Name the new architecture `global_shared` (new value of `network_type`), so experiment records distinguish it from `node_shared` and old checkpoints do not need compatibility branches.

## Current implementation

Implemented in `eb91a3a`.

`global_shared` reuses the `node_shared` encoder and global hidden state. It adds `node_policy_context`, which combines each node embedding with the broadcast global hidden state through a shared ReLU layer before producing fixation logits. Existing `node_shared` params do not have this layer, so checkpoint shape compatibility is handled by `network_type` and parameter shape rather than by bumping `PIPELINE_COMPAT_VERSION`.

## Rationale

The new head preserves the main symmetry property of `node_shared`: fixation logits remain permutation equivariant because the same global vector and same policy head are applied to every node. Relabeling nodes relabels the node embeddings and logits, while the pooled global representation is invariant.

Using `global_hidden` rather than raw `global_features` keeps the policy head compact and lets the existing global encoder learn a useful summary before it is combined with local node information. It also avoids giving the node policy a wide input whose scale and composition depend directly on every scalar global observation flag.

A nonlinear head is required for meaningful local/global interactions. The intended modeling change is not merely "the agent is more likely to continue rather than terminate"; it is "conditional on continuing, the preferred fixation target can depend on global trial state."

## Expected effects

The main behavioral effect should be more context-sensitive fixation choices. Late in a trial, or after a high-value terminal path has already been found, the model can score candidate fixations differently than it would early in a trial or when the current best path is weak.

This may improve fits to human fixation patterns if humans condition saccade choice on trial-level state, especially in analyses of child choice, pruning, value-conditioned fixations, and backup behavior.

The change may also make learned policies less independently interpretable. A node logit would no longer be a score of local node features alone; it would be a score of local node features under a global planning context.

## Training dynamics

The policy gradient for fixation actions will become more coupled across nodes. A loss term on one chosen fixation can update the chosen node's local embedding path, the global encoder, and indirectly the contribution of other observable nodes through pooled summaries. The current architecture already has this coupling through terminate/value gradients; this proposal adds it to fixation-policy gradients.

The additional expressivity may make optimization easier for context-dependent policies, but it may also increase sensitivity to learning-rate and entropy settings. The model may discover sharper fixation policies earlier because global summaries can gate local feature use.

Because this changes parameter shapes, existing `node_shared` checkpoints should not be loaded into the new architecture. Runs using the new architecture should be trained from scratch and recorded under the new network type.

## Runtime cost

The runtime cost should be modest for the current task sizes. The existing fixation head is a single shared linear projection from `hidden_size` to one logit per node. The proposed head adds one hidden layer over `[node_embedding, global_hidden]` for every node.

For `num_nodes = 15`, `hidden_size = 64`, and a `policy_hidden_size = 64`, the added forward-pass work is roughly:

```text
15 * (2 * 64 * 64) ~= 123k multiply-adds
```

This is small compared with full rollout and training overhead, but it is not free under many vectorized environments. Memory use also increases by the new policy-head parameters and per-node policy activations.

## Non-goals

This ADR does not decide whether the new architecture should replace `node_shared` as the default.

This ADR does not add new observation fields. It only changes how existing global information is made available to fixation logits.

This ADR does not preserve checkpoint compatibility between `node_shared` and the proposed architecture.
