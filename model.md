# Model specification

This document specifies the model: a working-memory-limited tree-search cognitive
architecture and the actor-critic policies used to act in it. Training and
optimization details are excluded.

## Task Environment

Each episode is a search problem on a rooted full binary tree with `N = num_nodes`
nodes. `N` is odd, giving `(N + 1) / 2` terminal leaves. Tree shapes are sampled
by starting with a single leaf and repeatedly expanding one uniformly selected
current leaf into two children until the tree has `N` nodes. Left-right
mirror-equivalent shapes are treated as the same shape, with probabilities
aggregated over expansion histories.

When `shuffle_nodes` is enabled, node labels are randomly permuted on each
episode and sibling order is randomly swapped. These operations change the
labels and child ordering exposed to the policy, not the underlying tree shape.

Every non-root node has a point value sampled independently and uniformly from
`point_set`; the root has value zero. For a node `i`, let `r_i` be its point
value and let `g_i` be the cumulative value on the path from the root to the
parent of `i`, excluding `i`. A terminal path ending at leaf `l` has raw value
`g_l + r_l`.

The environment is a cognitive architecture for tree search. It keeps track of
the current fixation, remembered path-prefix values `g_i`, node visit counts,
per-node working-memory activation, fixation recency traces, discovered-node
state, seen-terminal markers, and persistent per-node value estimates `Q_i`.
These stored values are part of the environment state, not policy-network
parameters.

At reset, all visit counts, recency traces, activations, terminal markers, and
stored Q-values are zero. The root is discovered, its `g` value is zero, and all
other `g` values are initialized to the minimum possible path value. The root is
then fixated once before the first observation is emitted; this initial fixation
does not run a memory-corruption event.

## Actions and Timing

The policy has one fixation action for each node and one terminate action. The
terminate action is always legal. Before the last available time step, a node
fixation is legal when the node is active in working memory; the root fixation
action is also legal even if the root is inactive. On the last available time
step, all fixation actions are masked and only termination remains legal.

Each action increments elapsed time. Before the action-specific transition,
fixation recency traces decay by `recency_decay`. A fixation action costs `cost`
reward units and runs a look transition at the selected node. The terminate
action samples a final move path and ends the episode. Episodes also end when
elapsed time reaches `t_max`.

## Working Memory and Discovery

A look transition at node `i` first makes `i` the current fixation. Working-memory
activation then decays:

```text
A_j <- clip(wm_decay * A_j, 0, 1)
```

After decay, each node is retained with probability equal to its activation;
dropped nodes have activation zero. The fixated node is set to activation one,
and its parent and children are refreshed to at least `wm_neighbor_activation`.
The root is not specially refreshed unless it is the fixated node or a neighbor
of the fixated node.

If `disable_persistence` is enabled, inactive node-specific memory is cleared
after this activation update and before the current fixation writes new
information. Clearing sets inactive Q-values to zero, visit counts to zero,
`g` values to the minimum possible path value, recency traces to zero, and
terminal markers to false.

If persistence is enabled and corruption is not disabled, memory corruption runs
after activation update and before the current fixation writes new information.
The corruption scope is controlled by `activation_prevents_corruption`:

- when true, only discovered inactive nodes are corruptible;
- when false, all discovered nodes are corruptible, including active nodes.

For corruptible nodes, Q-values are multiplied by `q_decay` and perturbed by
Gaussian noise with standard deviation `q_drift`. With probability `forget_rate`,
node-specific memory is reset: Q-value, visit count, `g` value, and recency are
cleared. When `forget_discovered` is true, this stochastic forgetting also clears
the node's discovered status and activation. Terminal memory is cleared
deterministically for all corruptible nodes.

After any clearing or corruption, the current fixation writes new information.
The fixated node is active, its children are refreshed to at least
`wm_neighbor_activation`, the fixated node and its children become discovered,
the children receive `g_child = g_i + r_i`, the fixated node's visit count is
incremented, its recency is set to one, and its terminal marker is set from the
tree structure. Activations of nodes that are no longer discovered are cleared.

This constraint makes search working-memory limited: information can be acted on
only while it remains active, except that the root remains a legal re-fixation
anchor before timeout.

## Stored Value Dynamics

Fixating a node updates the environment's stored Q-values. The update starts at
the fixated node with backup weight one, then walks upward through ancestors.
For a current node `i`, the backup target is:

```text
T_i = observed_reward_i + sum_c p(c | i) * child_value_c
```

where `observed_reward_i` is `r_i` only when `n_visits_i > 0`; otherwise it is
zero. This means a node's point reward is available to backups only after the
node has been directly observed by fixation, and it shares the same forgetting
dynamics as the visit count.

The child policy used in the backup target is controlled by backup-source
gating:

- if `activation_gates_backup_source` is false, all stored child Q-values are
  used;
- if it is true and `excluded_child_value` is `None`, inactive children are
  excluded from both the softmax support and the epsilon-random support;
- if it is true and `excluded_child_value` is numeric, inactive children remain
  in the child set with that substituted value.

Invalid child slots are always excluded from backup support.

The softmax/random mixture is:

```text
p(c | i) = (1 - eps_move) * softmax(beta_move * child_value_c)
         + eps_move * uniform_over_supported_children
```

Terminal nodes have no supported children, so their target is their observed
reward. The Q-value update is:

```text
Q_i <- Q_i + learning_rate * backup_weight * (T_i - Q_i)
```

The fixated node is always eligible for this update. Ancestor backups stop at the
root, after `backup_steps` ancestor updates, when the backup weight becomes
negligible, or, when `activation_gates_backup_sink` is enabled, when the next
ancestor is not active in working memory. Each ancestor multiplies the backup
weight by `lamda_backup`.

## Observations

The observation always exposes local fixation information:

- the current fixation as a one-hot vector;
- the point value at the current fixation;
- one-hot or multi-hot indicators for the current node's parent, children, and
  root.

Additional fields are included or omitted by static environment configuration:

- `g_values`: remembered path-prefix values;
- `q_values`: remembered value estimates;
- `n_visits`: node visit counts;
- `is_terminal`: seen-terminal indicators;
- `recency`: fixation recency traces;
- `time_elapsed`: elapsed time.

When an included field is node-specific, it is masked by the observation mask.
If `activation_masks_observation` is true, the observation mask is current
working-memory activation. Otherwise it is discovered-node state. Masked entries
are returned as zero.

A node becomes discovered when it is the root at reset, when it is fixated, or
when it is a child of the fixated node. Node point values enter the observation
directly only for the current fixation. Remembered `g` values expose path-prefix
information for nodes that are observable under the current observation mask.

## Termination and Final Move Reward

Termination samples a final traversal from the root using the current stored
Q-values. The terminate action first runs a look at the root with Q-updates
disabled. At each internal node, the next child is sampled from:

```text
p(c | i) = (1 - eps_move) * softmax(beta_move * Q_c) + eps_move / 2
```

The sampled child point values are accumulated until a terminal leaf is reached.
Each node reached during this final traversal is also looked at with Q-updates
disabled, so activation, discovery, recency, visit counts, terminal markers, and
memory corruption can still change during the movement phase.

The termination reward is:

```text
scale_factor * sampled_raw_path_value
  - move_cost_scale * cost * sampled_path_length
```

No ordinary fixation cost is subtracted for the terminate action. The sampled
path is recorded in `info["choice_path"]`.

## Policy Networks

All implemented policies are actor-critic networks that produce one logit per
fixation action, one terminate logit, and one scalar state-value estimate.
Illegal actions are masked outside the network before sampling or greedy
selection.

### Flat MLP

The flat MLP concatenates all included observation fields and passes the result
through two ReLU hidden layers. Separate linear heads produce policy logits and
the value estimate. This architecture is sensitive to arbitrary node labeling.

### Shared-Node Architecture

The `node_shared` architecture applies a shared encoder to each node. For node
`i`, the node-level input contains:

```text
fixation_i, parent_i, child_i, root_i,
[g_i], [Q_i], [n_visits_i], [is_terminal_i], [recency_i],
observable_i
```

The bracketed fields are present only when the corresponding observation field
is enabled. `observable_i` is the observation-mask indicator, not the legal-action
indicator.

The same two-layer ReLU MLP maps every node input to an embedding `h_i`. A shared
linear head maps each embedding to that node's fixation logit. The terminate
logit and value estimate are computed from global features: the mean and
elementwise maximum of node embeddings over observable nodes, concatenated with
the current fixation point value and, when enabled, elapsed time. A one-layer
ReLU MLP processes this global vector, followed by separate linear heads for the
terminate logit and state value.

### Global-Shared Architecture

The `global_shared` architecture uses the same node encoder and global features
as `node_shared`, but also feeds global context back into each node's fixation
logit. The global hidden vector is broadcast to every node, concatenated with
that node's embedding, processed by a shared ReLU layer, and then mapped to the
node fixation logit. This lets fixation logits depend on permutation-invariant
global context while preserving permutation equivariance.

For both shared architectures, node logits are permutation equivariant with
respect to node relabeling. The terminate and value outputs are permutation
invariant because their global inputs pool over observable-node embeddings.

## Default Configuration

The default environment uses 15 nodes, `t_max = 50`, `scale_factor = 0.125`,
shuffled node labels, persistence enabled, activation-masked actions,
activation-gated backup sink and source, corruption enabled, activation-protected
corruption, activation-masked observations, all optional observation fields
enabled, and no substituted value for excluded children.

The default environment parameters are:

```text
beta_move = 40.0
eps_move = 0.0
learning_rate = 1.0
lamda_backup = 1.0
backup_steps = 100
wm_decay = 1.0
wm_neighbor_activation = 1.0
forget_rate = 0.0
q_drift = 0.0
q_decay = 1.0
recency_decay = 0.5
cost = 0.01
move_cost_scale = 0.0
point_set = (-8, -4, -2, -1, 1, 2, 4, 8)
```

The default policy is `global_shared` with hidden size 64.
