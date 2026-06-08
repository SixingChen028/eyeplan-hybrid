# Model specification

This document specifies the model: a working-memory-limited tree-search environment and the actor-critic policies used to act in it. Training and optimization details are excluded.

## Task Environment

Each episode is a search problem on a rooted full binary tree with `N = num_nodes` nodes. `N` is odd, giving `(N + 1) / 2` terminal leaves. Tree shapes are sampled by starting with a single leaf and repeatedly expanding one uniformly selected current leaf into two children until the tree has `N` nodes. Left-right mirror-equivalent shapes are treated as the same shape, with probabilities aggregated over expansion histories. When `shuffle_nodes` is enabled, node labels are randomly permuted on each episode; the tree structure is unchanged.

Every non-root node has a point value sampled independently and uniformly from `point_set`; the root has value zero. For a node `i`, let `r_i` be its point value and let `g_i` be the cumulative value on the path from the root to the parent of `i`, excluding `i`. A terminal path ending at leaf `l` has raw value `g_l + r_l`.

The environment is a cognitive architecture for tree search. It keeps track of the current fixation, node visit counts, per-node working-memory activation, recency traces, and persistent per-node value estimates `Q_i`. These stored values are part of the environment state, not policy-network parameters.

At reset, all visit counts, recency traces, activations, and stored Q-values are zero. The root is then fixated once before the first observation is emitted.

## Actions and Timing

The policy has one fixation action for each node and one terminate action. A node can be fixated only if it is currently active in working memory. The terminate action is always legal. On the last available time step, all fixation actions are masked and only termination remains legal.

Each action increments elapsed time. Before the action-specific transition, fixation recency traces decay by `recency_decay`. A fixation action then costs `cost` reward units and updates the current fixation, working memory, and stored values. The terminate action returns the expected final move reward and ends the episode. Episodes also end when elapsed time reaches `t_max`.

## Working Memory

When node `i` is fixated, working-memory activation first decays:

```text
A_j <- clip(wm_decay * A_j, 0, 1)
```

The fixated node, its parent, its children, and the root are then set to activation one. After this refresh, each node is retained with probability equal to its activation; dropped nodes have activation zero. Thus fixated local neighborhoods are available immediately, while partially active nodes can decay out of working memory stochastically.

Only active nodes can be selected for later fixation. This constraint makes tree search a working-memory-limited process: information can be acted on only while it remains active or can be reached again through active nodes.

## Stored Value Dynamics

Fixating a node updates the environment's stored Q-values. For node `i`, the target is:

```text
T_i = r_i + max_c Q_c
```

where `c` ranges over the children of `i`; terminal nodes have target `T_i = r_i`. The fixated node is updated by:

```text
Q_i <- Q_i + learning_rate * (T_i - Q_i)
```

The update can then back up along ancestors. The first ancestor backup has weight `lamda_backup`, and each subsequent backup multiplies the weight by `lamda_backup` again. Backups stop at the root, after `backup_steps` ancestor updates, when the weight becomes negligible, or, if `wm_backup` is enabled, when the next ancestor is not active in working memory. Ancestor updates use step size `learning_rate * backup_weight`.

When `wm_backup` is enabled, inactive children contribute `Q = 0` in ancestor backup targets. When it is disabled, ancestor backups use all stored child Q-values.

After fixation and backup, Q-values for inactive nodes can degrade. Inactive Q-values are multiplied by `q_decay` and perturbed by Gaussian noise with standard deviation `q_drift`. Inactive Q-values, visit counts, and fixation recency traces are independently reset to zero with probability `forget_rate`. Active node memory is not degraded by this step.

## Observations

The observation exposes local tree-search information and selected memory variables. It always includes:

- the current fixation as a one-hot vector;
- the point value at the current fixation;
- one-hot or multi-hot indicators for the current node's parent, children, and root;
- path-prefix values `g_i` for observable nodes, with unobservable entries set to zero;
- all stored Q-values;
- node visit counts;
- indicators for terminal nodes that have been visited;
- elapsed time.

A node is known when its parent has been visited; the root is known from reset. Fixating a node therefore makes its children known open nodes. Node point values enter the observation directly only for the current fixation, while known `g_i` values expose path-prefix information for frontier nodes.

Optional observation fields are controlled by environment configuration:

- `recency`: per-node fixation recency traces.

## Termination Reward

Termination evaluates a final traversal from the root using the current stored Q-values. At each internal node, the next child is selected by a softmax over child Q-values, mixed with uniform choice:

```text
p(c | i) = (1 - eps_move) * softmax(beta_move * Q_c) + eps_move / 2
```

The environment returns the expected raw path value under this traversal rule, scaled by `scale_factor`. It does not sample or record a realized terminal path. Fixation costs are accumulated through prior fixation actions; no additional cost is subtracted on termination.

## Policy Networks

Both implemented policies are actor-critic networks that produce one logit per fixation action, one terminate logit, and one scalar state-value estimate. Illegal actions are masked before sampling or greedy selection.

### Flat MLP

The flat MLP concatenates all observation fields and passes the result through two ReLU hidden layers. Separate linear heads produce policy logits and the value estimate. This architecture is sensitive to the arbitrary node labeling.

### Shared-Node Architecture

The default architecture applies a shared encoder to each node. For node `i`, the node-level input contains:

```text
fixation_i, parent_i, child_i, root_i, g_i, Q_i, n_visits_i,
is_terminal_i, [recency_i], legal_i
```

The same two-layer ReLU MLP maps every node input to an embedding `h_i`. A shared linear head maps each `h_i` to that node's fixation logit.

The terminate logit and value estimate are computed from global features. The model pools embeddings over currently legal fixation actions using both a mean and an elementwise maximum, then concatenates these pooled vectors with the current fixation point value and elapsed time. A one-layer ReLU MLP processes this global vector, followed by separate linear heads for the terminate logit and state value.

Because node embeddings and node logits use shared weights, the fixation logits are permutation equivariant with respect to node relabeling. Because the terminate and value heads use pooled legal-node embeddings, those outputs are invariant to node relabeling, up to the non-node scalar features.

## Default Configuration

The default environment uses 15 nodes, `t_max = 50`, `scale_factor = 0.125`, shuffled node labels, recency observations, and working-memory-restricted backups. The default dynamic parameters are:

```text
beta_move = 40.0
eps_move = 0.0
learning_rate = 1.0
lamda_backup = 1.0
backup_steps = 100
wm_decay = 1.0
forget_rate = 0.0
q_drift = 0.0
q_decay = 1.0
recency_decay = 0.5
cost = 0.01
point_set = (-8, -4, -2, -1, 1, 2, 4, 8)
```

The default policy is the shared-node architecture with hidden size 64.
