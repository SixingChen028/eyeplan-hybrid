# Sigma Backup

## Goal

Make the value backup in the working-memory environment more principled using ideas from Sutton chapter 7, especially n-step tree backup and Q(sigma).

The relevant MDP is not the full fixation/look environment. It is the deterministic tree MDP induced by the decision tree represented in the current state. Because this MDP is tree-structured, we can represent `Q(p, c)` as `Q(c)`: the value of taking the action from parent `p` to child `c` is stored on the child node.

## Current Backup

For parent `p` with children `c1` and `c2`, the current backup is greedy:

```text
target(p) = r(p) + max(Q(c1), Q(c2))
Q(p) <- Q(p) + alpha * w * [target(p) - Q(p)]
```

Here `r(p)` is the point value at `p`, `alpha` is the learning rate, and `w` is the ancestor backup weight, currently derived from `lamda_backup`.

This estimates the value of arriving at `p` assuming the internal mover will choose the best child according to stored Q values. That may be intentional, but it is not aligned with the stochastic movement policy used elsewhere if `beta_move` and `eps_move` imply non-greedy choices.

## Q(sigma) Backup

When we look at node `s`, we can construct an artificial tree-action history: the sequence of child actions that would take the agent from the root to `s`. For an ancestor parent `p`, let `c1` be the child on this constructed path and `c2` the sibling.

Let:

```text
G(c1) = improved downstream return already computed for the constructed child branch
pi1 = pi(c1 | p)
pi2 = pi(c2 | p)
pi1 + pi2 = 1
sigma in [0, 1]
```

Then the local Q(sigma)-style target is:

```text
target_sigma(p)
  = r(p)
  + [sigma + (1 - sigma) * pi1] * G(c1)
  + (1 - sigma) * pi2 * Q(c2)
```

Special cases:

```text
sigma = 1:
target(p) = r(p) + G(c1)
```

This is the pure sampled/Sarsa-like backup if the constructed branch is treated as on-policy data.

```text
sigma = 0:
target(p) = r(p) + pi1 * G(c1) + pi2 * Q(c2)
```

This is the pure tree-backup / expected backup. The constructed child receives the improved downstream return; the sibling contributes its current stored estimate.

## Off-Policy Correction

The arbitrary look sequence is not naturally a rollout from the tree MDP. However, each look constructs an artificial action history. If we treat that construction as the behavior policy, then along the constructed branch:

```text
b(c1 | p) = 1
rho = pi(c1 | p) / b(c1 | p) = pi1
```

For a sigma=1 off-policy sampled backup, the principled control-variate target is not:

```text
target(p) = r(p) + pi1 * G(c1)
```

That omits the sibling baseline and would systematically lose value from unconstructed branches. The corrected target is:

```text
target(p)
  = r(p) + V_pi(p) + rho * [G(c1) - Q(c1)]

V_pi(p) = pi1 * Q(c1) + pi2 * Q(c2)
rho = pi1
```

which simplifies to:

```text
target(p) = r(p) + pi1 * G(c1) + pi2 * Q(c2)
```

Under this artificial-history interpretation, the corrected sigma=1 backup has the same local form as tree backup: update the constructed branch with its improved return, and keep siblings as expected bootstrapped values.

## Policy Choice

Use the same target policy as the internal movement policy:

```text
pi(c | p) = softmax_epsilon(Q(c), beta_move, eps_move)
```

This makes backup values consistent with `_expected_move_reward` and `_sample_move_path`.

## Implementation Direction

1. Add a backup type/config option rather than replacing the current greedy backup immediately.
2. Implement a tree-backup target first, equivalent to `sigma = 0`.
3. Reuse the existing ancestor loop in `_update_q`; replace the ancestor target computation.
4. Keep `backup_steps` as the n-step horizon.
5. Decide whether `lamda_backup` remains a learning-rate decay or is replaced by `sigma_backup`.
6. Preserve `wm_backup` as a gate on whether ancestors and sibling Q values are available in working memory.
7. Add focused tests for a parent `p` with two children `c1`, `c2`, checking greedy, tree-backup, and off-policy corrected sigma=1 targets.

## Open Decisions

- Is `Q` meant to estimate optimal path value or expected value under the stochastic movement policy?
- Should inactive children be treated as unavailable, as `Q = 0`, or as current stored values unavailable to the backup?
- Should `sigma_backup` be a global parameter, a function of activation, or a function of recency/visit confidence?
- Should we keep `lamda_backup` in addition to `sigma_backup`, or treat the Q(sigma) target as the replacement for lambda-style depth weighting?
- Should unvisited children contribute `0`, their current initialized Q value, or a prior based on visible point information?

## Recommended First Experiment

Implement a new backup mode:

```text
backup_type = "tree"
sigma_backup = 0.0
pi = movement softmax/epsilon policy
```

Then compare against the current greedy backup. If the expected backup behaves sensibly, sweep:

```text
sigma_backup in {0.0, 0.5, 1.0}
```

and consider a cognitive variant:

```text
sigma_backup = activation(c1)
```

so the backup interpolates between sampled branch propagation and expected sibling-aware propagation based on working-memory availability.
