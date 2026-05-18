# Sigma Backup

## Goal

Make the value backup in the working-memory environment more principled using ideas from Sutton chapter 7, especially n-step tree backup and $Q(\sigma)$.

The relevant MDP is not the full fixation/look environment. It is the deterministic tree MDP induced by the decision tree represented in the current state. Because this MDP is tree-structured, we can represent $Q(p, c)$ as $Q(c)$: the value of taking the action from parent $p$ to child $c$ is stored on the child node.

## Current Backup

For parent $p$ with children $c_1$ and $c_2$, the current backup is greedy:

$$
\mathrm{target}(p) = r(p) + \max(Q(c_1), Q(c_2))
$$

$$
Q(p) \leftarrow Q(p) + \alpha \cdot w \cdot [\mathrm{target}(p) - Q(p)]
$$

Here $r(p)$ is the point value at $p$, $\alpha$ is the learning rate, and $w$ is the ancestor backup weight, currently derived from $\lambda_{\mathrm{backup}}$.

This estimates the value of arriving at $p$ assuming the internal mover will choose the best child according to stored $Q$ values. That may be intentional, but it is not aligned with the stochastic movement policy used elsewhere if $\beta_{\mathrm{move}}$ and $\epsilon_{\mathrm{move}}$ imply non-greedy choices.

## $Q(\sigma)$ Backup

When we look at node $s$, we can construct an artificial tree-action history: the sequence of child actions that would take the agent from the root to $s$. For an ancestor parent $p$, let $c_1$ be the child on this constructed path and $c_2$ the sibling.

Let:

$$
\begin{aligned}
G(c_1) &= \text{improved downstream return already computed for the constructed child branch} \\
\pi_1 &= \pi(c_1 \mid p) \\
\pi_2 &= \pi(c_2 \mid p) \\
\pi_1 + \pi_2 &= 1 \\
\sigma &\in [0, 1]
\end{aligned}
$$

Then the local $Q(\sigma)$-style target is:

$$
\begin{aligned}
\mathrm{target}_\sigma(p)
  &= r(p) \\
  &\quad + [\sigma + (1 - \sigma) \cdot \pi_1] \cdot G(c_1) \\
  &\quad + (1 - \sigma) \cdot \pi_2 \cdot Q(c_2)
\end{aligned}
$$

Special cases:

$$
\sigma = 1:\quad \mathrm{target}(p) = r(p) + G(c_1)
$$

This is the pure sampled/Sarsa-like backup if the constructed branch is treated as on-policy data.

$$
\sigma = 0:\quad \mathrm{target}(p) = r(p) + \pi_1 \cdot G(c_1) + \pi_2 \cdot Q(c_2)
$$

This is the pure tree-backup / expected backup. The constructed child receives the improved downstream return; the sibling contributes its current stored estimate.

## Off-Policy Correction

The arbitrary look sequence is not naturally a rollout from the tree MDP. However, each look constructs an artificial action history. If we treat that construction as the behavior policy, then along the constructed branch:

$$
\begin{aligned}
b(c_1 \mid p) &= 1 \\
\rho &= \pi(c_1 \mid p) / b(c_1 \mid p) = \pi_1
\end{aligned}
$$

For a $\sigma=1$ off-policy sampled backup, the principled control-variate target is not:

$$
\mathrm{target}(p) = r(p) + \pi_1 \cdot G(c_1)
$$

That omits the sibling baseline and would systematically lose value from unconstructed branches. The corrected target is:

$$
\begin{aligned}
\mathrm{target}(p)
  &= r(p) + V_\pi(p) + \rho \cdot [G(c_1) - Q(c_1)] \\
V_\pi(p) &= \pi_1 \cdot Q(c_1) + \pi_2 \cdot Q(c_2) \\
\rho &= \pi_1
\end{aligned}
$$

which simplifies to:

$$
\mathrm{target}(p) = r(p) + \pi_1 \cdot G(c_1) + \pi_2 \cdot Q(c_2)
$$

Under this artificial-history interpretation, the corrected $\sigma=1$ backup has the same local form as tree backup: update the constructed branch with its improved return, and keep siblings as expected bootstrapped values.

## Policy Choice

Use the same target policy as the internal movement policy:

$$
\pi(c \mid p) = \mathrm{softmax\_epsilon}(Q(c), \beta_{\mathrm{move}}, \epsilon_{\mathrm{move}})
$$

This makes backup values consistent with `_expected_move_reward` and `_sample_move_path`.

## Implementation Direction

1. Add a backup type/config option rather than replacing the current greedy backup immediately.
2. Implement a tree-backup target first, equivalent to $\sigma = 0$.
3. Reuse the existing ancestor loop in `_update_q`; replace the ancestor target computation.
4. Keep `backup_steps` as the n-step horizon.
5. Decide whether $\lambda_{\mathrm{backup}}$ remains a learning-rate decay or is replaced by $\sigma_{\mathrm{backup}}$.
6. Preserve `wm_backup` as a gate on whether ancestors and sibling $Q$ values are available in working memory.
7. Add focused tests for a parent $p$ with two children $c_1$, $c_2$, checking greedy, tree-backup, and off-policy corrected $\sigma=1$ targets.

## Open Decisions

- Is $Q$ meant to estimate optimal path value or expected value under the stochastic movement policy?
- Should inactive children be treated as unavailable, as $Q = 0$, or as current stored values unavailable to the backup?
- Should $\sigma_{\mathrm{backup}}$ be a global parameter, a function of activation, or a function of recency/visit confidence?
- Should we keep $\lambda_{\mathrm{backup}}$ in addition to $\sigma_{\mathrm{backup}}$, or treat the $Q(\sigma)$ target as the replacement for lambda-style depth weighting?
- Should unvisited children contribute $0$, their current initialized $Q$ value, or a prior based on visible point information?

## Recommended First Experiment

Implement a new backup mode:

$$
\begin{aligned}
\mathrm{backup\_type} &= \text{"tree"} \\
\sigma_{\mathrm{backup}} &= 0.0 \\
\pi &= \text{movement softmax/epsilon policy}
\end{aligned}
$$

Then compare against the current greedy backup. If the expected backup behaves sensibly, sweep:

$$
\sigma_{\mathrm{backup}} \in \{0.0, 0.5, 1.0\}
$$

and consider a cognitive variant:

$$
\sigma_{\mathrm{backup}} = \mathrm{activation}(c_1)
$$

so the backup interpolates between sampled branch propagation and expected sibling-aware propagation based on working-memory availability.
