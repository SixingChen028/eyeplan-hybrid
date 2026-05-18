# Working-Memory Backups

## Goal

Make the value backup in the working-memory environment more principled using the two useful extremes from Sutton chapter 7:

1. Tree backup when both children of a parent are available in working memory.
2. Sampled n-step Sarsa-style backup, with importance sampling if needed, when only the constructed child branch is available in working memory.

The relevant MDP is not the full fixation/look environment. It is the deterministic tree MDP induced by the decision tree represented in the current state. Because this MDP is tree-structured, we can represent \(Q(p, c)\) as \(Q(c)\): the value of taking the action from parent \(p\) to child \(c\) is stored on the child node.

When we look at node \(s\), we construct an artificial tree-action history: the sequence of child actions that would take the agent from the root to \(s\). For an ancestor parent \(p\), let:

$$
\begin{aligned}
c_1 &= \text{the child on the constructed path to } s \\
c_2 &= \text{the sibling child} \\
r(p) &= \text{the point value at } p \\
G(c_1) &= \text{the improved downstream return for the constructed child branch} \\
\pi_1 &= \pi(c_1 \mid p) \\
\pi_2 &= \pi(c_2 \mid p)
\end{aligned}
$$

Use the same target policy as the internal movement policy unless we intentionally define a working-memory-restricted policy:

$$
\pi(c \mid p) = \mathrm{softmax\_epsilon}(Q(c), \beta_{\mathrm{move}}, \epsilon_{\mathrm{move}})
$$

## Current Behavior

The current backup is greedy. For a parent \(p\) with children \(c_1\) and \(c_2\):

$$
\mathrm{target}(p) = r(p) + \max(Q(c_1), Q(c_2))
$$

$$
Q(p) \leftarrow Q(p) + \alpha \cdot w \cdot [\mathrm{target}(p) - Q(p)]
$$

Here \(\alpha\) is the learning rate and \(w\) is the ancestor backup weight, currently derived from \(\lambda_{\mathrm{backup}}\).

If `wm_backup` is active, inactive children are treated as having value zero in ancestor backups.

### Current Case 1: Both Children in WM

If both \(c_1\) and \(c_2\) are active:

$$
\mathrm{target}_{\mathrm{current}}(p) = r(p) + \max(Q(c_1), Q(c_2))
$$

This is an optimal/greedy backup. It ignores the stochastic movement policy.

### Current Case 2: One Child in WM

If only the constructed child \(c_1\) is active and \(c_2\) is inactive:

$$
\mathrm{target}_{\mathrm{current}}(p) = r(p) + \max(Q(c_1), 0)
$$

This is not tree backup and not sampled Sarsa. It is a greedy backup with an implicit zero-valued missing sibling. That can be useful as a heuristic, but it is not a principled target for either the full tree MDP or a sampled branch return.

## Planned Behavior

Separate the backup rule by what is actually available in working memory.

## Case 1: Both Children in WM -> Tree Backup

When both \(c_1\) and \(c_2\) are in working memory, use a tree-backup / expected target:

$$
\mathrm{target}_{\mathrm{tree}}(p)
  = r(p) + \pi_1 \cdot G(c_1) + \pi_2 \cdot Q(c_2)
$$

The constructed child \(c_1\) gets the improved downstream return \(G(c_1)\). The sibling \(c_2\) contributes its current stored estimate \(Q(c_2)\).

This is the right target when both children are cognitively available because the backup can represent the expected continuation under the movement policy.

If both child branches had improved returns available, the fully expected target would be:

$$
\mathrm{target}_{\mathrm{tree}}(p)
  = r(p) + \pi_1 \cdot G(c_1) + \pi_2 \cdot G(c_2)
$$

but a single look usually improves only the constructed branch.

## Case 2: One Child in WM -> Sampled Backup

When only the constructed child \(c_1\) is in working memory, the backup should not pretend that the sibling value is available.

There are two coherent interpretations.

### Option A: Working-Memory-Restricted Target Policy

If the target policy is defined over WM-available children only, then:

$$
\pi_{\mathrm{WM}}(c_1 \mid p) = 1
$$

and the sampled target is:

$$
\mathrm{target}_{\mathrm{sample}}(p) = r(p) + G(c_1)
$$

No importance sampling is needed because the target policy and the constructed behavior both choose the only available child.

This estimates the value of the cognitive architecture given the current WM contents, not the value of the full tree MDP under the full movement policy.

### Option B: Full Target Policy With Importance Sampling

If the target remains the full movement policy over both children, then the sampled branch should be corrected by an importance ratio:

$$
\rho_1 = \frac{\pi(c_1 \mid p)}{b(c_1 \mid p)}
$$

The tempting artificial-history argument is:

$$
b(c_1 \mid p) = 1
\quad\Rightarrow\quad
\rho_1 = \pi_1
$$

This is only conditionally true after the look has already selected a branch. For statistical off-policy correction, \(b(c_1 \mid p)\) should be the probability that the behavior process constructs the \(c_1\) branch before conditioning on the observed look. That probability is induced by the look policy, not by the deterministic path reconstruction itself.

If we intentionally define the constructed backup as a deterministic query rather than sampled data, then \(b = 1\) can be used as a modeling convention. In that convention, the one-child sampled update should be written as an importance-weighted update, not as a plain Bellman target:

$$
Q(p) \leftarrow Q(p) + \alpha \cdot w \cdot \rho_1 \cdot [r(p) + G(c_1) - Q(p)]
$$

Equivalently, if the implementation requires a target:

$$
\mathrm{target}_{\mathrm{eff}}(p)
  = Q(p) + \rho_1 \cdot [r(p) + G(c_1) - Q(p)]
$$

This is different from:

$$
r(p) + \pi_1 \cdot G(c_1)
$$

which scales the continuation but drops the correct baseline against the current estimate.

## Problem With the Control-Variate Sigma=1 Target

The control-variate form for off-policy sampled backups is:

$$
\begin{aligned}
\mathrm{target}_{\mathrm{cv}}(p)
  &= r(p) + V_\pi(p) + \rho_1 \cdot [G(c_1) - Q(c_1)] \\
V_\pi(p) &= \pi_1 \cdot Q(c_1) + \pi_2 \cdot Q(c_2)
\end{aligned}
$$

If we use the artificial-history convention \(b(c_1 \mid p) = 1\), then \(\rho_1 = \pi_1\) and:

$$
\mathrm{target}_{\mathrm{cv}}(p)
  = r(p) + \pi_1 \cdot G(c_1) + \pi_2 \cdot Q(c_2)
$$

This is exactly the local tree-backup target.

That is a problem for the intended WM split. The control-variate target requires \(Q(c_2)\), the sibling value. But the one-child-in-WM case is defined by \(c_2\) not being available. Therefore the control-variate sigma=1 rule does not give a distinct principled one-child backup. It collapses back to tree backup when the sibling is available, and it is not implementable without an assumption about the missing sibling when the sibling is unavailable.

Actionable conclusion: use tree backup for the both-children-in-WM case, and use either a WM-restricted sampled target or a non-control-variate importance-weighted sampled update for the one-child-in-WM case.

## Implementation Direction

1. Add a backup mode/config option rather than replacing the current greedy backup immediately.
2. In ancestor backup, identify the constructed child \(c_1\) as the child equal to the current node in the upward backup loop; the sibling is \(c_2\).
3. If both children are active, use tree backup:

$$
\mathrm{target}(p) = r(p) + \pi_1 \cdot G(c_1) + \pi_2 \cdot Q(c_2)
$$

4. If only \(c_1\) is active, choose one of:

$$
\mathrm{target}(p) = r(p) + G(c_1)
$$

for a WM-restricted target policy, or:

$$
Q(p) \leftarrow Q(p) + \alpha \cdot w \cdot \rho_1 \cdot [r(p) + G(c_1) - Q(p)]
$$

for a full-policy sampled backup with importance weighting.

5. Do not use the control-variate target for the one-child case unless we explicitly define how to access or impute \(Q(c_2)\).
6. Keep `backup_steps` as the n-step horizon.
7. Decide whether \(\lambda_{\mathrm{backup}}\) remains a learning-rate decay for ancestor depth.
8. Add focused tests for:

- both children active -> expected/tree backup;
- only constructed child active -> sampled backup;
- inactive sibling with positive \(Q(c_2)\) does not affect the one-child sampled target;
- current greedy mode remains available for comparison.

## Open Decisions

- Should the one-child case estimate the value of the WM-restricted cognitive state, or the full tree MDP under the full movement policy?
- If using full-policy importance sampling, how do we define or estimate \(b(c_1 \mid p)\) from the look policy?
- Should inactive siblings be completely unavailable, or can their stored Q values be used even when they are not in working memory?
- Should \(\lambda_{\mathrm{backup}}\) remain as ancestor-depth learning-rate decay?
- Should unvisited active children contribute their initialized \(Q\) value, a prior, or zero?

## Recommended First Experiment

Implement a new backup mode with two explicit WM cases:

1. Both children active: tree backup under the movement softmax/epsilon policy.
2. Only constructed child active: WM-restricted sampled target, \(r(p) + G(c_1)\).

This avoids the unresolved behavior-policy denominator and makes the cognitive interpretation clear. Then compare against current greedy backup before adding full-policy importance sampling.\(
\)