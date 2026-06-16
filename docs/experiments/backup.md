# Backup Rule Experiment

## Goal

The goal is to replace the current greedy ancestor backup with policy-value backups that better match the internal movement policy while preserving the working-memory interpretation of the environment. The main empirical question is whether backups should estimate the full movement-policy value using stored child values, or a working-memory-limited value using only children currently available in working memory. We postpone weighted sampled updates until the appropriate weighting convention is clearer.

## Candidate Rules

Let p be an ancestor parent, c1 the constructed child on the currently looked branch, and c2 its sibling. Let Q_p = Q(p), r = r(p), G_1 = G(c1), Q_2 = Q(c2), and eta = alpha * w, where w is the ancestor backup weight. Backups are asynchronous and bottom-up: G_1 is the updated constructed-branch return available when backing up into p.

With the exception of `full`, a node's value is only updated if it is active in working memory.
All rules use the `full` backup when both children are in working memory (and the node is eligible for backup).

### `full`

Fully ignores working-memory availability for ancestor backup eligibility, sibling values, and policy calculation. This is the unconstrained policy-value backup.

    pi_1, pi_2 = softmax_epsilon([G_1, Q_2])
    Q'_p = Q_p + eta * (r + pi_1 * G_1 + pi_2 * Q_2 - Q_p)

### `wm_both`

    pi = softmax_epsilon([G_1, Q_2]).
    Q'_p = Q_p + eta * (r + pi_1 * G_1 + pi_2 * Q_2 - Q_p)

Ignores working-memory availability for the sibling when computing the target, but still requires the ancestor parent to be active in working memory before backing up into it. This estimates the full movement-policy value conditional on the parent being available in working memory.

### `wm_zero`

    pi = softmax_epsilon([G_1, 0]).
    Q'_p = Q_p + eta * (r + pi_1 * G_1 - Q_p)

Policy-value analogue of the current working-memory backup; inactive sibling is zero-filled for both value and policy.

### `wm_partial`

    Q'_p = Q_p + eta * (r + G_1 - Q_p)

Keller-style partial backup; renormalizes over available working-memory children.

### `wm_weighted` (TODO)

    pi_1 = ???
    Q'_p = Q_p + eta * pi_1 * (r + G_1 - Q(p))

Branch-target update with a policy-derived learning-rate weight. The weight can be derived as an importance-sampling weight (rho) where the behavior policy is assumed to always visit the fixated child. This isconsistent with the artificial construction of an action history that leads to the child (see sigma_backup.md). The key question is how pi_1 is determined. The naive solution requires access to both Q values, which violates the WM constraint.

## Implementation Notes

- `backup_mode` lives in the environment constructor and config static keys.
- Ancestor target logic is centralized in `DecisionTreeEnv._backup_target(state, node, params)`.
- Ancestor parent eligibility is handled in `_update_q`: `full` ignores working-memory activation for parents; the working-memory modes (`wm_both`, `wm_zero`, and `wm_partial`) require the parent to be active.
- `backup_steps` remains the ancestor horizon and `lamda_backup` remains the depth-dependent learning-rate decay.
- Existing configs were migrated from `wm_backup` to `backup_mode`. Former `wm_backup = false` configs use `backup_mode = "full"`; former `wm_backup = true` configs use `backup_mode = "wm_zero"`.

## Results

commit: 15dbe9f
config: pareto_backup
```py
backup_mode = ["full", "wm_both", "wm_zero", "wm_partial"]
wm_decay = [0.0, 0.5, 0.75, 1.0]
cost = [0.001, 0.01, 0.02, 0.04, 0.08, 0.16]
```

### Performance

![](/Users/fred/projects/eyeplan/analysis/r/results/figs/misc/pareto/pareto_backup.png)

**All methods should perform the same when wm_decay is 1. Do they?**

YES

**full and wm_both should perform better when decay < 1. Do they?**

YES. But the difference is not enormous.

**How do wm_zero and wm_partial compare?**

wm_zero is better, especially with fast decay and high cost

### Fit to human data

pareto_backup results are unclear. `full` underpredicts parent saccades, but the others over-predict parent-parent and don't predict the negative slope for reward on parent

http://localhost:5173/eyeplan/v10/backup_neigh100

- partial doesn't get reward effect
- full under-predicts parent
- non-full barely captures negative slope for parent by reward
- only partial gets action value
  - uniquely captures rev fix interaction
  - none of the models do very well on future value, but partial is best
- partial gets last_look-reward best; both and zero do okay
- partial almost gets flat n_fix in the both-seen case
- partial does slightly worse on nfix-value
- none of the models get reward-simple_type very well
  - cross over is always far too low (-20 vs 0 in human)

### Conclusion

Use partial as default. However, keep in mind that it might be doing better primarily because it provides a source of noise and a reason to double check the path before committing to it. We shoult revisit the choice of backup rule when we add other sources of noise, e.g. q_decay.