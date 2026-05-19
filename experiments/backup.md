# Backup Rule Experiment

## Goal

The goal is to replace the current greedy ancestor backup with policy-value backups that better match the internal movement policy while preserving the working-memory interpretation of the environment. The main empirical question is whether backups should estimate the full movement-policy value using stored child values, or a working-memory-limited value using only children currently available in working memory. We postpone weighted sampled updates until the appropriate weighting convention is clearer.

## Candidate Rules

Let \(p\) be an ancestor parent, \(c_1\) the constructed child on the currently looked branch, and \(c_2\) its sibling. Let \(Q_p = Q(p)\), \(r = r(p)\), \(G_1 = G(c_1)\), \(Q_2 = Q(c_2)\), and \(\eta = \alpha w\), where \(w\) is the ancestor backup weight. Backups are asynchronous and bottom-up: \(G_1\) is the updated constructed-branch return available when backing up into \(p\).

When both children are active in working memory, all modes use the same policy/tree backup:

\[
\pi_1, \pi_2 = \operatorname{softmax}_\epsilon([G_1, Q_2])
\]

\[
Q'_p = Q_p + \eta \left[r + \pi_1 G_1 + \pi_2 Q_2 - Q_p\right]
\]

When only the constructed child \(c_1\) is active:

| Mode | New \(Q_p\) value | Note |
|---|---|---|
| `full` | \(Q'_p = Q_p + \eta \left[r + \pi_1 G_1 + \pi_2 Q_2 - Q_p\right]\), where \(\pi = \operatorname{softmax}_\epsilon([G_1, Q_2])\) | Ignores working-memory availability for the sibling; estimates the full movement-policy value. |
| `wm_zero` | \(Q'_p = Q_p + \eta \left[r + \pi^0_1 G_1 - Q_p\right]\), where \(\pi^0 = \operatorname{softmax}_\epsilon([G_1, 0])\) | Policy-value analogue of the current working-memory backup; inactive sibling is zero-filled for both value and policy. |
| `wm_partial` | \(Q'_p = Q_p + \eta \left[r + G_1 - Q_p\right]\) | Keller-style partial backup; renormalizes over available working-memory children. |
| `wm_weighted` | Deferred | Branch-target update with a policy or behavior-derived learning-rate weight; omitted until the weighting convention is specified. |


## Implementation Notes

- `backup_mode` lives in the environment constructor and config static keys.
- Ancestor target logic is centralized in `JaxDecisionTreeEnv._backup_target(state, node, params)`.
- Ancestor parent eligibility is handled in `_update_q`: `full` ignores working-memory activation for parents; the working-memory modes require the parent to be active.
- `backup_steps` remains the ancestor horizon and `lamda_backup` remains the depth-dependent learning-rate decay.
- Existing configs were migrated from `wm_backup` to `backup_mode`. Former `wm_backup = false` configs use `backup_mode = "full"`; former `wm_backup = true` configs use `backup_mode = "wm_zero"`.

## Results

TBD.
