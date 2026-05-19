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

## Implementation Sketch

Add a backup mode parameter with three active values:

```toml
backup_mode = "wm_zero"  # one of "full", "wm_zero", "wm_partial"
```

`wm_zero` should reproduce the current behavior in the hard-policy limit where \(\beta_{\mathrm{move}}\) is large and \(\epsilon_{\mathrm{move}} = 0\). It should replace the existing greedy max target with a policy expectation over the zero-filled child values.

The likely implementation path is:

1. Add `backup_mode` to config defaults, config validation, sweep expansion, and environment construction.
2. Keep `wm_backup` temporarily only if needed for backward compatibility with existing configs; otherwise replace it with `backup_mode`.
3. Factor ancestor backup target computation into a helper that receives the parent, constructed child, current `q_values`, activation, points, and params.
4. Use the existing async bottom-up ordering: update the fixation node first, then back up through ancestors using the latest `q_values`.
5. For both-active children, call the common policy/tree target for every mode.
6. For one-child-active parents:
   - `full`: include the inactive sibling's stored \(Q_2\) in both the softmax and target.
   - `wm_zero`: use zero for the inactive sibling in both the softmax and target.
   - `wm_partial`: use only the constructed child target \(r + G_1\).
7. Preserve `backup_steps` as the ancestor horizon and `lamda_backup` as the depth-dependent learning-rate decay.
8. Add focused tests for the one-child case of each mode, plus a both-active test showing the modes agree.

Open implementation decision:

- Whether `backup_mode` should fully replace `wm_backup` now, or whether configs should temporarily map `wm_backup = false` to `full` and `wm_backup = true` to `wm_zero`.

## Results

TBD.
