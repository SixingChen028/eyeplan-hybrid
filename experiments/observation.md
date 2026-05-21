# Observation ablations

## Goal

Determine which observation components are actually needed for strong model performance and for predicting human behavior. The observation space is too large for an exhaustive component search, so this experiment should use targeted ablations that are interpretable and cheap enough to run across the same evaluation pipeline as the main working-memory models.

The main comparison should answer two questions:

1. Which individual observation components cause performance or human-prediction quality to drop when removed?
2. Can the model perform well when given only the information that is plausibly necessary for the current fixation decision, excluding extra state summaries such as remembered Q values?

## Current observation

For `network_type = "node_shared"`, the per-node observation for each candidate node `S` is:

| Component | Description from `node_shared` perspective | Flag |
| --- | --- | --- |
| `fixation` | Is `S` currently fixated? | Always included |
| `parent` | Is `S` the parent of the fixated node? | Always included |
| `child` | Is `S` a child of the fixated node? | Always included |
| `root` | Is `S` the root node? | Always included |
| `g_values` | Sum of rewards up to `S`, if `S` is known; otherwise `0` | `use_g_values_obs` |
| `q_values` | Remembered value estimate for `S` | `use_q_values_obs` |
| `n_visits` | Number of times `S` has been fixated | `use_n_visits_obs` |
| `is_terminal` | Is `S` a seen terminal node? | `use_is_terminal_obs` |
| `recency` | How recently was `S` fixated? | `use_recency_obs` |
| `legal_feature` | Is fixation on `S` currently legal? | Always included |

The `node_shared` network also receives these global observation components:

| Component | Description | Flag |
| --- | --- | --- |
| `fixation_point` | Reward at the currently fixated node | Always included |
| `time_elapsed` | Current elapsed step count | `use_time_elapsed_obs` |
| `best_open_value` | Best known sum of rewards up to an unvisited but reachable node | `use_best_open_value_obs` |
| `best_terminal_value` | Best observed complete path reward among seen terminal nodes | `use_best_terminal_value_obs` |

The always-included structural components define what is currently available in working memory: the fixation, its local tree neighborhood, the root marker, and the legal action mask. The ablations below focus on optional components because they already have configuration flags and can be varied without changing network code.

## Baseline

Use the best current `node_shared` configuration as the reference condition. Keep the training budget, seeds, environment settings, and evaluation scripts fixed across all observation variants.

The reference observation should include every currently enabled optional component:

```toml
network_type = "node_shared"
use_recency_obs = true
use_best_open_value_obs = true
use_best_terminal_value_obs = true
use_g_values_obs = true
use_q_values_obs = true
use_n_visits_obs = true
use_is_terminal_obs = true
use_time_elapsed_obs = true
```

If the canonical run for the surrounding experiment has `use_recency_obs = false`, run both the canonical baseline and the full-observation baseline above. Recency is part of the observation question even if it is not part of the current default.

## Targeted ablations

### Leave-one-out

Run one ablation per optional component, setting exactly one flag to `false` and leaving every other optional observation enabled:

| Condition | Changed flag | Interpretation |
| --- | --- | --- |
| `no_g_values` | `use_g_values_obs = false` | Tests whether explicit known path totals are needed, or whether the policy can infer enough from fixation history and current reward. |
| `no_q_values` | `use_q_values_obs = false` | Tests whether remembered value estimates are necessary as observations, rather than only as hidden environment state that shapes backup dynamics. |
| `no_n_visits` | `use_n_visits_obs = false` | Tests whether the policy needs explicit revisit counts. |
| `no_is_terminal` | `use_is_terminal_obs = false` | Tests whether the policy needs an explicit marker for observed terminal nodes. |
| `no_recency` | `use_recency_obs = false` | Tests whether short-term fixation history beyond `n_visits` matters. |
| `no_time_elapsed` | `use_time_elapsed_obs = false` | Tests whether decisions depend on an explicit clock rather than local state and legal actions. |
| `no_best_open_value` | `use_best_open_value_obs = false` | Tests whether a global summary of the best reachable unvisited path is useful. |
| `no_best_terminal_value` | `use_best_terminal_value_obs = false` | Tests whether a global summary of the best observed complete path is useful. |

The most important planned contrast is `no_q_values` versus baseline. If performance and human-prediction quality remain stable without Q values in the observation, then Q values are better interpreted as part of the cognitive architecture's persistent-memory dynamics rather than information directly available to the policy.

### Remove extraneous summaries

Run a minimal-observation condition that removes components that summarize latent state, accumulated history, or computed value information beyond the current local working-memory view:

```toml
use_recency_obs = false
use_best_open_value_obs = false
use_best_terminal_value_obs = false
use_g_values_obs = false
use_q_values_obs = false
use_n_visits_obs = false
use_is_terminal_obs = false
use_time_elapsed_obs = false
```

This leaves only:

- Current fixation indicator.
- Parent and children of the currently fixated node.
- Root indicator.
- Legal action indicator.
- Reward at the currently fixated node.

This is the strongest test of whether the policy can operate from the local working-memory state alone. It intentionally excludes Q values, global best-value summaries, visit counts, terminal markers, elapsed time, recency, and known path totals.

### Value-only and history-only follow-ups

If the leave-one-out and minimal-observation runs suggest that multiple components matter, use two grouped follow-ups before trying more combinations:

| Condition | Enabled optional components | Purpose |
| --- | --- | --- |
| `value_summaries_only` | `g_values`, `q_values`, `best_open_value`, `best_terminal_value` | Tests whether value information alone explains most of the baseline gain. |
| `history_markers_only` | `n_visits`, `is_terminal`, `recency`, `time_elapsed` | Tests whether history and progress markers explain most of the baseline gain. |

These two follow-ups are not a full factorial search. They are meant to localize any large effect seen in the leave-one-out runs.

## Metrics

Use the same metrics for every condition:

- Training reward and final evaluation reward.
- Human behavior prediction quality on held-out trials.
- Action-level likelihood or cross-entropy, if available.
- Agreement with human fixation transitions, especially inspect-parent, inspect-child, revisit, and terminate decisions.
- Parameter count and observation dimensionality, so performance changes can be interpreted relative to information removal rather than only network capacity.

Report each condition as a delta from the baseline, not only as an absolute score. A compact summary table should include mean, standard error or bootstrap interval, and the number of seeds.

## Run design

Use the same seeds for all observation variants. Prioritize enough seeds to distinguish a real ablation effect from training variance. If compute is limited, use a two-stage design:

1. Run all leave-one-out and minimal-observation conditions with a small seed set.
2. Re-run the baseline, `no_q_values`, minimal-observation, and any large-effect ablations with the full seed set.

The first stage identifies candidates; the second stage estimates the effect sizes that are most likely to matter for the paper.

## Interpretation

Interpret a component as important only if its removal affects either final task performance or held-out human-prediction quality consistently across seeds. Components that improve reward but worsen human prediction should be treated as computational conveniences rather than psychologically meaningful observations.

The minimal-observation condition is the key boundary case. If it performs well and predicts humans well, the model does not need the extraneous observation summaries. If it performs poorly, use the leave-one-out and grouped follow-ups to identify the smallest set of additional observation dimensions that recovers baseline behavior.
