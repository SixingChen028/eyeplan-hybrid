# Observation ablations

## Goal

Determine which observation components are actually needed for strong model performance and for predicting human behavior. The observation space is too large for an exhaustive component search, so this experiment uses a small set of cumulative conditions that add optional components in interpretable stages.


## Current observation

For `network_type = "node_shared"`, the per-node observation for each candidate node `S` is:

| Component        | Description                                                      |
| ---------------- | ---------------------------------------------------------------- |
| `fixation`       | Is `S` currently fixated?                                        |
| `parent`         | Is `S` the parent of the fixated node?                           |
| `child`          | Is `S` a child of the fixated node?                              |
| `root`           | Is `S` the root node?                                            |
| `g_values`       | Sum of rewards up to `S`, if `S` is known; otherwise `0`         |
| `q_values`       | Remembered value estimate for `S`                                |
| `n_visits`       | Number of times `S` has been fixated                             |
| `is_terminal`    | Is `S` a seen terminal node?                                     |
| `recency`        | How recently was `S` fixated?                                    |
| `legal_feature`  | Is fixation on `S` currently legal?                              |

The `node_shared` network also receives these global observation components.

| Component             | Description                                           |
| --------------------- | ------------------------------------------------------|
| `fixation_point`      | Reward at the currently fixated node                  |
| `time_elapsed`        | Current elapsed step count                            |
| `best_open_value`     | Best sum of rewards up to an unvisited but known node |
| `best_terminal_value` | Best observed complete path reward                    |

The always-included structural components define what is currently available in working memory: the fixation, its local tree neighborhood, the root marker, and the legal action mask. The ablations below focus on optional components because they already have configuration flags and can be varied without changing network code.

## Config

Use `config/0521_obs.toml`. Shared training settings are fixed across all observation variants:

```toml
backup_mode = "wm_partial"
wm_decay = [0, 0.5, 1]
cost = [0.005, 0.01, 0.02]
seed = [1, 2]
beta_e_final = 0.01
num_envs = 64
num_updates = 30_000
network_type = "node_shared"
hidden_size = 64
```

## Conditions

All runs inherit the same base observation flags. Every optional component starts disabled except `use_time_elapsed_obs`, which stays enabled in the shared params. This base leaves only the immediate observation (reward and neighbors) and a clock.

Each run adds optional components on top of that base:

- value: q_values
- mcts: q_values, n_visits
- basic: q_values, n_visits, g_values (roughtly LRTA*)
- everything: duh

## Results

### 0521_obs

http://localhost:5173/eyeplan/v10/ablations?minimal=true

- performance
  - everything is slightly better than basic
  - value and mcts are substantially worse, especially for low cost
  - value suffers a lot when wm is non-perfect
- reward: only value gets it; basic is non-monotonic (BUT analysis might be wrong)
- saccade types
  - value and mcts (no g) miss siblings entirely
  - basic overestimates parent, underestimates root (BUT check other wm values)
  - **BIG** mcts and basic nail the reward effect, cross over at 0
- action value
  - only everything gets it; value gets action value but not future (interaction)
  - basic is marginal
- nfix and seen
  - **BIG** value predicts strong positive effect but still gets seen vs unseen (quite close)

Conclusions
- taking out or weakening N might help a lot (from which_child/n_fix)
- we should also explore space between basic and everything with better decay values

Remaining questions:
1. which "extra" components in everything (vs. basic) have any impact?
2. which ones are important for the reward/crossover effect
3. how to explain value wins? can we just take out n_visits?

### 0521_obs2

We explore the space between basic and everything by adding one at a time. This targets questions 1 and 2 above.

Very little was learned here.

### 0521_obs3

Opposite approach: start with everything and *remove* one at a time.

drop open_value
drop terminal_value
mb keep recency ; dropping recency hurdes type~reward plot
dropping is_terminal hurts nfix-seen a lot; also future value
