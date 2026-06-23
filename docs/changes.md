# Change Log

Record every important result-producing change here. An important change is any intentional change that can affect the result of running train.py, evaluate.py or simulate.py, excluding trivial changes such as using a different random seed. This includes changes to the cognitive architecture, network architecture, observation schema, defaults, training/evaluation semantics, and result-producing scripts. It excludes changes to scripts 

The compatibility version is an integer epoch attached to runs and checkpoint weights. Bump it only when a change makes existing checkpoint weights incompatible with the current code. Compatible changes stay under the current version. Here, "incompatible" means that simulating an old run with the new code would mean evaluating a policy on an environment that is different from the one it was trained on (excluding RNG behavior).

## Version 9

- Exclude invalid child slots from value-backup target support. This fixes terminal
  node backups when `activation_gates_backup_source = true` and
  `excluded_child_value` is numeric: terminal targets are now the node's observed
  reward rather than observed reward plus the substituted child value. Bumped
  `COMPAT_VERSION` 8 -> 9.

## Version 8

- Clear activation for nodes that remain undiscovered after a look update. This
  prevents an already-forgotten parent from being reactivated as a fixation
  neighbor without being rediscovered, preserving the invariant that active
  nodes are discovered. Bumped `COMPAT_VERSION` 7 -> 8.

## Version 6

- Gate a node's own reward in the value backup on whether it has been observed.
  `_backup_target` now contributes `points[node]` only when `n_visits[node] > 0`,
  so a node that is active merely as a neighbor of the fixated node (and was never
  fixated, or has been forgotten) no longer injects its reward into its own Q value.
  The reward thus shares the memory dynamics of `n_visits`: set on fixation, cleared
  by forgetting/corruption and by `disable_persistence` WM eviction.
- Subject `g_values` to forgetting like `n_visits`: forgotten nodes reset their
  `g_values` to `min_path_value` in `_corrupt_memory`, and `_look` re-derives
  children's `g` from the parent's `g`, so forgetting propagates downstream. Bumped
  `COMPAT_VERSION` 5 -> 6.

## Version 5

- Store incremental remembered path values online. Undiscovered nodes now keep the
  initial minimum path value until their parent is fixated and discovers them.
- Apply memory corruption after working-memory activation refresh and before the
  current fixation's learning updates; reset skips the initial corruption event;
  ADR 0006.
- Add `activation_prevents_corruption`, defaulting to true. When false,
  corruption applies to all discovered node-specific memory, including nodes
  currently active in working memory.
- Fix `generate_wm_decay_backtrack_rollouts.py` to pass the full environment
  static parameter set, including `activation_prevents_corruption`.

## Version 4

- Add discovered-node state to the cognitive architecture. When
  `activation_masks_observation = false`, observations now expose discovered nodes rather
  than all nodes, and inactive-memory corruption only applies to discovered nodes.

## Version 3

- Replace `activation_protects_memory` with inverted `disable_corruption`, remove `persist_terminal`, and restore the previous semantics where disabled corruption also keeps terminal memory persistent; ADR 0005 is reverted/superseded.

## Version 2

- Add `persist_terminal` and change `activation_protects_memory = false` to corrupt node-specific memory regardless of activation; ADR 0005. Reverted/superseded in Version 3.
- Remove `best_open_value` and `best_terminal_value` observations from the environment, network inputs, defaults, and downstream helpers; ADR 0004; commit `91279dc`.
- Treat the root as an ordinary node for working-memory activation, observation masking, memory protection, and value-backup gating, while keeping it always legal as a fixation action before timeout; commit `1795937`.

## Version 1

- Apply `q_drift` noise during the movement phase; commit `4877f66`.
- Update `forget_rate` to clear inactive node memory, including fixation recency and terminal memory; commit `0c3184d`.
- Add `fixation_recency` and `is_terminal` to detailed simulation outputs; commit `271c5da`.
- Add `move_cost_scale` path-length movement penalty parameter; commit `b091c46`.
- Add the initial compatibility epoch; commit `5a83026`.
- Factor activation touch-point parameters; ADR 0001; commit `11d174c`.
- Rename `wm_only` to `disable_persistence`; ADR 0002; commits `b13b075`, `ca0c535`.
- Add `global_shared` architecture; ADR 0003; commit `eb91a3a`; checkpoint shape compatibility is distinguished by `network_type`.
