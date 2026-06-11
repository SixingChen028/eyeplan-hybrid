# Change Log

Record every important result-producing change here. An important change is any intentional change that can affect the result of running training, evaluation, or simulation, excluding trivial changes such as using a different random seed. This includes changes to the cognitive architecture, network architecture, observation schema, defaults, training/evaluation semantics, and result-producing scripts.

The compatibility version is an integer epoch attached to runs and checkpoint weights. Bump it only when a change makes existing checkpoint weights incompatible with the current code. Compatible changes stay under the current version.

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
Current code defines `COMPAT_VERSION = 4`; do not document a later epoch unless the constant is also bumped and enforced by checkpoint loading.
