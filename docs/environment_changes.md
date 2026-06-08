# Environment Change Log

Record every intentional change to `modules/environment.py` here, including compatible feature additions, removals, renames, semantic changes, and clarifications. This log is meant to make the evolution of the cognitive architecture easy to inspect without reconstructing it from raw commits.

The environment compatibility version is an integer epoch attached to runs and checkpoint weights. Bump it only when a change to the cognitive architecture or environment semantics makes existing checkpoint weights incompatible. Compatible environment changes stay under the current version.

Use entries in this format:

```md
- <change phrase>; ADR <number>; commits `<sha>`, `<sha>`.
```

Prefer concise change phrases such as "rename `wm_only` to `disable_persistence`", "add `wm_neighbor_activation`", or "remove `best_open_value` and `best_terminal_value` observations".

## Version 1

- Add the initial environment compatibility epoch; commit `5a83026`.
- Factor activation touch-point parameters; ADR 0001; commit `11d174c`.
- Rename `wm_only` to `disable_persistence`; ADR 0002; commits `b13b075`, `ca0c535`.
- Add `global_shared` architecture; ADR 0003; commit `eb91a3a`; checkpoint shape compatibility is distinguished by `network_type`.
- Remove `best_open_value` and `best_terminal_value` observations; ADR 0004; commit `91279dc`.

Current code defines `ENVIRONMENT_COMPAT_VERSION = 1`; do not document a later epoch unless the constant is also bumped and enforced by checkpoint loading.
