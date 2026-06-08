# Environment Change Tracker

The environment compatibility version is an integer epoch. Bump it only when a change to the cognitive architecture or environment semantics makes existing checkpoint weights incompatible.

Compatible environment feature additions and clarifications should be recorded under the current version without bumping the epoch.

## Version 1

- Initial environment compatibility epoch, introduced in `5a83026`.
- Includes the activation touch-point parameters from ADR 0001 as implemented in `11d174c`.
- Includes the `wm_only` to `disable_persistence` rename from ADR 0002 as implemented in `b13b075` and `ca0c535`.
- Includes the `global_shared` architecture from ADR 0003 as implemented in `eb91a3a`; checkpoint shape compatibility is distinguished by `network_type`.
- Includes removal of `best_open_value` and `best_terminal_value` observations from ADR 0004 as implemented in `91279dc`.

Current code defines `ENVIRONMENT_COMPAT_VERSION = 1`; do not document a later epoch unless the constant is also bumped and enforced by checkpoint loading.
