# ADR 0004: Remove best-value observations

Date: 2026-06-08

Status: Accepted

## Context

The environment exposed two scalar observation fields:

- `best_open_value`: the largest path-prefix value among known, unvisited nodes.
- `best_terminal_value`: the largest complete path value among visited terminal nodes.

These fields were intended as compact summary statistics over the cognitive architecture's available search state. In practice, their semantics became unclear once persistence and working-memory availability were factored into separate modes.

## Decision

Remove `best_open_value` and `best_terminal_value` from the environment observation, network inputs, and default configuration.

Do not replace them with new scalar observations in the current implementation.

## Rationale

The fields are difficult to implement faithfully across different persistence modes. A scalar summary needs to answer which information is still available, which remembered values persist after leaving working memory, and which forgotten state should be excluded. Those choices are not just implementation details; they define cognitive architecture assumptions.

They also likely do not correspond to iteratively updated summary statistics in the way originally intended. The current implementation recomputed them from node-level state at observation time, which makes them derived views rather than memory variables updated by the architecture.

Non-persistent, working-memory-supported versions can be learned by the `global_shared` architecture from node-level inputs and pooled global context. For example, the model can learn summaries like a maximum over observable terminal-path features. A caveat is that the old terminal value was `g_i + r_i`; an exact learned analogue needs access to the terminal reward/value signal as well as `g_values` and `is_terminal`.

## Consequences

Observation shape no longer depends on `use_best_open_value_obs` or `use_best_terminal_value_obs`.

Historical configs that still specify those flags are stale under the new environment schema and should be updated intentionally before reuse.

If persistent best-value summaries are needed later, add them back as explicit iteratively updated environment state rather than recomputing them as observation-time derived fields.
