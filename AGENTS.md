# Project Instructions

## Rules

All changes to environment.py must be directly instructed or approved by the user. If
making the requested change requires (or motivates) an additional change, stop and ask for
permission.

Questions about how the environment or its parameters work must be answered with the
utmost care. The environment dynamics involve complex interactions between parameters.
Think very carefully about your answers to such questions. Ruthlessly check your own
assumptions and clearly indicate them to the user.

## Purpose

This repository defines the optimal/rational model for a larger project studying
how working memory constraints shape human planning strategies.

In this codebase, the "environment" should be understood as a cognitive
architecture. It is internal to the agent, but it is not under the agent's
direct control. The core assumptions of the cognitive architecture are:

- Operations can only be performed on information currently available in working memory.
- Information in working memory decays rapidly.
- Value information can be stored in and retrieved from persistent memory.

Keep this framing explicit when naming concepts, writing comments, adding config
options, or documenting behavior. Avoid language that implies the environment is
an external task world unless that distinction is intentional. 
Note that WM stands for working memory (e.g. `wm_decay`). 

## Repository Layout

- `modules/` contains the main Python implementation, including the JAX decision
  tree environment, model code, baselines, config helpers, simulation utilities,
  and result layout helpers.
- `config/` contains TOML experiment configurations.
- `tests/` contains pytest coverage for the environment, tree generation,
  training pipeline, baselines, and script helpers.
- Top-level scripts such as `train.py`, `simulate.py`, `plot_training.py`, and
  `generate_sbatch.py` are command-line entry points.

## Running code

Use .venv/bin/python if available. Next, try /Users/fred/.venvs/default/bin/python3.
Fall back to creating a new environment as described below.

## Setup

Use Python 3.12. The project setup script creates a local virtual environment
and installs dependencies. 

```sh
./install_deps.sh
```

For GPU JAX support (only if user requests)

```sh
./install_deps.sh --gpu
```

## Verification

Run tests from the repository root:

```sh
pytest
```

For focused changes, run the smallest relevant pytest target first, then run the
full suite before committing when the change affects shared model behavior.

Don't add tests for minor changes. Instead, test the behavior by running bespoke
code at change time. Changes to environment.py should almost always have a
new/updated test.

## Implementation Guidelines

- Write simple, concise, modular Python.
- Prefer existing local patterns over new abstractions.
- Factor shared control flow and data preparation into one path when cases only
  differ in small decisions. Branch as close as possible to the actual
  difference instead of duplicating whole helper functions or dispatch layers.
- Keep JAX code compatible with JIT/vectorized execution where surrounding code
  expects it.
- Do not edit files under `config/` unless instructed to. If edits might be
  necessary, offer it as a suggestion.
- Do not add compatibility branches for outdated conventions unless the
  user explicitly asks for backward compatibility. This includes config variables and
  files.
- If a change makes previous config files or runs invalid, bump `COMPAT_VERSION`. This
  includes changes that make the environment or network behave differently, even if the
  checkpointed parameters could run without errors. It thus includes most changes to
  `network.py` or `environment.py`. Exceptions are refactors and *optional* new
  functionality that does not change default behavior.

## Project memory

Before substantive work:

1. Read `/Users/fred/lib/obsidian/vault/Projects/EyePlan/EyePlan.md`.
2. Run the discovery command below.
3. Open only the records relevant to the current task.

```bash
/Users/fred/lib/obsidian/vault/Scripts/project-memory EyePlan
```

Consult `/Users/fred/lib/obsidian/vault/Meta/Projects.md` before creating or updating project memory. After substantive work, update the relevant project Task. Keep repository-local technical documentation here and link it from the vault when it matters across repositories.
