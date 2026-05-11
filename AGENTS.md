# Project Instructions

## Purpose

This repository defines the optimal/rational model for a larger project studying
how working memory constraints shape human planning strategies.

In this codebase, the "environment" should be understood as a cognitive
architecture. It is internal to the agent, but it is not under the agent's
direct control: the model can choose how to deliberate, but it cannot choose to
increase its working memory capacity or reduce the cost of deliberation.

The core assumptions of the cognitive architecture are:

- Operations can only be performed on information currently available in working
  memory.
- Information in working memory decays rapidly.
- Value information can be stored in and retrieved from persistent memory.

Keep this framing explicit when naming concepts, writing comments, adding config
options, or documenting behavior. Avoid language that implies the environment is
an external task world unless that distinction is intentional.

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

## Implementation Guidelines

- Write simple, concise, modular Python.
- Prefer existing local patterns over new abstractions.
- Keep JAX code compatible with JIT/vectorized execution where surrounding code
  expects it.
- Treat configuration names as part of the experiment record. Rename or remove
  config fields only when the corresponding migration/update is intentional.
- Do not add compatibility branches for outdated conventions unless the
  user explicitly asks for backward compatibility.
