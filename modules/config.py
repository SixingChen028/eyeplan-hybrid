from __future__ import annotations

import itertools
import tomllib
from pathlib import Path

DEFAULTS_PATH = Path(__file__).resolve().parents[1] / "config" / "_DEFAULTS.toml"

PARAM_DEFAULTS = {
    "environment": {
        # Number of nodes in each generated decision tree.
        "num_nodes": 15,
        # Maximum number of environment steps per episode.
        "t_max": 50,
        # Multiplier applied to raw point rewards before returning environment rewards.
        "scale_factor": 0.125,
        # Whether to randomly permute node labels for each generated tree.
        "shuffle_nodes": True,
        # Whether observations include per-node fixation recency values.
        "use_recency_obs": True,
        # Whether value backups use only active working-memory nodes.
        "wm_backup": True,
        # Inverse temperature for softmax move probabilities in environment dynamics.
        "beta_move": 40.0,
        # Uniform random-move mixture rate in environment dynamics.
        "eps_move": 0.0,
        # Environment Q-value update step size.
        "learning_rate": 1.0,
        # Decay factor for ancestor value backups.
        "lamda_backup": 1.0,
        # Maximum number of ancestor levels updated during value backup.
        "backup_steps": 100,
        # Per-step decay applied to working-memory activation.
        "wm_decay": 1.0,
        # Probability of clearing inactive Q-values after each step.
        "q_drop_rate": 0.0,
        # Standard deviation of Gaussian drift added to inactive Q-values.
        "q_drift": 0.0,
        # Per-step decay applied to inactive Q-values.
        "q_decay": 1.0,
        # Per-step decay applied to fixation recency observations.
        "recency_decay": 0.5,
        # Per-step movement cost subtracted from environment reward.
        "cost": 0.01,
    },
    "training": {
        # Optimizer learning rate for A2C training.
        "lr": 0.0005,
        # Discount factor used for return estimation.
        "gamma": 1.0,
        # GAE lambda used for advantage estimation.
        "lamda": 0.8,
        # Coefficient for the value loss term.
        "beta_v": 0.05,
        # Initial entropy coefficient at the start of training.
        "beta_e_init": 0.02,
        # Final entropy coefficient at the end of training.
        "beta_e_final": 0.001,
        # Global gradient norm clipping threshold.
        "max_grad_norm": 2.0,
        # Number of A2C optimization updates to run.
        "num_updates": 50000,
        # Number of parallel training environments.
        "num_envs": 256,
        # Number of environment steps collected per update.
        "rollout_length": 50,
        # Number of evaluation episodes run after training.
        "eval_episodes": 102400,
        # Number of updates between progress prints.
        "print_frequency": 100,
        # Maximum update chunk size to compile; non-positive uses the requested chunk size.
        "max_compiled_updates_per_chunk": -1,
    },
    "network": {
        # Policy/value network architecture identifier.
        "network_type": "mlp",
        # Hidden layer width for network architectures that use dense hidden layers.
        "hidden_size": 256,
    },
    "meta": {
        # Random seed for a training run.
        "seed": 15,
        # Directory where run outputs are written.
        "result_path": "./results",
        # Experiment name; defaults to the config file stem when omitted.
        "experiment": None,
        # Optional sbatch array axes selected from sweep parameters.
        "array_vars": None,
    },
}

ENV_DYNAMIC_PARAM_KEYS = (
    "beta_move",
    "eps_move",
    "learning_rate",
    "lamda_backup",
    "backup_steps",
    "wm_decay",
    "q_drop_rate",
    "q_drift",
    "q_decay",
    "recency_decay",
    "cost",
)
ENV_STATIC_PARAM_KEYS = tuple(
    key for key in PARAM_DEFAULTS["environment"] if key not in ENV_DYNAMIC_PARAM_KEYS
)
TRAIN_SWEEP_KEYS = (
    "lr",
    "gamma",
    "lamda",
    "beta_v",
    "beta_e_init",
    "beta_e_final",
    "max_grad_norm",
)
MODEL_SHAPE_PARAM_KEYS = (
    "network_type",
    "hidden_size",
    "num_updates",
    "num_envs",
    "rollout_length",
    "eval_episodes",
    "max_compiled_updates_per_chunk",
)
RUN_PARAM_KEYS = (
    "seed",
    "print_frequency",
)

REQUIRED_PARAM_KEYS = (
    *RUN_PARAM_KEYS,
    *ENV_STATIC_PARAM_KEYS,
    *ENV_DYNAMIC_PARAM_KEYS,
    *MODEL_SHAPE_PARAM_KEYS,
    *TRAIN_SWEEP_KEYS,
)

ENV_SWEEP_KEYS = set(ENV_DYNAMIC_PARAM_KEYS)
SWEEP_KEYS = ENV_SWEEP_KEYS | set(TRAIN_SWEEP_KEYS) | {"seed"}
SHAPE_KEYS = set(ENV_STATIC_PARAM_KEYS) | set(MODEL_SHAPE_PARAM_KEYS)


def load_canonical_defaults(defaults_path: Path = DEFAULTS_PATH) -> tuple[dict, dict]:
    if not defaults_path.exists():
        raise FileNotFoundError(f"Canonical defaults file not found: {defaults_path}")
    with defaults_path.open("rb") as file:
        defaults = tomllib.load(file)

    meta = defaults.get("meta")
    params = defaults.get("params")
    if not isinstance(meta, dict):
        raise ValueError("config/_DEFAULTS.toml must include a [meta] table.")
    if not isinstance(params, dict):
        raise ValueError("config/_DEFAULTS.toml must include a [params] table.")

    missing_keys = [key for key in REQUIRED_PARAM_KEYS if key not in params]
    if missing_keys:
        raise ValueError(
            "config/_DEFAULTS.toml is missing required [params] keys: "
            + ", ".join(sorted(missing_keys))
        )

    return dict(meta), dict(params)


DEFAULT_META, DEFAULT_PARAMS = load_canonical_defaults()


def load_config(path: str) -> tuple[Path, dict]:
    config_path = Path(path)
    if not config_path.exists() and config_path.suffix != ".toml":
        candidate = Path("config") / f"{path}.toml"
        if candidate.exists():
            config_path = candidate
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with config_path.open("rb") as file:
        return config_path, tomllib.load(file)


def is_list(value) -> bool:
    return isinstance(value, list)


def parse_unit_interval(value, *, name: str) -> float:
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError as error:
            raise ValueError(f"{name} must be a number in [0, 1].") from error
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} numeric values must satisfy 0 <= {name} <= 1.")
    return value


def validate_params(params: dict) -> None:
    unknown_keys = sorted(set(params) - set(DEFAULT_PARAMS))
    if unknown_keys:
        raise ValueError("Unknown [params] keys: " + ", ".join(unknown_keys))

    for key, value in params.items():
        if not is_list(value):
            continue
        if len(value) == 0:
            raise ValueError(f"params.{key} must not be an empty array.")
        if key in SHAPE_KEYS:
            raise ValueError(
                f"params.{key} cannot be an array in train.py because it changes compiled shapes."
            )
        if key not in SWEEP_KEYS:
            raise ValueError(f"params.{key} is not a supported parallel sweep parameter.")

    for key in ("recency_decay", "q_decay"):
        value = params.get(key, 0.0)
        values = value if is_list(value) else [value]
        for item in values:
            parse_unit_interval(item, name=key)


def expand_sweep(params: dict) -> tuple[dict, list[dict], list[str]]:
    unknown_keys = sorted(set(params) - set(DEFAULT_PARAMS))
    if unknown_keys:
        raise ValueError("Unknown [params] keys: " + ", ".join(unknown_keys))

    merged = dict(DEFAULT_PARAMS)
    merged.update(params)
    validate_params(merged)

    sweep_items = [
        (key, value)
        for key, value in merged.items()
        if is_list(value)
    ]
    fixed = {
        key: value
        for key, value in merged.items()
        if not is_list(value)
    }

    if not sweep_items:
        combos = [dict(fixed)]
        combos[0]["seed"] = int(combos[0]["seed"])
        return fixed, combos, []

    varied_keys = [key for key, _ in sweep_items]
    combos: list[dict] = []
    for values in itertools.product(*(value for _, value in sweep_items)):
        combo = dict(fixed)
        combo.update(dict(zip(varied_keys, values)))
        combo["seed"] = int(combo["seed"])
        combos.append(combo)
    return fixed, combos, varied_keys


def resolve_training_geometry(params: dict) -> tuple[int, int, int]:
    if "num_updates" in params and "num_envs" in params and "rollout_length" in params:
        return int(params["num_updates"]), int(params["num_envs"]), int(params["rollout_length"])

    raise ValueError(
        "Training geometry must be specified as num_updates + num_envs + rollout_length."
    )


def parse_cli_value(raw: str, template_value):
    if isinstance(template_value, bool):
        lowered = raw.strip().lower()
        if lowered in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "f", "no", "n", "off"}:
            return False
        raise ValueError(f"Invalid boolean value: {raw!r}")
    if isinstance(template_value, int) and not isinstance(template_value, bool):
        return int(raw)
    if isinstance(template_value, float):
        return float(raw)
    return raw


def apply_cli_param_overrides(params: dict, override_tokens: list[str]) -> dict:
    if not override_tokens:
        return dict(params)

    unknown_keys = sorted(set(params) - set(DEFAULT_PARAMS))
    if unknown_keys:
        raise ValueError("Unknown [params] keys: " + ", ".join(unknown_keys))

    merged = dict(DEFAULT_PARAMS)
    merged.update(params)
    updated = dict(params)
    pairs: list[tuple[str, str]] = []
    idx = 0
    while idx < len(override_tokens):
        token = override_tokens[idx]
        if not token.startswith("--"):
            raise ValueError(f"Invalid override key {token!r}; expected '--<name>'.")
        if "=" in token:
            key_token, value_token = token.split("=", 1)
            if value_token == "":
                raise ValueError(f"Missing value for override {key_token!r}.")
            pairs.append((key_token, value_token))
            idx += 1
            continue
        if idx + 1 >= len(override_tokens):
            raise ValueError("Parameter overrides must be provided as '--key value' or '--key=value'.")
        pairs.append((token, override_tokens[idx + 1]))
        idx += 2

    for key_token, value_token in pairs:
        if not key_token.startswith("--"):
            raise ValueError(f"Invalid override key {key_token!r}; expected '--<name>'.")
        key = key_token[2:]
        if key not in merged:
            raise ValueError(f"Unknown parameter override: {key}")
        updated[key] = parse_cli_value(value_token, merged[key])
    return updated
