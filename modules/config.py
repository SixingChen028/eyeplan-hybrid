from __future__ import annotations

import itertools
import tomllib
from pathlib import Path

PARAM_CLASSES = ("environment", "training", "network", "meta")
PARAM_RUNTIME_CLASSES = ("environment", "training", "network")

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
        # Whether observations include best-open-path scalar value.
        "use_best_open_value_obs": True,
        # Whether observations include best-seen-terminal scalar value.
        "use_best_terminal_value_obs": True,
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
        # Set of points to sample from for each node.
        "point_set": (-8, -4, -2, -1, 1, 2, 4, 8),
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
        # Random seed for a training run.
        "seed": 15,
    },
    "network": {
        # Policy/value network architecture identifier. mlp|node_shared
        "network_type": "mlp",
        # Hidden layer width for network architectures that use dense hidden layers.
        "hidden_size": 256,
    },
    "meta": {
        # Directory where run outputs are written.
        "result_path": "./results",
        # Experiment name; defaults to the config file stem when omitted.
        "experiment": None,
        # Optional sbatch array axes selected from sweep parameters.
        "array_vars": None,
        # Number of updates between progress prints.
        "print_frequency": 100,
        # Maximum update chunk size to compile; non-positive uses the requested chunk size.
        "max_compiled_updates_per_chunk": -1,
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
    "seed",
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
)
REQUIRED_PARAM_KEYS = (
    *ENV_STATIC_PARAM_KEYS,
    *ENV_DYNAMIC_PARAM_KEYS,
    *MODEL_SHAPE_PARAM_KEYS,
    *TRAIN_SWEEP_KEYS,
)

SWEEP_KEYS = set(ENV_DYNAMIC_PARAM_KEYS) | set(TRAIN_SWEEP_KEYS)
SWEEP_KEY_ORDER = ("seed", *ENV_DYNAMIC_PARAM_KEYS, *(key for key in TRAIN_SWEEP_KEYS if key != "seed"))
SHAPE_KEYS = set(ENV_STATIC_PARAM_KEYS) | set(MODEL_SHAPE_PARAM_KEYS)


def _flatten_defaults(param_classes: tuple[str, ...]) -> dict:
    return {
        key: value
        for param_class in param_classes
        for key, value in PARAM_DEFAULTS[param_class].items()
    }


DEFAULT_META = dict(PARAM_DEFAULTS["meta"])
DEFAULT_PARAMS = _flatten_defaults(PARAM_RUNTIME_CLASSES)


PARAM_CLASS_BY_KEY = {
    key: param_class
    for param_class in PARAM_CLASSES
    for key in PARAM_DEFAULTS[param_class]
}


def load_canonical_defaults() -> tuple[dict, dict]:
    return dict(DEFAULT_META), dict(DEFAULT_PARAMS)


def _ensure_table(config: dict, table_name: str) -> dict:
    value = config.get(table_name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Expected [{table_name}] to be a table.")
    return value


def _validate_section_keys(config: dict, table_name: str, allowed_keys: set[str]) -> None:
    table = _ensure_table(config, table_name)
    unknown_keys = sorted(set(table) - allowed_keys)
    if unknown_keys:
        raise ValueError(f"Unknown [{table_name}] keys: " + ", ".join(unknown_keys))


def _split_legacy_params(params: dict) -> dict[str, dict]:
    split = {param_class: {} for param_class in PARAM_CLASSES}
    unknown_keys = sorted(set(params) - set(PARAM_CLASS_BY_KEY))
    if unknown_keys:
        raise ValueError("Unknown [params] keys: " + ", ".join(unknown_keys))

    for key, value in params.items():
        split[PARAM_CLASS_BY_KEY[key]][key] = value
    return split


def normalize_config(config: dict) -> dict:
    allowed_top_level = set(PARAM_CLASSES) | {"params", "sbatch"}
    unknown_tables = sorted(set(config) - allowed_top_level)
    if unknown_tables:
        raise ValueError("Unknown config tables: " + ", ".join(unknown_tables))

    for param_class in PARAM_CLASSES:
        _validate_section_keys(config, param_class, set(PARAM_DEFAULTS[param_class]))

    legacy_sections = {param_class: {} for param_class in PARAM_CLASSES}
    if "params" in config:
        params = _ensure_table(config, "params")
        legacy_sections = _split_legacy_params(params)

    normalized = {
        param_class: dict(PARAM_DEFAULTS[param_class])
        for param_class in PARAM_CLASSES
    }
    for param_class in PARAM_CLASSES:
        normalized[param_class].update(legacy_sections[param_class])
        normalized[param_class].update(_ensure_table(config, param_class))

    meta = normalized["meta"]
    point_set = normalized["environment"].get("point_set")
    if isinstance(point_set, list):
        normalized["environment"]["point_set"] = tuple(point_set)

    params = {
        key: value
        for param_class in PARAM_RUNTIME_CLASSES
        for key, value in normalized[param_class].items()
    }

    return {
        **normalized,
        "meta": meta,
        "params": params,
    }


def load_config(path: str) -> tuple[Path, dict]:
    config_path = Path(path)
    if not config_path.exists() and config_path.suffix != ".toml":
        candidate = Path("config") / f"{path}.toml"
        if candidate.exists():
            config_path = candidate
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with config_path.open("rb") as file:
        return config_path, normalize_config(tomllib.load(file))


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

    sweep_items = [(key, merged[key]) for key in SWEEP_KEY_ORDER if is_list(merged[key])]
    fixed = {
        key: value
        for key, value in merged.items()
        if not is_list(value)
    }

    if not sweep_items:
        return fixed, [dict(fixed)], []

    varied_keys = [key for key, _ in sweep_items]
    combos: list[dict] = []
    for values in itertools.product(*(value for _, value in sweep_items)):
        combo = dict(fixed)
        combo.update(dict(zip(varied_keys, values)))
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
