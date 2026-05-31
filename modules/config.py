from __future__ import annotations

import itertools
import tomllib
from pathlib import Path

BACKUP_MODES = ("full", "wm_both", "wm_zero", "wm_partial")
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
        # Whether inactive nodes retain no node-specific information.
        "wm_only": False,
        # Whether seen-terminal indicators persist after a terminal node leaves working memory.
        "persist_terminal": False,
        # Whether observations include per-node fixation recency values.
        "use_recency_obs": True,
        # Whether observations include best-open-path scalar value.
        "use_best_open_value_obs": True,
        # Whether observations include best-seen-terminal scalar value.
        "use_best_terminal_value_obs": True,
        # Whether observations include per-node path values.
        "use_g_values_obs": True,
        # Whether observations include per-node remembered value estimates.
        "use_q_values_obs": True,
        # Whether observations include per-node visit counts.
        "use_n_visits_obs": True,
        # Whether observations include per-node seen-terminal indicators.
        "use_is_terminal_obs": True,
        # Whether observations include elapsed time.
        "use_time_elapsed_obs": True,
        # Policy backup mode for ancestor value updates.
        "backup_mode": "wm_partial",
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
        # Activation assigned to the fixated node's parent and children.
        "wm_neighbor_activation": 1.0,
        # Probability of clearing inactive Q-values, visit counts, and fixation recency after each step.
        "forget_rate": 0.0,
        # Standard deviation of Gaussian drift added to inactive Q-values.
        "q_drift": 0.0,
        # Per-step decay applied to inactive Q-values.
        "q_decay": 1.0,
        # Per-step decay applied to fixation recency observations.
        "recency_decay": 0.5,
        # Per-step movement cost subtracted from environment reward.
        "cost": 0.01,
        # Multiplier for path-length move penalty, applied as move_cost_scale * cost * path length.
        "move_cost_scale": 0.0,
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
        "num_updates": 30000,
        # Number of parallel training environments.
        "num_envs": 128,
        # Number of environment steps collected per update.
        "rollout_length": 50,
        # Random seed for a training run.
        "seed": 1,
    },
    "network": {
        # Policy/value network architecture identifier. mlp|node_shared
        "network_type": "node_shared",
        # Hidden layer width for network architectures that use dense hidden layers.
        "hidden_size": 64,
    },
    "meta": {
        # Directory where run outputs are written.
        "result_path": "./results",
        # Experiment name; defaults to the config file stem when omitted.
        "experiment": None,
        # Optional human-readable run label written to run metadata for downstream analysis.
        "label": None,
        # Optional sbatch array axes: param name list, "ALL" for every sweep array, or None for shape keys only.
        "array_vars": None,
        # Number of updates between progress prints.
        "print_frequency": 100,
        # Maximum update chunk size to compile; non-positive uses the requested chunk size.
        "max_compiled_updates_per_chunk": -1,
        # Terminate if training has not started within this many seconds; non-positive disables.
        # On timeout, train.py prints JAX/GPU diagnostics before exiting.
        "startup_training_timeout_seconds": 300,
        # Skip parameter combinations that already have complete outputs under the experiment.
        "skip_existing": False,
        # Whether to run post-training policy evaluation.
        "run_eval": False,
        # Number of evaluation episodes to run when evaluation is requested.
        "eval_episodes": 102400,
    },
}

ENV_DYNAMIC_PARAM_KEYS = (
    "persist_terminal",
    "beta_move",
    "eps_move",
    "learning_rate",
    "lamda_backup",
    "backup_steps",
    "wm_decay",
    "wm_neighbor_activation",
    "forget_rate",
    "q_drift",
    "q_decay",
    "recency_decay",
    "cost",
    "move_cost_scale",
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
)
REQUIRED_PARAM_KEYS = (
    *ENV_STATIC_PARAM_KEYS,
    *ENV_DYNAMIC_PARAM_KEYS,
    *MODEL_SHAPE_PARAM_KEYS,
    *TRAIN_SWEEP_KEYS,
)

SWEEP_KEYS = set(ENV_DYNAMIC_PARAM_KEYS) | set(TRAIN_SWEEP_KEYS)
SWEEP_KEY_ORDER = ("seed", *ENV_DYNAMIC_PARAM_KEYS, *(key for key in TRAIN_SWEEP_KEYS if key != "seed"))
# Shape/static arrays are valid in TOML configs for job generators such as
# generate_sbatch.py, which lift them into separate Slurm-array tasks. They are
# rejected only by train.py's in-process expand_sweep path because one compiled
# vmapped run cannot vary static environment/model behavior.
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
    allowed_top_level = set(PARAM_CLASSES) | {"params", "conditions", "sbatch", "local"}
    unknown_tables = sorted(set(config) - allowed_top_level)
    if unknown_tables:
        raise ValueError("Unknown config tables: " + ", ".join(unknown_tables))

    for param_class in PARAM_CLASSES:
        _validate_section_keys(config, param_class, set(PARAM_DEFAULTS[param_class]))

    legacy_sections = {param_class: {} for param_class in PARAM_CLASSES}
    if "params" in config:
        params = _ensure_table(config, "params")
        legacy_sections = _split_legacy_params(params)
    conditions = normalize_conditions(config.get("conditions", []))

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
        "conditions": conditions,
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


def is_scalar(value) -> bool:
    return isinstance(value, (bool, int, float, str))


def normalize_conditions(raw_conditions) -> list[dict]:
    if raw_conditions is None:
        return []
    if not isinstance(raw_conditions, list):
        raise ValueError("Expected [[conditions]] to be an array of tables.")

    conditions: list[dict] = []
    allowed_condition_keys = set(DEFAULT_PARAMS) | {"label"}
    for condition_idx, raw_condition in enumerate(raw_conditions):
        if not isinstance(raw_condition, dict):
            raise ValueError(f"conditions[{condition_idx}] must be a table.")
        unknown_keys = sorted(set(raw_condition) - allowed_condition_keys)
        if unknown_keys:
            raise ValueError(f"Unknown conditions[{condition_idx}] keys: " + ", ".join(unknown_keys))

        condition: dict = {}
        for key, value in raw_condition.items():
            if isinstance(value, list):
                template = DEFAULT_PARAMS.get(key)
                if not isinstance(template, tuple):
                    raise ValueError(f"conditions[{condition_idx}].{key} must be scalar.")
                for item_idx, item in enumerate(value):
                    if not is_scalar(item):
                        raise ValueError(
                            f"conditions[{condition_idx}].{key}[{item_idx}] must be scalar, "
                            f"got {type(item).__name__}."
                        )
                condition[key] = tuple(value)
                continue
            if not is_scalar(value):
                raise ValueError(f"conditions[{condition_idx}].{key} must be scalar.")
            condition[key] = value
        conditions.append(condition)
    return conditions


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


def validate_backup_mode(value, *, name: str = "backup_mode") -> None:
    if value not in BACKUP_MODES:
        raise ValueError(f"{name} must be one of {BACKUP_MODES}.")


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

    value = params.get("wm_neighbor_activation", 1.0)
    values = value if is_list(value) else [value]
    for item in values:
        parsed = parse_unit_interval(item, name="wm_neighbor_activation")
        if parsed == 0.0:
            raise ValueError("wm_neighbor_activation numeric values must satisfy 0 < wm_neighbor_activation <= 1.")

    validate_backup_mode(params.get("backup_mode"), name="backup_mode")


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


def select_condition_params(
    params: dict,
    conditions: list[dict],
    condition_index: int | None,
) -> tuple[dict, str | None, int | None]:
    if conditions and condition_index is None:
        raise ValueError("Config contains [[conditions]]; pass --condition <index>.")
    if not conditions:
        if condition_index is not None:
            raise ValueError("--condition was provided, but the config has no [[conditions]].")
        return dict(params), None, None
    if condition_index is None or condition_index < 0 or condition_index >= len(conditions):
        raise ValueError(f"condition index must be in [0, {len(conditions) - 1}], got {condition_index}.")

    condition = conditions[condition_index]
    selected_params = dict(params)
    selected_params.update({key: value for key, value in condition.items() if key != "label"})
    label = condition.get("label")
    return selected_params, None if label is None else str(label), condition_index


def expand_config_runs(
    config: dict,
    *,
    condition_index: int | None = None,
    override_tokens: list[str] | None = None,
) -> tuple[dict, list[dict], list[str], str | None, int | None]:
    params, condition_label, selected_condition_index = select_condition_params(
        config.get("params", {}),
        config.get("conditions", []),
        condition_index,
    )
    params = apply_cli_param_overrides(params, [] if override_tokens is None else override_tokens)
    fixed, runs, varied_keys = expand_sweep(params)
    return fixed, runs, varied_keys, condition_label, selected_condition_index


def resolve_training_geometry(params: dict) -> tuple[int, int, int]:
    if "num_updates" in params and "num_envs" in params and "rollout_length" in params:
        return int(params["num_updates"]), int(params["num_envs"]), int(params["rollout_length"])

    raise ValueError(
        "Training geometry must be specified as num_updates + num_envs + rollout_length."
    )


def _parse_cli_scalar(raw: str, template_value):
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


def parse_cli_value(raw: str, template_value):
    if isinstance(template_value, list):
        if not template_value:
            raise ValueError("Cannot parse CLI override against an empty list template.")
        return _parse_cli_scalar(raw, template_value[0])

    if isinstance(template_value, tuple):
        if not template_value:
            raise ValueError("Cannot parse CLI override against an empty tuple template.")
        items = [item.strip() for item in raw.split(",")]
        if any(item == "" for item in items):
            raise ValueError(f"Invalid tuple override value: {raw!r}")
        return tuple(_parse_cli_scalar(item, template_value[0]) for item in items)

    return _parse_cli_scalar(raw, template_value)


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
