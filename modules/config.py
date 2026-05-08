from __future__ import annotations

import itertools
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

DEFAULTS_PATH = Path(__file__).resolve().parents[1] / "config" / "_DEFAULTS.toml"

ParameterClass = Literal["environment", "training", "network", "meta"]


@dataclass(frozen=True)
class ParameterDefinition:
    parameter_class: ParameterClass
    default: Any
    sweepable: bool
    description: str


PARAMETER_DEFINITIONS: dict[str, ParameterDefinition] = {
    "num_nodes": ParameterDefinition(
        "environment",
        15,
        False,
        "Number of nodes in each generated decision tree.",
    ),
    "t_max": ParameterDefinition(
        "environment",
        50,
        False,
        "Maximum number of environment steps per episode.",
    ),
    "scale_factor": ParameterDefinition(
        "environment",
        0.125,
        False,
        "Multiplier applied to raw point rewards before returning environment rewards.",
    ),
    "shuffle_nodes": ParameterDefinition(
        "environment",
        True,
        False,
        "Whether to randomly permute node labels for each generated tree.",
    ),
    "use_recency_obs": ParameterDefinition(
        "environment",
        True,
        False,
        "Whether observations include per-node fixation recency values.",
    ),
    "wm_backup": ParameterDefinition(
        "environment",
        True,
        False,
        "Whether value backups use only active working-memory nodes.",
    ),
    "beta_move": ParameterDefinition(
        "environment",
        40.0,
        True,
        "Inverse temperature for softmax move probabilities in environment dynamics.",
    ),
    "eps_move": ParameterDefinition(
        "environment",
        0.0,
        True,
        "Uniform random-move mixture rate in environment dynamics.",
    ),
    "learning_rate": ParameterDefinition(
        "environment",
        1.0,
        True,
        "Environment Q-value update step size.",
    ),
    "lamda_backup": ParameterDefinition(
        "environment",
        1.0,
        True,
        "Decay factor for ancestor value backups.",
    ),
    "backup_steps": ParameterDefinition(
        "environment",
        100,
        True,
        "Maximum number of ancestor levels updated during value backup.",
    ),
    "wm_decay": ParameterDefinition(
        "environment",
        1.0,
        True,
        "Per-step decay applied to working-memory activation.",
    ),
    "q_drop_rate": ParameterDefinition(
        "environment",
        0.0,
        True,
        "Probability of clearing inactive Q-values after each step.",
    ),
    "q_drift": ParameterDefinition(
        "environment",
        0.0,
        True,
        "Standard deviation of Gaussian drift added to inactive Q-values.",
    ),
    "q_decay": ParameterDefinition(
        "environment",
        1.0,
        True,
        "Per-step decay applied to inactive Q-values.",
    ),
    "recency_decay": ParameterDefinition(
        "environment",
        0.5,
        True,
        "Per-step decay applied to fixation recency observations.",
    ),
    "cost": ParameterDefinition(
        "environment",
        0.01,
        True,
        "Per-step movement cost subtracted from environment reward.",
    ),
    "lr": ParameterDefinition(
        "training",
        0.0005,
        True,
        "Optimizer learning rate for A2C training.",
    ),
    "gamma": ParameterDefinition(
        "training",
        1.0,
        True,
        "Discount factor used for return estimation.",
    ),
    "lamda": ParameterDefinition(
        "training",
        0.8,
        True,
        "GAE lambda used for advantage estimation.",
    ),
    "beta_v": ParameterDefinition(
        "training",
        0.05,
        True,
        "Coefficient for the value loss term.",
    ),
    "beta_e_init": ParameterDefinition(
        "training",
        0.02,
        True,
        "Initial entropy coefficient at the start of training.",
    ),
    "beta_e_final": ParameterDefinition(
        "training",
        0.001,
        True,
        "Final entropy coefficient at the end of training.",
    ),
    "max_grad_norm": ParameterDefinition(
        "training",
        2.0,
        True,
        "Global gradient norm clipping threshold.",
    ),
    "num_updates": ParameterDefinition(
        "training",
        50000,
        False,
        "Number of A2C optimization updates to run.",
    ),
    "num_envs": ParameterDefinition(
        "training",
        256,
        False,
        "Number of parallel training environments.",
    ),
    "rollout_length": ParameterDefinition(
        "training",
        50,
        False,
        "Number of environment steps collected per update.",
    ),
    "eval_episodes": ParameterDefinition(
        "training",
        102400,
        False,
        "Number of evaluation episodes run after training.",
    ),
    "print_frequency": ParameterDefinition(
        "training",
        100,
        False,
        "Number of updates between progress prints.",
    ),
    "max_compiled_updates_per_chunk": ParameterDefinition(
        "training",
        -1,
        False,
        "Maximum update chunk size to compile; non-positive uses the requested chunk size.",
    ),
    "network_type": ParameterDefinition(
        "network",
        "mlp",
        False,
        "Policy/value network architecture identifier.",
    ),
    "hidden_size": ParameterDefinition(
        "network",
        256,
        False,
        "Hidden layer width for network architectures that use dense hidden layers.",
    ),
    "seed": ParameterDefinition(
        "meta",
        15,
        True,
        "Random seed for a training run.",
    ),
    "result_path": ParameterDefinition(
        "meta",
        "./results",
        False,
        "Directory where run outputs are written.",
    ),
    "experiment": ParameterDefinition(
        "meta",
        None,
        False,
        "Experiment name; defaults to the config file stem when omitted.",
    ),
    "array_vars": ParameterDefinition(
        "meta",
        None,
        False,
        "Optional sbatch array axes selected from sweep parameters.",
    ),
}


def _parameter_keys(
    parameter_class: ParameterClass,
    *,
    sweepable: bool | None = None,
) -> tuple[str, ...]:
    return tuple(
        key
        for key, definition in PARAMETER_DEFINITIONS.items()
        if definition.parameter_class == parameter_class
        and (sweepable is None or definition.sweepable == sweepable)
    )


ENV_STATIC_PARAM_KEYS = _parameter_keys("environment", sweepable=False)
ENV_DYNAMIC_PARAM_KEYS = _parameter_keys("environment", sweepable=True)
TRAIN_SWEEP_KEYS = _parameter_keys("training", sweepable=True)
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
