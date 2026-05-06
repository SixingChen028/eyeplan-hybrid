from __future__ import annotations

import itertools
import tomllib
from pathlib import Path

DEFAULTS_PATH = Path(__file__).resolve().parents[1] / "config" / "_DEFAULTS.toml"

ENV_STATIC_PARAM_KEYS = (
    "num_nodes",
    "t_max",
    "scale_factor",
    "shuffle_nodes",
    "use_recency_obs",
)

ENV_DYNAMIC_PARAM_KEYS = (
    "beta_move",
    "eps_move",
    "learning_rate",
    "lamda_backup",
    "backup_steps",
    "wm_decay",
    "wm_backup",
    "q_drop_rate",
    "q_drift",
    "q_decay",
    "recency_decay",
    "cost",
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
    "mask_fixation",
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
