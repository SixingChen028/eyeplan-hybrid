from __future__ import annotations

from pathlib import Path
import tomllib

DEFAULTS_PATH = Path(__file__).resolve().parents[1] / "config" / "_DEFAULTS.toml"

REQUIRED_PARAM_KEYS = (
    "algo",
    "jobid",
    "seed",
    "network_type",
    "hidden_size",
    "num_nodes",
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
    "t_max",
    "cost",
    "scale_factor",
    "shuffle_nodes",
    "recency_decay",
    "mask_fixation",
    "num_updates",
    "num_envs",
    "rollout_length",
    "eval_episodes",
    "lr",
    "max_grad_norm",
    "gamma",
    "lamda",
    "beta_v",
    "beta_e",
    "beta_e_init",
    "beta_e_final",
    "print_frequency",
    "checkpoint_frequency",
    "log_full_metrics",
    "max_compiled_updates_per_chunk",
)

ENV_STATIC_PARAM_KEYS = (
    "num_nodes",
    "t_max",
    "scale_factor",
    "shuffle_nodes",
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


def load_canonical_defaults() -> tuple[dict, dict]:
    if not DEFAULTS_PATH.exists():
        raise FileNotFoundError(f"Canonical defaults file not found: {DEFAULTS_PATH}")
    with DEFAULTS_PATH.open("rb") as file:
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

