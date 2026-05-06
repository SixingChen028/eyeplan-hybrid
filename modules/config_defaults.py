from __future__ import annotations

from modules import config

DEFAULTS_PATH = config.DEFAULTS_PATH
REQUIRED_PARAM_KEYS = config.REQUIRED_PARAM_KEYS
ENV_STATIC_PARAM_KEYS = config.ENV_STATIC_PARAM_KEYS
ENV_DYNAMIC_PARAM_KEYS = config.ENV_DYNAMIC_PARAM_KEYS


def load_canonical_defaults() -> tuple[dict, dict]:
    return config.load_canonical_defaults(DEFAULTS_PATH)
