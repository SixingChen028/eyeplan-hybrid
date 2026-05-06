from pathlib import Path

import pytest

import simulate
from modules import config


def test_train_uses_canonical_defaults():
    meta, params = config.load_canonical_defaults()
    assert config.DEFAULT_META == meta
    assert config.DEFAULT_PARAMS == params


def test_load_canonical_defaults_missing_file():
    missing_path = Path("/tmp/does-not-exist-defaults.toml")
    with pytest.raises(FileNotFoundError):
        config.load_canonical_defaults(missing_path)


def test_load_canonical_defaults_missing_required_key(tmp_path):
    bad_defaults_path = tmp_path / "_DEFAULTS.toml"
    bad_defaults_path.write_text("[meta]\nresult_path = \"./results\"\n\n[params]\nseed=1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required \\[params\\] keys"):
        config.load_canonical_defaults(bad_defaults_path)


def test_simulate_requires_scale_factor_in_metadata():
    metadata_args = {
        "num_nodes": 15,
        "t_max": 100,
        "shuffle_nodes": True,
        "use_recency_obs": False,
        "recency_decay": 0.0,
    }
    with pytest.raises(ValueError, match="scale_factor"):
        simulate._build_env_from_metadata_args(metadata_args)


def test_simulate_reports_all_missing_env_dynamic_keys():
    metadata_args = {
        "num_nodes": 15,
        "t_max": 100,
        "scale_factor": 0.125,
        "shuffle_nodes": True,
        "use_recency_obs": False,
    }
    with pytest.raises(ValueError) as error:
        simulate._build_env_params_from_metadata_args(
            simulate._build_env_from_metadata_args(
                {**metadata_args, "recency_decay": 0.0}
            ),
            metadata_args,
        )
    message = str(error.value)
    assert "beta_move" in message
    assert "cost" in message
    assert "recency_decay" in message
