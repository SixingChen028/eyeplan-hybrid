from pathlib import Path

import pytest

import simulate
import train
from modules import config_defaults


def test_train_uses_canonical_defaults():
    meta, params = config_defaults.load_canonical_defaults()
    assert train.DEFAULT_META == meta
    assert train.DEFAULT_PARAMS == params


def test_load_canonical_defaults_missing_file(monkeypatch):
    missing_path = Path("/tmp/does-not-exist-defaults.toml")
    monkeypatch.setattr(config_defaults, "DEFAULTS_PATH", missing_path)
    with pytest.raises(FileNotFoundError):
        config_defaults.load_canonical_defaults()


def test_load_canonical_defaults_missing_required_key(tmp_path, monkeypatch):
    bad_defaults_path = tmp_path / "_DEFAULTS.toml"
    bad_defaults_path.write_text("[meta]\nresult_path = \"./results\"\n\n[params]\nseed=1\n", encoding="utf-8")
    monkeypatch.setattr(config_defaults, "DEFAULTS_PATH", bad_defaults_path)
    with pytest.raises(ValueError, match="missing required \\[params\\] keys"):
        config_defaults.load_canonical_defaults()


def test_simulate_requires_scale_factor_in_metadata():
    metadata_args = {
        "num_nodes": 15,
        "t_max": 100,
        "shuffle_nodes": True,
        "recency_decay": "off",
    }
    with pytest.raises(ValueError, match="scale_factor"):
        simulate._build_env_from_metadata_args(metadata_args)


def test_simulate_reports_all_missing_env_dynamic_keys():
    metadata_args = {
        "num_nodes": 15,
        "t_max": 100,
        "scale_factor": 0.125,
        "shuffle_nodes": True,
    }
    with pytest.raises(ValueError) as error:
        simulate._build_env_params_from_metadata_args(
            simulate._build_env_from_metadata_args(
                {**metadata_args, "recency_decay": "off"}
            ),
            metadata_args,
        )
    message = str(error.value)
    assert "beta_move" in message
    assert "cost" in message
    assert "recency_decay" in message
