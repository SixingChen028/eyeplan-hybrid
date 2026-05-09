import pytest

import simulate
from modules import config
from modules.train_results import env_from_args


def test_train_uses_canonical_defaults():
    meta, params = config.load_canonical_defaults()
    assert config.DEFAULT_META == meta
    assert config.DEFAULT_PARAMS == params


def test_canonical_defaults_come_from_param_defaults():
    assert config.DEFAULT_META == config.PARAM_DEFAULTS["meta"]
    assert config.DEFAULT_PARAMS["num_nodes"] == config.PARAM_DEFAULTS["environment"]["num_nodes"]
    assert config.DEFAULT_PARAMS["lr"] == config.PARAM_DEFAULTS["training"]["lr"]
    assert config.DEFAULT_PARAMS["network_type"] == config.PARAM_DEFAULTS["network"]["network_type"]


def test_normalize_config_rejects_unknown_section_key():
    with pytest.raises(ValueError, match=r"Unknown \[training\] keys: num_episodes"):
        config.normalize_config({"training": {"num_episodes": 8}})


def test_normalize_config_converts_point_set_list_to_tuple():
    normalized = config.normalize_config({"environment": {"point_set": [1, 3, 9]}})
    assert normalized["params"]["point_set"] == (1, 3, 9)


def test_simulate_requires_scale_factor_in_metadata():
    metadata_args = {
        "num_nodes": 15,
        "t_max": 100,
        "shuffle_nodes": True,
        "use_recency_obs": False,
        "use_best_open_value_obs": True,
        "use_best_terminal_value_obs": True,
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
        "use_best_open_value_obs": True,
        "use_best_terminal_value_obs": True,
        "wm_backup": True,
        "point_set": [-8, -4, -2, -1, 1, 2, 4, 8],
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


def test_simulate_build_env_uses_point_set():
    metadata_args = {
        "num_nodes": 15,
        "t_max": 100,
        "scale_factor": 0.125,
        "shuffle_nodes": True,
        "use_recency_obs": False,
        "use_best_open_value_obs": True,
        "use_best_terminal_value_obs": True,
        "wm_backup": True,
        "point_set": [-3, -1, 1, 3],
    }
    env = simulate._build_env_from_metadata_args(metadata_args)
    assert tuple(float(value) for value in env.point_set.tolist()) == (-3.0, -1.0, 1.0, 3.0)


def test_train_results_env_from_args_uses_point_set():
    env = env_from_args(
        {
            "num_nodes": 15,
            "t_max": 100,
            "scale_factor": 0.125,
            "shuffle_nodes": True,
            "use_recency_obs": False,
            "use_best_open_value_obs": True,
            "use_best_terminal_value_obs": True,
            "wm_backup": True,
            "point_set": [-5, -2, 2, 5],
        }
    )
    assert tuple(float(value) for value in env.point_set.tolist()) == (-5.0, -2.0, 2.0, 5.0)
