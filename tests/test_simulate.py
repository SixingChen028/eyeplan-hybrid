from simulate import _simulator_cache_key


def test_simulator_cache_key_is_hashable_with_nested_list_values():
    metadata_args = {
        "num_nodes": 3,
        "t_max": 5,
        "scale_factor": 1.0,
        "shuffle_nodes": False,
        "use_recency_obs": True,
        "use_best_open_value_obs": False,
        "use_best_terminal_value_obs": False,
        "wm_backup": True,
        "point_set": [1.0, [2.0, 3.0]],
    }

    key = _simulator_cache_key(metadata_args)
    assert isinstance(key, tuple)
    hash(key)
