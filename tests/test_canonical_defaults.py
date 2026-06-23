import pytest

import simulate
from modules import config
from modules.compat import COMPAT_VERSION
from modules.train_results import env_from_args


def _make_simulate_run_dir(tmp_path):
    run_dir = tmp_path / "results" / "runs" / "test" / "seed1_20260425_122806_pfs1"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text("{}")
    return run_dir


def test_train_uses_canonical_defaults():
    meta, params = config.load_canonical_defaults()
    assert config.DEFAULT_META == meta
    assert config.DEFAULT_PARAMS == params


def test_canonical_defaults_come_from_param_defaults():
    assert config.DEFAULT_META == config.PARAM_DEFAULTS["meta"]
    assert "label" in config.DEFAULT_META
    assert config.DEFAULT_PARAMS["num_nodes"] == config.PARAM_DEFAULTS["environment"]["num_nodes"]
    assert config.DEFAULT_PARAMS["activation_prevents_corruption"] is True
    assert config.DEFAULT_PARAMS["lr"] == config.PARAM_DEFAULTS["training"]["lr"]
    assert config.DEFAULT_PARAMS["network_type"] == config.PARAM_DEFAULTS["network"]["network_type"]


def test_normalize_config_rejects_unknown_section_key():
    with pytest.raises(ValueError, match=r"Unknown \[training\] keys: num_episodes"):
        config.normalize_config({"training": {"num_episodes": 8}})


def test_normalize_config_rejects_unknown_condition_key():
    with pytest.raises(ValueError, match=r"Unknown conditions\[0\] keys: num_episodes"):
        config.normalize_config({"conditions": [{"num_episodes": 8}]})


def test_normalize_config_rejects_old_memory_protection_section_key():
    with pytest.raises(ValueError, match=r"Unknown \[environment\] keys: activation_protects_memory"):
        config.normalize_config({"environment": {"activation_protects_memory": True}})


def test_normalize_config_accepts_activation_prevents_corruption():
    normalized = config.normalize_config({"environment": {"activation_prevents_corruption": False}})
    assert normalized["params"]["activation_prevents_corruption"] is False


def test_normalize_config_rejects_old_terminal_persistence_section_key():
    with pytest.raises(ValueError, match=r"Unknown \[environment\] keys: persist_terminal"):
        config.normalize_config({"environment": {"persist_terminal": True}})


def test_normalize_config_rejects_old_memory_keys_in_params_and_conditions():
    with pytest.raises(ValueError, match=r"Unknown \[params\] keys: activation_protects_memory"):
        config.normalize_config({"params": {"activation_protects_memory": True}})
    with pytest.raises(ValueError, match=r"Unknown conditions\[0\] keys: persist_terminal"):
        config.normalize_config({"conditions": [{"persist_terminal": True}]})


def test_compat_version_is_bumped_for_forgotten_parent_activation_fix():
    assert COMPAT_VERSION == 8


def test_normalize_config_converts_point_set_list_to_tuple():
    normalized = config.normalize_config({"environment": {"point_set": [1, 3, 9]}})
    assert normalized["params"]["point_set"] == (1, 3, 9)


def test_normalize_config_converts_condition_point_set_list_to_tuple():
    normalized = config.normalize_config({"conditions": [{"point_set": [1, 3, 9]}]})
    assert normalized["conditions"][0]["point_set"] == (1, 3, 9)


def test_normalize_config_keeps_condition_sweep_array_as_list():
    normalized = config.normalize_config(
        {"conditions": [{"wm_decay": [0.6, 0.8, 0.9]}]}
    )
    assert normalized["conditions"][0]["wm_decay"] == [0.6, 0.8, 0.9]


def test_normalize_config_rejects_array_for_non_sweepable_condition_key():
    with pytest.raises(ValueError, match=r"conditions\[0\].label cannot be swept"):
        config.normalize_config({"conditions": [{"label": ["a", "b"]}]})


def test_expand_config_runs_expands_condition_sweep_array():
    normalized = config.normalize_config(
        {
            "params": {"cost": [0.01, 0.02]},
            "conditions": [{"label": "wm_only", "wm_decay": [0.6, 0.8]}],
        }
    )

    fixed, runs, varied_keys, label, condition_index = config.expand_config_runs(
        normalized,
        condition_index=0,
    )

    assert sorted(varied_keys) == ["cost", "wm_decay"]
    assert {run["wm_decay"] for run in runs} == {0.6, 0.8}
    assert {run["cost"] for run in runs} == {0.01, 0.02}
    assert len(runs) == 4


def test_expand_config_runs_selects_condition_before_sweep_expansion():
    normalized = config.normalize_config(
        {
            "params": {
                "seed": [1, 2],
                "cost": [0.01, 0.02],
                "use_recency_obs": False,
            },
            "conditions": [
                {
                    "label": "recency",
                    "cost": 0.03,
                    "use_recency_obs": True,
                }
            ],
        }
    )

    fixed, runs, varied_keys, label, condition_index = config.expand_config_runs(
        normalized,
        condition_index=0,
    )

    assert fixed["cost"] == 0.03
    assert fixed["use_recency_obs"] is True
    assert varied_keys == ["seed"]
    assert [run["seed"] for run in runs] == [1, 2]
    assert all(run["cost"] == 0.03 for run in runs)
    assert label == "recency"
    assert condition_index == 0


def test_expand_config_runs_requires_condition_index_when_conditions_exist():
    normalized = config.normalize_config({"conditions": [{"label": "basic"}]})

    with pytest.raises(ValueError, match=r"pass --condition <index>"):
        config.expand_config_runs(normalized)


def test_cli_override_uses_array_element_type():
    params = {"cost": [0.01, 0.02], "num_envs": [64, 128], "shuffle_nodes": [True, False]}

    updated = config.apply_cli_param_overrides(
        params,
        ["--cost=0.03", "--num_envs=256", "--shuffle_nodes=false"],
    )

    assert updated["cost"] == 0.03
    assert updated["num_envs"] == 256
    assert updated["shuffle_nodes"] is False


def test_cli_override_parses_tuple_values():
    updated = config.apply_cli_param_overrides(
        {"point_set": (-8, -4, 4, 8)},
        ["--point_set=-2,2"],
    )

    assert updated["point_set"] == (-2, 2)


def test_validate_params_rejects_zero_wm_neighbor_activation():
    params = dict(config.DEFAULT_PARAMS)
    params["wm_neighbor_activation"] = 0.0

    with pytest.raises(ValueError, match="wm_neighbor_activation"):
        config.validate_params(params)


def test_simulate_skips_existing_output_by_default(tmp_path, monkeypatch, capsys):
    run_dir = _make_simulate_run_dir(tmp_path)
    output_path = run_dir / "data_simulation.json"
    output_path.write_text("{}\n")
    calls = []

    def fake_simulate_run(**kwargs):
        calls.append(kwargs)
        return "params.p", 15, 10, 10

    monkeypatch.setattr(simulate, "_simulate_run", fake_simulate_run)
    monkeypatch.setattr(
        "sys.argv",
        ["simulate.py", str(run_dir), "--results_root", str(tmp_path / "results")],
    )

    simulate.main()

    assert calls == []
    assert "skip existing" in capsys.readouterr().out


def test_simulate_overwrite_reruns_existing_output(tmp_path, monkeypatch):
    run_dir = _make_simulate_run_dir(tmp_path)
    output_path = run_dir / "data_simulation.json"
    output_path.write_text("{}\n")
    calls = []

    def fake_simulate_run(**kwargs):
        calls.append(kwargs)
        return "params.p", 15, 10, 10

    monkeypatch.setattr(simulate, "_simulate_run", fake_simulate_run)
    monkeypatch.setattr(
        "sys.argv",
        ["simulate.py", str(run_dir), "--results_root", str(tmp_path / "results"), "--overwrite"],
    )

    simulate.main()

    assert len(calls) == 1
    assert calls[0]["output_path"] == str(output_path)
    assert calls[0]["num_nodes"] is None
    assert calls[0]["t_max"] is None
    assert calls[0]["allow_unversioned_params"] is False
    assert calls[0]["allow_compat_mismatch"] is False


def test_simulate_default_output_name_includes_environment_size_overrides(tmp_path, monkeypatch):
    run_dir = _make_simulate_run_dir(tmp_path)
    calls = []

    def fake_simulate_run(**kwargs):
        calls.append(kwargs)
        return "params.p", 15, 10, 10

    monkeypatch.setattr(simulate, "_simulate_run", fake_simulate_run)
    monkeypatch.setattr(simulate, "_read_metadata_args", lambda run_dir: {"seed": 1})
    monkeypatch.setattr(
        "sys.argv",
        [
            "simulate.py",
            str(run_dir),
            "--results_root",
            str(tmp_path / "results"),
            "--num_nodes",
            "31",
            "--t_max",
            "80",
        ],
    )

    simulate.main()

    assert len(calls) == 1
    assert calls[0]["output_path"] == str(run_dir / "data_simulation_num_nodes31_t_max80.json")


def test_simulate_passes_environment_size_overrides(tmp_path, monkeypatch):
    run_dir = _make_simulate_run_dir(tmp_path)
    calls = []

    def fake_simulate_run(**kwargs):
        calls.append(kwargs)
        return "params.p", 15, 10, 10

    monkeypatch.setattr(simulate, "_simulate_run", fake_simulate_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "simulate.py",
            str(run_dir),
            "--results_root",
            str(tmp_path / "results"),
            "--num_nodes",
            "31",
            "--t_max",
            "80",
        ],
    )

    simulate.main()

    assert len(calls) == 1
    assert calls[0]["num_nodes"] == 31
    assert calls[0]["t_max"] == 80


def test_simulate_passes_allow_unversioned_params_flag(tmp_path, monkeypatch):
    run_dir = _make_simulate_run_dir(tmp_path)
    calls = []

    def fake_simulate_run(**kwargs):
        calls.append(kwargs)
        return "params.p", 15, 10, 10

    monkeypatch.setattr(simulate, "_simulate_run", fake_simulate_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "simulate.py",
            str(run_dir),
            "--results_root",
            str(tmp_path / "results"),
            "--allow-unversioned-params",
        ],
    )

    simulate.main()

    assert len(calls) == 1
    assert calls[0]["allow_unversioned_params"] is True


def test_simulate_passes_allow_compat_mismatch_flag(tmp_path, monkeypatch):
    run_dir = _make_simulate_run_dir(tmp_path)
    calls = []

    def fake_simulate_run(**kwargs):
        calls.append(kwargs)
        return "params.p", 15, 10, 10

    monkeypatch.setattr(simulate, "_simulate_run", fake_simulate_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "simulate.py",
            str(run_dir),
            "--results_root",
            str(tmp_path / "results"),
            "--allow-compat-mismatch",
        ],
    )

    simulate.main()

    assert len(calls) == 1
    assert calls[0]["allow_compat_mismatch"] is True


def test_simulate_num_nodes_override_requires_node_shared_params():
    with pytest.raises(ValueError, match="node_shared"):
        simulate._metadata_args_with_simulation_overrides(
            {},
            {"fc1": {}},
            num_nodes=31,
            t_max=None,
        )
