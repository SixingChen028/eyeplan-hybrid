import pytest

import simulate
from modules import config
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
    assert config.DEFAULT_PARAMS["lr"] == config.PARAM_DEFAULTS["training"]["lr"]
    assert config.DEFAULT_PARAMS["network_type"] == config.PARAM_DEFAULTS["network"]["network_type"]


def test_normalize_config_rejects_unknown_section_key():
    with pytest.raises(ValueError, match=r"Unknown \[training\] keys: num_episodes"):
        config.normalize_config({"training": {"num_episodes": 8}})


def test_normalize_config_converts_point_set_list_to_tuple():
    normalized = config.normalize_config({"environment": {"point_set": [1, 3, 9]}})
    assert normalized["params"]["point_set"] == (1, 3, 9)


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
