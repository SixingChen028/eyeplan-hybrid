"""Recovery task/fold selection happens before large coefficient allocations."""

import hashlib
import json
import sys

import numpy as np
import pytest

from action_recovery import (
    assert_independent_manifest_repeats, main, select_recovery_trials, undirected_repeat_groups,
)
from modules import action_cache_fitting


def write_cache(tmp_path, name, manifest, arrays):
    array_path = tmp_path / f"{name}.npz"
    np.savez(array_path, **arrays)
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({"cache_version": 1, "arrays_file": array_path.name,
                                "arrays_sha256": hashlib.sha256(array_path.read_bytes()).hexdigest(), **manifest}))
    return path


def test_trial_subset_preserves_original_designs():
    trials = [{"pid": "ab"[i // 5], "trial_index": i % 5} for i in range(10)]
    keys, compact, original = select_recovery_trials(trials, [2, 4], 1, 72)
    expected = sorted(np.random.default_rng(72).permutation(10)[:4].tolist())
    assert original["fixed_4_trial_design"] == expected
    assert original["participant_a"] == list(range(5))
    for design in original:
        assert [keys[index] for index in compact[design]] == [
            (trials[index]["pid"], trials[index]["trial_index"]) for index in original[design]
        ]
    assert len(keys) == len(set(range(5)) | set(expected))


def test_undirected_fold_grouping_and_validation():
    assert undirected_repeat_groups(4, 2) == [[0, 1], [2, 3]]
    assert undirected_repeat_groups(2, 2) == [[0], [1]]
    assert undirected_repeat_groups(4, 2, [[0, 2], [1, 3]]) == [[0, 2], [1, 3]]
    with pytest.raises(ValueError, match="equal learned-cache folds"):
        undirected_repeat_groups(3, 2)
    with pytest.raises(ValueError, match="disjoint"):
        undirected_repeat_groups(4, 2, [[0, 1], [1, 3]])


def test_cli_streams_only_union_and_requested_fold_groups(tmp_path, monkeypatch):
    trials = [{"pid": "ab"[i // 3], "trial_index": i % 3, "paths": [[1], [2]], "observed_index": 0}
              for i in range(6)]
    ordinary_arrays, estimates = {}, []
    gamma_arrays, gamma_estimates = {}, []
    for index, trial in enumerate(trials):
        repeats = []
        for repeat in range(3):
            key = f"t{index}r{repeat}"
            ordinary_arrays[key + "_mean"] = np.array([[.8, .5], [.2, .5]])
            repeats.append({"key": key, "samples": 10, "seed": index * 10 + repeat})
        estimates.append(repeats)
        key = f"t{index}"
        gamma_arrays[key] = np.tile(np.array([[.6, .5], [.4, .5]]), (6, 3, 1, 1))
        gamma_estimates.append({"key": key, "seeds": [1000 + index * 10 + repeat for repeat in range(6)]})
    ordinary = write_cache(tmp_path, "ordinary", {"trials": trials, "runs": [
        {"run_dir": "runs/policy", "estimates": estimates},
    ]}, ordinary_arrays)
    gamma = write_cache(tmp_path, "gamma", {"trials": trials, "method": "undirected_gamma_trajectory_cache",
                                          "runs": [{"wm_decay": .8, "wm_neighbor_activation": .5,
                                                    "max_fixations": 3, "estimates": gamma_estimates}]}, gamma_arrays)
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"candidates": [{"id": "policy", "model": "main", "parameters": {},
                                                      "runs": [{"run_dir": "runs/policy"}]}]}))
    called = {}
    for name in ["stream_cache_candidates", "stream_undirected_candidates"]:
        original = getattr(action_cache_fitting, name)
        def record(*args, _name=name, _original=original, **kwargs):
            called[_name] = kwargs
            result = _original(*args, **kwargs)
            assert len(result[0]["trials"]) == len(kwargs["trial_keys"])
            assert len(result[-1]) == 2
            return result
        monkeypatch.setattr(action_cache_fitting, name, record)
    output = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", ["action_recovery.py", str(ordinary), "--undirected-cache", str(gamma),
                                      "--candidate-manifest", str(inventory), "--output", str(output),
                                      "--gamma-shapes", "1", "--gamma-means", "4", "--trial-counts", "2",
                                      "--participant-designs", "1", "--fit-repeats", "0", "--truth-repeat", "2",
                                      "--true-lapses", ".1", "--lapse-step", ".2", "--repetitions", "2"])
    main()
    result = json.loads(output.read_text())
    expected_keys, _, expected_original = select_recovery_trials(trials, [2], 1, 20261011)
    assert len(expected_keys) < len(trials)
    assert called["stream_cache_candidates"] == {"trial_keys": expected_keys, "repeat_indices": [0, 2]}
    assert called["stream_undirected_candidates"] == {"trial_keys": expected_keys,
                                                      "repeat_groups": [[0, 1], [4, 5]]}
    assert result["trial_sets"] == expected_original
    assert result["provenance"]["source_trial_count"] == 6
    assert result["provenance"]["undirected_repeat_groups"] == [[0, 1], [2, 3], [4, 5]]
    assert {row["true_family"] for row in result["datasets"]} == {"main", "undirected"}
    assert result["provenance"]["rng_audit"]["independent"]
    assert result["provenance"]["rng_audit"]["truth_seed_count"] == 3 * len(expected_keys)
    # Cross-family collisions must fail before coefficient loaders run.
    manifest = json.loads(ordinary.read_text())
    manifest["runs"][0]["estimates"][0][2]["seed"] = 1000
    ordinary.write_text(json.dumps(manifest))
    called.clear()
    monkeypatch.setattr(sys, "argv", [str(tmp_path / "collision.json") if arg == str(output) else arg
                                      for arg in sys.argv])
    with pytest.raises(ValueError, match="RNG overlap"):
        main()
    assert not called
    # Grouped guard checks every constituent seed, including the second truth-group member.
    manifest = json.loads(gamma.read_text())
    manifest["runs"][0]["estimates"][0]["seeds"][5] = manifest["runs"][0]["estimates"][0]["seeds"][0]
    gamma.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="share MC seeds"):
        assert_independent_manifest_repeats([gamma], 2, [0], repeat_groups=[[0, 1], [2, 3], [4, 5]])
