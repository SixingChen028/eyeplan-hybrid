"""Exercise separately cached, empirically selected generating models through the CLI."""

import hashlib
import json
import sys

import numpy as np
import pytest

import run_split_action_recovery as cli
from modules.action_fitting import freeze_participant_split


def write_json(path, value):
    path.write_text(json.dumps(value))
    return path


def write_cache(tmp_path, name, trials, *, undirected=False, seed=0):
    arrays, runs = {}, []
    for policy in range(1 if undirected else 3):
        estimates = []
        for index, trial in enumerate(trials):
            if undirected:
                key = f"t{index}"
                arrays[key] = np.tile(np.array([[.65, .5], [.35, .5]]), (4, 3, 1, 1))
                estimates.append({"key": key, "samples_per_repeat": 12,
                                  "seeds": [seed + index * 10 + repeat for repeat in range(4)]})
            else:
                entries = []
                for repeat in range(2):
                    key = f"p{policy}t{index}r{repeat}"
                    probability = [.85, .15, .7][policy]
                    arrays[key + "_mean"] = np.array([[probability, .5], [1 - probability, .5]])
                    entries.append({"key": key, "samples": 10,
                                    "seed": seed + policy * 100 + index * 10 + repeat})
                estimates.append(entries)
        runs.append({"run_dir": "/torch/runs/gamma", "wm_decay": .8, "wm_neighbor_activation": .5,
                     "metadata_sha256": "same_gamma_metadata",
                     "max_fixations": 3, "estimates": estimates} if undirected else
                    {"run_dir": f"/torch/runs/policy{policy}", "estimates": estimates})
    array_path = tmp_path / f"{name}.npz"
    np.savez(array_path, **arrays)
    cached_trials = [{"pid": trial["pid"], "trial_index": trial["trial_index"],
                      "paths": trial["paths"], "observed_index": trial["observed_index"],
                      "moves": trial["paths"][trial["observed_index"]]} for trial in trials] if undirected else trials
    return write_json(tmp_path / f"{name}.json", {
        "cache_version": 1, "data_sha256": "same_frozen_data", "trials": cached_trials,
        "git_sha": name, "source_sha256": {"simulator.py": "gamma_source" if undirected else "learned_source"},
        "runs": runs, "arrays_file": array_path.name,
        "arrays_sha256": hashlib.sha256(array_path.read_bytes()).hexdigest(),
        "method": "undirected_gamma_trajectory_cache" if undirected else
                  "planning_mc_bernstein_moments_exact_execution",
    })


@pytest.fixture
def recovery_inputs(tmp_path):
    # Deliberately use original data ordering that differs from the loader's sorted order.
    trials = [{"record_index": i, "pid": "cba"[i // 2], "trial_index": i % 2,
               "paths": [[1], [2]], "observed_index": i % 2, "problem_sha256": f"problem{i}"} for i in range(6)]
    paths = [write_cache(tmp_path, name, trials, undirected=undirected, seed=seed)
             for name, undirected, seed in [("fit", False, 0), ("gamma", True, 1000),
                                            ("truth", False, 10000), ("truthgamma", True, 11000)]]
    inventory = write_json(tmp_path / "candidates.json", {"candidates": [
        {"id": f"policy{i}", "model": ["main", "unbounded", "main"][i], "scope": "primary",
         "parameters": {"cost": i + 1}, "runs": [{"run_dir": f"runs/policy{i}"}]} for i in range(3)
    ]})
    gamma_parameters = {"wm_decay": .8, "wm_neighbor_activation": .5,
                        "gamma_shape": 1.7, "gamma_mean": 2.3, "gamma_scale": 2.3 / 1.7}
    split = freeze_participant_split(trials, fractions=(1 / 3, 1 / 3, 1 / 3))
    split_path = write_json(tmp_path / "split.json", split)
    report = write_json(tmp_path / "fits.json", {
        "split": split,
        "provenance": {"data_sha256": "same_frozen_data", "candidate_manifest_sha256": cli.file_hash(inventory),
                       "cache_manifests": cli.manifest_provenance(paths[:2])},
        "fits": {"selected_by_validation": {"main": "policy0", "unbounded": "policy1", "undirected": "empirical"},
                 "fits": [{"candidate_id": "policy0", "family": "main", "parameters": {"cost": 1},
                           "lapse_rate": .12},
                          {"candidate_id": "policy1", "family": "unbounded", "parameters": {"cost": 2},
                           "lapse_rate": .23},
                          {"candidate_id": "empirical", "family": "undirected", "parameters": gamma_parameters,
                           "lapse_rate": .34}]},
        "refined_undirected_candidate_specs": [{"id": "empirical", "model": "undirected",
                                                "parameters": gamma_parameters}],
    })
    output = tmp_path / "recovery.json"
    argv = ["run_split_action_recovery.py", "--cache", str(paths[0]), "--undirected-cache", str(paths[1]),
            "--truth-cache", str(paths[2]), "--truth-undirected-cache", str(paths[3]),
            "--candidate-manifest", str(inventory), "--fit-results", str(report), "--split", str(split_path),
            "--output", str(output), "--gamma-shapes", "1", "--gamma-means", "2", "--repetitions", "1",
            "--profile-points", "3", "--optimizer-maxiter", "3", "--checkpoint-dir", str(tmp_path / "resume")]
    return paths, split_path, report, output, argv


def test_primary_cli_selects_truths_and_pools_every_repeat(recovery_inputs, monkeypatch):
    paths, _, _, output, argv = recovery_inputs
    monkeypatch.setattr(sys, "argv", argv)
    captured = {}
    def replay(candidates, truths, trials, split, **kwargs):
        captured.update(kwargs)
        assert {candidate.candidate_id for candidate in candidates} == {
            "policy0", "policy1", "policy2", "undirected_decay0.8_neighbor0.5_shape1_mean2",
        }
        assert {candidate.candidate_id for candidate in truths} == {"policy0", "policy1", "empirical"}
        assert {candidate.family for candidate in truths} == {"main", "unbounded", "undirected"}
        for candidate in [*candidates, *truths]:
            assert candidate.probabilities(.1).shape == (6, 2)
            np.testing.assert_allclose(candidate.probabilities(.1).sum(axis=1), 1)
        assert [trial["pid"] for trial in trials] == ["a", "a", "b", "b", "c", "c"]
        assert all("record_index" in trial and "problem_sha256" in trial for trial in trials)
        return {"datasets": []}
    monkeypatch.setattr(cli, "split_recovery", replay)
    cli.main()
    result = json.loads(output.read_text())
    assert captured["truth_lapses"] == {"policy0": .12, "policy1": .23, "empirical": .34}
    assert captured["continuous_gamma_paths"] == [paths[1]]
    assert captured["checkpoint_dir"].name == "resume"
    provenance = result["provenance"]
    assert provenance["rng_audit"]["independent"]
    assert provenance["rng_audit"]["truth_seed_count"] == 6 * (2 * 2 + 4)
    assert provenance["rng_audit"]["fit_seed_count"] == 6 * (3 * 2 + 4)
    assert provenance["full_pooled_precision"]["fitting_undirected"]["sample_counts"] == [[12, 12, 12, 12]]
    assert provenance["full_pooled_precision"]["generating_learned"]["raw_repeat_count"] == 2


def test_primary_cli_rejects_cross_checkpoint_overlap_before_arrays(recovery_inputs, monkeypatch):
    paths, _, _, _, argv = recovery_inputs
    manifest = json.loads(paths[2].read_text())
    # True policy0 reuses a fit policy2 seed on the same empirical task.
    manifest["runs"][0]["estimates"][0][0]["seed"] = 200
    paths[2].write_text(json.dumps(manifest))
    monkeypatch.setattr(sys, "argv", argv)
    def forbidden(*args, **kwargs):
        raise AssertionError("coefficient arrays loaded before RNG audit")
    monkeypatch.setattr(cli, "stream_cache_candidates", forbidden)
    with pytest.raises(ValueError, match="RNG overlap"):
        cli.main()


def test_primary_cli_rejects_changed_frozen_task_hash(recovery_inputs, monkeypatch):
    _, split_path, report_path, _, argv = recovery_inputs
    split = json.loads(split_path.read_text())
    split["trial_keys_sha256"] = "incorrect"
    split["split_sha256"] = cli.canonical_hash({key: value for key, value in split.items() if key != "split_sha256"})
    split_path.write_text(json.dumps(split))
    report = json.loads(report_path.read_text())
    report["split"] = split
    report_path.write_text(json.dumps(report))
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="trial-key hash"):
        cli.main()


def test_primary_cli_replays_continuous_protocol(recovery_inputs, monkeypatch):
    paths, _, _, output, argv = recovery_inputs
    ordinary_trial = json.loads(paths[0].read_text())["trials"][0]
    gamma_trial = json.loads(paths[1].read_text())["trials"][0]
    assert {"record_index", "problem_sha256"} <= ordinary_trial.keys()
    assert "moves" in gamma_trial and not {"record_index", "problem_sha256"} & gamma_trial.keys()
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    result = json.loads(output.read_text())
    assert result["gamma_recovery"] == "continuous_at_every_working_memory_law"
    assert len(result["datasets"]) == 3
    assert result["actual_trial_counts"] == {"train": 2, "validation": 2, "test": 2}
    assert {row["true_candidate"] for row in result["datasets"]} == {"policy0", "policy1", "empirical"}
    assert all(set(row["family_test_log_likelihoods"]) == {"main", "unbounded", "undirected", "random"}
               for row in result["datasets"])
    assert all(row["selected_family_fits"]["undirected"]["parameter_method"]
               == "continuous_gamma_and_lapse_train_then_fixed_gamma_lapse_profile" for row in result["datasets"])


@pytest.mark.parametrize("failure, message", [
    ("paths", "trial paths differs"), ("observed_index", "trial observed_index differs"),
    ("problem_sha256", "trial problem hash differs"), ("data_sha256", "data hash differs"),
    ("unknown_trial", "problem absent"), ("missing_trial", "complete frozen task"),
    ("source_sha256", "simulator source hashes differ"),
    ("Gamma_metadata", "source metadata hash or stopping horizon"),
    ("Gamma_horizon", "source metadata hash or stopping horizon"),
])
def test_primary_cli_rejects_semantic_task_and_data_mismatches_before_arrays(recovery_inputs, monkeypatch,
                                                                          failure, message):
    paths, _, _, _, argv = recovery_inputs
    manifest = json.loads(paths[3].read_text())
    if failure in {"paths", "observed_index", "problem_sha256"}:
        manifest["trials"][0][failure] = {"paths": [[2], [1]], "observed_index": 1,
                                          "problem_sha256": "changed_problem"}[failure]
    elif failure == "data_sha256":
        manifest["data_sha256"] = "changed_data"
    elif failure == "unknown_trial":
        manifest["trials"][0]["pid"] = "unknown"
    elif failure == "source_sha256":
        manifest["source_sha256"]["simulator.py"] = "changed_source"
    elif failure == "Gamma_metadata":
        manifest["runs"][0]["metadata_sha256"] = "changed_metadata"
    elif failure == "Gamma_horizon":
        manifest["runs"][0]["max_fixations"] = 4
    else:
        manifest["trials"].pop()
    paths[3].write_text(json.dumps(manifest))
    monkeypatch.setattr(sys, "argv", argv)
    def forbidden(*args, **kwargs):
        raise AssertionError("coefficient arrays loaded before task/data validation")
    monkeypatch.setattr(cli, "stream_cache_candidates", forbidden)
    with pytest.raises(ValueError, match=message):
        cli.main()


def test_primary_source_contract_allows_different_heads_and_sample_counts(recovery_inputs, monkeypatch):
    paths, _, _, output, argv = recovery_inputs
    assert json.loads(paths[0].read_text())["git_sha"] != json.loads(paths[2].read_text())["git_sha"]
    for path in paths[2:]:
        manifest = json.loads(path.read_text())
        for run in manifest["runs"]:
            for entry in run["estimates"]:
                if isinstance(entry, list):
                    for repeat in entry:
                        repeat["samples"] *= 2
                else:
                    entry["samples_per_repeat"] *= 2
        path.write_text(json.dumps(manifest))
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(cli, "split_recovery", lambda *args, **kwargs: {"datasets": []})
    cli.main()
    precision = json.loads(output.read_text())["provenance"]["full_pooled_precision"]
    assert precision["fitting_learned"]["pooled_samples_per_checkpoint_or_wm_law"] == [20]
    assert precision["generating_learned"]["pooled_samples_per_checkpoint_or_wm_law"] == [40]
    assert precision["fitting_undirected"]["pooled_samples_per_checkpoint_or_wm_law"] == [48]
    assert precision["generating_undirected"]["pooled_samples_per_checkpoint_or_wm_law"] == [96]
