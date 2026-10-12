"""Functional checks for participant-held-out action fitting and seed mixtures."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from modules.action_cache_fitting import coefficient_candidates, stream_cache_candidates
from modules.action_fitting import (
    MIN_LAPSE_RATE, Candidate, candidates_from_cache, evaluate_test, fit_all_candidates, fit_lapse, freeze_participant_split,
    observed_log_probabilities, participant_bootstrap, posterior_predictive_checks, repeat_score_diagnostics,
    refine_lapse_profile,
)


def bernoulli_candidate(name, base_probability, family="main"):
    base_probability = np.asarray(base_probability)

    def probabilities(lapse):
        left = (1 - lapse) * base_probability + lapse * .5
        return np.stack([left, 1 - left], axis=1)

    return Candidate(name, family, {"strength": float(base_probability.mean())}, probabilities)


def test_split_keeps_participants_together_and_freezes_order():
    records = [{"pid": f"p{pid}", "trial_index": t} for pid in range(20) for t in range(pid % 3 + 1)]
    split = freeze_participant_split(records)
    assert split["participant_counts"] == {"train": 12, "validation": 4, "test": 4}
    for name, rows in split["indices"].items():
        assert all(split["participant_assignment"][records[row]["pid"]] == name for row in rows)
    assert freeze_participant_split(records) == split
    assert freeze_participant_split(list(reversed(records)))["participant_assignment"] == split["participant_assignment"]


def test_continuous_lapse_fit_and_boundary():
    observed = np.r_[np.zeros(40, dtype=int), np.ones(10, dtype=int)]
    candidate = bernoulli_candidate("known", np.full(50, .875))
    fit = fit_lapse(candidate, observed)
    assert abs(fit["lapse_rate"] - .2) < 1e-5
    assert not fit["lapse_at_boundary"]
    assert fit["profile"]["approx_95_interval"][0] <= .2 <= fit["profile"]["approx_95_interval"][1]
    boundary = fit_lapse(candidate, np.zeros(50, dtype=int))
    assert boundary["lapse_at_boundary"]
    assert boundary["lapse_rate"] == .01


@pytest.mark.parametrize("edge,left_count", [("lower", 1600), ("upper", 1002)])
def test_binary_branch_lapse_refines_boundary_adjacent_maximum(edge, left_count):
    grid = np.linspace(MIN_LAPSE_RATE, 1, 101)
    expected_lapse = grid[0] + .4 * (grid[1] - grid[0]) if edge == "lower" else grid[-1] - .4 * (grid[-1] - grid[-2])
    observed = np.r_[np.zeros(left_count, dtype=int), np.ones(2000 - left_count, dtype=int)]
    frequency = left_count / len(observed)
    base_probability = (frequency - expected_lapse * .5) / (1 - expected_lapse)
    candidate = bernoulli_candidate("near-boundary", np.full(len(observed), base_probability))
    scores = np.array([observed_log_probabilities(candidate, lapse, observed).sum() for lapse in grid])
    assert np.argmax(scores) in (0, len(grid) - 1)

    fit = fit_lapse(candidate, observed, profile_points=len(grid))
    assert fit == refine_lapse_profile(candidate, observed, None, grid, scores)
    assert fit["lapse_rate"] == pytest.approx(expected_lapse, abs=2e-6)
    assert not fit["lapse_at_boundary"]
    assert fit["total_log_likelihood"] >= scores.max()
    expected_score = left_count * np.log(frequency) + (len(observed) - left_count) * np.log(1 - frequency)
    assert fit["total_log_likelihood"] == pytest.approx(expected_score, abs=1e-9)


@pytest.mark.parametrize("left_count,edge", [(50, "lower"), (25, "upper")])
def test_binary_branch_lapse_keeps_exact_boundary_optimum(left_count, edge):
    expected_lapse = MIN_LAPSE_RATE if edge == "lower" else 1.
    observed = np.r_[np.zeros(left_count, dtype=int), np.ones(50 - left_count, dtype=int)]
    candidate = bernoulli_candidate("boundary", np.full(len(observed), .875))
    fit = fit_lapse(candidate, observed)
    assert fit["lapse_rate"] == expected_lapse
    assert fit["lapse_at_boundary"]
    assert fit["total_log_likelihood"] == observed_log_probabilities(candidate, expected_lapse, observed).sum()


def test_flat_lapse_profile_preserves_lower_bound_without_refinement():
    evaluations = []

    def probabilities(lapse):
        evaluations.append(lapse)
        return np.array([[.5, .5]])

    candidate = Candidate("constant", "main", {}, probabilities)
    fit = fit_lapse(candidate, [0], profile_points=11)
    assert len(evaluations) == 11
    assert fit["lapse_rate"] == MIN_LAPSE_RATE
    assert fit["profile"]["likelihood_ratio_95_support"] == np.linspace(MIN_LAPSE_RATE, 1, 11).tolist()


def test_multimodal_lapse_profile_keeps_coarse_maximum_and_refines_each_peak():
    left = np.array([.27, .71, .9, .22, .94, .79, .13, .5])
    coefficients = np.stack([left, 1 - left])[None, :, :]
    trials = [{"paths": [list(range(7)), list(range(7, 14))]}]
    candidate = coefficient_candidates(
        [{"id": "multimodal", "model": "main", "parameters": {}}], [coefficients], trials,
    )[0]
    # The three-point bracket spans two peaks. Its bounded search settles on a
    # lower peak, so the already evaluated middle point must remain available.
    coarse = fit_lapse(candidate, [0], profile_points=3)
    middle = np.linspace(MIN_LAPSE_RATE, 1, 3)[1]
    assert coarse["lapse_rate"] == middle
    assert coarse["total_log_likelihood"] == pytest.approx(np.log(candidate.probabilities(middle)[0, 0]))

    dense = fit_lapse(candidate, [0], profile_points=101)
    reference_grid = np.linspace(MIN_LAPSE_RATE, 1, 10001)
    reference_scores = [observed_log_probabilities(candidate, lapse, [0])[0] for lapse in reference_grid]
    assert dense["total_log_likelihood"] >= max(reference_scores) - 1e-9
    assert dense["lapse_rate"] > .4


def test_validation_selects_without_using_test_responses():
    first = bernoulli_candidate("first", [.9, .9, .9, .9, .9, .1])
    second = bernoulli_candidate("second", [.9, .9, .9, .9, .1, .9])
    observed = np.zeros(6, dtype=int)
    splits = {"train": [0, 1, 2, 3], "validation": [4], "test": [5]}
    fitted = fit_all_candidates([first, second], observed, splits, profile_points=11)
    assert fitted["selected_by_validation"] == {"main": "first"}
    report = evaluate_test([first, second], fitted, observed, splits, list("aaabcc"), np.full(6, .5),
                           bootstrap_repetitions=20)
    assert report["total_log_likelihoods"]["main"] < report["total_log_likelihoods"]["random"]
    changed = observed.copy()
    changed[5] = 1
    assert fit_all_candidates([first, second], changed, splits, profile_points=11) == fitted


def test_seed_probabilities_averaged_before_log():
    paths = ((1,), (2,))
    manifest = {"trials": [{"paths": [[1], [2]], "observed_index": 0}],
                "runs": [{"run_dir": "seed1"}, {"run_dir": "seed2"}]}
    caches = [[[
        SimpleNamespace(paths=paths, mean_coefficients=np.array([[left, .5], [1 - left, .5]]), samples=10)
    ]] for left in [.1, .9]]
    specification = {"candidates": [{"id": "mixture", "model": "main", "parameters": {},
                                      "runs": [{"run_dir": "seed1"}, {"run_dir": "seed2"}]}]}
    candidates, observed = candidates_from_cache(manifest, caches, specification)
    assert np.allclose(observed_log_probabilities(candidates[0], .01, observed), np.log(.5))
    assert not np.isclose(np.log(.5), (np.log(.104) + np.log(.896)) / 2)


def test_streaming_shards_aggregate_seeds_and_repeats(tmp_path):
    paths = []
    for seed, left in enumerate([.1, .9]):
        trial = {"pid": "p", "trial_index": 0, "paths": [[1], [2]], "observed_index": 0}
        arrays_path = tmp_path / f"seed{seed}.npz"
        np.savez_compressed(arrays_path, a_mean=np.array([[left, .5], [1 - left, .5]]),
                            b_mean=np.array([[left + .02, .5], [.98 - left, .5]]))
        manifest = {"cache_version": 1, "arrays_file": arrays_path.name,
                    "arrays_sha256": hashlib.sha256(arrays_path.read_bytes()).hexdigest(), "trials": [trial],
                    "runs": [{"run_dir": f"seed{seed}", "estimates": [[
                        {"key": "a", "samples": 2}, {"key": "b", "samples": 6},
                    ]]}]}
        manifest_path = tmp_path / f"seed{seed}.json"
        manifest_path.write_text(json.dumps(manifest))
        paths.append(manifest_path)
    specification = {"candidates": [{"id": "mixture", "model": "main", "parameters": {},
                                      "runs": [{"run_dir": "seed0"}, {"run_dir": "seed1"}]}]}
    manifest, candidates, observed, repeats = stream_cache_candidates(paths, specification)
    assert len(manifest["trials"]) == 1
    assert np.allclose(candidates[0].probabilities(.01)[0], [.51485, .48515])
    assert np.allclose(repeats[0][0].probabilities(.01), [[.5, .5]])
    assert np.allclose(repeats[1][0].probabilities(.01), [[.5198, .4802]])
    full = observed_log_probabilities(candidates[0], .2, observed)
    candidates[0].evaluate_observed = None
    assert np.allclose(full, observed_log_probabilities(candidates[0], .2, observed))


def test_subset_candidate_uses_requested_trials_only():
    trials = [{"paths": [[1], [2]]}] * 5
    coefficients = np.broadcast_to(np.array([[.8, .5], [.2, .5]]), (5, 2, 2)).copy()
    candidate = coefficient_candidates([{"id": "test", "model": "main", "parameters": {}}],
                                       [coefficients], trials)[0]
    assert np.allclose(candidate.probabilities(.2, [3, 1]), candidate.probabilities(.2)[[3, 1]])


def test_participant_bootstrap_preserves_cluster_weight_and_pairing():
    result = participant_bootstrap([1, 0, 0, 0], ["a", "b", "b", "b"], repetitions=100, seed=7)
    assert result["mean_log_score"] == [.25]
    assert result["equal_participant_mean_log_score"] == [.5]
    assert result == participant_bootstrap([1, 0, 0, 0], ["a", "b", "b", "b"], repetitions=100, seed=7)
    zero = participant_bootstrap(np.zeros(4), ["a", "b", "b", "b"], repetitions=100, seed=7)
    assert zero["mean_log_score_95_interval"] == [[0, 0]]


def test_predictive_metrics_use_entire_path_distribution():
    records = [{"pid": "p", "graph": [[1, 2], [], []], "start": 0,
                "rewards": [0, 10, -10], "moves": [2]}]
    candidate = bernoulli_candidate("test", [.8])
    fitted = {"fits": [{"candidate_id": "test", "lapse_rate": .2}],
              "selected_by_validation": {"main": "test"}}
    result = posterior_predictive_checks([candidate], fitted, [1], records, [0], bootstrap_repetitions=20)
    reward = result["main"]["metrics"]["path_reward_raw_units"]["mean_log_score"]
    assert np.allclose(reward, [-10, 4.8, 14.8])
    optimal = result["main"]["metrics"]["optimal_path_rate"]["mean_log_score"]
    assert np.allclose(optimal, [0, .74, .74])
    precision = repeat_score_diagnostics([candidate], [[candidate], [candidate]], fitted, [1], {"test": [0]})
    assert precision["candidates"]["test"]["test"]["approx_pooled_log_likelihood_mc_se"] == 0


def make_undirected_shards(tmp_path, repeat_count=2):
    records = [{"pid": f"p{i}", "trial_index": 0, "graph": [[1, 2], [], []], "start": 0,
                "rewards": [0, 1, -1], "moves": [1 if i % 2 else 2]} for i in range(6)]
    data_path = tmp_path / "data.json"
    data_path.write_text(json.dumps(records))
    paths = []
    for shard in range(2):
        arrays, trials, entries = {}, [], []
        for i in range(shard * 3, shard * 3 + 3):
            left = np.stack([np.array([.55, .7, .8, .9]) + .02 * repeat for repeat in range(repeat_count)])
            raw = np.zeros((repeat_count, 4, 2, 2))
            raw[:, :, 0, 0] = left
            raw[:, :, 1, 0] = 1 - left
            raw[:, :, :, 1] = .5
            key = f"trial{i}"
            arrays[key] = raw
            trials.append({"pid": f"p{i}", "trial_index": 0, "moves": records[i]["moves"],
                           "paths": [[1], [2]], "observed_index": 0 if records[i]["moves"] == [1] else 1})
            entries.append({"key": key, "samples_per_repeat": 10,
                            "seeds": [i * repeat_count + repeat for repeat in range(repeat_count)]})
        array_path = tmp_path / f"gamma{shard}.npz"
        np.savez_compressed(array_path, **arrays)
        manifest = {"cache_version": 1, "method": "undirected_gamma_trajectory_cache",
                    "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                    "arrays_file": array_path.name, "arrays_sha256": hashlib.sha256(array_path.read_bytes()).hexdigest(),
                    "trials": trials, "runs": [{"wm_decay": .9, "wm_neighbor_activation": .5,
                                                "max_fixations": 4, "estimates": entries}]}
        path = tmp_path / f"gamma{shard}.json"
        path.write_text(json.dumps(manifest))
        paths.append(path)
    return data_path, paths


def test_streamed_gamma_and_observed_refinement_coefficients(tmp_path):
    from modules.action_cache_fitting import (
        align_candidate_trials, stream_undirected_candidates, undirected_observed_coefficients,
    )
    from modules.undirected_likelihood_cache import gamma_stopping_probabilities

    _, paths = make_undirected_shards(tmp_path)
    manifest, candidates, repeats = stream_undirected_candidates(paths, [1], [2])
    mass = gamma_stopping_probabilities(1, 2, 4)
    expected_left = (mass @ np.array([.56, .71, .81, .91])) * .8 + .1
    assert np.allclose(candidates[0].probabilities(.2)[:, 0], expected_left)
    assert len(repeats) == 2
    keys = [[trial["pid"], trial["trial_index"]] for trial in reversed(manifest["trials"])]
    coefficients, lengths = undirected_observed_coefficients(paths, (.9, .5), keys)
    assert coefficients.shape == (4, 6, 2)
    assert np.all(lengths == 1)
    aligned = align_candidate_trials(candidates, manifest["trials"], list(reversed(manifest["trials"])))
    assert np.allclose(aligned[0].probabilities(.2), candidates[0].probabilities(.2)[::-1])
    refined_manifest, refined, _ = stream_undirected_candidates(
        paths, candidate_specs_by_law={(.9, .5): [{"gamma_shape": 1, "gamma_mean": 2}]},
    )
    assert refined[0].candidate_id.endswith("_refined")
    assert refined_manifest["trials"] == manifest["trials"]
    assert np.allclose(refined[0].probabilities(.2), candidates[0].probabilities(.2))


def test_cli_refines_gamma_using_training_only(tmp_path):
    from pathlib import Path
    import subprocess
    import sys

    data_path, cache_paths = make_undirected_shards(tmp_path)
    candidate_path = tmp_path / "candidates.json"
    candidate_path.write_text(json.dumps({"candidates": [{"id": "law", "model": "undirected"}]}))
    output = tmp_path / "fit.json"
    split = tmp_path / "split.json"
    result = subprocess.run([
        sys.executable, "fit_action_models.py", "--data", str(data_path),
        "--undirected-cache", *map(str, cache_paths), "--gamma-shapes", "1", "--gamma-means", "2", "4",
        "--candidates", str(candidate_path), "--split", str(split), "--output", str(output),
        "--profile-points", "11", "--bootstrap-repetitions", "20",
    ], cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["gamma_refinement"][0]["training_trials"] == report["split"]["trial_counts"]["train"]
    assert len(report["coarse_gamma_fits"]) == 2
    assert len(report["refined_undirected_candidate_specs"]) == 1
    assert report["fits"]["selected_by_validation"]["undirected"].endswith("_refined")
    assert report["numerical_precision"]["independent_repeat_count"] == 2


def test_disjoint_streamed_repeat_groups_preserve_full_mean_and_subset_scoring(tmp_path):
    from modules.action_fitting import independent_repeat_groups
    from modules.action_cache_fitting import stream_undirected_candidates

    _, paths = make_undirected_shards(tmp_path, repeat_count=4)
    _, pooled, raw = stream_undirected_candidates(paths, [1], [2])
    groups = independent_repeat_groups(len(raw), 2)
    _, grouped_pooled, grouped = stream_undirected_candidates(paths, [1], [2], repeat_groups=groups)
    assert groups == [[0, 1], [2, 3]]
    assert sorted(index for group in groups for index in group) == list(range(4))
    assert len(raw) == 4 and len(grouped) == 2
    observed = np.array([0, 1, 0, 1, 0, 1])
    for lapse in [.01, .2, 1]:
        raw_mean = np.mean([repeat[0].probabilities(lapse) for repeat in raw], axis=0)
        grouped_mean = np.mean([fold[0].probabilities(lapse) for fold in grouped], axis=0)
        np.testing.assert_allclose(grouped_mean, raw_mean)
        np.testing.assert_allclose(grouped_pooled[0].probabilities(lapse), pooled[0].probabilities(lapse))
        for fold, members in zip(grouped, groups):
            expected = np.mean([raw[index][0].probabilities(lapse) for index in members], axis=0)
            np.testing.assert_allclose(fold[0].probabilities(lapse), expected)
        np.testing.assert_allclose(grouped[0][0].probabilities(lapse, [1]),
                                   grouped[0][0].probabilities(lapse)[[1]])
        selected = grouped[0][0].probabilities(lapse)[np.arange(len(observed)), observed]
        np.testing.assert_allclose(np.exp(observed_log_probabilities(grouped[0][0], lapse, observed)), selected)


def test_cli_pairs_four_gamma_repeats_with_two_ordinary_repeats(tmp_path):
    from pathlib import Path
    import subprocess
    import sys

    data_path, gamma_paths = make_undirected_shards(tmp_path, repeat_count=4)
    records = json.loads(data_path.read_text())
    trials, entries, arrays = [], [], {}
    for i, record in enumerate(records):
        problem = {key: record[key] for key in ("graph", "start", "rewards")}
        trials.append({"pid": record["pid"], "trial_index": 0, "paths": [[1], [2]],
                       "observed_index": 0 if record["moves"] == [1] else 1,
                       "problem_sha256": hashlib.sha256(json.dumps(problem, sort_keys=True).encode()).hexdigest()})
        repeats = []
        for repeat, left in enumerate([.65, .75]):
            key = f"t{i}_r{repeat}"
            arrays[key + "_mean"] = np.array([[left, .5], [1 - left, .5]])
            repeats.append({"key": key, "samples": 10})
        entries.append(repeats)
    array_path = tmp_path / "ordinary.npz"
    np.savez_compressed(array_path, **arrays)
    ordinary_path = tmp_path / "ordinary.json"
    ordinary_path.write_text(json.dumps({
        "cache_version": 1, "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "arrays_file": array_path.name, "arrays_sha256": hashlib.sha256(array_path.read_bytes()).hexdigest(),
        "trials": trials, "runs": [{"run_dir": "seed", "estimates": entries}],
    }))
    candidate_path = tmp_path / "candidates.json"
    candidate_path.write_text(json.dumps({"candidates": [
        {"id": "main", "model": "main", "parameters": {}, "runs": [{"run_dir": "seed"}]},
        {"id": "law", "model": "undirected"},
    ]}))
    for coarse_only in [True, False]:
        output = tmp_path / f"fit_{coarse_only}.json"
        command = [
            sys.executable, "fit_action_models.py", "--data", str(data_path), "--cache", str(ordinary_path),
            "--undirected-cache", *map(str, gamma_paths), "--gamma-shapes", "1", "--gamma-means", "2", "4",
            "--candidates", str(candidate_path), "--split", str(tmp_path / "split.json"), "--output", str(output),
            "--profile-points", "11", "--bootstrap-repetitions", "20",
        ]
        if coarse_only:
            command.append("--coarse-gamma-only")
        result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        report = json.loads(output.read_text())
        grouping = {"ordinary": [[0], [1]], "undirected": [[0, 1], [2, 3]]}
        assert report["provenance"]["numerical_repeat_groups"] == grouping
        assert report["numerical_precision"]["repeat_grouping"] == grouping
        assert report["numerical_precision"]["independent_repeat_count"] == 2
        for candidate in report["numerical_precision"]["candidates"].values():
            assert len(candidate["test"]["repeat_total_log_likelihoods"]) == 2
        # Pooled complete-path likelihood still uses the mean of all four raw Gamma repeats.
        from modules.action_cache_fitting import stream_undirected_candidates
        if coarse_only:
            _, gamma_candidates, _ = stream_undirected_candidates(gamma_paths, [1], [2, 4])
        else:
            parameter = report["refined_undirected_candidate_specs"][0]["parameters"]
            _, gamma_candidates, _ = stream_undirected_candidates(gamma_paths, candidate_specs_by_law={
                (.9, .5): [{"gamma_shape": parameter["gamma_shape"], "gamma_mean": parameter["gamma_mean"]}],
            })
        fitted = {fit["candidate_id"]: fit for fit in report["fits"]["fits"]}
        observed = np.asarray([trial["observed_index"] for trial in trials])
        for candidate in gamma_candidates:
            fit = fitted[candidate.candidate_id]
            rows = report["split"]["indices"]["train"]
            expected = observed_log_probabilities(candidate, fit["lapse_rate"], observed, rows).sum()
            np.testing.assert_allclose(expected, fit["total_log_likelihood"], atol=1e-10)
