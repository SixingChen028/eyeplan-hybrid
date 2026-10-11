"""Functional checks for participant-held-out action fitting and seed mixtures."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np

from modules.action_cache_fitting import coefficient_candidates, stream_cache_candidates
from modules.action_fitting import (
    Candidate, candidates_from_cache, evaluate_test, fit_all_candidates, fit_lapse, freeze_participant_split,
    observed_log_probabilities, participant_bootstrap, posterior_predictive_checks, repeat_score_diagnostics,
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
