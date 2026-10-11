"""Tiny independent-cache rescoring fixtures; no simulation or optimization."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from action_truth_confirmation import baseline_paths, confirmation_report
from modules.undirected_likelihood_cache import gamma_stopping_probabilities


def write_bundle(directory, name, method, trials, run, arrays):
    array_path = directory / f"{name}.npz"
    np.savez_compressed(array_path, **arrays)
    manifest = {"cache_version": 1, "method": method, "data_sha256": "same_complete_data",
                "source_sha256": {"frozen_simulator.py": "same_source"},
                "arrays_file": array_path.name, "arrays_sha256": hashlib.sha256(array_path.read_bytes()).hexdigest(),
                "trials": trials, "runs": [run]}
    path = directory / f"{name}.json"
    path.write_text(json.dumps(manifest))
    return path


def fixture(tmp_path):
    trials = [{"pid": f"p{i}", "trial_index": 0, "paths": [[1], [2]], "observed_index": i % 2,
               "problem_sha256": f"problem{i}"} for i in range(3)]
    ordinary, gamma = [], []
    ordinary_ps, gamma_ps, gamma_repeat_ps = [], [], []
    for new in (False, True):
        shift = .02 if new else 0
        arrays, estimates = {}, []
        for t in range(3):
            entries = []
            for r in range(2):
                qs = np.asarray([.2, .4, .6, .8]) + shift + .01 * r
                samples = np.stack([np.stack([qs, np.full(4, .5)], 1),
                                    np.stack([1 - qs, np.full(4, .5)], 1)], 1)
                centered = samples - samples.mean(axis=0)
                key = f"t{t}_r{r}"
                arrays[key + "_mean"] = samples.mean(axis=0)
                arrays[key + "_m2"] = np.einsum("npi,npj->pij", centered, centered)
                entries.append({"key": key, "samples": 4, "seed": (3000 if new else 1000) + t * 10 + r})
            estimates.append(entries)
        run = {"run_dir": "/workspace/seed1", "metadata_sha256": "ordinary_meta", "params_sha256": "policy",
               "estimates": estimates}
        ordinary.append(write_bundle(tmp_path, f"ordinary_{new}", "planning_mc_bernstein_moments_exact_execution",
                                     trials, run, arrays))
        left = .8 * (.505 + shift) + .1
        ordinary_ps.append(np.array([left, 1 - left, left]))
        arrays, estimates = {}, []
        raw_repeats = 2 if new else 4
        for t in range(3):
            left = np.stack([np.array([.4, .6, .8]) + .01 * r + shift for r in range(raw_repeats)])
            raw = np.zeros((raw_repeats, 3, 2, 2))
            raw[:, :, 0, 0], raw[:, :, 1, 0] = left, 1 - left
            raw[:, :, :, 1] = .5
            key = f"t{t}"
            arrays[key] = raw
            estimates.append({"key": key, "samples_per_repeat": 4,
                              "seeds": [(4000 if new else 2000) + t * 10 + r for r in range(raw_repeats)]})
        run = {"run_dir": "/workspace/gamma", "metadata_sha256": "gamma_meta", "max_fixations": 3,
               "wm_decay": .9, "wm_neighbor_activation": .5, "estimates": estimates}
        gamma.append(write_bundle(tmp_path, f"gamma_{new}", "undirected_gamma_trajectory_cache", trials, run, arrays))
        mass = gamma_stopping_probabilities(1, 2, 3)
        repeat_left = .75 * (left @ mass) + .125
        repeat_ps = np.stack([repeat_left, 1 - repeat_left, repeat_left], 1)
        gamma_repeat_ps.append(repeat_ps)
        gamma_ps.append(repeat_ps.mean(axis=0))
    selected = {"main": "main", "undirected": "gamma"}
    parameters = {"wm_decay": .9, "wm_neighbor_activation": .5,
                  "gamma_shape": 1, "gamma_scale": 2, "gamma_mean": 2}
    rows = []
    for name, family, params, lapse, p in [("main", "main", {}, .2, ordinary_ps[0]),
                                          ("gamma", "undirected", parameters, .25, gamma_ps[0])]:
        rows.append({"candidate_id": name, "family": family, "parameters": params, "lapse_rate": lapse,
                     "total_log_likelihood": float(np.log(p[0])), "validation_log_likelihood": float(np.log(p[1]))})
    baseline = ordinary[:1] + gamma[:1]
    fit = {"fits": {"selected_by_validation": selected, "fits": rows},
           "test": {"total_log_likelihoods": {"main": float(np.log(ordinary_ps[0][2])),
                                                "undirected": float(np.log(gamma_ps[0][2]))}},
           "split": {"split_sha256": "split", "participant_assignment": dict(zip(["p0", "p1", "p2"],
                                                                                   ["train", "validation", "test"])),
                     "trial_counts": {"train": 1, "validation": 1, "test": 1}},
           "provenance": {"data_sha256": "same_complete_data", "cache_manifests": [
               {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in baseline
           ]},
           "numerical_precision": {"candidates": {
               name: {split: {"approx_pooled_log_likelihood_mc_se": .02} for split in ["train", "validation", "test"]}
               for name in ["main", "gamma"]
           }}}
    grouped_gamma_scores = np.log(gamma_repeat_ps[0].reshape(2, 2, 3).mean(axis=1))
    for t, split in enumerate(["train", "validation", "test"]):
        fit["numerical_precision"]["candidates"]["gamma"][split] = {
            "approx_pooled_log_likelihood_mc_se": float(grouped_gamma_scores[:, t].std(ddof=1) / np.sqrt(2)),
        }
    inventory = {"candidates": [{"id": "main", "runs": [{"run_dir": "seed1", "metadata_sha256": "ordinary_meta",
                                                            "params_sha256": "policy"}]}]}
    precision = {"candidates": {"main": {"splits": {
        split: {"delta_total_log_likelihood_mc_se": .01} for split in ["train", "validation", "test"]
    }}}}
    return fit, inventory, ordinary[1:], gamma[1:], baseline, precision, ordinary_ps[1], gamma_ps[1]


def test_fixed_confirmation_scores_and_combined_precision(tmp_path):
    fit, inventory, ordinary, gamma, baseline, precision, expected_main, expected_gamma = fixture(tmp_path)
    original = json.dumps(fit, sort_keys=True)
    assert baseline_paths(fit) == baseline
    report = confirmation_report(fit, inventory, ordinary, gamma, baseline, precision)
    assert not report["parameters_refitted"] and not report["models_reselected"]
    assert report["independence"]["independent"]
    assert json.dumps(fit, sort_keys=True) == original
    for row, expected in zip(report["candidates"], [expected_main, expected_gamma]):
        for t, split in enumerate(["train", "validation", "test"]):
            summary = row["splits"][split]
            np.testing.assert_allclose(summary["confirmation_total_log_likelihood"], np.log(expected[t]))
            np.testing.assert_allclose(summary["combined_delta_mc_se"],
                                       np.hypot(summary["confirmation_mc_se"], summary["baseline_mc_se"]))
    ordinary_row = report["candidates"][0]["splits"]["test"]
    assert ordinary_row["baseline_mc_se_method"] == "ordinary_within_history_M2_delta"
    assert ordinary_row["baseline_mc_se"] == .01
    gamma_row = report["candidates"][1]["splits"]["test"]
    assert gamma_row["confirmation_mc_se_method"].startswith("Gamma_")
    assert gamma_row["confirmation_analytic_mc_se"] is None
    assert gamma_row["baseline_mc_se_method"] == "Gamma_independent_raw_trajectory_repeats_only"
    assert gamma_row["baseline_independent_repeat_count"] == 4
    assert gamma_row["confirmation_independent_repeat_count"] == 2
    np.testing.assert_allclose(gamma_row["baseline_rescored_minus_original"], 0, atol=1e-14)
    with np.load(baseline[1].with_suffix(".npz")) as arrays:
        raw = arrays["t2"]
    mass = gamma_stopping_probabilities(1, 2, raw.shape[1])
    raw_probabilities = .75 * (raw[:, :, 0, 0] @ mass) + .125
    raw_scores = np.log(raw_probabilities)
    np.testing.assert_allclose(gamma_row["baseline_repeat_total_log_likelihoods"], raw_scores)
    np.testing.assert_allclose(gamma_row["baseline_mc_se"], raw_scores.std(ddof=1) / 2)
    grouped_probabilities = raw_probabilities.reshape(2, 2).mean(axis=1)
    np.testing.assert_allclose(grouped_probabilities.mean(), raw_probabilities.mean())
    grouped_se = fit["numerical_precision"]["candidates"]["gamma"]["test"]["approx_pooled_log_likelihood_mc_se"]
    np.testing.assert_allclose(grouped_se, np.log(grouped_probabilities).std(ddof=1) / np.sqrt(2))
    assert not np.isclose(gamma_row["baseline_mc_se"], grouped_se)
    gap = report["paired_family_gaps"][0]["splits"]["test"]
    np.testing.assert_allclose(gap["confirmation_gap"], np.log(expected_gamma[2]) - np.log(expected_main[2]))
    assert abs(gap["change_in_gap"]) > 0


@pytest.mark.parametrize("failure", ["same_problem_seed", "source_hash", "Gamma_metadata", "trial_path",
                                     "baseline_Gamma_score"])
def test_confirmation_rejects_nonindependent_or_mismatched_sources(tmp_path, failure):
    fit, inventory, ordinary, gamma, baseline, precision, _, _ = fixture(tmp_path)
    path = gamma[0]
    manifest = json.loads(path.read_text())
    if failure == "same_problem_seed":
        manifest["runs"][0]["estimates"][0]["seeds"][0] = 1000
    elif failure == "source_hash":
        manifest["source_sha256"]["frozen_simulator.py"] = "changed"
    elif failure == "Gamma_metadata":
        manifest["runs"][0]["metadata_sha256"] = "changed"
    elif failure == "baseline_Gamma_score":
        fit["fits"]["fits"][1]["total_log_likelihood"] += .01
    else:
        manifest["trials"][0]["paths"] = [[2], [1]]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        confirmation_report(fit, inventory, ordinary, gamma, baseline, precision)
