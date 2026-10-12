"""Frozen Gamma ranking, raw-repeat precision, and provenance integration checks."""

import copy
import hashlib
import json

import numpy as np
import pytest

from action_gamma_precision import gamma_precision_report, top_gamma_fits
from modules.action_fitting import select_fitted_candidates
from modules.undirected_likelihood_cache import gamma_stopping_probabilities
from tests.test_action_truth_confirmation import fixture, write_bundle


def gamma_fixture(tmp_path):
    fit, _, _, _, baseline, _, _, _ = fixture(tmp_path)
    path = baseline[1]
    manifest = json.loads(path.read_text())
    with np.load(path.with_suffix(".npz")) as arrays:
        original_arrays = {key: arrays[key] for key in arrays.files}
    paths, expected = [path], {}
    mass = gamma_stopping_probabilities(1, 2, 3)
    for index, neighbor in enumerate([.5, .375, .25, .75]):
        arrays = {key: raw.copy() for key, raw in original_arrays.items()}
        for raw in arrays.values():
            raw[:, :, 0, 0] += .01 * index
            raw[:, :, 1, 0] -= .01 * index
        probabilities = np.stack([.75 * (arrays[f"t{t}"][:, :, t % 2, 0] @ mass) + .125 for t in range(3)], 1)
        expected[f"gamma{index}" if index else "gamma"] = probabilities
        if not index:
            continue
        run = copy.deepcopy(manifest["runs"][0])
        run["wm_neighbor_activation"] = neighbor
        run["metadata_sha256"] = f"gamma_metadata{index}"
        for entry in run["estimates"]:
            entry["seeds"] = [seed + 10000 * index for seed in entry["seeds"]]
        added = write_bundle(tmp_path, f"additional_gamma{index}", manifest["method"], manifest["trials"], run, arrays)
        paths.append(added)
        row = copy.deepcopy(fit["fits"]["fits"][1])
        row.update(candidate_id=f"gamma{index}", total_log_likelihood=float(np.log(probabilities.mean(axis=0)[0])),
                   validation_log_likelihood=float(np.log(probabilities.mean(axis=0)[1])))
        row["parameters"]["wm_neighbor_activation"] = neighbor
        fit["fits"]["fits"].append(row)
        fit["provenance"]["cache_manifests"].append({"path": str(added),
            "sha256": hashlib.sha256(added.read_bytes()).hexdigest()})
    # The fourth-ranked law is intentionally unavailable as arrays. The diagnostic
    # must restrict array loading to its top-three frozen laws.
    paths[-1].with_suffix(".npz").write_bytes(b"unused law arrays")
    return fit, paths, expected


def test_frozen_top_three_raw_repeat_scores_and_gap_precision(tmp_path):
    fit, paths, expected = gamma_fixture(tmp_path)
    before = json.dumps(fit, sort_keys=True)
    report = gamma_precision_report(fit, paths, top_k=3)
    assert json.dumps(fit, sort_keys=True) == before
    assert not report["parameters_refitted"] and not report["models_reselected"]
    assert [row["candidate_id"] for row in report["candidates"]] == ["gamma", "gamma1", "gamma2"]
    assert len(report["used_cache_manifests"]) == 3
    assert report["same_problem_rng_audit"]["streams_checked"] == 3 * 3 * 4
    assert report["same_problem_rng_audit"]["within_law_repeats_independent"]
    for candidate in report["candidates"]:
        raw_probabilities = expected[candidate["candidate_id"]]
        for t, split in enumerate(["train", "validation", "test"]):
            row = candidate["splits"][split]
            raw_scores = np.log(raw_probabilities[:, t])
            np.testing.assert_allclose(row["raw_repeat_total_log_likelihoods"], raw_scores)
            np.testing.assert_allclose(row["raw_repeat_total_log_likelihood_mc_se"], raw_scores.std(ddof=1) / 2)
            np.testing.assert_allclose(row["pooled_total_log_likelihood"], np.log(raw_probabilities[:, t].mean()))
            assert row["raw_repeat_count"] == 4 and row["repeat_variance_degrees_of_freedom"] == 3
            if split == "test" and candidate["candidate_id"] != "gamma":
                assert row["frozen_fit_total_log_likelihood"] is None
            else:
                np.testing.assert_allclose(row["pooled_minus_frozen_fit"], 0, atol=1e-14)
            relative_se = raw_probabilities[:, t].std(ddof=1) / 2 / raw_probabilities[:, t].mean()
            np.testing.assert_allclose(row["repeat_delta_log_mc_se_quantiles"]["1"], relative_se)
            np.testing.assert_allclose(row["repeat_delta_leading_Jensen_bias_diagnostic"], -.5 * relative_se**2)
    pair = report["validation_comparisons"][0]
    left, right = [row["splits"]["validation"] for row in report["candidates"][:2]]
    np.testing.assert_allclose(pair["validation_log_likelihood_gap"],
                               left["pooled_total_log_likelihood"] - right["pooled_total_log_likelihood"])
    assert pair["rng_coupling"] == "independent_streams"
    np.testing.assert_allclose(pair["validation_gap_mc_se"],
                               np.hypot(left["raw_repeat_total_log_likelihood_mc_se"],
                                        right["raw_repeat_total_log_likelihood_mc_se"]))


def common_rng_fixture(tmp_path):
    fit, paths, expected = gamma_fixture(tmp_path)
    source = json.loads(paths[0].read_text())["runs"][0]["estimates"]
    for path in paths[1:3]:
        previous_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = json.loads(path.read_text())
        for entry, original in zip(manifest["runs"][0]["estimates"], source, strict=True):
            entry["seeds"] = original["seeds"].copy()
        path.write_text(json.dumps(manifest))
        reference = next(row for row in fit["provenance"]["cache_manifests"] if row["sha256"] == previous_hash)
        reference["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return fit, paths, expected


def test_common_rng_schedule_uses_paired_repeat_likelihood_gaps(tmp_path):
    fit, paths, _ = common_rng_fixture(tmp_path)
    report = gamma_precision_report(fit, paths)
    assert all(pair["coupling"] == "exact_paired_common_random_numbers"
               for pair in report["same_problem_rng_audit"]["cross_law_pairs"])
    pair = report["validation_comparisons"][0]
    left, right = [row["splits"]["validation"] for row in report["candidates"][:2]]
    gaps = np.asarray(left["raw_repeat_total_log_likelihoods"]) - right["raw_repeat_total_log_likelihoods"]
    covariance = np.cov(left["raw_repeat_total_log_likelihoods"], right["raw_repeat_total_log_likelihoods"])[0, 1]
    np.testing.assert_allclose(pair["raw_repeat_validation_log_likelihood_gaps"], gaps)
    np.testing.assert_allclose(pair["validation_gap_mc_se"], gaps.std(ddof=1) / 2)
    np.testing.assert_allclose(pair["validation_gap_mc_se"]**2,
                               left["raw_repeat_total_log_likelihood_mc_se"]**2
                               + right["raw_repeat_total_log_likelihood_mc_se"]**2 - 2 * covariance / 4)
    assert not np.isclose(pair["validation_gap_mc_se"],
                          np.hypot(left["raw_repeat_total_log_likelihood_mc_se"],
                                   right["raw_repeat_total_log_likelihood_mc_se"]))
    assert pair["repeat_variance_degrees_of_freedom"] == 3
    assert pair["method"] == "paired_raw_repeat_total_log_likelihood_gap_SE"


@pytest.mark.parametrize("failure", ["within_law_duplicate", "partial_coupling", "misaligned_pairing"])
def test_common_rng_schedule_rejects_duplicate_or_incomplete_pairing(tmp_path, failure):
    fit, paths, _ = common_rng_fixture(tmp_path)
    path = paths[1]
    previous_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = json.loads(path.read_text())
    seeds = manifest["runs"][0]["estimates"][0]["seeds"]
    if failure == "within_law_duplicate":
        seeds[1] = seeds[0]
    elif failure == "partial_coupling":
        seeds[0] += 100000
    else:
        seeds.reverse()
    path.write_text(json.dumps(manifest))
    reference = next(row for row in fit["provenance"]["cache_manifests"] if row["sha256"] == previous_hash)
    reference["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="RNG"):
        gamma_precision_report(fit, paths)


@pytest.mark.parametrize("failure", ["manifest_hash", "same_problem_rng", "pooled_fit_score"])
def test_gamma_precision_rejects_invalid_provenance_or_independence(tmp_path, failure):
    fit, paths, _ = gamma_fixture(tmp_path)
    if failure == "pooled_fit_score":
        fit["fits"]["fits"][1]["validation_log_likelihood"] += .01
    else:
        path = paths[1]
        manifest = json.loads(path.read_text())
        if failure == "manifest_hash":
            manifest["data_sha256"] = "different"
        else:
            manifest["runs"][0]["estimates"][0]["seeds"][0] = 2000
        path.write_text(json.dumps(manifest))
        if failure == "same_problem_rng":
            fit["provenance"]["cache_manifests"][-3]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        gamma_precision_report(fit, paths)


def test_gamma_validation_ties_match_empirical_selection(tmp_path):
    fit, _, _ = gamma_fixture(tmp_path)
    for row in fit["fits"]["fits"]:
        if row["family"] == "undirected":
            row["validation_log_likelihood"] = -5.
    ranked = top_gamma_fits(fit, 3)
    selected = select_fitted_candidates(fit["fits"]["fits"])["selected_by_validation"]["undirected"]
    assert [row["candidate_id"] for row in ranked] == ["gamma3", "gamma2", "gamma1"]
    assert ranked[0]["candidate_id"] == selected
