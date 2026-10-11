"""Paired sensitivity comparisons from stored participant score sufficient statistics."""

from copy import deepcopy

import numpy as np
import pytest

from modules.action_fitting import participant_bootstrap
from tools.report_action_fits import sensitivity_comparison


def stored_fits():
    pids = np.array(["a", "b", "b", "c", "c", "c"])
    primary_scores = np.array([-1.1, -1.2, -.8, -.3, -.6, -.7])
    differences = np.array([.4, -.1, -.3, .2, .3, .7])
    bootstrap = participant_bootstrap(primary_scores, pids, repetitions=97, seed=31)

    def report(scope, scores, candidate_id):
        return {
            "scope": scope, "split": {"trial_counts": {"train": 7, "validation": 4, "test": len(pids)},
                                       "split_sha256": "same_frozen_split"},
            "provenance": {"data_sha256": "same_data", "candidate_manifest_sha256": "same_inventory"},
            "fits": {"selected_by_validation": {"main": candidate_id}, "fits": [
                {"candidate_id": candidate_id, "family": "main", "parameters": {"wm_decay": .8},
                 "lapse_rate": .3, "total_log_likelihood": -9., "validation_log_likelihood": -5.},
            ]},
            "test": {"scores": bootstrap, "total_log_likelihoods": {"main": scores.sum()},
                     "participant_scores": [
                         {"pid": pid, "trial_count": int((pids == pid).sum()),
                          "total_log_scores": {"main": scores[pids == pid].sum()}}
                         for pid in sorted(set(pids))
                     ]},
        }

    return report("primary", primary_scores, "primary"), report(
        "sensitivity", primary_scores + differences, "matched"), pids, differences


def test_aggregate_sensitivity_bootstrap_matches_trial_level_with_shuffled_participants():
    primary, sensitivity, pids, differences = stored_fits()
    sensitivity["test"]["participant_scores"].reverse()
    result = sensitivity_comparison(primary, sensitivity)
    expected = participant_bootstrap(differences, pids, repetitions=97, seed=31)
    assert result["test_mean_log_score_difference"] == pytest.approx(expected["mean_log_score"][0])
    assert result["test_mean_log_score_difference_95_interval"] == pytest.approx(
        expected["mean_log_score_95_interval"][0])
    assert result["equal_participant_mean_log_score_difference"] == pytest.approx(
        expected["equal_participant_mean_log_score"][0])
    assert result["bootstrap_probability_above_zero"] == expected["bootstrap_probability_above_zero"][0]
    assert result["total_log_likelihoods"]["test"]["sensitivity_minus_primary"] == pytest.approx(differences.sum())
    assert [row["pid"] for row in result["participant_scores"]] == sorted(set(pids))


@pytest.mark.parametrize("mismatch", ["split", "data", "participant_count", "stored_total"])
def test_sensitivity_rejects_incompatible_frozen_reports(mismatch):
    primary, original, _, _ = stored_fits()
    sensitivity = deepcopy(original)
    if mismatch == "split":
        sensitivity["split"]["split_sha256"] = "different"
    elif mismatch == "data":
        sensitivity["provenance"]["data_sha256"] = "different"
    elif mismatch == "participant_count":
        sensitivity["test"]["participant_scores"][0]["trial_count"] += 1
    else:
        sensitivity["test"]["total_log_likelihoods"]["main"] += 1
    with pytest.raises(ValueError):
        sensitivity_comparison(primary, sensitivity)
