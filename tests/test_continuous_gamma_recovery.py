import numpy as np
import pytest

from continuous_gamma_recovery import continuous_recovery, execution_weights, screened_starts
from modules.undirected_likelihood_cache import gamma_stopping_probabilities


def trajectory_fixture():
    targets, trials_per_condition = 20, 80
    trials, arrays = [], []
    for threshold in (1, 3, 6, 10, 16):
        left = 0.5 + 0.45 * (np.arange(targets) >= threshold)
        coefficients = np.stack([np.stack([left, np.full(targets, 0.5)], axis=-1),
                                 np.stack([1 - left, np.full(targets, 0.5)], axis=-1)], axis=1)
        for _ in range(trials_per_condition):
            trials.append({"paths": [[1], [2]]})
            arrays.append(coefficients)
    return trials, np.stack(arrays, axis=1)


def test_continuous_recovery_uses_sampled_paths_and_reports_bounded_stopping_law():
    trials, coefficients = trajectory_fixture()
    result = continuous_recovery(
        trials, coefficients, coefficients.copy(),
        true_settings=[{"gamma_shape": 2, "gamma_mean": 8, "lapse_rate": 0.15}],
        trial_sets={"actual_design": list(range(len(trials)))}, repetitions=2, seed=11,
        gamma_shapes=[1, 2, 4], gamma_means=[4, 8, 16], optimizer_maxiter=100,
    )
    assert len(result["datasets"]) == 2
    assert result["cells"][0]["datasets"] == 2
    for row in result["datasets"]:
        fit = row["fit"]
        assert np.isfinite(fit["train_log_likelihood"])
        assert 1 <= fit["mean_total_fixations"] <= coefficients.shape[0]
        assert 0 <= fit["maximum_target_fraction"] <= 1
        assert fit["training_trials"] == len(trials)
        assert len(row["observed_path_indices"]) == len(trials)
        assert len(fit["starts"]) >= 3
        mass = gamma_stopping_probabilities(2, 4, coefficients.shape[0])
        assert row["truth"]["mean_total_fixations"] == pytest.approx(1 + mass @ np.arange(len(mass)))
        fitted_mass = gamma_stopping_probabilities(fit["gamma_shape"], fit["gamma_scale"], coefficients.shape[0])
        weights = execution_weights(trials, fit["lapse_rate"], coefficients.shape[2], coefficients.shape[3])
        probabilities = np.einsum("t,tnpk,npk->np", fitted_mass, coefficients, weights)
        expected_score = np.log(probabilities[np.arange(len(trials)), row["observed_path_indices"]]).sum()
        assert fit["train_log_likelihood"] == pytest.approx(expected_score)
    assert "lapse_rate" in result["summary_by_design"][0]["parameters"]


def test_execution_weights_handle_unequal_path_degrees_and_padding():
    trials = [{"paths": [[1], [2, 3], [2, 4]]}, {"paths": [[1], [2]]}]
    weights = execution_weights(trials, 0.2, 3, 3)
    np.testing.assert_allclose(weights[0, 0], [0.8, 0.2, 0])
    np.testing.assert_allclose(weights[0, 1], [0.64, 0.32, 0.04])
    np.testing.assert_array_equal(weights[1, 2], 0)


def test_screen_chooses_distinct_gamma_starts_and_rejects_zero_probability():
    coefficients = np.ones((3, 10, 2)) * 0.5
    starts = screened_starts(coefficients, np.ones(10, dtype=int), [1, 2], [4, 8])
    assert len({(start["gamma_shape"], start["gamma_mean"]) for start in starts}) == 3
    with pytest.raises(ValueError, match="positive"):
        screened_starts(np.zeros_like(coefficients), np.ones(10, dtype=int), [1], [4])
