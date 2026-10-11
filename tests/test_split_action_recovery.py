import numpy as np
import pytest

from modules.action_fitting import Candidate, fit_all_candidates, observed_log_probabilities
from split_action_recovery import coarse_gamma_fits, sparse_responses, split_recovery


def fixture():
    trials = [{"pid": str(i // 12), "trial_index": i % 12, "paths": [[1], [2]], "observed_index": 0}
              for i in range(36)]
    split = {"participant_assignment": {"0": "train", "1": "validation", "2": "test"},
             "trial_counts": {"train": 12, "validation": 12, "test": 12}}
    candidates = []
    for name, family, base, decay in [("a", "main", [.95, .05], .8),
                                       ("b", "main", [.1, .9], 1.),
                                       ("u", "unbounded", [.2, .8], 1.)]:
        base = np.tile(base, (len(trials), 1))
        candidates.append(Candidate(name, family, {"wm_decay": decay},
                                    lambda lapse, base=base: (1 - lapse) * base + lapse / 2))
    return trials, split, candidates


def test_sparse_scores_are_complete_trial_log_likelihoods():
    observations = np.array([[0, 1, 0], [1, 1, 0]])
    logs = np.log([[.8, .2], [.4, .6], [.7, .3]])
    scores = sparse_responses(observations, np.array([0, 2]), 2) @ logs.ravel()
    np.testing.assert_allclose(scores, [np.log(.8 * .7), np.log(.2 * .7)])


def test_primary_replays_empirical_selection_and_frozen_test_scores():
    trials, split, candidates = fixture()
    result = split_recovery(candidates, candidates[:1], trials, split, truth_lapses={"a": .17},
                            repetitions=4, seed=2, profile_points=11)
    indices = {name: np.arange(i * 12, (i + 1) * 12)
               for i, name in enumerate(("train", "validation", "test"))}
    assert result["actual_trial_counts"] == split["trial_counts"]
    for row in result["datasets"]:
        observed = np.array(row["observed_path_indices"])
        manual = fit_all_candidates(candidates, observed, indices, profile_points=11)
        selected = row["selected_by_validation"]
        assert {key: value for key, value in selected.items() if key != "random"} == manual["selected_by_validation"]
        assert "random" in row["family_test_log_likelihoods"]
        for fit in manual["fits"]:
            if selected[fit["family"]] != fit["candidate_id"]:
                continue
            candidate = next(item for item in candidates if item.candidate_id == fit["candidate_id"])
            test = observed_log_probabilities(candidate, fit["lapse_rate"], observed, indices["test"]).sum()
            assert row["family_test_log_likelihoods"][fit["family"]] == pytest.approx(test)
            assert row["selected_family_fits"][fit["family"]]["lapse_rate"] == fit["lapse_rate"]
        scores = row["family_test_log_likelihoods"]
        assert row["test_gap_to_best_competitor"] == pytest.approx(
            scores["main"] - max(value for family, value in scores.items() if family != "main"))
    assert result["cells"][0]["model_recovery_wilson_95_interval"]


def test_gamma_refinement_uses_training_observations_only_and_resumes(tmp_path, monkeypatch):
    import modules.undirected_likelihood_cache as gamma_module
    from modules.action_cache_fitting import coefficient_candidates

    trials, split, ordinary = fixture()
    raw = np.zeros((2, 3, len(trials), 2, 2))
    for target, base in enumerate(([.9, .1], [.5, .5], [.1, .9])):
        raw[:, target, :, :, 0] = base
        raw[:, target, :, :, 1] = .5
    specifications = [{"id": f"gamma{i}", "model": "undirected", "parameters": {
        "wm_decay": .8, "wm_neighbor_activation": .5, "gamma_shape": 1., "gamma_mean": mean,
        "gamma_scale": mean}} for i, mean in enumerate((1., 3., 8.))]
    coefficients = [np.einsum("t,tnpk->npk", gamma_module.gamma_stopping_probabilities(
        spec["parameters"]["gamma_shape"], spec["parameters"]["gamma_scale"], 3), raw.mean(axis=0))
        for spec in specifications]
    gamma_candidates = coefficient_candidates(specifications, coefficients, trials)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("fixture")
    loader_calls, optimizer_calls = [], []
    def load(paths, decay, neighbor, trial_keys):
        loader_calls.append((decay, neighbor))
        return trials, raw, np.zeros((2, len(trials)))
    original = gamma_module.refine_gamma_coefficients
    def refine(coefficients, lengths, starts, **kwargs):
        optimizer_calls.append(len(lengths))
        assert coefficients.shape == (3, 12, 2)
        return original(coefficients, lengths, starts, **kwargs)
    monkeypatch.setattr(gamma_module, "load_undirected_trajectory_arrays", load)
    monkeypatch.setattr(gamma_module, "refine_gamma_coefficients", refine)
    kwargs = dict(truth_lapses={"gamma0": .2}, repetitions=2, seed=9, profile_points=7,
                  continuous_gamma_paths=[manifest], optimizer_maxiter=20, checkpoint_dir=tmp_path / "checkpoints")
    first = split_recovery(ordinary + gamma_candidates, gamma_candidates[:1], trials, split, **kwargs)
    second = split_recovery(ordinary + gamma_candidates, gamma_candidates[:1], trials, split, **kwargs)
    assert first == second
    assert loader_calls == [(.8, .5)]
    assert optimizer_calls == [12, 12]
    assert first["gamma_recovery"] == "continuous_at_every_working_memory_law"
    kwargs["seed"] = 10
    with pytest.raises(ValueError, match="checkpoint differs"):
        split_recovery(ordinary + gamma_candidates, gamma_candidates[:1], trials, split, **kwargs)


def test_missing_split_rejected_before_fitting():
    trials, split, candidates = fixture()
    split["participant_assignment"]["2"] = "train"
    with pytest.raises(ValueError, match="every frozen participant split"):
        split_recovery(candidates, candidates[:1], trials, split, truth_lapses={"a": .1})


def test_sparse_initialization_profiles_match_empirical_continuous_lapse():
    from modules.action_fitting import fit_lapse, refine_lapse_profile

    trials, split, candidates = fixture()
    candidate = candidates[0]
    candidate.family = "undirected"
    observations = np.tile([0, 0, 1], 12)[None, :]
    indices = {name: np.arange(i * 12, (i + 1) * 12)
               for i, name in enumerate(("train", "validation", "test"))}
    records = coarse_gamma_fits([candidate], observations, indices, (36, 2), profile_points=11)
    reference = fit_lapse(candidate, observations[0], indices["train"], profile_points=11)
    assert records[0][0]["lapse_rate"] == pytest.approx(reference["lapse_rate"], abs=1e-7)
    assert records[0][0]["total_log_likelihood"] == pytest.approx(reference["total_log_likelihood"], abs=1e-10)
    grid = np.linspace(.01, 1, 11)
    scores = [observed_log_probabilities(candidate, lapse, observations[0], indices["train"]).sum()
              for lapse in grid]
    assert refine_lapse_profile(candidate, observations[0], indices["train"], grid, scores) == reference
    with pytest.raises(ValueError, match="precomputed lapse profile"):
        refine_lapse_profile(candidate, observations[0], indices["train"], grid, scores[:-1])
