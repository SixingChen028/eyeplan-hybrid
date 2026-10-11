import hashlib
import json

import numpy as np
import pytest

from modules.action_cache_fitting import selected_repeat_groups, stream_cache_candidates, stream_undirected_candidates


def write_bundle(tmp_path, name, arrays, manifest):
    array_path = tmp_path / f"{name}.npz"
    np.savez(array_path, **arrays)
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({**manifest, "cache_version": 1, "arrays_file": array_path.name,
                               "arrays_sha256": hashlib.sha256(array_path.read_bytes()).hexdigest()}))
    return path


def test_undirected_stream_subsets_trials_and_groups_before_retaining_arrays(tmp_path):
    trials = [{"pid": "p", "trial_index": i, "paths": [[1], [2]], "observed_index": 0} for i in range(2)]
    arrays, entries = {}, []
    for trial in range(2):
        raw = np.empty((4, 3, 2, 2))
        for repeat, left in enumerate([0.1, 0.3, 0.7, 0.9]):
            raw[repeat] = np.tile(np.column_stack([[left, 1 - left], [0.5, 0.5]]), (3, 1, 1))
        arrays[f"trial{trial}"] = raw
        entries.append({"key": f"trial{trial}", "seeds": list(range(4 * trial, 4 * trial + 4))})
    path = write_bundle(tmp_path, "undirected", arrays, {
        "method": "undirected_gamma_trajectory_cache", "trials": trials,
        "runs": [{"wm_decay": 0.75, "wm_neighbor_activation": 0.25, "max_fixations": 3, "estimates": entries}],
    })
    manifest, pooled, repeats = stream_undirected_candidates(
        [path], [1], [4], trial_keys=[("p", 1)], repeat_groups=[[0, 1], [2, 3]],
    )
    assert [trial["trial_index"] for trial in manifest["trials"]] == [1]
    assert manifest["repeat_groups"] == [[0, 1], [2, 3]]
    assert len(repeats) == 2
    np.testing.assert_allclose(pooled[0].probabilities(0.2), [[0.5, 0.5]])
    np.testing.assert_allclose(repeats[0][0].probabilities(0.2), [[0.26, 0.74]])
    np.testing.assert_allclose(repeats[1][0].probabilities(0.2), [[0.74, 0.26]])
    _, selected, selected_repeats = stream_undirected_candidates(
        [path], [1], [4], trial_keys=[("p", 1), ("p", 0)], repeat_indices=[3],
    )
    assert len(selected_repeats) == 1
    np.testing.assert_allclose(selected[0].probabilities(0.2), [[0.82, 0.18], [0.82, 0.18]])


def test_ordinary_stream_subsets_trials_and_sample_weighted_repeat_means(tmp_path):
    trials = [{"pid": "p", "trial_index": i, "paths": [[1], [2]], "observed_index": i} for i in range(2)]
    arrays, estimates = {}, []
    for trial in range(2):
        entries = []
        for repeat, left in enumerate([0.1, 0.9]):
            key = f"trial{trial}repeat{repeat}"
            arrays[key + "_mean"] = np.column_stack([[left, 1 - left], [0.5, 0.5]])
            entries.append({"key": key, "samples": [2, 6][repeat], "seed": 2 * trial + repeat})
        estimates.append(entries)
    path = write_bundle(tmp_path, "ordinary", arrays, {
        "trials": trials, "runs": [{"run_dir": "run", "estimates": estimates}],
    })
    specifications = {"candidates": [{"id": "a", "model": "main", "parameters": {}, "runs": [{"run_dir": "run"}]}]}
    manifest, pooled, observed, repeats = stream_cache_candidates([path], specifications, trial_keys=[("p", 1)])
    assert observed.tolist() == [1]
    np.testing.assert_allclose(pooled[0].probabilities(0.2), [[0.66, 0.34]])
    assert len(repeats) == 2
    manifest, pooled, _, repeats = stream_cache_candidates(
        [path], specifications, trial_keys=[("p", 1)], repeat_indices=[1],
    )
    assert manifest["repeat_groups"] == [[1]]
    assert len(repeats) == 1
    np.testing.assert_allclose(pooled[0].probabilities(0.2), [[0.82, 0.18]])


@pytest.mark.parametrize("groups", [[], [[]], [[0], [0]], [[-1]], [[4]]])
def test_invalid_repeat_grouping_is_rejected(groups):
    with pytest.raises(ValueError, match="nonempty, disjoint, and in range"):
        selected_repeat_groups(4, repeat_groups=groups)


@pytest.mark.parametrize("fast_observed", [False, True])
def test_aligned_likelihood_remaps_supplied_responses_and_subset_order(fast_observed):
    from modules.action_cache_fitting import align_candidate_trials, coefficient_candidates
    from modules.action_fitting import Candidate, observed_log_probabilities

    source = [{"pid": "p", "trial_index": i, "paths": [[1], [2]], "observed_index": i % 2}
              for i in range(4)]
    target_order = np.array([2, 0, 3, 1])
    target = [source[index] for index in target_order]
    coefficients = np.stack([np.column_stack([base, [.5, .5]]) for base in
                             ([.9, .1], [.3, .7], [.6, .4], [.2, .8])])
    original = coefficient_candidates([{"id": "gamma", "model": "undirected", "parameters": {}}],
                                      [coefficients], source)[0]
    if not fast_observed:
        original = Candidate(original.candidate_id, original.family, original.parameters,
                             original.evaluate, original.evaluate_subset)
    aligned = align_candidate_trials([original], source, target)[0]
    synthetic = 1 - np.array([trial["observed_index"] for trial in target])
    source_synthetic = synthetic[np.argsort(target_order)]
    for indices in (None, np.array([0, 2]), np.array([3, 1, 0])):
        source_rows = target_order if indices is None else target_order[indices]
        expected = observed_log_probabilities(original, .23, source_synthetic, source_rows)
        np.testing.assert_allclose(observed_log_probabilities(aligned, .23, synthetic, indices), expected)
    with pytest.raises(ValueError, match="aligned trial count"):
        aligned.evaluate_observed(.23, synthetic[:-1], None)
