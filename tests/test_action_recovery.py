from types import SimpleNamespace
import hashlib
import json
import sys

import numpy as np
import pytest

from action_recovery import (
    assert_independent_manifest_repeats, assert_independent_repeats, choose_profiled_maxima, main,
    parameter_distance, profile_interval, recovery,
    sample_paths,
)
from modules.action_fitting import Candidate


def candidate(name, family, base, parameters=None):
    base = np.asarray(base)
    return Candidate(name, family, parameters or {}, lambda lapse: (1 - lapse) * base + lapse / base.shape[1])


def test_generation_matches_categorical_distribution():
    truth = np.array([[0.8, 0.2, 0], [0.1, 0.3, 0.6]])
    observed = sample_paths(truth, 10000, np.random.default_rng(1))
    np.testing.assert_allclose(np.bincount(observed[:, 0], minlength=3) / len(observed), truth[0], atol=0.015)
    np.testing.assert_allclose(np.bincount(observed[:, 1], minlength=3) / len(observed), truth[1], atol=0.015)


def test_recovery_identifies_model_cognitive_setting_and_lapse():
    n = 600
    a = np.tile([[0.95, 0.05], [0.7, 0.3]], (n // 2, 1))
    b = np.tile([[0.7, 0.3], [0.95, 0.05]], (n // 2, 1))
    candidates = [candidate("a", "optimal", a, {"wm_decay": 0.8}),
                  candidate("b", "optimal", b, {"wm_decay": 1.0}),
                  candidate("opposite", "undirected", 1 - a, {"wm_decay": 1.0})]
    result = recovery(candidates, candidates[:2], lapse_grid=np.linspace(0.01, 1, 100),
                      true_lapses=[0.2], trial_sets={"actual_design": list(range(n))}, repetitions=40, seed=4)
    assert all(cell["model_recovery_rate"] > 0.95 for cell in result["cells"])
    assert all(cell["within_family_candidate_recovery_rate"] > 0.95 for cell in result["cells"])
    assert all(abs(cell["within_family_lapse_bias"]) < 0.04 for cell in result["cells"])
    assert all(cell["within_family_profile_interval_coverage"] >= 0.8 for cell in result["cells"])
    assert result["parameter_summary"][0]["parameters"]["wm_decay"]["correlation"] > 0.95


def test_recovery_exposes_memory_lapse_nonidentifiability():
    # Two cognitive settings produce the same response law after adjusting lapse.
    candidates = [candidate("strong", "optimal", np.tile([0.9, 0.1], (100, 1)), {"wm_decay": 1.0}),
                  candidate("weak", "optimal", np.tile([0.7, 0.3], (100, 1)), {"wm_decay": 0.7})]
    result = recovery(candidates, candidates[:1], lapse_grid=np.unique(np.r_[np.linspace(0.01, 1, 100), 0.5]),
                      true_lapses=[0.5], trial_sets={"actual_design": list(range(100))}, repetitions=20, seed=8)
    assert abs(result["identifiability"][0]["expected_self_minus_best_other_cognitive_setting_log_likelihood"]) < 0.01
    assert result["cells"][0]["within_family_mean_interval_width"] > 0.3


def test_numerical_ties_are_not_broken_by_candidate_order():
    scores, counts = np.ones((1000, 2)), np.ones((1000, 2), dtype=int)
    models = choose_profiled_maxima(scores, counts, np.random.default_rng(12))
    assert 400 < np.sum(models == 0) < 600
    scores, counts = np.ones((1000, 4)), np.ones((1000, 4), dtype=int)
    models = choose_profiled_maxima(scores, counts, np.random.default_rng(13), np.array(["a", "a", "a", "b"]))
    assert 400 < np.sum(models == 3) < 600
    counts[:, 0] = 3
    models = choose_profiled_maxima(scores[:, :2], counts[:, :2], np.random.default_rng(14))
    assert 680 < np.sum(models == 0) < 820


def test_profile_envelope_and_numeric_parameter_scaling():
    lower, upper = profile_interval(np.array([[0, -4, 0], [-5, 0, -5]]), np.array([0.01, 0.5, 1]))
    np.testing.assert_allclose(lower, [0.01, 0.5])
    np.testing.assert_allclose(upper, [1, 0.5])
    candidates = [candidate("a", "optimal", [[0.5, 0.5]], {"wm_decay": 0.7, "cost": 0.1}),
                  candidate("b", "optimal", [[0.5, 0.5]], {"wm_decay": 1.0, "cost": 0.1})]
    assert parameter_distance(candidates[0], candidates[1], candidates) == pytest.approx(1)


def test_independent_cache_repeat_enforcement():
    caches = [[[SimpleNamespace(seed=1), SimpleNamespace(seed=2)]]]
    assert_independent_repeats(caches, 1, [0])
    with pytest.raises(ValueError, match="disjoint"):
        assert_independent_repeats(caches, 0, [0])
    with pytest.raises(ValueError, match="missing"):
        assert_independent_repeats(caches, 2, [0])
    caches[0][0][1].seed = 1
    with pytest.raises(ValueError, match="share MC seeds"):
        assert_independent_repeats(caches, 1, [0])


def test_invalid_normalization_is_not_silently_repaired():
    with pytest.raises(ValueError, match="sum to one"):
        sample_paths([[0.2, 0.2]], 1, np.random.default_rng(1))


def test_streamed_cache_recovery_cli(tmp_path, monkeypatch):
    trials = [{"pid": "p1", "trial_index": i, "paths": [[1], [2]], "observed_index": i % 2}
              for i in range(4)]
    runs, arrays, specifications = [], {}, []
    for run_index, base in enumerate(([0.9, 0.1], [0.1, 0.9])):
        run_dir = f"runs/policy{run_index}"
        estimates = []
        for trial in range(4):
            repeats = []
            for repeat in range(2):
                key = f"r{run_index}t{trial}s{repeat}"
                arrays[key + "_mean"] = np.column_stack([base, [0.5, 0.5]])
                repeats.append({"key": key, "samples": 100, "seed": run_index * 100 + trial * 2 + repeat})
            estimates.append(repeats)
        runs.append({"run_dir": run_dir, "estimates": estimates})
        specifications.append({"id": f"policy{run_index}", "model": ["main", "unbounded"][run_index],
                               "parameters": {}, "runs": [{"run_dir": run_dir}]})
    arrays_path = tmp_path / "cache.npz"
    np.savez(arrays_path, **arrays)
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(json.dumps({"cache_version": 1, "arrays_file": arrays_path.name,
                                     "arrays_sha256": hashlib.sha256(arrays_path.read_bytes()).hexdigest(),
                                     "trials": trials, "runs": runs}))
    candidates_path = tmp_path / "candidates.json"
    candidates_path.write_text(json.dumps({"candidates": specifications}))
    output = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", ["action_recovery.py", str(cache_path),
                                      "--candidate-manifest", str(candidates_path),
                                      "--output", str(output), "--trial-counts", "4", "--true-lapses", "0.1", "1",
                                      "--lapse-step", "0.1", "--repetitions", "4"])
    main()
    result = json.loads(output.read_text())
    assert len(result["cells"]) == 4
    assert len(result["datasets"]) == 16
    assert result["provenance"]["independent_generative_and_fitting_mc"]
    assert_independent_manifest_repeats([cache_path], 1, [0])
    with pytest.raises(ValueError, match="disjoint"):
        assert_independent_manifest_repeats([cache_path], 0, [0])
