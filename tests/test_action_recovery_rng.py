import hashlib
import json

import pytest

from modules.action_recovery_rng import (
    assert_independent_manifest_rng,
    audit_manifest_rng,
    iter_manifest_seeds,
)


def bundle(tmp_path, name, runs, *, trials=None, undirected=False, trial_start=0):
    trials = trials or [{"pid": "p", "trial_index": 7, "record_index": 20}]
    path = tmp_path / f"{name}.json"
    prepared = []
    for index, (run_dir, seeds) in enumerate(runs):
        run = {"run_dir": run_dir, "checkpoint_identity": f"checkpoint:{run_dir}",
               "wm_decay": 0.5 + index / 4, "wm_neighbor_activation": 0.25}
        run["estimates"] = ([{"seeds": values} for values in seeds] if undirected else
                            [[{"seed": value} for value in values] for values in seeds])
        prepared.append(run)
    path.write_text(json.dumps({
        "method": "undirected_gamma_trajectory_cache" if undirected else
                  "planning_mc_bernstein_moments_exact_execution",
        "trials": trials, "runs": prepared, "trial_start": trial_start,
    }))
    return path


def test_collision_across_different_checkpoints_is_rejected(tmp_path):
    fit = bundle(tmp_path, "fit", [("a", [[10, 11]]), ("b", [[20, 21]])])
    truth = bundle(tmp_path, "truth", [("c", [[20, 31]])])
    report = audit_manifest_rng([fit], [truth])
    assert not report["independent"]
    assert report["collision_count"] == 1
    collision = report["collisions"][0]
    assert collision["fit"]["checkpoint"] == "checkpoint:b"
    assert collision["truth"]["checkpoint"] == "checkpoint:c"
    assert collision["fit"]["pid"] == "p"
    assert collision["fit"]["trial_index"] == 7
    assert collision["fit"]["seed"] == 20
    with pytest.raises(ValueError, match='"seed": 20'):
        assert_independent_manifest_rng([fit], [truth])


def test_cross_problem_coincidences_do_not_fail_scope_guard(tmp_path):
    trials = [{"pid": "p", "trial_index": 7}, {"pid": "q", "trial_index": 7}]
    fit = bundle(tmp_path, "fit", [("a", [[10], [20]])], trials=trials)
    truth = bundle(tmp_path, "truth", [("a", [[20], [10]])], trials=trials)
    report = assert_independent_manifest_rng([fit], [truth])
    assert report["truth_trial_count"] == 2
    assert report["collision_count"] == 0
    assert "weak dependence" in report["cross_problem_rng"]


def test_truth_filters_and_repeat_selection_match_used_generators(tmp_path):
    learned = bundle(tmp_path, "learned", [("a", [[10, 11]]), ("b", [[20, 21]])])
    undirected = bundle(tmp_path, "undirected", [("u", [[30, 31]]), ("u", [[40, 41]])], undirected=True)
    report = assert_independent_manifest_rng(
        [learned, undirected], [learned, undirected], fit_repeats=[0], truth_repeats=[1],
        truth_run_dirs=["a"], truth_wm_laws=[(0.75, 0.25)],
    )
    assert report["truth_seed_count"] == 2
    assert report["fit_seed_count"] == 4
    assert report["collision_count"] == 0
    report = audit_manifest_rng(
        [learned, undirected], [learned, undirected], truth_run_dirs=[], truth_wm_laws=[(0.75, 0.25)],
    )
    assert report["truth_seed_count"] == 2
    assert report["collision_count"] == 2


def test_full_fit_folds_compared_against_full_independent_truth_folds(tmp_path):
    fit = bundle(tmp_path, "fit", [("a", [[10, 11]]), ("b", [[20, 21]])])
    truth = bundle(tmp_path, "truth", [("a", [[30, 21]])])
    report = audit_manifest_rng([fit], [truth])
    assert report["collision_count"] == 1
    assert report["collisions"][0]["fit"]["repeat"] == 1
    assert report["collisions"][0]["truth"]["repeat"] == 1


def test_grouped_undirected_repeat_indices_share_global_cross_family_guard(tmp_path):
    learned = bundle(tmp_path, "learned", [("a", [[10, 11]])])
    undirected = bundle(tmp_path, "undirected", [("u", [[20, 21, 30, 31]])], undirected=True)
    report = assert_independent_manifest_rng(
        [learned, undirected], [learned, undirected], fit_repeats=[0], truth_repeats=[1],
        fit_undirected_repeats=[0, 1], truth_undirected_repeats=[2, 3],
    )
    assert report["fit_seed_count"] == report["truth_seed_count"] == 3
    # A collision between a truth trajectory seed and a different learned fit checkpoint must fail.
    other = bundle(tmp_path, "other", [("b", [[31]])])
    report = audit_manifest_rng(
        [learned, undirected, other], [learned, undirected], fit_repeats=[0], truth_repeats=[1],
        fit_undirected_repeats=[0, 1], truth_undirected_repeats=[2, 3],
    )
    assert report["collision_count"] == 1
    assert report["collisions"][0]["fit"]["run_dir"] == "b"


def test_minimal_metadata_accepted_by_loaders_can_be_audited(tmp_path):
    learned = bundle(tmp_path, "learned", [("a", [[10, 11]])])
    metadata = json.loads(learned.read_text())
    del metadata["method"]
    learned.write_text(json.dumps(metadata))
    undirected = bundle(tmp_path, "undirected", [("u", [[20, 21]])], undirected=True)
    metadata = json.loads(undirected.read_text())
    del metadata["runs"][0]["run_dir"]
    undirected.write_text(json.dumps(metadata))
    report = assert_independent_manifest_rng(
        [learned, undirected], [learned, undirected], fit_repeats=[0], truth_repeats=[1],
    )
    assert report["fit_seed_count"] == report["truth_seed_count"] == 2


def test_uint32_wrapping_and_bounded_detailed_reports(tmp_path):
    fit = bundle(tmp_path, "fit", [("a", [[5]]), ("b", [[5]])])
    truth = bundle(tmp_path, "truth", [("c", [[2**32 + 5]])])
    report = audit_manifest_rng([fit], [truth], max_collisions=1)
    assert report["collision_count"] == 2
    assert report["collisions_truncated"]
    assert len(report["collisions"]) == 1


def test_missing_fit_trials_fail_and_empty_truth_selection_fails(tmp_path):
    fit = bundle(tmp_path, "fit", [("a", [[10]])])
    truth = bundle(tmp_path, "truth", [("a", [[20]])], trials=[{"pid": "q", "trial_index": 7}])
    report = audit_manifest_rng([fit], [truth])
    assert not report["independent"]
    assert report["missing_fit_trials"] == [["q", 7]]
    with pytest.raises(ValueError, match="no generating cache seeds"):
        audit_manifest_rng([fit], [truth], truth_run_dirs=[])


def test_learned_proposed_seed_matches_cli_sha256_rule(tmp_path):
    path = bundle(tmp_path, "learned", [("a", [[10, 11]])])
    refs = list(iter_manifest_seeds([path], proposed_base_seed=123))
    expected = [int.from_bytes(hashlib.sha256(f"123:checkpoint:a:20:{repeat}".encode()).digest()[:4],
                               "little") for repeat in range(2)]
    assert [reference.seed for reference in refs] == expected
    fit = bundle(tmp_path, "fit", [("b", [[expected[1]]])])
    report = audit_manifest_rng([fit], [path], proposed_truth_base_seed=123)
    assert report["collision_count"] == 1
    assert report["proposed_truth_base_seed"] == 123


def test_undirected_proposed_seed_matches_cli_shard_and_run_offsets(tmp_path):
    path = bundle(tmp_path, "undirected", [("u", [[0, 1]]), ("u", [[2, 3]])],
                  undirected=True, trial_start=15)
    with pytest.raises(ValueError, match="full input total_records"):
        list(iter_manifest_seeds([path], proposed_base_seed=100))
    refs = list(iter_manifest_seeds([path], proposed_base_seed=100, total_records=80,
                                    wm_laws=[(0.75, 0.25)]))
    assert [reference.seed for reference in refs] == [100 + (80 + 15) * 2 + repeat for repeat in range(2)]


@pytest.mark.parametrize("repeats", [[], [-1], [0, 0], [2]])
def test_invalid_repeat_indices_fail(tmp_path, repeats):
    path = bundle(tmp_path, "fit", [("a", [[10, 11]])])
    with pytest.raises(ValueError, match="repeat"):
        list(iter_manifest_seeds([path], repeats=repeats))
