"""Check recovery RNG separation from JSON metadata without opening cache arrays.

Separation is required across every checkpoint used on the same empirical problem.
Seeds coinciding on different problems are deliberately outside this check: treating
those Monte Carlo errors as weakly dependent remains an approximation.
"""

from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SeedReference:
    manifest: str
    pid: str
    trial_index: int
    run_dir: str
    checkpoint: str
    wm_law: tuple | None
    repeat: int
    seed: int

    @property
    def trial_key(self):
        return self.pid, self.trial_index


def _selected_run(run, undirected, run_dirs, wm_laws):
    # Candidate run directories select learned policies; WM laws select undirected policies.
    if undirected:
        return wm_laws is None or (float(run["wm_decay"]), float(run["wm_neighbor_activation"])) in wm_laws
    return run_dirs is None or str(run["run_dir"]) in run_dirs


def iter_manifest_seeds(paths, *, repeats=None, undirected_repeats=None, run_dirs=None, wm_laws=None, trial_keys=None,
                        proposed_base_seed=None, total_records=None):
    """Yield selected seeds, retaining at most one JSON manifest at a time.

    ``None`` selects all; empty run/law lists select no learned/undirected runs.
    Filters apply separately by model family, so use ``wm_laws=[]`` to exclude
    undirected generators and ``run_dirs=[]`` to exclude learned generators.
    ``proposed_base_seed`` reproduces the current cache CLIs for a proposed job;
    undirected templates require the full input ``total_records`` (before sharding).
    Template run order must match the proposed CLI run/WM order. Explicit
    ``undirected_repeats`` overrides ``repeats`` for grouped trajectory folds.
    """
    run_dirs = None if run_dirs is None else {str(path) for path in run_dirs}
    wm_laws = None if wm_laws is None else {tuple(map(float, law)) for law in wm_laws}
    trial_keys = None if trial_keys is None else {(str(pid), int(index)) for pid, index in trial_keys}
    repeats = None if repeats is None else tuple(repeats)
    undirected_repeats = None if undirected_repeats is None else tuple(undirected_repeats)
    for selection in (repeats, undirected_repeats):
        if selection is not None and (not selection or len(set(selection)) != len(selection)
                                      or any(index < 0 for index in selection)):
            raise ValueError("repeat selection must be nonempty, unique, and nonnegative")
    for path in paths:
        path = Path(path)
        manifest = json.loads(path.read_text())
        method = manifest.get("method")
        if method not in {"planning_mc_bernstein_moments_exact_execution", "undirected_gamma_trajectory_cache"}:
            raise ValueError(f"unsupported recovery cache method in {path}: {method}")
        undirected = method == "undirected_gamma_trajectory_cache"
        selected_repeats = undirected_repeats if undirected and undirected_repeats is not None else repeats
        if proposed_base_seed is not None and undirected and (total_records is None or total_records < 1):
            raise ValueError("proposed undirected seeds require full input total_records")
        for run_index, run in enumerate(manifest["runs"]):
            if not _selected_run(run, undirected, run_dirs, wm_laws):
                continue
            wm_law = (float(run["wm_decay"]), float(run["wm_neighbor_activation"])) if undirected else None
            checkpoint = run.get("checkpoint_identity", run.get("metadata_sha256", run["run_dir"]))
            for offset, (trial, entry) in enumerate(zip(manifest["trials"], run["estimates"], strict=True)):
                key = str(trial["pid"]), int(trial["trial_index"])
                if trial_keys is not None and key not in trial_keys:
                    continue
                seeds = entry["seeds"] if undirected else [repeat["seed"] for repeat in entry]
                selected = range(len(seeds)) if selected_repeats is None else selected_repeats
                for repeat in selected:
                    if repeat >= len(seeds):
                        raise ValueError(f"missing repeat {repeat}: {path}, trial {key}, run {run['run_dir']}")
                    seed = int(seeds[repeat])
                    if proposed_base_seed is not None:
                        if undirected:
                            record_index = int(manifest["trial_start"]) + offset
                            seed = int(proposed_base_seed) + (run_index * total_records + record_index) * len(seeds)
                            seed += repeat
                        else:
                            text = f"{proposed_base_seed}:{run['checkpoint_identity']}:{trial['record_index']}:{repeat}"
                            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")
                    # JAX PRNGKey uses uint32 seed material, including sequential seeds above 2**32.
                    yield SeedReference(str(path), *key, str(run["run_dir"]), str(checkpoint), wm_law,
                                        int(repeat), seed % 2**32)


def audit_manifest_rng(fit_paths, truth_paths, *, fit_repeats=None, truth_repeats=None,
                       fit_undirected_repeats=None, truth_undirected_repeats=None,
                       fit_run_dirs=None, truth_run_dirs=None, fit_wm_laws=None, truth_wm_laws=None,
                       trial_keys=None, max_collisions=100, proposed_truth_base_seed=None,
                       truth_total_records=None):
    """Report fitting/generating seed intersections keyed by actual (pid, trial_index).

    All fitting checkpoints are compared against all selected generating checkpoints.
    Retains only selected truth seeds and streams fitting metadata; collision details
    are capped while the total collision count remains exact. A proposed seed check
    uses truth manifests as templates, with unchanged CLI run order and repeat count.
    """
    if max_collisions < 0:
        raise ValueError("max_collisions must be nonnegative")
    truth_by_trial = defaultdict(lambda: defaultdict(list))
    truth_count = 0
    for reference in iter_manifest_seeds(
        truth_paths, repeats=truth_repeats, undirected_repeats=truth_undirected_repeats,
        run_dirs=truth_run_dirs, wm_laws=truth_wm_laws,
        trial_keys=trial_keys, proposed_base_seed=proposed_truth_base_seed, total_records=truth_total_records,
    ):
        truth_by_trial[reference.trial_key][reference.seed].append(reference)
        truth_count += 1
    if not truth_count:
        raise ValueError("no generating cache seeds selected for recovery RNG audit")
    collisions, collision_count, fit_count, covered = [], 0, 0, set()
    for reference in iter_manifest_seeds(
        fit_paths, repeats=fit_repeats, undirected_repeats=fit_undirected_repeats,
        run_dirs=fit_run_dirs, wm_laws=fit_wm_laws, trial_keys=truth_by_trial,
    ):
        fit_count += 1
        covered.add(reference.trial_key)
        for truth in truth_by_trial[reference.trial_key].get(reference.seed, ()):
            collision_count += 1
            if len(collisions) < max_collisions:
                collisions.append({"fit": asdict(reference), "truth": asdict(truth)})
    missing = sorted(set(truth_by_trial) - covered)
    return {
        "independent": collision_count == 0 and not missing,
        "scope": "same (pid, trial_index), across all selected checkpoints and WM laws",
        "cross_problem_rng": "unchecked; coincident uint32 seeds across different problems imply weak dependence",
        "truth_seed_count": truth_count, "fit_seed_count": fit_count,
        "truth_trial_count": len(truth_by_trial), "missing_fit_trials": [list(key) for key in missing],
        "collision_count": collision_count, "collisions": collisions,
        "collisions_truncated": collision_count > len(collisions),
        "proposed_truth_base_seed": proposed_truth_base_seed,
    }


def assert_independent_manifest_rng(*args, **kwargs):
    """Return the successful audit report, or raise with actionable collision details."""
    report = audit_manifest_rng(*args, **kwargs)
    if not report["independent"]:
        raise ValueError("recovery fitting/generating RNG overlap or missing fit trials: "
                         + json.dumps(report, sort_keys=True))
    return report
