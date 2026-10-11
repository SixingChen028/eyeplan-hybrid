"""Stream latent-MC cache shards into compact seed-averaged fitting candidates."""

import hashlib
import json
from math import comb
from pathlib import Path

import numpy as np

from modules.action_fitting import Candidate


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def coefficient_candidates(specifications, coefficients, trials):
    """Wrap compact coefficients with vectorized, own-path-degree evaluation."""
    max_paths = max(len(trial["paths"]) for trial in trials)
    max_degree = max(len(path) for trial in trials for path in trial["paths"])
    lengths = np.zeros((len(trials), max_paths), dtype=int)
    binomial = np.zeros((len(trials), max_paths, max_degree + 1))
    for t, trial in enumerate(trials):
        for p, path in enumerate(trial["paths"]):
            lengths[t, p] = len(path)
            binomial[t, p, :len(path) + 1] = [comb(len(path), k) for k in range(len(path) + 1)]
    powers = np.arange(max_degree + 1)[None, None, :]
    remaining = np.maximum(lengths[:, :, None] - powers, 0)
    candidates = []
    for specification, mean in zip(specifications, coefficients, strict=True):
        def subset(lapse, indices, mean=mean):
            weights = binomial[indices] * lapse**powers * (1 - lapse)**remaining[indices]
            return np.sum(mean[indices] * weights, axis=-1)

        def evaluate(lapse, mean=mean):
            weights = binomial * lapse**powers * (1 - lapse)**remaining
            return np.sum(mean * weights, axis=-1)

        def observed(lapse, observed_indices, indices, mean=mean):
            rows = np.arange(len(mean)) if indices is None else np.asarray(indices)
            paths = observed_indices[rows]
            weights = binomial[rows, paths] * lapse**powers[0] * (1 - lapse)**remaining[rows, paths]
            return np.sum(mean[rows, paths] * weights, axis=-1)

        candidates.append(Candidate(specification["id"], specification["model"], specification["parameters"],
                                    evaluate, subset, observed))
    return candidates


def stream_cache_candidates(paths, candidate_manifest):
    """Read one NPZ mean array at a time, discard per-sample covariance arrays.

    Returns minimal manifest, pooled candidates, observed indices, and candidate
    lists for each independent MC repeat. Equal-weight checkpoint mixtures are
    preserved even when repeats have different latent-sample counts.
    """
    specifications = candidate_manifest["candidates"]
    expected = {}
    for c, specification in enumerate(specifications):
        for run in specification["runs"]:
            key = str(Path(run["run_dir"]))
            if key in expected:
                raise ValueError(f"checkpoint belongs to multiple candidates: {key}")
            expected[key] = (c, run)

    def match_run(run_dir):
        matches = [key for key in expected if str(Path(run_dir)) == key or str(Path(run_dir)).endswith("/" + key)]
        if len(matches) > 1:
            raise ValueError(f"ambiguous cache run identity: {run_dir}")
        return matches[0] if matches else None

    trials_by_key = {}
    available_runs = set()
    repeat_count = None
    provenance = {}
    for path in paths:
        manifest = json.loads(Path(path).read_text())
        if manifest["cache_version"] != 1:
            raise ValueError("unsupported lapse cache version")
        for trial in manifest["trials"]:
            key = (str(trial["pid"]), int(trial["trial_index"]))
            if key in trials_by_key and trial != trials_by_key[key]:
                raise ValueError(f"inconsistent trial identity: {key}")
            trials_by_key[key] = trial
        for run in manifest["runs"]:
            key = match_run(run["run_dir"])
            if key is None:
                continue
            available_runs.add(key)
            _, identity = expected[key]
            for field in ("metadata_sha256", "params_sha256"):
                if field in identity and identity[field] != run[field]:
                    raise ValueError(f"checkpoint {field} differs from frozen candidate inventory: {key}")
            provenance[key] = {field: value for field, value in run.items() if field != "estimates"}
            for estimates in run["estimates"]:
                if repeat_count is None:
                    repeat_count = len(estimates)
                if len(estimates) != repeat_count or not estimates:
                    raise ValueError("all selected run/trial caches need the same positive independent-repeat count")
    missing = set(expected) - available_runs
    if missing:
        raise ValueError(f"missing {len(missing)} candidate checkpoint caches, first: {sorted(missing)[0]}")
    keys = sorted(trials_by_key)
    trials = [trials_by_key[key] for key in keys]
    indices = {key: index for index, key in enumerate(keys)}
    shape = (len(trials), max(len(t["paths"]) for t in trials),
             max(len(p) for t in trials for p in t["paths"]) + 1)
    pooled = [np.zeros(shape) for _ in specifications]
    repeats = [[np.zeros(shape) for _ in specifications] for _ in range(repeat_count)]
    seen = {key: np.zeros(len(trials), dtype=bool) for key in expected}
    for path in paths:
        path = Path(path)
        manifest = json.loads(path.read_text())
        array_path = path.parent / manifest["arrays_file"]
        if file_sha256(array_path) != manifest["arrays_sha256"]:
            raise ValueError(f"cache arrays hash mismatch: {array_path}")
        with np.load(array_path, allow_pickle=False) as arrays:
            for run in manifest["runs"]:
                key = match_run(run["run_dir"])
                if key is None:
                    continue
                c, _ = expected[key]
                for trial, estimates in zip(manifest["trials"], run["estimates"], strict=True):
                    t = indices[(str(trial["pid"]), int(trial["trial_index"]))]
                    if seen[key][t]:
                        raise ValueError(f"duplicate cache checkpoint/trial: {key} {t}")
                    seen[key][t] = True
                    sample_total = sum(entry["samples"] for entry in estimates)
                    for r, entry in enumerate(estimates):
                        mean = arrays[entry["key"] + "_mean"]
                        if mean.shape[0] != len(trial["paths"]) or mean.shape[1] > shape[-1]:
                            raise ValueError("cache coefficient shape differs from trial paths")
                        selection = (t, slice(0, mean.shape[0]), slice(0, mean.shape[1]))
                        pooled[c][selection] += mean * entry["samples"] / sample_total
                        repeats[r][c][selection] += mean
    for key, mask in seen.items():
        if not np.all(mask):
            raise ValueError(f"checkpoint {key} is missing {int((~mask).sum())} trial caches")
    for c, specification in enumerate(specifications):
        count = len(specification["runs"])
        pooled[c] /= count
        for repeat in repeats:
            repeat[c] /= count
    manifest = {"trials": trials, "runs": [provenance[key] for key in sorted(provenance)]}
    observed = np.asarray([trial["observed_index"] for trial in trials], dtype=int)
    candidates = coefficient_candidates(specifications, pooled, trials)
    repeat_candidates = [coefficient_candidates(specifications, means, trials) for means in repeats]
    return manifest, candidates, observed, repeat_candidates


def stream_undirected_candidates(paths, gamma_shapes=(.5, 1, 2, 4, 8), gamma_means=(2, 4, 8, 16, 32, 64),
                                 candidate_specs_by_law=None):
    """Integrate Gamma grids one trajectory array at a time, freeing raw paths.

    Every supplied WM law must cover exactly the same complete trial set. The
    returned candidates contain compact integrated Bernstein coefficients only.
    """
    from modules.undirected_likelihood_cache import gamma_stopping_probabilities

    trials_by_key, laws = {}, {}
    repeat_count = None
    for path in paths:
        manifest = json.loads(Path(path).read_text())
        if manifest["cache_version"] != 1 or manifest["method"] != "undirected_gamma_trajectory_cache":
            raise ValueError("unsupported undirected cache format")
        for trial in manifest["trials"]:
            key = (str(trial["pid"]), int(trial["trial_index"]))
            if key in trials_by_key and trial != trials_by_key[key]:
                raise ValueError(f"inconsistent undirected trial identity: {key}")
            trials_by_key[key] = trial
        for run in manifest["runs"]:
            law = (run["wm_decay"], run["wm_neighbor_activation"])
            if law in laws and laws[law] != run["max_fixations"]:
                raise ValueError("inconsistent undirected target horizon")
            laws[law] = run["max_fixations"]
            for entry in run["estimates"]:
                if repeat_count is None:
                    repeat_count = len(entry["seeds"])
                if len(entry["seeds"]) != repeat_count:
                    raise ValueError("undirected independent-repeat counts differ")
    if candidate_specs_by_law is not None:
        laws = {law: horizon for law, horizon in laws.items() if law in candidate_specs_by_law}
        if set(laws) != set(candidate_specs_by_law):
            raise ValueError("refined undirected WM law is missing from trajectory caches")
    keys = sorted(trials_by_key)
    trials = [trials_by_key[key] for key in keys]
    indices = {key: index for index, key in enumerate(keys)}
    dimensions = (len(trials), max(len(t["paths"]) for t in trials),
                  max(len(p) for t in trials for p in t["paths"]) + 1)
    shapes_and_means = [(float(shape), float(mean)) for shape in gamma_shapes for mean in gamma_means]
    specifications, coefficient_arrays, repeat_arrays = [], [], [[] for _ in range(repeat_count)]
    law_arrays, law_masses, seen = {}, {}, {}
    for law in sorted(laws):
        decay, neighbor = law
        settings = shapes_and_means if candidate_specs_by_law is None else [
            (spec["gamma_shape"], spec["gamma_mean"]) for spec in candidate_specs_by_law[law]
        ]
        means = np.zeros((len(settings), *dimensions))
        repeated = np.zeros((len(settings), repeat_count, *dimensions))
        law_arrays[law] = (means, repeated)
        law_masses[law] = np.stack([gamma_stopping_probabilities(shape, mean / shape, laws[law])
                                    for shape, mean in settings])
        seen[law] = np.zeros(len(trials), dtype=bool)
        for g, (shape, mean) in enumerate(settings):
            identifier = f"undirected_decay{decay:g}_neighbor{neighbor:g}_shape{shape:g}_mean{mean:g}"
            if candidate_specs_by_law is not None:
                identifier += "_refined"
            specifications.append({"id": identifier, "model": "undirected", "parameters": {
                "wm_decay": decay, "wm_neighbor_activation": neighbor,
                "gamma_shape": shape, "gamma_scale": mean / shape, "gamma_mean": mean,
            }})
            coefficient_arrays.append(means[g])
            for r in range(repeat_count):
                repeat_arrays[r].append(repeated[g, r])
    for path in paths:
        path = Path(path)
        manifest = json.loads(path.read_text())
        array_path = path.parent / manifest["arrays_file"]
        if file_sha256(array_path) != manifest["arrays_sha256"]:
            raise ValueError(f"undirected cache arrays hash mismatch: {array_path}")
        with np.load(array_path, allow_pickle=False) as arrays:
            for run in manifest["runs"]:
                law = (run["wm_decay"], run["wm_neighbor_activation"])
                if law not in law_arrays:
                    continue
                means, repeated = law_arrays[law]
                for trial, entry in zip(manifest["trials"], run["estimates"], strict=True):
                    t = indices[(str(trial["pid"]), int(trial["trial_index"]))]
                    if seen[law][t]:
                        raise ValueError(f"duplicate undirected WM/trial cache: {law} {t}")
                    seen[law][t] = True
                    raw = arrays[entry["key"]]
                    if raw.ndim != 4 or raw.shape[:2] != (repeat_count, laws[law]):
                        raise ValueError("undirected trajectory array shape differs from manifest")
                    integrated = np.einsum("rtpk,gt->grpk", raw, law_masses[law], optimize=True)
                    repeated[:, :, t, :raw.shape[2], :raw.shape[3]] = integrated
                    means[:, t, :raw.shape[2], :raw.shape[3]] = integrated.mean(axis=1)
    for law, mask in seen.items():
        if not np.all(mask):
            raise ValueError(f"undirected WM {law} is missing {int((~mask).sum())} trial caches")
    candidates = coefficient_candidates(specifications, coefficient_arrays, trials)
    repeat_candidates = [coefficient_candidates(specifications, arrays, trials) for arrays in repeat_arrays]
    return {"trials": trials, "wm_laws": [list(law) for law in sorted(laws)]}, candidates, repeat_candidates


def undirected_observed_coefficients(paths, law, trial_keys, *, repeat_indices=None):
    """Stream only observed-path target coefficients for one WM law.

    Output is [target, trial, coefficient], with independent-repeat means. This
    supports continuous Gamma optimization without retaining full trajectories.
    """
    indices = {tuple(key): index for index, key in enumerate(trial_keys)}
    result = None
    path_lengths = np.zeros(len(trial_keys), dtype=int)
    seen = np.zeros(len(trial_keys), dtype=bool)
    for path in paths:
        path = Path(path)
        manifest = json.loads(path.read_text())
        matching = [run for run in manifest["runs"]
                    if (run["wm_decay"], run["wm_neighbor_activation"]) == tuple(law)]
        if not matching:
            continue
        array_path = path.parent / manifest["arrays_file"]
        if file_sha256(array_path) != manifest["arrays_sha256"]:
            raise ValueError("undirected cache arrays hash mismatch")
        with np.load(array_path, allow_pickle=False) as arrays:
            for run in matching:
                for trial, entry in zip(manifest["trials"], run["estimates"], strict=True):
                    key = (str(trial["pid"]), int(trial["trial_index"]))
                    if key not in indices:
                        continue
                    t = indices[key]
                    if seen[t]:
                        raise ValueError("duplicate observed-coefficient trial")
                    raw = arrays[entry["key"]]
                    if repeat_indices is not None:
                        raw = raw[list(repeat_indices)]
                    observed = raw[:, :, trial["observed_index"]].mean(axis=0)
                    if result is None:
                        result = np.zeros((observed.shape[0], len(trial_keys), observed.shape[1]))
                    elif observed.shape[1] > result.shape[2]:
                        result = np.pad(result, ((0, 0), (0, 0), (0, observed.shape[1] - result.shape[2])))
                    result[:, t, :observed.shape[1]] = observed
                    path_lengths[t] = len(trial["paths"][trial["observed_index"]])
                    seen[t] = True
    if result is None or not np.all(seen):
        raise ValueError("continuous Gamma refinement is missing observed-path trajectory caches")
    return result, path_lengths


def align_candidate_trials(candidates, source_trials, target_trials):
    """Reorder callback indices without copying compact coefficient tensors."""
    source = {(str(trial["pid"]), int(trial["trial_index"])): i for i, trial in enumerate(source_trials)}
    target_keys = [(str(trial["pid"]), int(trial["trial_index"])) for trial in target_trials]
    if set(source) != set(target_keys):
        raise ValueError("ordinary and undirected trial sets differ")
    order = np.asarray([source[key] for key in target_keys])
    source_observed = np.asarray([trial["observed_index"] for trial in source_trials], dtype=int)
    for row, index in zip(target_trials, order):
        if row["paths"] != source_trials[index]["paths"] or row["observed_index"] != source_observed[index]:
            raise ValueError("ordinary and undirected path order or observations differ")
    aligned = []
    for candidate in candidates:
        def evaluate(lapse, candidate=candidate):
            return candidate.probabilities(lapse, order)

        def subset(lapse, indices, candidate=candidate):
            return candidate.probabilities(lapse, order[indices])

        def observed(lapse, observed_indices, indices, candidate=candidate):
            rows = order if indices is None else order[indices]
            if candidate.evaluate_observed is not None:
                return candidate.evaluate_observed(lapse, source_observed, rows)
            probabilities = candidate.probabilities(lapse, rows)
            return probabilities[np.arange(len(rows)), source_observed[rows]]

        aligned.append(Candidate(
            candidate.candidate_id, candidate.family, candidate.parameters, evaluate, subset, observed,
        ))
    return aligned
