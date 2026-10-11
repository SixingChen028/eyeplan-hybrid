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
