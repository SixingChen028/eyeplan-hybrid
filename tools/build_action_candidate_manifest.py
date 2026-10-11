"""Freeze the available compatibility-11 action-fitting candidates and file hashes."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess


MAIN_EXPERIMENTS = (
    "0910_principled_selection_first_round",
    "0911_principled_selection_lower_decay_followup",
    "0911_principled_selection_neighbor_followup",
    "0925_activation_one_decay_followup",
    "0925_neighbor_activation_extremes",
)
UNBOUNDED_EXPERIMENT = "0911_principled_selection_unbounded_first_round"
UNDIRECTED_EXPERIMENTS = (
    "0911_principled_selection_undirected_first_round_random_search",
    "0911_principled_selection_undirected_upper_decay_followup_random_search",
)
SENSITIVITY_EXPERIMENT = "0909_fixed_root_50k_env64"
IGNORED_RECIPE_KEYS = {
    "cost", "wm_decay", "wm_neighbor_activation", "seed", "label", "parallel_config", "parallel_varied_keys",
}


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number_id(value):
    return format(value, ".12g").replace(".", "p").replace("-", "m")


def collect(root, experiment, model, scope):
    groups = defaultdict(list)
    for path in sorted((root / "results/runs" / experiment).glob("*/metadata.json")):
        metadata = json.loads(path.read_text())
        args = metadata["args"]
        if metadata.get("compat_version") != 11 or args["num_nodes"] != 15:
            raise ValueError(f"Incompatible candidate metadata: {path}")
        key = (args["wm_decay"], args["wm_neighbor_activation"], args["cost"])
        groups[key].append((path, metadata))
    if not groups:
        raise ValueError(f"No candidate metadata found for {experiment}")
    candidates = []
    for coordinates, rows in sorted(groups.items()):
        recipes = [{key: value for key, value in metadata["args"].items() if key not in IGNORED_RECIPE_KEYS}
                   for _, metadata in rows]
        if any(recipe != recipes[0] for recipe in recipes):
            raise ValueError(f"Recipes differ across seeds in {experiment}: {coordinates}")
        required_seeds = {1} if model == "undirected" else set(range(1, 11 if model == "unbounded" else 31))
        by_seed = {metadata["args"]["seed"]: (path, metadata) for path, metadata in rows}
        if len(by_seed) != len(rows) or not required_seeds <= by_seed.keys():
            raise ValueError(f"Missing or repeated policy seeds in {experiment}: {coordinates}")
        runs = []
        for seed in sorted(required_seeds):
            path, metadata = by_seed[seed]
            params = None if model == "undirected" else path.parent / "net_jax.p"
            if params is not None and not params.is_file():
                raise FileNotFoundError(params)
            runs.append({
                "run_dir": str(path.parent.relative_to(root)), "training_seed": seed,
                "metadata_sha256": file_hash(path), "params_file": str(params.relative_to(root)) if params else None,
                "params_sha256": file_hash(params) if params else None, "git_sha": metadata["git_sha"],
            })
        decay, neighbor, cost = coordinates
        candidate = {
            "id": f"{model}_d{number_id(decay)}_n{number_id(neighbor)}_c{number_id(cost)}",
            "model": model, "scope": scope,
            "parameters": {"wm_decay": decay, "wm_neighbor_activation": neighbor, "cost": cost},
            "experiment": experiment, "runs": runs,
            "available_seeds": sorted(by_seed), "selected_seeds": sorted(required_seeds),
            "recipe": recipes[0],
        }
        if model == "undirected":
            law_keys = ("random_search_stop_gamma_shape", "random_search_stop_gamma_scale",
                        "random_search_stop_max_fixations", "lesion_policy")
            law = {key: rows[0][1]["args"][key] for key in law_keys}
            if any({key: metadata["args"][key] for key in law_keys} != law for _, metadata in rows):
                raise ValueError(f"Undirected law varies across seeds: {experiment}")
            candidate["stopping_law"] = law
            candidate["seed_interpretation"] = "simulation RNG only; no learned policy weights"
        candidates.append(candidate)
    return candidates


def build(root):
    candidates = []
    for experiment in MAIN_EXPERIMENTS:
        candidates.extend(collect(root, experiment, "main", "primary"))
    candidates.extend(collect(root, UNBOUNDED_EXPERIMENT, "unbounded", "primary"))
    for experiment in UNDIRECTED_EXPERIMENTS:
        candidates.extend(collect(root, experiment, "undirected", "primary"))
    candidates.extend(collect(root, SENSITIVITY_EXPERIMENT, "main", "sensitivity"))
    if len({candidate["id"] for candidate in candidates}) != len(candidates):
        raise ValueError("Candidate IDs must be unique")
    main_recipes = [candidate["recipe"] for candidate in candidates if candidate["model"] == "main"]
    if any(recipe != main_recipes[0] for recipe in main_recipes):
        raise ValueError("Main candidate training/evaluation recipes differ")
    expected = {"main": 16, "unbounded": 9, "undirected": 50}
    for model, count in expected.items():
        actual = sum(candidate["model"] == model and candidate["scope"] == "primary" for candidate in candidates)
        if actual != count:
            raise ValueError(f"Expected {count} primary {model} candidates, found {actual}")
    data_path = root / "results/diagnostics/action_likelihood_20261011/all_action_trials.json"
    data = json.loads(data_path.read_text())
    problem_keys = {json.dumps({key: row[key] for key in ("graph", "start", "rewards")}, sort_keys=True)
                    for row in data}
    return {
        "schema_version": 1, "compat_version": 11,
        "builder_code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "data": {"file": str(data_path.relative_to(root)), "sha256": file_hash(data_path),
                 "trial_count": len(data), "participant_count": len({row["pid"] for row in data}),
                 "unique_exact_problems": len(problem_keys),
                 "node_counts": sorted({len(row["graph"]) for row in data})},
        "primary_candidate_counts": expected, "primary_run_count": 620,
        "interpretation": "Equal-weight seed ensembles; irregular finite grid; looks excluded from target likelihood",
        "sensitivity": "The .95/.5/.005 main candidate has the same substantive 50K/env64 recipe",
        "candidates": candidates,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.output}: {manifest['primary_candidate_counts']}, 620 primary run records")
    print(f"sha256={file_hash(args.output)}")


if __name__ == "__main__":
    main()
