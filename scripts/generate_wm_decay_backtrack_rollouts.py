#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from modules.config import ENV_DYNAMIC_PARAM_KEYS, load_canonical_defaults
from modules.environment import JaxDecisionTreeEnv
from modules.simulation import append_simulation_trial, empty_simulation_data


WM_DECAYS = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_NAME = "wm_decay_backtrack_rollouts"
DEFAULT_TREE_VIEWER = Path("/Users/fred/projects/eyeplan/tree-viewer")


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def _build_env(params: dict[str, Any]) -> JaxDecisionTreeEnv:
    return JaxDecisionTreeEnv(
        num_nodes=int(params["num_nodes"]),
        t_max=int(params["t_max"]),
        scale_factor=float(params["scale_factor"]),
        shuffle_nodes=bool(params["shuffle_nodes"]),
        use_recency_obs=bool(params["use_recency_obs"]),
        use_best_open_value_obs=bool(params["use_best_open_value_obs"]),
        use_best_terminal_value_obs=bool(params["use_best_terminal_value_obs"]),
        wm_backup=bool(params["wm_backup"]),
        point_set=params["point_set"],
    )


def _make_env_params(env: JaxDecisionTreeEnv, params: dict[str, Any]):
    dynamic = {key: params[key] for key in ENV_DYNAMIC_PARAM_KEYS}
    return env.make_params(**dynamic)


def _policy_logits(action_mask: np.ndarray, action: int) -> list[float]:
    logits = np.where(action_mask, 0.0, -1_000_000.0).astype(np.float32)
    logits[action] = 8.0
    return logits.tolist()


def _choose_action(
    state,
    action_mask: np.ndarray,
    rng: np.random.Generator,
    *,
    backtrack_prob: float,
    min_steps_before_terminate: int,
    step_idx: int,
    num_nodes: int,
    t_max: int,
) -> int:
    current = int(state.fixation_node)
    child_nodes = np.asarray(state.child_nodes)
    parent_nodes = np.asarray(state.parent_nodes)

    if step_idx >= t_max - 1:
        return num_nodes

    root = int(state.root_node)
    parent = int(parent_nodes[current])
    can_backtrack = parent >= 0 and bool(action_mask[parent])

    children = [
        int(child)
        for child in child_nodes[current]
        if child >= 0 and bool(action_mask[int(child)])
    ]

    if can_backtrack and rng.random() < backtrack_prob:
        return parent

    if children:
        n_visits = np.asarray(state.n_visits)
        q_values = np.asarray(state.q_values)
        child_visits = n_visits[children]
        min_visits = np.min(child_visits)
        least_visited_children = [
            child for child, visits in zip(children, child_visits) if visits == min_visits
        ]
        child_scores = q_values[least_visited_children]
        best_score = np.max(child_scores)
        best_children = [
            child
            for child, score in zip(least_visited_children, child_scores)
            if score == best_score
        ]
        return int(rng.choice(best_children))

    if bool(action_mask[root]):
        return root

    if step_idx >= min_steps_before_terminate:
        return num_nodes

    if can_backtrack:
        return parent

    return num_nodes


def _run_trial(
    env: JaxDecisionTreeEnv,
    reset_fn,
    step_fn,
    choice_fn,
    key,
    rng: np.random.Generator,
    *,
    backtrack_prob: float,
    min_steps_before_terminate: int,
) -> dict[str, Any] | None:
    state, _, info = reset_fn(key)
    action_mask = np.asarray(info["mask"])

    action_seq: list[int] = []
    details = {
        "activations": [],
        "counts": [],
        "gs": [],
        "qs": [],
        "logits": [],
    }

    done = False
    step_idx = 0
    while not done and step_idx < env.t_max:
        action = _choose_action(
            state,
            action_mask,
            rng,
            backtrack_prob=backtrack_prob,
            min_steps_before_terminate=min_steps_before_terminate,
            step_idx=step_idx,
            num_nodes=env.num_nodes,
            t_max=env.t_max,
        )

        details["activations"].append(np.asarray(state.activation, dtype=np.float32).tolist())
        details["counts"].append(np.asarray(state.n_visits, dtype=np.int32).tolist())
        details["gs"].append(np.asarray(state.g_values, dtype=np.float32).tolist())
        details["qs"].append(np.asarray(state.q_values, dtype=np.float32).tolist())
        details["logits"].append(_policy_logits(action_mask, action))

        state, _, _, done, info = step_fn(state, jnp.asarray(action, dtype=jnp.int32))
        action_seq.append(action)
        action_mask = np.asarray(info["mask"])
        step_idx += 1

    if len(action_seq) == 0 or action_seq[-1] != env.num_nodes:
        return None

    choice_key = jax.random.fold_in(key, 999)
    choice_state = state._replace(rng_key=choice_key)
    _, choice_path, choice_path_len, _ = choice_fn(choice_state)

    return {
        "child_nodes": np.asarray(state.child_nodes),
        "root_node": int(state.root_node),
        "points": np.asarray(state.points),
        "action_seq": action_seq,
        "choice_seq": np.asarray(choice_path[: int(choice_path_len)], dtype=np.int32).tolist(),
        "details": details,
    }


def _simulate_decay(
    run_dir: Path,
    *,
    base_params: dict[str, Any],
    experiment_name: str,
    wm_decay: float,
    seed: int,
    num_trials: int,
    backtrack_prob: float,
    min_steps_before_terminate: int,
) -> int:
    params = dict(base_params)
    params.update(
        {
            "wm_decay": wm_decay,
            "experiment": experiment_name,
            "seed": seed,
        }
    )
    env = _build_env(params)
    env_params = _make_env_params(env, params)
    reset_fn = jax.jit(lambda key: env.reset(key, env_params))
    step_fn = jax.jit(lambda state, action: env.step(state, action, env_params))
    choice_fn = jax.jit(lambda state: env._sample_move_path(state, env_params))

    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "argv": ["scripts/generate_wm_decay_backtrack_rollouts.py"],
        "args": _jsonable(params),
        "cwd": str(Path.cwd()),
        "generated_by": "fixed_backtrack_rollout_policy",
    }
    with (run_dir / "metadata.json").open("w") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")

    data = empty_simulation_data(detailed=True)
    key = jax.random.PRNGKey(seed)
    rng = np.random.default_rng(seed + int(round(wm_decay * 1000)))
    exported = 0

    for trial_idx in range(num_trials):
        trial_key = jax.random.fold_in(key, trial_idx)
        trial = _run_trial(
            env,
            reset_fn,
            step_fn,
            choice_fn,
            trial_key,
            rng,
            backtrack_prob=backtrack_prob,
            min_steps_before_terminate=min_steps_before_terminate,
        )
        if trial is None:
            continue

        appended = append_simulation_trial(
            data,
            child_nodes=trial["child_nodes"],
            root_node=trial["root_node"],
            points=trial["points"],
            action_seq=trial["action_seq"],
            choice_seq=trial["choice_seq"],
            num_nodes=env.num_nodes,
            t_max=env.t_max,
            skip_timeout_trials=False,
            details=trial["details"],
        )
        exported += int(appended)

    with (run_dir / "data_simulation_detailed.json").open("w") as file:
        json.dump(data, file)
        file.write("\n")

    return exported


def _decay_slug(wm_decay: float) -> str:
    return f"wm_decay{wm_decay:g}".replace(".", "p")


def _read_viewer_index(index_path: Path) -> dict[str, Any]:
    if not index_path.exists():
        return {"defaultVersion": DEFAULT_NAME, "datasets": []}
    with index_path.open("r") as file:
        return json.load(file)


def _restore_viewer_index(index_path: Path, previous_index: dict[str, Any], dataset_name: str) -> None:
    datasets = [
        dataset
        for dataset in previous_index.get("datasets", [])
        if dataset.get("version") != dataset_name
    ]
    datasets.append(
        {
            "version": dataset_name,
            "label": f"Simulation {dataset_name}",
            "manifestPath": f"/simulations/{dataset_name}/index.json",
        }
    )
    next_index = {
        "defaultVersion": dataset_name,
        "datasets": datasets,
    }
    with index_path.open("w") as file:
        json.dump(next_index, file, indent=2)
        file.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--results-root", type=Path, default=Path("results/runs"))
    parser.add_argument("--tree-viewer", type=Path, default=DEFAULT_TREE_VIEWER)
    parser.add_argument("--seed", type=int, default=15)
    parser.add_argument("--num-trials", type=int, default=30)
    parser.add_argument("--backtrack-prob", type=float, default=0.2)
    parser.add_argument("--min-steps-before-terminate", type=int, default=20)
    args = parser.parse_args()

    _, base_params = load_canonical_defaults()
    source_dir = (args.results_root / args.name).resolve()
    if source_dir.exists():
        shutil.rmtree(source_dir)
    source_dir.mkdir(parents=True)

    for wm_decay in WM_DECAYS:
        run_dir = source_dir / _decay_slug(wm_decay)
        exported = _simulate_decay(
            run_dir,
            base_params=base_params,
            experiment_name=args.name,
            wm_decay=wm_decay,
            seed=args.seed,
            num_trials=args.num_trials,
            backtrack_prob=args.backtrack_prob,
            min_steps_before_terminate=args.min_steps_before_terminate,
        )
        print(f"{run_dir}: wrote {exported} trials", flush=True)

    viewer_index_path = args.tree_viewer / "assets/simulations/index.json"
    previous_viewer_index = _read_viewer_index(viewer_index_path)
    command = ["bun", "scripts/reformat-sim15.mjs", str(source_dir)]
    print(f"Running in {args.tree_viewer}: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=args.tree_viewer, check=True)
    _restore_viewer_index(viewer_index_path, previous_viewer_index, args.name)


if __name__ == "__main__":
    main()
