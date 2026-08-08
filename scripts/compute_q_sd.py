#!/usr/bin/env python3
"""Compute the standard deviation of true Q-values over sampled trees.

This is the anchor for the drift reparameterization in `make_params`: inactive
Q-values follow an AR(1) process whose asymptotic standard deviation is
`q_drift / sqrt(1 - q_decay^2)`. Pinning that to the spread of the values being
remembered makes a maximally drifted memory look like a random draw from the
value distribution -- the drift analogue of forgetting resetting a memory to
0.0, near the prior mean.

The result depends on `point_set` (linearly) and on `num_nodes` via the tree
templates, so `Q_SD` in environment.py must be updated if either changes.

    python scripts/compute_q_sd.py

`--simulate` additionally rolls out the environment under a uniform-random
fixation policy and reports the spread of remembered Q-values actually reached,
which should land near the analytic Q_SD. This is the check that would have
caught the 0616_compare_mem mistake, where q_drift=4 with q_decay=0.99 gave a
simulated sd of 15.2 and |q| up to 101 against a true-Q range of +-24.
"""
from __future__ import annotations

import argparse

import numpy as np

from modules.config import PARAM_DEFAULTS
from modules.tree_generation import build_tree_templates


def q_value_samples(num_nodes: int, point_set: tuple, *, num_trees: int, seed: int) -> np.ndarray:
    """Sample trees and return true Q-values under optimal play, shape (num_trees, num_nodes)."""
    templates = build_tree_templates(num_nodes)
    roots = np.asarray(templates.roots)
    child_nodes = np.asarray(templates.child_nodes)
    probabilities = np.asarray(templates.probabilities)
    points = np.asarray(point_set, dtype=float)

    rng = np.random.default_rng(seed)
    samples = np.empty((num_trees, num_nodes), dtype=float)

    for trial in range(num_trees):
        tree_idx = rng.choice(len(roots), p=probabilities)
        children = child_nodes[tree_idx]
        root = int(roots[tree_idx])

        node_points = points[rng.integers(0, len(points), size=num_nodes)]
        node_points[root] = 0.0

        # Post-order so every node's children are resolved before the node itself.
        order: list[int] = []
        stack = [(root, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            stack.append((node, True))
            if children[node, 0] >= 0:
                stack.append((int(children[node, 0]), False))
                stack.append((int(children[node, 1]), False))

        q_values = np.zeros(num_nodes, dtype=float)
        for node in order:
            if children[node, 0] < 0:
                q_values[node] = node_points[node]
            else:
                best_child = max(q_values[children[node, 0]], q_values[children[node, 1]])
                q_values[node] = node_points[node] + best_child
        samples[trial] = q_values

    return samples


def simulated_q_stats(num_nodes: int, wm_decay: float, q_drift: float, *, num_rollouts: int, num_steps: int):
    """Spread of remembered Q-values under a uniform-random fixation policy."""
    import jax

    from modules.rollout_invariants import collect_random_fixation_rollouts
    from modules.train_results import env_from_args, env_params_from_args

    run = dict(PARAM_DEFAULTS["environment"])
    run.update(PARAM_DEFAULTS["training"])
    run.update(PARAM_DEFAULTS["network"])
    run.update(
        num_nodes=num_nodes,
        t_max=num_steps + 5,
        wm_decay=wm_decay,
        wm_neighbor_activation=0.5,
        use_recency_obs=False,
        q_drift=q_drift,
        forget_rate=0.0,
    )
    env = env_from_args(run)
    params = env_params_from_args(env, run)
    trace = collect_random_fixation_rollouts(
        env, params, seed=1, num_rollouts=num_rollouts, num_steps=num_steps
    )
    states = jax.device_get(trace.states)
    remembered = np.asarray(states.q_values)[np.asarray(states.is_discovered)]
    return float(params.q_decay), remembered.std(), np.percentile(np.abs(remembered), 99.9)


def main() -> None:
    defaults = PARAM_DEFAULTS["environment"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-nodes", type=int, default=int(defaults["num_nodes"]))
    parser.add_argument("--point-set", type=str, default=",".join(str(p) for p in defaults["point_set"]))
    parser.add_argument("--num-trees", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Also simulate the environment under a random policy and report the realized spread.",
    )
    args = parser.parse_args()

    point_set = tuple(float(item) for item in args.point_set.split(","))
    samples = q_value_samples(args.num_nodes, point_set, num_trees=args.num_trees, seed=args.seed)

    print(f"num_nodes={args.num_nodes} point_set={point_set} num_trees={args.num_trees}")
    print(f"q_sd={samples.std():.4f}  mean={samples.mean():.4f}")
    print(f"percentiles(1,25,50,75,99)={np.percentile(samples, [1, 25, 50, 75, 99]).round(2).tolist()}")

    if not args.simulate:
        return

    print()
    print(f"simulated under a random policy (target sd {samples.std():.2f}, max |Q| {np.abs(samples).max():.0f})")
    print(f'{"wm_decay":>9} {"q_drift":>8} {"q_decay":>8} {"sd":>7} {"p99.9|q|":>9}')
    for wm_decay in (0.6, 0.95):
        for q_drift in (0.5, 1.0, 2.0, 3.0):
            q_decay, sd, p999 = simulated_q_stats(
                args.num_nodes, wm_decay, q_drift, num_rollouts=200, num_steps=95
            )
            print(f"{wm_decay:9} {q_drift:8} {q_decay:8.4f} {sd:7.2f} {p999:9.1f}", flush=True)


if __name__ == "__main__":
    main()
