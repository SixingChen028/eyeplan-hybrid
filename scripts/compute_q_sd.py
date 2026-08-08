#!/usr/bin/env python3
"""Compute the standard deviation of true Q-values over sampled trees.

This is the anchor for the drift reparameterization in `make_params`: inactive
Q-values follow an AR(1) process whose asymptotic standard deviation is
`q_drift / sqrt(1 - q_decay^2)`. Pinning that to the spread of the values being
remembered makes a maximally drifted memory look like a random draw from the
value distribution -- the drift analogue of forgetting resetting a memory to
0.0, near the prior mean.

The result depends on `point_set` (linearly) and on `num_nodes` via the tree
templates, so `Q_SD` in environment.py is guarded by an assert on both. Rerun
this script if either changes.

    python scripts/compute_q_sd.py
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


def main() -> None:
    defaults = PARAM_DEFAULTS["environment"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-nodes", type=int, default=int(defaults["num_nodes"]))
    parser.add_argument("--point-set", type=str, default=",".join(str(p) for p in defaults["point_set"]))
    parser.add_argument("--num-trees", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    point_set = tuple(float(item) for item in args.point_set.split(","))
    samples = q_value_samples(args.num_nodes, point_set, num_trees=args.num_trees, seed=args.seed)

    print(f"num_nodes={args.num_nodes} point_set={point_set} num_trees={args.num_trees}")
    print(f"q_sd={samples.std():.4f}  mean={samples.mean():.4f}")
    print(f"percentiles(1,25,50,75,99)={np.percentile(samples, [1, 25, 50, 75, 99]).round(2).tolist()}")


if __name__ == "__main__":
    main()
