from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np


Tree = tuple


def enumerate_tree_probs(num_nodes: int) -> dict[Tree, float]:
    if num_nodes % 2 == 0 or num_nodes < 1:
        raise ValueError("num_nodes must be a positive odd integer")

    states: dict[Tree, float] = {(): 1.0}

    def leaves_with_paths(t: Tree, path: tuple[int, ...] = ()):
        if t == ():
            yield path
            return
        yield from leaves_with_paths(t[0], path + (0,))
        yield from leaves_with_paths(t[1], path + (1,))

    def expand_at(t: Tree, path: tuple[int, ...]) -> Tree:
        if not path:
            return ((), ())
        side = path[0]
        children = list(t)
        children[side] = expand_at(children[side], path[1:])
        return tuple(children)

    def canonical(t: Tree) -> Tree:
        if t == ():
            return ()
        left = canonical(t[0])
        right = canonical(t[1])
        return tuple(sorted((left, right), key=repr))

    num_expansions = (num_nodes - 1) // 2
    for _ in range(num_expansions):
        new_states: dict[Tree, float] = defaultdict(float)
        for tree, prob in states.items():
            leaves = list(leaves_with_paths(tree))
            leaf_prob = prob / len(leaves)
            for path in leaves:
                new_tree = canonical(expand_at(tree, path))
                new_states[new_tree] += leaf_prob
        states = dict(new_states)

    return states


@dataclass(frozen=True)
class TreeTemplates:
    roots: np.ndarray
    child_nodes: np.ndarray
    parent_nodes: np.ndarray
    probabilities: np.ndarray


def _tree_to_arrays(tree: Tree, num_nodes: int) -> tuple[np.ndarray, np.ndarray, int]:
    child_nodes = -np.ones((num_nodes, 2), dtype=np.int32)
    parent_nodes = -np.ones((num_nodes,), dtype=np.int32)

    next_id = 0
    root_id = next_id
    next_id += 1
    queue: deque[tuple[Tree, int]] = deque([(tree, root_id)])

    while queue:
        subtree, node_id = queue.popleft()
        if subtree == ():
            continue

        left_id = next_id
        next_id += 1
        right_id = next_id
        next_id += 1

        child_nodes[node_id, 0] = left_id
        child_nodes[node_id, 1] = right_id
        parent_nodes[left_id] = node_id
        parent_nodes[right_id] = node_id

        queue.append((subtree[0], left_id))
        queue.append((subtree[1], right_id))

    if next_id != num_nodes:
        raise ValueError(f"Expected {num_nodes} nodes, got {next_id}")

    return child_nodes, parent_nodes, root_id


def build_tree_templates(num_nodes: int) -> TreeTemplates:
    probs_by_tree = enumerate_tree_probs(num_nodes)
    trees_and_probs = sorted(probs_by_tree.items(), key=lambda item: repr(item[0]))

    roots = np.zeros((len(trees_and_probs),), dtype=np.int32)
    child_nodes = np.zeros((len(trees_and_probs), num_nodes, 2), dtype=np.int32)
    parent_nodes = np.zeros((len(trees_and_probs), num_nodes), dtype=np.int32)
    probabilities = np.zeros((len(trees_and_probs),), dtype=np.float32)

    for i, (tree, prob) in enumerate(trees_and_probs):
        child_i, parent_i, root_i = _tree_to_arrays(tree, num_nodes)
        roots[i] = root_i
        child_nodes[i] = child_i
        parent_nodes[i] = parent_i
        probabilities[i] = np.float32(prob)

    total = float(probabilities.sum())
    if not np.isclose(total, 1.0):
        probabilities = probabilities / total

    return TreeTemplates(
        roots=roots,
        child_nodes=child_nodes,
        parent_nodes=parent_nodes,
        probabilities=probabilities,
    )
