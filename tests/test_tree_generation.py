from collections import Counter
import os

import jax
import numpy as np
import pytest

from modules.environment import JaxDecisionTreeEnv
from modules.tree_generation import enumerate_tree_probs

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_TREE_GENERATION_TESTS") != "1",
    reason="Set RUN_TREE_GENERATION_TESTS=1 to run tree generation sampling tests.",
)


def _canonical_from_arrays(child_nodes: np.ndarray, parent_nodes: np.ndarray):
    root_candidates = np.flatnonzero(parent_nodes < 0)
    assert len(root_candidates) == 1
    root = int(root_candidates[0])

    def canonical(node: int):
        left = int(child_nodes[node, 0])
        right = int(child_nodes[node, 1])
        if left < 0:
            return ()
        left_tree = canonical(left)
        right_tree = canonical(right)
        return tuple(sorted((left_tree, right_tree), key=repr))

    return canonical(root)


def test_num_nodes_9_sampling_matches_explicit_probabilities():
    num_nodes = 9
    expected = enumerate_tree_probs(num_nodes)
    env = JaxDecisionTreeEnv(
        num_nodes=num_nodes,
        t_max=8,
        scale_factor=1.0,
        shuffle_nodes=True,
        use_recency_obs=False,
        use_best_open_value_obs=True,
        use_best_terminal_value_obs=True,
        wm_backup=False,
        point_set=(-8, -4, -2, -1, 1, 2, 4, 8),
    )

    counts = Counter()
    key = jax.random.PRNGKey(0)
    num_samples = 30000

    for _ in range(num_samples):
        key, root, child_nodes, parent_nodes = env._build_tree(key)
        del root
        tree = _canonical_from_arrays(np.asarray(child_nodes), np.asarray(parent_nodes))
        counts[tree] += 1

    observed = {tree: c / num_samples for tree, c in counts.items()}

    assert set(observed) == set(expected)
    for tree, p_expected in expected.items():
        p_observed = observed[tree]
        assert abs(p_observed - p_expected) < 0.02
