"""Action-only likelihoods: sample planning, integrate path execution exactly."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from modules.environment import DecisionTreeEnv, DecisionTreeParams, DecisionTreeState
from modules.network import actor_critic_forward, sample_actions
from modules.random_search import RandomSearchSimulator


MIN_LAPSE_RATE = 0.01


@dataclass(frozen=True)
class TreeProblem:
    root: int
    points: np.ndarray
    children: np.ndarray
    parents: np.ndarray
    depths: np.ndarray
    paths: tuple[tuple[int, ...], ...]

    @classmethod
    def from_record(cls, record: dict):
        """Read zero-indexed graph/start/rewards, in the model's raw reward units."""
        graph = record["graph"]
        points = np.asarray(record["rewards"], dtype=np.float32)
        root = record["start"]
        n = len(graph)
        if not isinstance(root, int) or not 0 <= root < n:
            raise ValueError("start must be a zero-indexed node")
        if points.shape != (n,) or not np.all(np.isfinite(points)) or points[root] != 0:
            raise ValueError("rewards must be finite, match graph, and be zero at the root")
        children = np.full((n, 2), -1, dtype=np.int32)
        parents = np.full(n, -1, dtype=np.int32)
        for node, kids in enumerate(graph):
            if len(kids) not in (0, 2):
                raise ValueError("graph must be a full binary tree")
            for child in kids:
                if not isinstance(child, int) or not 0 <= child < n or child == root or parents[child] >= 0:
                    raise ValueError("graph must have unique parents and valid zero-indexed nodes")
                parents[child] = node
            if kids:
                children[node] = kids
        depths = np.full(n, -1, dtype=np.int32)
        paths = []

        def visit(node, path):
            if depths[node] >= 0:
                raise ValueError("graph must be acyclic")
            depths[node] = len(path)
            if children[node, 0] < 0:
                paths.append(tuple(path))
            else:
                for child in children[node]:
                    visit(int(child), [*path, int(child)])

        visit(root, [])
        if np.any(depths < 0):
            raise ValueError("graph must be connected")
        return cls(root, points, children, parents, depths, tuple(paths))

    def padded_paths(self):
        paths = np.full((len(self.paths), len(self.points)), -1, dtype=np.int32)
        for i, path in enumerate(self.paths):
            paths[i, :len(path)] = path
        return paths


def conditional_path_probabilities(
    state: DecisionTreeState, params: DecisionTreeParams, depths: jax.Array, paths: jax.Array,
    lapse_rate: float,
) -> jax.Array:
    """Exact probabilities of complete paths given memory immediately before stopping.

    Each child Q is its stopping value or zero. At depth d it must survive d-1
    unprotected execution fixations before its parent refreshes it. Distinct
    branch comparisons involve disjoint independent node-retention streams.
    """
    def decay_step(k, carry):
        activation, survival = carry
        activation = jnp.clip(activation * params.wm_decay, 0.0, 1.0)
        survival = survival * jnp.where(k < depths - 1, activation, 1.0)
        return activation, survival

    _, survival = jax.lax.fori_loop(
        0, state.activation.shape[0], decay_step, (state.activation, jnp.ones_like(state.activation)),
    )
    children = jnp.maximum(state.child_nodes, 0)
    q = state.q_values[children]
    retained = survival[children]
    outcomes = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=jnp.bool_)
    effective_q = jnp.where(outcomes[None, :, :], q[:, None, :], 0.0)
    weights = jnp.prod(jnp.where(outcomes[None, :, :], retained[:, None, :], 1.0 - retained[:, None, :]), -1)
    left = effective_q[:, :, 0]
    right = effective_q[:, :, 1]
    p_left = jnp.sum(weights * ((left > right) + 0.5 * (left == right)), axis=-1)
    p_right = jnp.sum(weights * ((right > left) + 0.5 * (left == right)), axis=-1)
    choices = (1.0 - lapse_rate) * jnp.stack([p_left, p_right], axis=-1) + lapse_rate / 2.0
    parents = jnp.maximum(state.parent_nodes, 0)
    is_right = state.child_nodes[parents, 1] == jnp.arange(state.activation.shape[0])
    edge_probability = choices[parents, is_right.astype(jnp.int32)]
    return jnp.prod(jnp.where(paths >= 0, edge_probability[jnp.maximum(paths, 0)], 1.0), axis=-1)


class PathLikelihoodEstimator:
    def __init__(self, env: DecisionTreeEnv, *, undirected: bool = False, lapse_rate: float = MIN_LAPSE_RATE):
        if not MIN_LAPSE_RATE <= lapse_rate <= 1.0:
            raise ValueError(f"lapse_rate must be in [{MIN_LAPSE_RATE}, 1]")
        self.env = env
        self.undirected = undirected
        self.lapse_rate = float(lapse_rate)
        # Reuse the ordinary Gamma target and uniform legal-fixation rule.
        self.random_search = RandomSearchSimulator(env, env_params=None) if undirected else None
        self._sample_batch = jax.jit(jax.vmap(self._sample_stopping_state, in_axes=(None,) * 6 + (0,)))
        self._probability_batch = jax.jit(self._batch_probabilities)

    def initial_state(self, root, points, children, parents, key, env_params):
        """Reset on a supplied problem, without sampling or revealing a new problem."""
        state = DecisionTreeState(
            root_node=root, points=points, child_nodes=children, parent_nodes=parents,
            fixation_node=root,
            g_values=jnp.full_like(points, self.env.min_path_value).at[root].set(0.0),
            q_values=jnp.zeros_like(points), n_visits=jnp.zeros(points.shape, dtype=jnp.int32),
            activation=jnp.zeros_like(points), is_terminal=jnp.zeros(points.shape, dtype=jnp.bool_),
            time_elapsed=jnp.int32(0), rng_key=key,
        )
        return self.env._look(state, root, env_params)

    def _sample_stopping_state(self, policy_params, env_params, root, points, children, parents, key):
        policy_key, memory_key, stop_key = jax.random.split(key, 3)
        state = self.initial_state(root, points, children, parents, memory_key, env_params)
        target = self.random_search._fixation_target(stop_key) if self.undirected else jnp.int32(0)

        def body(carry):
            state, policy_key, _ = carry
            policy_key, action_key = jax.random.split(policy_key)
            mask = self.env._get_action_mask(state)
            if self.undirected:
                fixation = self.random_search._sample_random_fixation(action_key, mask, state.fixation_node)
                action = jnp.where(state.time_elapsed < target, fixation, self.env.num_nodes)
            else:
                obs = jax.tree_util.tree_map(
                    lambda value: None if value is None else value[None, ...], self.env._get_obs(state),
                )
                logits, _ = actor_critic_forward(
                    policy_params, obs, mask[None, :], self.env._get_observation_mask(state)[None, :],
                )
                actions, _, _ = sample_actions(action_key, logits, mask[None, :])
                action = actions[0]
            stopped = action == self.env.num_nodes
            state = jax.lax.cond(
                stopped,
                lambda: state,
                lambda: self.env._look(state._replace(time_elapsed=state.time_elapsed + 1), action, env_params),
            )
            return state, policy_key, stopped

        state, _, _ = jax.lax.while_loop(lambda carry: ~carry[2], body, (state, policy_key, jnp.bool_(False)))
        return state

    def _batch_probabilities(self, policy_params, env_params, root, points, children, parents, depths, paths, keys):
        states = self._sample_batch(policy_params, env_params, root, points, children, parents, keys)
        probabilities = jax.vmap(conditional_path_probabilities, in_axes=(0, None, None, None, None))(
            states, env_params, depths, paths, self.lapse_rate,
        )
        return probabilities, states.time_elapsed

    def estimate(self, problem: TreeProblem, policy_params, env_params, *, samples: int, batch_size=512, seed=0):
        """Return all path probabilities and independent-sample MC standard errors.

        Inputs contain only the problem. Observed moves are scored afterwards;
        observed fixations, counts, and durations never enter this estimator.
        """
        if not 2 <= samples < 2**32 or batch_size < 1:
            raise ValueError("samples must be in [2, 2**32) and batch_size must be positive")
        if len(problem.points) != self.env.num_nodes:
            raise ValueError("problem size does not match checkpoint environment")
        if not np.all(np.isin(np.delete(problem.points, problem.root), np.asarray(self.env.point_set))):
            raise ValueError("problem rewards must use the checkpoint's raw point units")
        master_key = jax.random.PRNGKey(seed)
        count = 0
        mean = np.zeros(len(problem.paths), dtype=np.float64)
        m2 = np.zeros_like(mean)
        nonzero = np.zeros(len(problem.paths), dtype=np.int64)
        fixation_total = 0
        horizon_total = 0
        args = (problem.root, problem.points, problem.children, problem.parents, problem.depths, problem.padded_paths())
        for start in range(0, samples, batch_size):
            keys = jax.vmap(lambda i: jax.random.fold_in(master_key, i))(
                jnp.arange(start, start + batch_size, dtype=jnp.uint32),
            )
            probabilities, fixations = self._probability_batch(policy_params, env_params, *args, keys)
            n = min(batch_size, samples - start)
            probabilities = np.asarray(probabilities, dtype=np.float64)[:n]
            fixations = np.asarray(fixations)[:n]
            batch_mean = probabilities.mean(axis=0)
            delta = batch_mean - mean
            m2 += np.square(probabilities - batch_mean).sum(axis=0) + delta**2 * count * n / (count + n)
            count += n
            mean += delta * n / count
            nonzero += (probabilities > 0).sum(axis=0)
            fixation_total += int(fixations.sum())
            horizon_total += int((fixations == self.env.t_max - 1).sum())
        se = np.sqrt(m2 / ((count - 1) * count))
        return {
            "lapse_rate": self.lapse_rate,
            "paths": [list(path) for path in problem.paths],
            "probabilities": mean.tolist(), "standard_errors": se.tolist(),
            "nonzero_samples": nonzero.tolist(), "samples": count,
            "mean_latent_fixations": fixation_total / count,
            "horizon_fraction": horizon_total / count,
            "probability_sum": float(mean.sum()),
        }
