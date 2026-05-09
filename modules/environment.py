import math
import jax
import jax.numpy as jnp
from typing import NamedTuple

from modules.tree_generation import build_tree_templates


def safe_get(arr: jax.Array, idx: jax.Array, *, fill_value) -> jax.Array:
    return arr.at[idx].get(
        mode="fill",
        fill_value=fill_value,
        wrap_negative_indices=False,
    )


def safe_set(arr: jax.Array, idx: jax.Array, value) -> jax.Array:
    return arr.at[idx].set(
        value,
        mode="drop",
        wrap_negative_indices=False,
    )


class JaxDecisionTreeState(NamedTuple):
    rng_key: jax.Array
    time_elapsed: jax.Array
    fixation_node: jax.Array
    root_node: jax.Array
    points: jax.Array
    child_nodes: jax.Array
    parent_nodes: jax.Array
    q_values: jax.Array
    g_values: jax.Array
    n_visits: jax.Array
    fixation_recency: jax.Array
    activation: jax.Array


class JaxDecisionTreeParams(NamedTuple):
    beta_move: jax.Array
    eps_move: jax.Array
    learning_rate: jax.Array
    lamda_backup: jax.Array
    backup_steps: jax.Array
    wm_decay: jax.Array
    q_drop_rate: jax.Array
    q_drift: jax.Array
    q_decay: jax.Array
    recency_decay: jax.Array
    cost: jax.Array


class JaxDecisionTreeEnv:
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        num_nodes: int,
        t_max: int,
        scale_factor: float,
        shuffle_nodes: bool,
        use_recency_obs: bool,
        use_best_open_value_obs: bool,
        use_best_terminal_value_obs: bool,
        wm_backup: bool,
        point_set: tuple,
    ):
        self.num_nodes = int(num_nodes)
        self.t_max = int(t_max)
        self.scale_factor = float(scale_factor)
        self.shuffle_nodes = bool(shuffle_nodes)
        self.use_recency_obs = bool(use_recency_obs)
        self.use_best_open_value_obs = bool(use_best_open_value_obs)
        self.use_best_terminal_value_obs = bool(use_best_terminal_value_obs)
        self.wm_backup = bool(wm_backup)

        self.point_set = jnp.asarray(point_set, dtype=jnp.float32)
        self.empty_path = -jnp.ones((self.num_nodes,), dtype=jnp.int32)

        templates = build_tree_templates(self.num_nodes)
        self._tree_roots = jnp.asarray(templates.roots, dtype=jnp.int32)
        self._tree_child_nodes = jnp.asarray(templates.child_nodes, dtype=jnp.int32)
        self._tree_parent_nodes = jnp.asarray(templates.parent_nodes, dtype=jnp.int32)
        self._tree_probabilities = jnp.asarray(templates.probabilities, dtype=jnp.float32)

        dummy_key = jnp.zeros((2,), dtype=jnp.uint32)
        self.observation_shape = self._get_obs(self._sample_initial_state(dummy_key)).shape
        self.action_size = self.num_nodes + 1

    def make_params(
        self,
        *,
        beta_move: float,
        eps_move: float,
        learning_rate: float,
        lamda_backup: float,
        backup_steps: int,
        wm_decay: float,
        q_drop_rate: float,
        q_drift: float,
        q_decay,
        recency_decay,
        cost: float,
    ) -> JaxDecisionTreeParams:

        assert beta_move >= 0.0, "beta_move must be non-negative."
        assert 0.0 <= eps_move <= 1.0, "eps_move must between 0 and 1"
        assert learning_rate >= 0.0, "learning_rate must be non-negative."
        assert 0.0 <= lamda_backup <= 1.0, "lamda_backup must be between 0 and 1."
        assert backup_steps >= 0, "backup_steps must be non-negative."
        assert 0.0 <= wm_decay <= 1.0, "wm_decay must be between 0 and 1."
        assert 0.0 <= q_drop_rate <= 1.0, "q_drop_rate must be between 0 and 1."
        assert q_drift >= 0.0, "q_drift must be non-negative."
        assert 0.0 <= q_decay <= 1.0, "q_decay must be between 0 and 1."
        assert 0.0 <= recency_decay <= 1.0, "recency_decay must be between 0 and 1."
        assert cost >= 0.0, "cost must be non-negative."

        return JaxDecisionTreeParams(
            beta_move=jnp.asarray(beta_move, dtype=jnp.float32),
            eps_move=jnp.asarray(eps_move, dtype=jnp.float32),
            learning_rate=jnp.asarray(learning_rate, dtype=jnp.float32),
            lamda_backup=jnp.asarray(lamda_backup, dtype=jnp.float32),
            backup_steps=jnp.asarray(backup_steps, dtype=jnp.int32),
            wm_decay=jnp.asarray(wm_decay, dtype=jnp.float32),
            q_drop_rate=jnp.asarray(q_drop_rate, dtype=jnp.float32),
            q_drift=jnp.asarray(q_drift, dtype=jnp.float32),
            q_decay=jnp.asarray(q_decay, dtype=jnp.float32),
            recency_decay=jnp.asarray(recency_decay, dtype=jnp.float32),
            cost=jnp.asarray(cost, dtype=jnp.float32),
        )

    def _zeros(self, dtype: jnp.dtype = jnp.float32) -> jax.Array:
        return jnp.zeros((self.num_nodes,), dtype=dtype)

    def _one_hot(self, label: jax.Array) -> jax.Array:
        label = jnp.asarray(label, dtype=jnp.int32)
        idx = jnp.maximum(label, 0)
        mask = label >= 0
        return jax.nn.one_hot(idx, self.num_nodes, dtype=jnp.float32) * mask.astype(jnp.float32)

    def _softmax(self, x: jax.Array, params: JaxDecisionTreeParams) -> jax.Array:
        z = params.beta_move * (x - jnp.max(x, axis=-1, keepdims=True))
        p = jnp.exp(z)
        p = p / jnp.sum(p, axis=-1, keepdims=True)

        p = (1.0 - params.eps_move) * p + params.eps_move * (1.0 / x.shape[-1])

        return p

    def _sample_tree(self, key: jax.Array):
        key, tree_key = jax.random.split(key)
        tree_idx = jax.random.choice(
            tree_key,
            self._tree_probabilities.shape[0],
            p=self._tree_probabilities,
        )

        root = self._tree_roots[tree_idx]
        child_nodes = self._tree_child_nodes[tree_idx]
        parent_nodes = self._tree_parent_nodes[tree_idx]

        if self.shuffle_nodes:
            key, perm_key = jax.random.split(key)
            perm = jax.random.permutation(perm_key, jnp.arange(self.num_nodes, dtype=jnp.int32))

            child_safe = jnp.maximum(child_nodes, 0)
            parent_safe = jnp.maximum(parent_nodes, 0)
            mapped_children = jnp.where(child_nodes >= 0, perm[child_safe], -1)
            mapped_parents = jnp.where(parent_nodes >= 0, perm[parent_safe], -1)

            child_nodes = jnp.full_like(child_nodes, -1).at[perm].set(mapped_children)
            parent_nodes = jnp.full_like(parent_nodes, -1).at[perm].set(mapped_parents)
            root = perm[root]

        return key, root, child_nodes, parent_nodes

    def _compute_path_values(self, parent_nodes, points):
        g_values = self._zeros()

        def body_fn(_, g_values):
            parent_safe = jnp.maximum(parent_nodes, 0)
            parent_values = g_values[parent_safe] + points[parent_safe]
            return jnp.where(parent_nodes >= 0, parent_values, 0.0)

        return jax.lax.fori_loop(0, self.num_nodes, body_fn, g_values)

    def _bellman_target(self, q_values, child_nodes, points, node, activation=None):
        children = child_nodes[node]
        child_q = safe_get(q_values, children, fill_value=0.0)
        if activation is not None:
            child_active = safe_get(activation, children, fill_value=0.0)
            # NOTE: inactive nodes are treated as having Q=0 rather than not existing (Q=-inf)
            # this is an assumption; maybe it should be optimized over
            child_q = jnp.where(child_active > 0.0, child_q, 0.0)
        return points[node] + jnp.max(child_q)

    def _update_q(self, state, params: JaxDecisionTreeParams):
        node = state.fixation_node
        q_values = state.q_values
        target = self._bellman_target(state.q_values, state.child_nodes, state.points, node)
        node_step = params.learning_rate * (target - q_values[node])
        q_values = q_values.at[node].add(node_step)

        def cond_fn(carry):
            current, weight, steps, _ = carry
            parent = state.parent_nodes[current]
            has_parent = parent >= 0
            has_budget = steps < params.backup_steps
            parent_active = safe_get(state.activation, parent, fill_value=0.0) > 0.0
            wm_allows_backup = (not self.wm_backup) | parent_active
            return (weight > 1e-6) & (current != state.root_node) & has_parent & has_budget & wm_allows_backup

        def body_fn(carry):
            current, weight, steps, q_values = carry
            ancestor = state.parent_nodes[current]
            if self.wm_backup:
                target = self._bellman_target(
                    q_values,
                    state.child_nodes,
                    state.points,
                    ancestor,
                    state.activation,
                )
            else:
                target = self._bellman_target(
                    q_values,
                    state.child_nodes,
                    state.points,
                    ancestor,
                )
            step_size = params.learning_rate * weight
            new_value = q_values[ancestor] + step_size * (target - q_values[ancestor])
            q_values = q_values.at[ancestor].set(new_value)
            return ancestor, weight * params.lamda_backup, steps + 1, q_values

        _, _, _, q_values = jax.lax.while_loop(
            cond_fn,
            body_fn,
            (
                node,
                jnp.asarray(params.lamda_backup, dtype=q_values.dtype),
                jnp.asarray(0, dtype=jnp.int32),
                q_values,
            ),
        )
        return state._replace(q_values=q_values)

    def _look(self, state: JaxDecisionTreeState, node: jax.Array, params: JaxDecisionTreeParams) -> JaxDecisionTreeState:
        state = state._replace(
            fixation_node=node,
            n_visits=state.n_visits.at[node].add(1),
            fixation_recency=state.fixation_recency.at[node].set(1.0),
        )
        state = self._update_q(state, params)
        state = self._update_activation(state, params)
        state = self._corrupt_q_values(state, params)
        return state

    def _update_activation(self, state: JaxDecisionTreeState, params: JaxDecisionTreeParams) -> JaxDecisionTreeState:
        node = state.fixation_node

        # apply decay
        activation = state.activation * params.wm_decay
        activation = jnp.clip(activation, 0.0, 1.0)

        # activate fixated, parent, children
        activation = activation.at[node].set(1.0)
        activation = safe_set(activation, state.parent_nodes[node], 1.0)
        activation = safe_set(activation, state.child_nodes[node], 1.0)
        # root is always active
        activation = activation.at[state.root_node].set(1.0)

        # stochastically drop nodes from WM
        key, drop_key = jax.random.split(state.rng_key)
        keep = jax.random.uniform(drop_key, shape=(self.num_nodes,)) < activation
        activation = jnp.where(keep, activation, 0.0)

        return state._replace(activation=activation)

    def _corrupt_q_values(self, state: JaxDecisionTreeState, params: JaxDecisionTreeParams):
        key = state.rng_key
        inactive = state.activation == 0.0

        # add noise/drift to q values outside of WM
        key, q_drift_key = jax.random.split(key)
        q_values = jnp.where(inactive, state.q_values * params.q_decay, state.q_values)
        q_noise = jax.random.normal(q_drift_key, shape=(self.num_nodes,)) * params.q_drift
        q_values = jnp.where(inactive, q_values + q_noise, q_values)

        # stochastically drop q values outside of WM
        key, q_drop_key = jax.random.split(key)
        q_drop_mask = inactive & (
            jax.random.uniform(q_drop_key, shape=(self.num_nodes,)) < params.q_drop_rate
        )
        q_values = jnp.where(q_drop_mask, 0.0, q_values)

        return state._replace(
            q_values=q_values,
            rng_key=key,
        )

    def _get_obs(self, state: JaxDecisionTreeState) -> jax.Array:
        fixation_parent = state.parent_nodes[state.fixation_node]
        fixation_children = state.child_nodes[state.fixation_node]
        known_mask = safe_get(state.n_visits > 0, state.parent_nodes, fill_value=True)
        visible_g_values_raw = jnp.where(known_mask, state.g_values, 0.0)
        is_terminal_seen_raw = (state.child_nodes[:, 0] < 0) & (state.n_visits > 0)

        is_terminal_seen = is_terminal_seen_raw.astype(jnp.float32)
        fixation_child_mask = self._one_hot(fixation_children[0]) + self._one_hot(
            fixation_children[1]
        )

        parts = [
            self._one_hot(state.fixation_node),
            jnp.array([state.points[state.fixation_node]], dtype=jnp.float32),
            self._one_hot(fixation_parent),
            fixation_child_mask,
            self._one_hot(state.root_node),
            visible_g_values_raw,
            state.q_values,
            state.n_visits.astype(jnp.float32),
            is_terminal_seen,
        ]
        if self.use_best_open_value_obs or self.use_best_terminal_value_obs:
            open_obs = -10.0
            if self.use_best_open_value_obs:
                unseen_mask = state.n_visits == 0
                open_mask = known_mask & unseen_mask
                open_obs = jnp.where(
                    jnp.any(open_mask),
                    jnp.max(jnp.where(open_mask, state.g_values, -jnp.inf)),
                    -10.0,
                )
            parts.append(jnp.array([open_obs], dtype=jnp.float32))
        if self.use_best_terminal_value_obs:
            total_values = state.g_values + state.points
            best_terminal_value = jnp.where(
                jnp.any(is_terminal_seen_raw),
                jnp.max(jnp.where(is_terminal_seen_raw, total_values, -jnp.inf)),
                -10.0,
            )
            parts.append(jnp.array([best_terminal_value], dtype=jnp.float32))
        if self.use_recency_obs:
            parts.append(state.fixation_recency)
        parts.append(jnp.array([state.time_elapsed], dtype=jnp.float32))
        return jnp.concatenate(parts)

    def _get_action_mask(self, state: JaxDecisionTreeState) -> jax.Array:
        fixation_allowed = state.time_elapsed != (self.t_max - 1)
        node_mask = (state.activation > 0) & fixation_allowed
        term_mask = jnp.array([True], dtype=jnp.bool_)
        return jnp.concatenate([node_mask, term_mask], axis=0)

    def _expected_move_reward(self, state: JaxDecisionTreeState, params: JaxDecisionTreeParams):
        expected = self._zeros()

        def body_fn(_, expected):
            children = state.child_nodes
            has_children = children[:, 0] >= 0
            safe_children = jnp.maximum(children, 0)
            q_children = state.q_values[safe_children]
            probs = self._softmax(q_children, params)
            child_returns = state.points[safe_children] + expected[safe_children]
            node_expected = jnp.sum(probs * child_returns, axis=-1)
            return jnp.where(has_children, node_expected, 0.0)

        expected = jax.lax.fori_loop(0, math.ceil(self.num_nodes / 2), body_fn, expected)
        return expected[state.root_node]

    def _sample_move_path(self, state: JaxDecisionTreeState, params: JaxDecisionTreeParams):
        path = self.empty_path

        init = (
            state.root_node,
            jnp.array(0.0, dtype=jnp.float32),
            path,
            jnp.int32(0),
            state.rng_key,
        )

        def cond_fn(carry):
            node, _, _, _, _ = carry
            return state.child_nodes[node, 0] >= 0

        def body_fn(carry):
            node, cum_reward, path, path_len, key = carry
            key, choice_key = jax.random.split(key)

            children = state.child_nodes[node]
            q_children = state.q_values[children]
            probs = self._softmax(q_children, params)
            idx = jax.random.choice(choice_key, 2, p=probs)
            child = children[idx]

            cum_reward = cum_reward + state.points[child]
            path = path.at[path_len].set(child)
            path_len = path_len + 1

            return child, cum_reward, path, path_len, key

        _, cum_reward, path, path_len, key = jax.lax.while_loop(cond_fn, body_fn, init)

        return cum_reward, path, path_len, key

    def _sample_points(self, key: jax.Array, root: jax.Array):
        key, points_key = jax.random.split(key)
        point_idx = jax.random.randint(
            points_key,
            shape=(self.num_nodes,),
            minval=0,
            maxval=self.point_set.shape[0],
        )
        points = self.point_set[point_idx]
        points = points.at[root].set(0.0)
        return key, points

    def _sample_initial_state(self, key: jax.Array):
        key, root, child_nodes, parent_nodes = self._sample_tree(key)
        key, points = self._sample_points(key, root)

        return JaxDecisionTreeState(
            rng_key=key,
            time_elapsed=jnp.int32(0),
            fixation_node=root,
            root_node=root,
            points=points,
            child_nodes=child_nodes,
            parent_nodes=parent_nodes,
            q_values=self._zeros(),
            g_values=self._compute_path_values(parent_nodes, points),
            n_visits=self._zeros(jnp.int32),
            fixation_recency=self._zeros(),
            activation=self._zeros(),
        )

    def reset(self, key: jax.Array, params: JaxDecisionTreeParams):
        state = self._sample_initial_state(key)
        state = self._look(state, state.root_node, params)
        obs = self._get_obs(state)
        info = {"mask": self._get_action_mask(state)}
        return state, obs, info

    def step(self, state: JaxDecisionTreeState, action: jax.Array, params: JaxDecisionTreeParams):
        state = state._replace(
            time_elapsed=state.time_elapsed + 1,
            fixation_recency=state.fixation_recency * params.recency_decay,
        )
        state, reward = jax.lax.cond(
            action < self.num_nodes,
            lambda: (self._look(state, action, params), -params.cost),
            lambda: (state, self._expected_move_reward(state, params) * self.scale_factor),
        )
        done = (action == self.num_nodes) | (state.time_elapsed == self.t_max)
        obs = self._get_obs(state)
        info = {"mask": self._get_action_mask(state)}

        return state, obs, reward, done, info
