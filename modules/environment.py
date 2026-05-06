import jax
import jax.numpy as jnp
from typing import NamedTuple


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
    wm_backup: jax.Array


def make_decision_tree_params(
    env: "JaxDecisionTreeEnv",
    beta_move: float = 4.0,
    eps_move: float = 0.02,
    learning_rate: float = 0.2,
    lamda_backup: float = 0.0,
    backup_steps: int = 100,
    wm_decay: float = 0.8,
    q_drop_rate: float = 0.0,
    q_drift: float = 0.0,
    q_decay=0.0,
    recency_decay="off",
    cost: float = 0.01,
    wm_backup: bool = False,
) -> JaxDecisionTreeParams:
    q_drift = float(q_drift)
    if q_drift < 0.0:
        raise ValueError("q_drift must be non-negative.")
    q_decay_auto, q_decay_value = JaxDecisionTreeEnv._parse_q_decay(q_decay)
    _, recency_decay_auto, recency_decay_value = JaxDecisionTreeEnv._parse_recency_decay(recency_decay)
    resolved_q_decay = (
        env._q_decay_value(q_drift, env.scale_factor)
        if q_decay_auto
        else jnp.asarray(q_decay_value, dtype=jnp.float32)
    )
    resolved_recency_decay = (
        jnp.where(float(wm_decay) == 1.0, 0.5, float(wm_decay))
        if recency_decay_auto
        else jnp.asarray(recency_decay_value, dtype=jnp.float32)
    )
    return JaxDecisionTreeParams(
        beta_move=jnp.asarray(beta_move, dtype=jnp.float32),
        eps_move=jnp.asarray(eps_move, dtype=jnp.float32),
        learning_rate=jnp.asarray(learning_rate, dtype=jnp.float32),
        lamda_backup=jnp.asarray(lamda_backup, dtype=jnp.float32),
        backup_steps=jnp.asarray(backup_steps, dtype=jnp.int32),
        wm_decay=jnp.asarray(wm_decay, dtype=jnp.float32),
        q_drop_rate=jnp.asarray(q_drop_rate, dtype=jnp.float32),
        q_drift=jnp.asarray(q_drift, dtype=jnp.float32),
        q_decay=jnp.asarray(resolved_q_decay, dtype=jnp.float32),
        recency_decay=jnp.asarray(resolved_recency_decay, dtype=jnp.float32),
        cost=jnp.asarray(cost, dtype=jnp.float32),
        wm_backup=jnp.asarray(wm_backup, dtype=jnp.bool_),
    )


class JaxDecisionTreeEnv:
    metadata = {"render_modes": ["human", "rgb_array"]}

    @staticmethod
    def _parse_recency_decay(recency_decay) -> tuple[bool, bool, float]:
        if isinstance(recency_decay, str):
            value = recency_decay.strip().lower()
            if value == "off":
                return False, False, 0.0
            if value == "auto":
                return True, True, 0.0
            try:
                recency_decay = float(value)
            except ValueError as error:
                raise ValueError("recency_decay must be 'off', 'auto', or a number in [0, 1).") from error

        value = float(recency_decay)
        if not 0.0 <= value < 1.0:
            raise ValueError("recency_decay numeric values must satisfy 0 <= recency_decay < 1.")
        return True, False, value

    @staticmethod
    def _parse_q_decay(q_decay) -> tuple[bool, float]:
        if isinstance(q_decay, str):
            value = q_decay.strip().lower()
            if value == "auto":
                return True, 0.0
            try:
                q_decay = float(value)
            except ValueError as error:
                raise ValueError("q_decay must be 'auto' or a number in [0, 1].") from error

        value = float(q_decay)
        if not 0.0 <= value <= 1.0:
            raise ValueError("q_decay numeric values must satisfy 0 <= q_decay <= 1.")
        return False, value

    def __init__(
        self,
        num_nodes: int = 15,
        t_max: int = 100,
        scale_factor: float = 1 / 8,
        shuffle_nodes: bool = True,
        use_recency_obs: bool = False,
        point_set=None,
    ):
        self.num_nodes = int(num_nodes)
        self.t_max = int(t_max)
        self.scale_factor = float(scale_factor)
        self.shuffle_nodes = bool(shuffle_nodes)
        self.use_recency_obs = bool(use_recency_obs)

        if point_set is None:
            point_set = [-8, -4, -2, -1, 1, 2, 4, 8]
        self.point_set = jnp.asarray(point_set, dtype=jnp.float32)
        self.empty_path = -jnp.ones((self.num_nodes,), dtype=jnp.int32)

        observation_size = (
            self.num_nodes
            + 1
            + self.num_nodes * 2
            + self.num_nodes
            + self.num_nodes
            + self.num_nodes
            + self.num_nodes
            + self.num_nodes
            + 2
            + (self.num_nodes if self.use_recency_obs else 0)
            + 1
        )
        self.observation_shape = (observation_size,)
        self.action_size = self.num_nodes + 1

    def _q_prior_var(self, scale_factor: jax.Array) -> jax.Array:
        scale = jnp.asarray(scale_factor, dtype=jnp.float32)
        values = self.point_set * scale
        return jnp.var(values)

    def _q_decay_value(self, q_drift: jax.Array, scale_factor: jax.Array) -> jax.Array:
        drift_var = jnp.square(jnp.asarray(q_drift, dtype=jnp.float32))
        prior_var = jnp.maximum(self._q_prior_var(scale_factor), jnp.asarray(1e-8, dtype=jnp.float32))
        return drift_var / (drift_var + prior_var)

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

    def _build_tree(self, key: jax.Array):
        nodes = jnp.arange(self.num_nodes, dtype=jnp.int32)

        def shuffled_fn(k):
            k, perm_key = jax.random.split(k)
            return k, jax.random.permutation(perm_key, nodes)

        if self.shuffle_nodes:
            key, nodes = shuffled_fn(key)

        root = nodes[0]
        child_nodes = -jnp.ones((self.num_nodes, 2), dtype=jnp.int32)
        parent_nodes = -jnp.ones((self.num_nodes,), dtype=jnp.int32)

        leaf_nodes = -jnp.ones((self.num_nodes,), dtype=jnp.int32)
        leaf_nodes = leaf_nodes.at[0].set(root)
        leaf_count = jnp.int32(1)

        num_edges = (self.num_nodes - 1) // 2

        def body_fn(i, carry):
            key, child_nodes, parent_nodes, leaf_nodes, leaf_count = carry

            key, parent_key = jax.random.split(key)
            parent_idx = jax.random.randint(parent_key, shape=(), minval=0, maxval=leaf_count)

            node_idx = 1 + 2 * i
            children = jax.lax.dynamic_slice(nodes, (node_idx,), (2,))
            parent = leaf_nodes[parent_idx]

            child_nodes = child_nodes.at[parent].set(children)
            parent_nodes = parent_nodes.at[children[0]].set(parent)
            parent_nodes = parent_nodes.at[children[1]].set(parent)

            last_idx = leaf_count - 1
            leaf_nodes = leaf_nodes.at[parent_idx].set(leaf_nodes[last_idx])
            leaf_nodes = leaf_nodes.at[last_idx].set(children[0])
            leaf_nodes = leaf_nodes.at[leaf_count].set(children[1])
            leaf_count = leaf_count + 1

            return key, child_nodes, parent_nodes, leaf_nodes, leaf_count

        key, child_nodes, parent_nodes, _, _ = jax.lax.fori_loop(
            0,
            num_edges,
            body_fn,
            (key, child_nodes, parent_nodes, leaf_nodes, leaf_count),
        )

        return key, root, child_nodes, parent_nodes

    def _compute_path_values(self, parent_nodes, points):
        g_values = jnp.zeros((self.num_nodes,), dtype=jnp.float32)

        def body_fn(_, g_values):
            parent_safe = jnp.maximum(parent_nodes, 0)
            parent_values = g_values[parent_safe] + points[parent_safe]
            return jnp.where(parent_nodes >= 0, parent_values, 0.0)

        return jax.lax.fori_loop(0, self.num_nodes, body_fn, g_values)

    def _known_mask(self, parent_nodes, root_node, n_visits):
        expanded = n_visits > 0
        expanded = expanded.at[root_node].set(True)
        parent_safe = jnp.maximum(parent_nodes, 0)
        known = (parent_nodes >= 0) & expanded[parent_safe]
        return known.at[root_node].set(True)

    def _bellman_target(self, q_values, child_nodes, points, node, activation=None):
        children = child_nodes[node]
        child_q = q_values.at[children].get(
            mode="fill",
            fill_value=0.0,
            wrap_negative_indices=False,
        )
        if activation is not None:
            child_active = activation.at[children].get(
                mode="fill",
                fill_value=0.0,
                wrap_negative_indices=False,
            )
            child_q = jnp.where(child_active > 0.0, child_q, 0.0)
        return points[node] + jnp.max(child_q)

    def _update_q(
        self,
        q_values,
        child_nodes,
        parent_nodes,
        root_node,
        points,
        node,
        activation,
        params: JaxDecisionTreeParams,
    ):
        target = self._bellman_target(q_values, child_nodes, points, node)
        node_step = params.learning_rate * (target - q_values[node])
        q_values = q_values.at[node].add(node_step)

        def cond_fn(carry):
            current, weight, steps, _ = carry
            parent = parent_nodes[current]
            has_parent = parent >= 0
            has_budget = steps < params.backup_steps
            return (weight > 1e-6) & (current != root_node) & has_parent & has_budget

        def body_fn(carry):
            current, weight, steps, q_values = carry
            ancestor = parent_nodes[current]
            target = jax.lax.cond(
                params.wm_backup,
                lambda _: self._bellman_target(q_values, child_nodes, points, ancestor, activation),
                lambda _: self._bellman_target(q_values, child_nodes, points, ancestor),
                operand=None,
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
        return q_values

    def _decay_fixation_recency(
        self,
        state: JaxDecisionTreeState,
        params: JaxDecisionTreeParams,
    ) -> JaxDecisionTreeState:
        return state._replace(fixation_recency=state.fixation_recency * params.recency_decay)

    def _look(
        self,
        state: JaxDecisionTreeState,
        node: jax.Array,
        params: JaxDecisionTreeParams,
    ) -> JaxDecisionTreeState:
        n_visits = state.n_visits.at[node].add(1)
        q_values = self._update_q(
            state.q_values,
            state.child_nodes,
            state.parent_nodes,
            state.root_node,
            state.points,
            node,
            state.activation,
            params,
        )

        return state._replace(
            q_values=q_values,
            n_visits=n_visits,
            fixation_recency=state.fixation_recency.at[node].set(
                jnp.where(params.recency_decay > 0.0, 1.0, 0.0),
            ),
        )

    def _update_activation(
        self,
        state: JaxDecisionTreeState,
        node: jax.Array,
        params: JaxDecisionTreeParams,
    ) -> JaxDecisionTreeState:
        key, drop_key = jax.random.split(state.rng_key)
        key, q_drop_key = jax.random.split(key)

        activation = state.activation * params.wm_decay
        activation = jnp.clip(activation, 0.0, 1.0)
        activation = activation.at[node].set(1.0)

        parent = state.parent_nodes[node]
        activation = activation.at[parent].set(
            1.0,
            mode="drop",
            wrap_negative_indices=False,
        )

        children = state.child_nodes[node]
        activation = activation.at[children].set(
            1.0,
            mode="drop",
            wrap_negative_indices=False,
        )

        activation = activation.at[state.root_node].set(1.0)

        keep = jax.random.uniform(drop_key, shape=(self.num_nodes,)) < activation
        activation = jnp.where(keep, activation, 0.0)
        q_drop_mask = (activation == 0.0) & (
            jax.random.uniform(q_drop_key, shape=(self.num_nodes,)) < params.q_drop_rate
        )
        inactive = activation == 0.0
        q_drift_key = jax.random.fold_in(q_drop_key, 1)
        q_values = jnp.where(inactive, state.q_values * (1.0 - params.q_decay), state.q_values)
        q_noise = jax.random.normal(q_drift_key, shape=(self.num_nodes,)) * params.q_drift
        q_values = jnp.where(inactive, q_values + q_noise, q_values)
        q_values = jnp.where(q_drop_mask, 0.0, q_values)

        return state._replace(
            rng_key=key,
            activation=activation,
            q_values=q_values,
        )

    def get_obs(self, state: JaxDecisionTreeState) -> jax.Array:
        fixation_parent = state.parent_nodes[state.fixation_node]
        fixation_children = state.child_nodes[state.fixation_node]
        known_mask = self._known_mask(state.parent_nodes, state.root_node, state.n_visits)
        visible_g_values_raw = jnp.where(known_mask, state.g_values, 0.0)
        unseen_mask = state.n_visits == 0
        open_mask = known_mask & unseen_mask
        is_terminal_seen_raw = (state.child_nodes[:, 0] < 0) & (state.n_visits > 0)
        best_open_value = jnp.where(
            jnp.any(open_mask),
            jnp.max(jnp.where(open_mask, state.g_values, -jnp.inf)),
            -10.0,
        )
        total_values = state.g_values + state.points
        best_terminal_value = jnp.where(
            jnp.any(is_terminal_seen_raw),
            jnp.max(jnp.where(is_terminal_seen_raw, total_values, -jnp.inf)),
            -10.0,
        )

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
            jnp.array([best_open_value], dtype=jnp.float32),
            jnp.array([best_terminal_value], dtype=jnp.float32),
        ]
        if self.use_recency_obs:
            parts.append(state.fixation_recency)
        parts.append(jnp.array([state.time_elapsed], dtype=jnp.float32))
        return jnp.concatenate(parts)

    def get_action_mask(self, state: JaxDecisionTreeState) -> jax.Array:
        fixation_allowed = state.time_elapsed != (self.t_max - 1)
        raw_node_mask = (state.activation > 0) & fixation_allowed
        raw_node_mask = raw_node_mask.at[state.root_node].set(fixation_allowed)
        mask = jnp.zeros((self.action_size,), dtype=jnp.bool_)
        mask = mask.at[: self.num_nodes].set(raw_node_mask)
        mask = mask.at[-1].set(True)
        return mask

    def _expected_move_reward(
        self,
        state: JaxDecisionTreeState,
        params: JaxDecisionTreeParams,
    ):
        expected = jnp.zeros((self.num_nodes,), dtype=jnp.float32)

        def body_fn(_, expected):
            children = state.child_nodes
            has_children = children[:, 0] >= 0
            safe_children = jnp.maximum(children, 0)
            q_children = state.q_values[safe_children]
            probs = self._softmax(q_children, params)
            child_returns = state.points[safe_children] + expected[safe_children]
            node_expected = jnp.sum(probs * child_returns, axis=-1)
            return jnp.where(has_children, node_expected, 0.0)

        expected = jax.lax.fori_loop(0, self.num_nodes, body_fn, expected)
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

    def reset_with_params(self, key: jax.Array, params: JaxDecisionTreeParams):
        key, root, child_nodes, parent_nodes = self._build_tree(key)

        key, points_key = jax.random.split(key)
        point_idx = jax.random.randint(
            points_key,
            shape=(self.num_nodes,),
            minval=0,
            maxval=self.point_set.shape[0],
        )
        points = self.point_set[point_idx]
        points = points.at[root].set(0.0)
        g_values = self._compute_path_values(parent_nodes, points)

        recency_initial_value = jnp.where(params.recency_decay > 0.0, 1.0, 0.0)
        state = JaxDecisionTreeState(
            rng_key=key,
            time_elapsed=jnp.int32(0),
            fixation_node=root,
            root_node=root,
            points=points,
            child_nodes=child_nodes,
            parent_nodes=parent_nodes,
            q_values=jnp.zeros((self.num_nodes,), dtype=jnp.float32).at[root].set(0.0),
            g_values=g_values,
            n_visits=jnp.zeros((self.num_nodes,), dtype=jnp.int32),
            fixation_recency=jnp.zeros((self.num_nodes,), dtype=jnp.float32).at[root].set(recency_initial_value),
            activation=jnp.zeros((self.num_nodes,), dtype=jnp.float32),
        )

        state = self._update_activation(state, state.root_node, params)

        obs = self.get_obs(state)
        info = {"mask": self.get_action_mask(state)}
        return state, obs, info

    def step_with_params(
        self,
        state: JaxDecisionTreeState,
        action: jax.Array,
        params: JaxDecisionTreeParams,
    ):
        action = jnp.asarray(action, dtype=jnp.int32)
        raw_action = action
        state = state._replace(time_elapsed=state.time_elapsed + 1)
        state = self._decay_fixation_recency(state, params)
        reward = -jnp.asarray(params.cost, dtype=jnp.float32)

        def fixation_branch(payload):
            state, reward = payload
            state = self._look(state, raw_action, params)
            state = state._replace(fixation_node=raw_action)
            state = self._update_activation(state, raw_action, params)
            return state, reward

        def move_branch(payload):
            state, reward = payload
            cum_reward = self._expected_move_reward(state, params)
            reward = cum_reward * self.scale_factor
            return state, reward

        state, reward = jax.lax.cond(
            action < self.num_nodes,
            fixation_branch,
            move_branch,
            (state, reward),
        )

        done = (action == self.num_nodes) | (state.time_elapsed == self.t_max)
        obs = self.get_obs(state)
        info = {"mask": self.get_action_mask(state)}

        return state, obs, reward, done, jnp.array(False), info
