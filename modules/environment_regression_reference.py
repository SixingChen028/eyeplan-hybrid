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
    scale_factor: jax.Array
    shuffle_nodes: jax.Array
    wm_backup: jax.Array


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
    def _parse_q_decay(q_decay) -> float:
        if isinstance(q_decay, str):
            try:
                q_decay = float(q_decay.strip())
            except ValueError as error:
                raise ValueError("q_decay must be a number in [0, 1].") from error

        value = float(q_decay)
        if not 0.0 <= value <= 1.0:
            raise ValueError("q_decay numeric values must satisfy 0 <= q_decay <= 1.")
        return value

    def __init__(
        self,
        num_nodes: int = 15,
        beta_move: float = 4.0,
        eps_move: float = 0.02,
        learning_rate: float = 0.2,
        lamda_backup: float = 0.0,
        backup_steps: int = 100,
        wm_decay: float = 0.8,
        q_drop_rate: float = 0.0,
        q_drift: float = 0.0,
        q_decay=0.0,
        t_max: int = 100,
        cost: float = 0.01,
        scale_factor: float = 1 / 8,
        shuffle_nodes: bool = True,
        recency_decay="off",
        wm_backup: bool = False,
        point_set=None,
    ):
        self.num_nodes = int(num_nodes)
        self.beta_move = float(beta_move)
        self.eps_move = float(eps_move)
        self.learning_rate = float(learning_rate)
        self.lamda_backup = float(lamda_backup)
        self.backup_steps = int(backup_steps)
        self.wm_decay = float(wm_decay)
        self.q_drop_rate = float(q_drop_rate)
        self.q_drift = float(q_drift)
        if self.q_drift < 0.0:
            raise ValueError("q_drift must be non-negative.")
        self.q_decay = self._parse_q_decay(q_decay)
        self.t_max = int(t_max)
        self.cost = float(cost)
        self.scale_factor = float(scale_factor)
        self.shuffle_nodes = True
        self.use_recency_obs, self.recency_decay_auto, self.recency_decay = self._parse_recency_decay(recency_decay)
        self.wm_backup = bool(wm_backup)

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

    def default_params(self) -> JaxDecisionTreeParams:
        return JaxDecisionTreeParams(
            beta_move=jnp.asarray(self.beta_move, dtype=jnp.float32),
            eps_move=jnp.asarray(self.eps_move, dtype=jnp.float32),
            learning_rate=jnp.asarray(self.learning_rate, dtype=jnp.float32),
            lamda_backup=jnp.asarray(self.lamda_backup, dtype=jnp.float32),
            backup_steps=jnp.asarray(self.backup_steps, dtype=jnp.int32),
            wm_decay=jnp.asarray(self.wm_decay, dtype=jnp.float32),
            q_drop_rate=jnp.asarray(self.q_drop_rate, dtype=jnp.float32),
            q_drift=jnp.asarray(self.q_drift, dtype=jnp.float32),
            q_decay=jnp.asarray(self.q_decay, dtype=jnp.float32),
            recency_decay=jnp.asarray(self._recency_decay_value(), dtype=jnp.float32),
            cost=jnp.asarray(self.cost, dtype=jnp.float32),
            scale_factor=jnp.asarray(self.scale_factor, dtype=jnp.float32),
            shuffle_nodes=jnp.asarray(self.shuffle_nodes, dtype=jnp.bool_),
            wm_backup=jnp.asarray(self.wm_backup, dtype=jnp.bool_),
        )

    def _one_hot(self, label: jax.Array) -> jax.Array:
        label = jnp.asarray(label, dtype=jnp.int32)
        idx = jnp.maximum(label, 0)
        mask = label >= 0
        return jax.nn.one_hot(idx, self.num_nodes, dtype=jnp.float32) * mask.astype(jnp.float32)

    def _softmax(self, x: jax.Array, params: JaxDecisionTreeParams | None = None) -> jax.Array:
        beta_move = self.beta_move if params is None else params.beta_move
        eps_move = self.eps_move if params is None else params.eps_move

        z = beta_move * (x - jnp.max(x, axis=-1, keepdims=True))
        p = jnp.exp(z)
        p = p / jnp.sum(p, axis=-1, keepdims=True)

        p = (1.0 - eps_move) * p + eps_move * (1.0 / x.shape[-1])

        return p

    def _build_tree(self, key: jax.Array, params: JaxDecisionTreeParams | None = None):
        nodes = jnp.arange(self.num_nodes, dtype=jnp.int32)
        if params is None and self.shuffle_nodes:
            key, perm_key = jax.random.split(key)
            nodes = jax.random.permutation(perm_key, nodes)
        elif params is not None:
            def shuffled_fn(k):
                k, perm_key = jax.random.split(k)
                return k, jax.random.permutation(perm_key, nodes)

            key, nodes = jax.lax.cond(
                params.shuffle_nodes,
                shuffled_fn,
                lambda k: (k, nodes),
                key,
            )

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
        params: JaxDecisionTreeParams | None = None,
    ):
        learning_rate = self.learning_rate if params is None else params.learning_rate
        lamda_backup = self.lamda_backup if params is None else params.lamda_backup
        backup_steps = self.backup_steps if params is None else params.backup_steps
        wm_backup = self.wm_backup if params is None else params.wm_backup

        target = self._bellman_target(q_values, child_nodes, points, node)
        node_step = learning_rate * (target - q_values[node])
        q_values = q_values.at[node].add(node_step)

        def cond_fn(carry):
            current, weight, steps, _ = carry
            parent = parent_nodes[current]
            has_parent = parent >= 0
            has_budget = steps < backup_steps
            return (weight > 1e-6) & (current != root_node) & has_parent & has_budget

        def body_fn(carry):
            current, weight, steps, q_values = carry
            ancestor = parent_nodes[current]
            target = jax.lax.cond(
                wm_backup,
                lambda _: self._bellman_target(q_values, child_nodes, points, ancestor, activation),
                lambda _: self._bellman_target(q_values, child_nodes, points, ancestor),
                operand=None,
            )
            step_size = learning_rate * weight
            new_value = q_values[ancestor] + step_size * (target - q_values[ancestor])
            q_values = q_values.at[ancestor].set(new_value)
            return ancestor, weight * lamda_backup, steps + 1, q_values

        _, _, _, q_values = jax.lax.while_loop(
            cond_fn,
            body_fn,
            (
                node,
                jnp.asarray(lamda_backup, dtype=q_values.dtype),
                jnp.asarray(0, dtype=jnp.int32),
                q_values,
            ),
        )
        return q_values

    def _decay_fixation_recency(
        self,
        state: JaxDecisionTreeState,
        params: JaxDecisionTreeParams | None = None,
    ) -> JaxDecisionTreeState:
        recency_decay = self._recency_decay_value(params)
        return state._replace(fixation_recency=state.fixation_recency * recency_decay)

    def _recency_decay_value(self, params: JaxDecisionTreeParams | None = None):
        if params is not None:
            return params.recency_decay
        if self.recency_decay_auto:
            return jnp.where(self.wm_decay == 1.0, 0.5, self.wm_decay)
        return jnp.asarray(self.recency_decay, dtype=jnp.float32)

    def _look(
        self,
        state: JaxDecisionTreeState,
        node: jax.Array,
        params: JaxDecisionTreeParams | None = None,
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
                jnp.where(self._recency_decay_value(params) > 0.0, 1.0, 0.0),
            ),
        )

    def _update_activation(
        self,
        state: JaxDecisionTreeState,
        node: jax.Array,
        params: JaxDecisionTreeParams | None = None,
    ) -> JaxDecisionTreeState:
        key, drop_key = jax.random.split(state.rng_key)
        key, q_drop_key = jax.random.split(key)

        wm_decay = self.wm_decay if params is None else params.wm_decay
        q_drop_rate = self.q_drop_rate if params is None else params.q_drop_rate
        q_drift = self.q_drift if params is None else params.q_drift
        q_decay = self.q_decay if params is None else params.q_decay
        activation = state.activation * wm_decay
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
            jax.random.uniform(q_drop_key, shape=(self.num_nodes,)) < q_drop_rate
        )
        inactive = activation == 0.0
        q_drift_key = jax.random.fold_in(q_drop_key, 1)
        q_values = jnp.where(inactive, state.q_values * q_decay, state.q_values)
        q_noise = jax.random.normal(q_drift_key, shape=(self.num_nodes,)) * q_drift
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
        params: JaxDecisionTreeParams | None = None,
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

    def _sample_move_path(self, state: JaxDecisionTreeState, params: JaxDecisionTreeParams | None = None):
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
        key, root, child_nodes, parent_nodes = self._build_tree(key, params)

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

        recency_initial_value = jnp.where(self._recency_decay_value(params) > 0.0, 1.0, 0.0)
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

    def reset(self, key: jax.Array):
        return self.reset_with_params(key, self.default_params())

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
            reward = cum_reward * params.scale_factor
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

    def step(self, state: JaxDecisionTreeState, action: jax.Array):
        return self.step_with_params(state, action, self.default_params())
