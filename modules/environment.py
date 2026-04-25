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
    planner_known: jax.Array
    planner_expanded: jax.Array
    q_values: jax.Array
    g_values: jax.Array
    n_visits: jax.Array
    activation: jax.Array
    active_mask: jax.Array
    chosen_path: jax.Array
    chosen_path_len: jax.Array


class JaxDecisionTreeEnv:
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        num_nodes: int = 15,
        beta_move: float = 4.0,
        eps_move: float = 0.02,
        learning_rate: float = 0.2,
        lamda_backup: float = 0.0,
        wm_decay: float = 0.8,
        t_max: int = 100,
        cost: float = 0.01,
        scale_factor: float = 1 / 8,
        shuffle_nodes: bool = True,
        point_set=None,
    ):
        self.num_nodes = int(num_nodes)
        self.beta_move = float(beta_move)
        self.eps_move = float(eps_move)
        self.learning_rate = float(learning_rate)
        self.lamda_backup = float(lamda_backup)
        self.wm_decay = float(wm_decay)
        self.t_max = int(t_max)
        self.cost = float(cost)
        self.scale_factor = float(scale_factor)
        self.shuffle_nodes = bool(shuffle_nodes)

        if point_set is None:
            point_set = [-8, -4, -2, -1, 1, 2, 4, 8]
        self.point_set = jnp.asarray(point_set, dtype=jnp.float32)
        self.empty_path = -jnp.ones((self.num_nodes,), dtype=jnp.int32)

        observation_size = (
            self.num_nodes
            + 1
            + self.num_nodes * 3
            + self.num_nodes
            + self.num_nodes
            + self.num_nodes
            + self.num_nodes
            + 1
        )
        self.observation_shape = (observation_size,)
        self.action_size = self.num_nodes + 1

    def _one_hot(self, label: jax.Array) -> jax.Array:
        label = jnp.asarray(label, dtype=jnp.int32)
        idx = jnp.maximum(label, 0)
        mask = label >= 0
        return jax.nn.one_hot(idx, self.num_nodes, dtype=jnp.float32) * mask.astype(jnp.float32)

    def _softmax(self, x: jax.Array) -> jax.Array:
        if x.size == 0:
            return x

        z = self.beta_move * (x - jnp.max(x))
        p = jnp.exp(z)
        p = p / jnp.sum(p)

        if self.eps_move > 0.0:
            p = (1.0 - self.eps_move) * p + self.eps_move * (1.0 / x.shape[0])

        return p

    def _clear_chosen_path(self, state: JaxDecisionTreeState) -> JaxDecisionTreeState:
        return state._replace(
            chosen_path=self.empty_path,
            chosen_path_len=jnp.int32(0),
        )

    def _build_tree(self, key: jax.Array):
        nodes = jnp.arange(self.num_nodes, dtype=jnp.int32)
        if self.shuffle_nodes:
            key, perm_key = jax.random.split(key)
            nodes = jax.random.permutation(perm_key, nodes)

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

    def _update_g(self, g_values, parent_nodes, root_node, points, node):
        def root_fn(g):
            return g.at[node].set(0.0)

        def non_root_fn(g):
            parent = parent_nodes[node]
            new_value = g[parent] + points[parent]
            return g.at[node].set(new_value)

        return jax.lax.cond(node == root_node, root_fn, non_root_fn, g_values)

    def _expand(self, planner_known, planner_expanded, g_values, node, parent_nodes, child_nodes, root_node, points):
        planner_expanded = planner_expanded.at[node].set(True)
        children = child_nodes[node]

        def child_fn(carry, child):
            planner_known, g_values = carry
            safe_child = jnp.maximum(child, 0)
            valid_child = child >= 0
            unseen_child = ~planner_known[safe_child]
            should_add = valid_child & unseen_child

            def add_fn(state):
                planner_known, g_values = state
                planner_known = planner_known.at[child].set(True)
                g_values = self._update_g(g_values, parent_nodes, root_node, points, child)
                return planner_known, g_values

            planner_known, g_values = jax.lax.cond(
                should_add,
                add_fn,
                lambda x: x,
                (planner_known, g_values),
            )

            return (planner_known, g_values), None

        (planner_known, g_values), _ = jax.lax.scan(
            child_fn,
            (planner_known, g_values),
            children,
        )

        return planner_known, planner_expanded, g_values

    def _bellman_target(self, q_values, child_nodes, points, node):
        children = child_nodes[node]
        has_child = children[0] >= 0

        def leaf_target(_):
            return points[node]

        def non_leaf_target(_):
            child_q = q_values[children]
            return points[node] + jnp.max(child_q)

        return jax.lax.cond(has_child, non_leaf_target, leaf_target, operand=None)

    def _update_q(self, q_values, planner_expanded, child_nodes, parent_nodes, root_node, points, node):
        def do_update(q_values):
            target = self._bellman_target(q_values, child_nodes, points, node)
            node_step = self.learning_rate * (target - q_values[node])
            q_values = q_values.at[node].add(node_step)

            def cond_fn(carry):
                current, weight, _ = carry
                parent = parent_nodes[current]
                has_parent = parent >= 0
                return (weight > 1e-6) & (current != root_node) & has_parent

            def body_fn(carry):
                current, weight, q_values = carry
                ancestor = parent_nodes[current]
                target = self._bellman_target(q_values, child_nodes, points, ancestor)
                step_size = self.learning_rate * weight
                new_value = q_values[ancestor] + step_size * (target - q_values[ancestor])
                q_values = q_values.at[ancestor].set(new_value)
                return ancestor, weight * self.lamda_backup, q_values

            _, _, q_values = jax.lax.while_loop(
                cond_fn,
                body_fn,
                (node, jnp.asarray(self.lamda_backup, dtype=q_values.dtype), q_values),
            )
            return q_values

        return jax.lax.cond(planner_expanded[node], do_update, lambda x: x, q_values)

    def _look(self, state: JaxDecisionTreeState, node: jax.Array) -> JaxDecisionTreeState:
        n_visits = state.n_visits.at[node].add(1)
        g_values = self._update_g(state.g_values, state.parent_nodes, state.root_node, state.points, node)

        planner_known = state.planner_known
        planner_expanded = state.planner_expanded

        children = state.child_nodes[node]
        child0 = children[0]
        child1 = children[1]
        child0_known = planner_known[jnp.maximum(child0, 0)]
        child1_known = planner_known[jnp.maximum(child1, 0)]
        hidden_child = ((child0 >= 0) & (~child0_known)) | ((child1 >= 0) & (~child1_known))

        should_expand = ((~planner_expanded[node]) & planner_known[node]) | (
            planner_expanded[node] & hidden_child
        )

        planner_known, planner_expanded, g_values = jax.lax.cond(
            should_expand,
            lambda x: self._expand(x[0], x[1], x[2], node, state.parent_nodes, state.child_nodes, state.root_node, state.points),
            lambda x: x,
            (planner_known, planner_expanded, g_values),
        )

        q_values = self._update_q(
            state.q_values,
            planner_expanded,
            state.child_nodes,
            state.parent_nodes,
            state.root_node,
            state.points,
            node,
        )

        return state._replace(
            planner_known=planner_known,
            planner_expanded=planner_expanded,
            g_values=g_values,
            q_values=q_values,
            n_visits=n_visits,
        )

    def _update_activation(self, state: JaxDecisionTreeState, node: jax.Array) -> JaxDecisionTreeState:
        key, drop_key = jax.random.split(state.rng_key)

        activation = state.activation * self.wm_decay
        activation = jnp.clip(activation, 0.0, 1.0)
        activation = activation.at[node].set(1.0)

        parent = state.parent_nodes[node]
        activation = jax.lax.cond(
            parent >= 0,
            lambda x: x.at[parent].set(1.0),
            lambda x: x,
            activation,
        )

        children = state.child_nodes[node]
        child0 = children[0]
        child1 = children[1]

        activation = jax.lax.cond(
            child0 >= 0,
            lambda x: x.at[child0].set(1.0),
            lambda x: x,
            activation,
        )
        activation = jax.lax.cond(
            child1 >= 0,
            lambda x: x.at[child1].set(1.0),
            lambda x: x,
            activation,
        )

        activation = activation.at[state.root_node].set(1.0)

        keep = jax.random.uniform(drop_key, shape=(self.num_nodes,)) < activation
        activation = jnp.where(keep, activation, 0.0)

        return state._replace(
            rng_key=key,
            activation=activation,
            active_mask=keep,
        )

    def get_obs(self, state: JaxDecisionTreeState) -> jax.Array:
        fixation_parent = state.parent_nodes[state.fixation_node]
        fixation_children = state.child_nodes[state.fixation_node]

        obs = jnp.concatenate(
            [
                self._one_hot(state.fixation_node),
                jnp.array([state.points[state.fixation_node]], dtype=jnp.float32),
                self._one_hot(fixation_parent),
                self._one_hot(fixation_children[0]),
                self._one_hot(fixation_children[1]),
                self._one_hot(state.root_node),
                state.g_values,
                state.q_values,
                state.n_visits.astype(jnp.float32),
                jnp.array([state.time_elapsed], dtype=jnp.float32),
            ]
        )

        return obs

    def get_action_mask(self, state: JaxDecisionTreeState) -> jax.Array:
        gated = state.planner_known & state.active_mask
        gated = gated.at[state.root_node].set(True)

        mask = jnp.zeros((self.action_size,), dtype=jnp.bool_)
        mask = mask.at[: self.num_nodes].set(gated)
        mask = mask.at[-1].set(True)

        terminal_mask = jnp.zeros((self.action_size,), dtype=jnp.bool_)
        terminal_mask = terminal_mask.at[-1].set(True)
        return jnp.where(state.time_elapsed == (self.t_max - 1), terminal_mask, mask)

    def _move(self, state: JaxDecisionTreeState):
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
            probs = self._softmax(q_children)
            idx = jax.random.choice(choice_key, 2, p=probs)
            child = children[idx]

            cum_reward = cum_reward + state.points[child]
            path = path.at[path_len].set(child)
            path_len = path_len + 1

            return child, cum_reward, path, path_len, key

        _, cum_reward, path, path_len, key = jax.lax.while_loop(cond_fn, body_fn, init)

        return cum_reward, path, path_len, key

    def reset(self, key: jax.Array):
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

        planner_known = jnp.zeros((self.num_nodes,), dtype=jnp.bool_)
        planner_known = planner_known.at[root].set(True)

        root_children = child_nodes[root]
        root_children_safe = jnp.maximum(root_children, 0)
        root_children_valid = root_children >= 0
        planner_known = planner_known.at[root_children_safe].set(root_children_valid)

        planner_expanded = jnp.zeros((self.num_nodes,), dtype=jnp.bool_)
        planner_expanded = planner_expanded.at[root].set(True)

        state = JaxDecisionTreeState(
            rng_key=key,
            time_elapsed=jnp.int32(0),
            fixation_node=root,
            root_node=root,
            points=points,
            child_nodes=child_nodes,
            parent_nodes=parent_nodes,
            planner_known=planner_known,
            planner_expanded=planner_expanded,
            q_values=jnp.zeros((self.num_nodes,), dtype=jnp.float32).at[root].set(0.0),
            g_values=jnp.zeros((self.num_nodes,), dtype=jnp.float32),
            n_visits=jnp.zeros((self.num_nodes,), dtype=jnp.int32),
            activation=jnp.zeros((self.num_nodes,), dtype=jnp.float32),
            active_mask=jnp.zeros((self.num_nodes,), dtype=jnp.bool_),
            chosen_path=self.empty_path,
            chosen_path_len=jnp.int32(0),
        )

        state = self._update_activation(state, state.root_node)

        obs = self.get_obs(state)
        info = {"mask": self.get_action_mask(state)}
        return state, obs, info

    def step(self, state: JaxDecisionTreeState, action: jax.Array):
        action = jnp.asarray(action, dtype=jnp.int32)
        state = state._replace(time_elapsed=state.time_elapsed + 1)
        action = jnp.where(state.time_elapsed == self.t_max, jnp.int32(self.num_nodes), action)
        reward = jnp.array(-self.cost, dtype=jnp.float32)

        def fixation_branch(payload):
            state, reward = payload
            state = self._look(state, action)
            state = state._replace(fixation_node=action)
            state = self._update_activation(state, action)
            state = self._clear_chosen_path(state)
            return state, reward

        def move_branch(payload):
            state, reward = payload
            cum_reward, path, path_len, key = self._move(state)
            reward = cum_reward * self.scale_factor
            state = state._replace(
                rng_key=key,
                chosen_path=path,
                chosen_path_len=path_len,
            )
            return state, reward

        def noop_branch(payload):
            state, reward = payload
            state = self._clear_chosen_path(state)
            return state, reward

        state, reward = jax.lax.cond(
            action < self.num_nodes,
            fixation_branch,
            lambda x: jax.lax.cond(action == self.num_nodes, move_branch, noop_branch, x),
            (state, reward),
        )

        done = (action == self.num_nodes) | (state.time_elapsed == self.t_max)
        obs = self.get_obs(state)
        info = {"mask": self.get_action_mask(state)}

        return state, obs, reward, done, jnp.array(False), info
