import math
import jax
import jax.numpy as jnp
from typing import NamedTuple

from modules.tree_generation import build_tree_templates

# When changing this file in a way that can affect results, update
# docs/changes.md. If existing checkpoint weights become incompatible,
# also bump COMPAT_VERSION.


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


class DecisionTreeState(NamedTuple):
    # problem definition (static within episode)
    root_node: jax.Array
    points: jax.Array
    child_nodes: jax.Array
    parent_nodes: jax.Array
    # search state
    fixation_node: jax.Array
    g_values: jax.Array
    q_values: jax.Array
    n_visits: jax.Array
    fixation_recency: jax.Array
    activation: jax.Array
    is_discovered: jax.Array
    is_terminal: jax.Array
    time_elapsed: jax.Array
    # implementation detail
    rng_key: jax.Array


class MoveTrace(NamedTuple):
    actions: jax.Array
    activations: jax.Array
    counts: jax.Array
    gs: jax.Array
    qs: jax.Array
    fixation_recency: jax.Array
    is_terminal: jax.Array
    is_discovered: jax.Array
    length: jax.Array

class DecisionTreeParams(NamedTuple):
    wm_decay: jax.Array
    wm_neighbor_activation: jax.Array
    recency_decay: jax.Array
    cost: jax.Array

class DecisionTreeObs(NamedTuple):
    fixation: jax.Array
    fixation_point: jax.Array
    parent: jax.Array
    child: jax.Array
    root: jax.Array
    g_values: jax.Array | None
    q_values: jax.Array | None
    n_visits: jax.Array | None
    is_terminal: jax.Array | None
    recency: jax.Array | None
    time_elapsed: jax.Array | None


class DecisionTreeEnv:
    """Cognitive architecture with one WM store and full backups through available ancestors.

    Every fixation decays/retains activation, refreshes its neighborhood, clears
    unavailable memory, and learns the current observation. Movement skips backups.
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        num_nodes: int,
        t_max: int,
        scale_factor: float,
        use_recency_obs: bool,
        use_g_values_obs: bool,
        use_q_values_obs: bool,
        use_n_visits_obs: bool,
        use_is_terminal_obs: bool,
        use_time_elapsed_obs: bool,
        point_set: tuple,
    ):
        self.num_nodes = int(num_nodes)
        self.t_max = int(t_max)
        self.scale_factor = float(scale_factor)
        self.use_recency_obs = bool(use_recency_obs)
        self.use_g_values_obs = bool(use_g_values_obs)
        self.use_q_values_obs = bool(use_q_values_obs)
        self.use_n_visits_obs = bool(use_n_visits_obs)
        self.use_is_terminal_obs = bool(use_is_terminal_obs)
        self.use_time_elapsed_obs = bool(use_time_elapsed_obs)

        self.point_set = jnp.asarray(point_set, dtype=jnp.float32)
        self.empty_path = -jnp.ones((self.num_nodes,), dtype=jnp.int32)

        self.max_height = math.ceil(self.num_nodes / 2)
        self.min_path_value = self.max_height * min(point_set)

        templates = build_tree_templates(self.num_nodes)
        self._tree_roots = jnp.asarray(templates.roots, dtype=jnp.int32)
        self._tree_child_nodes = jnp.asarray(templates.child_nodes, dtype=jnp.int32)
        self._tree_parent_nodes = jnp.asarray(templates.parent_nodes, dtype=jnp.int32)
        self._tree_probabilities = jnp.asarray(templates.probabilities, dtype=jnp.float32)

        dummy_key = jnp.zeros((2,), dtype=jnp.uint32)
        self.observation_template = self._get_obs(self._sample_initial_state(dummy_key))
        self.action_size = self.num_nodes + 1

    def make_params(
        self,
        *,
        wm_decay: float,
        wm_neighbor_activation: float,
        recency_decay: float,
        cost: float,
    ) -> DecisionTreeParams:
        assert 0.0 <= wm_decay <= 1.0, "wm_decay must be between 0 and 1."
        assert 0.0 < wm_neighbor_activation <= 1.0, "wm_neighbor_activation must be positive and at most 1."
        assert 0.0 <= recency_decay <= 1.0, "recency_decay must be between 0 and 1."
        assert cost >= 0.0, "cost must be non-negative."

        return DecisionTreeParams(
            wm_decay=jnp.asarray(wm_decay, dtype=jnp.float32),
            wm_neighbor_activation=jnp.asarray(wm_neighbor_activation, dtype=jnp.float32),
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

    def _greedy_child_probs(self, child_values: jax.Array) -> jax.Array:
        maximal = child_values == jnp.max(child_values)
        return maximal.astype(jnp.float32) / jnp.sum(maximal)

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

        key, perm_key, swap_key = jax.random.split(key, 3)
        perm = jax.random.permutation(perm_key, jnp.arange(self.num_nodes, dtype=jnp.int32))

        child_safe = jnp.maximum(child_nodes, 0)
        parent_safe = jnp.maximum(parent_nodes, 0)
        mapped_children = jnp.where(child_nodes >= 0, perm[child_safe], -1)
        mapped_parents = jnp.where(parent_nodes >= 0, perm[parent_safe], -1)

        child_nodes = jnp.full_like(child_nodes, -1).at[perm].set(mapped_children)
        parent_nodes = jnp.full_like(parent_nodes, -1).at[perm].set(mapped_parents)
        root = perm[root]

        swap_children = jax.random.bernoulli(swap_key, shape=(self.num_nodes,))
        child_nodes = jnp.where(swap_children[:, None], jnp.flip(child_nodes, axis=1), child_nodes)

        return key, root, child_nodes, parent_nodes

    def _clear_inactive_memory(self, state: DecisionTreeState):
        active = state.activation > 0.0
        g_values = jnp.where(active, state.g_values, self.min_path_value)
        g_values = g_values.at[state.root_node].set(0.0)
        return state._replace(
            q_values=jnp.where(active, state.q_values, 0.0),
            n_visits=jnp.where(active, state.n_visits, 0),
            g_values=g_values,
            fixation_recency=jnp.where(active, state.fixation_recency, 0.0),
            is_terminal=state.is_terminal & active,
        )

    def _backup_target(self, state, node):
        children = state.child_nodes[node]
        child_active = (children >= 0) & (safe_get(state.activation, children, fill_value=0.0) > 0.0)
        child_q = safe_get(state.q_values, children, fill_value=0.0)
        continuation = jnp.max(jnp.where(child_active, child_q, -jnp.inf))
        continuation = jnp.where(jnp.any(child_active), continuation, 0.0)
        # A node's reward is available only while its direct visit is remembered.
        reward = jnp.where(state.n_visits[node] > 0, state.points[node], 0.0)
        return reward + continuation

    def _update_q(self, state):
        def cond_fn(carry):
            current, _ = carry
            return (current >= 0) & (safe_get(state.activation, current, fill_value=0.0) > 0.0)

        def body_fn(carry):
            current, q_values = carry
            target = self._backup_target(state._replace(q_values=q_values), current)
            return state.parent_nodes[current], q_values.at[current].set(target)

        _, q_values = jax.lax.while_loop(cond_fn, body_fn, (state.fixation_node, state.q_values))
        return state._replace(q_values=q_values)

    def _look(
        self,
        state: DecisionTreeState,
        node: jax.Array,
        params: DecisionTreeParams,
        *,
        skip_q_update: bool = False,
    ):

        state = state._replace(fixation_node=node)
        state = self._update_activation(state, params)
        state = self._clear_inactive_memory(state)
        children = state.child_nodes[node]
        child_activation = jnp.maximum(
            safe_get(state.activation, children, fill_value=0.0),
            params.wm_neighbor_activation,
        )
        activation = state.activation.at[node].set(1.0)
        activation = safe_set(activation, children, child_activation)
        is_discovered = state.is_discovered.at[node].set(True)
        is_discovered = safe_set(is_discovered, children, True)
        activation = jnp.where(is_discovered, activation, 0.0)
        state = state._replace(
            g_values=safe_set(state.g_values, children, state.g_values[node] + state.points[node]),
            n_visits=state.n_visits.at[node].add(1),
            fixation_recency=state.fixation_recency.at[node].set(1.0),
            activation=activation,
            is_discovered=is_discovered,
            is_terminal=state.is_terminal.at[node].set(state.child_nodes[node, 0] < 0),
        )
        if not skip_q_update:
            state = self._update_q(state)
        return state

    def _update_activation(self, state: DecisionTreeState, params: DecisionTreeParams) -> DecisionTreeState:
        node = state.fixation_node

        # apply decay
        activation = state.activation * params.wm_decay
        activation = jnp.clip(activation, 0.0, 1.0)

        # stochastically drop nodes from WM
        key, drop_key = jax.random.split(state.rng_key)
        keep = jax.random.uniform(drop_key, shape=(self.num_nodes,)) < activation
        activation = jnp.where(keep, activation, 0.0)

        # activate fixated, parent, children
        activation = activation.at[node].set(1.0)
        parent = state.parent_nodes[node]
        children = state.child_nodes[node]
        parent_activation = jnp.maximum(
            safe_get(activation, parent, fill_value=0.0),
            params.wm_neighbor_activation,
        )
        child_activation = jnp.maximum(
            safe_get(activation, children, fill_value=0.0),
            params.wm_neighbor_activation,
        )
        activation = safe_set(activation, parent, parent_activation)
        activation = safe_set(activation, children, child_activation)

        return state._replace(
            rng_key=key,
            activation=activation,
        )

    def _get_obs(self, state: DecisionTreeState) -> DecisionTreeObs:
        observation_mask = self._get_observation_mask(state)

        child1, child2 = state.child_nodes[state.fixation_node]
        return DecisionTreeObs(
            fixation=self._one_hot(state.fixation_node),
            fixation_point=jnp.array([state.points[state.fixation_node]], dtype=jnp.float32),
            parent=self._one_hot(state.parent_nodes[state.fixation_node]),
            child=self._one_hot(child1) + self._one_hot(child2),
            root=self._one_hot(state.root_node),
            g_values=(
                jnp.where(observation_mask, state.g_values, 0.0)
                if self.use_g_values_obs
                else None
            ),
            q_values=(
                jnp.where(observation_mask, state.q_values, 0.0)
                if self.use_q_values_obs
                else None
            ),
            n_visits=(
                jnp.where(observation_mask, state.n_visits, 0).astype(jnp.float32)
                if self.use_n_visits_obs
                else None
            ),
            is_terminal=(
                (state.is_terminal & observation_mask).astype(jnp.float32)
                if self.use_is_terminal_obs
                else None
            ),
            recency=(
                jnp.where(observation_mask, state.fixation_recency, 0.0)
                if self.use_recency_obs
                else None
            ),
            time_elapsed=(
                jnp.array([state.time_elapsed], dtype=jnp.float32)
                if self.use_time_elapsed_obs
                else None
            ),
        )

    def _get_observation_mask(self, state: DecisionTreeState) -> jax.Array:
        return state.activation > 0.0

    def _get_action_mask(self, state: DecisionTreeState) -> jax.Array:
        fixation_allowed = state.time_elapsed != (self.t_max - 1)
        node_mask = (state.activation > 0) & fixation_allowed
        node_mask = node_mask.at[state.root_node].set(fixation_allowed)
        term_mask = jnp.array([True], dtype=jnp.bool_)
        return jnp.concatenate([node_mask, term_mask], axis=0)

    def _get_info(self, state: DecisionTreeState) -> dict[str, jax.Array]:
        return {
            "mask": self._get_action_mask(state),
            "observation_mask": self._get_observation_mask(state),
        }

    def _empty_move_trace(self) -> MoveTrace:
        matrix_shape = (self.num_nodes, self.num_nodes)
        return MoveTrace(
            actions=-jnp.ones((self.num_nodes,), dtype=jnp.int32),
            activations=jnp.zeros(matrix_shape, dtype=jnp.float32),
            counts=jnp.zeros(matrix_shape, dtype=jnp.int32),
            gs=jnp.zeros(matrix_shape, dtype=jnp.float32),
            qs=jnp.zeros(matrix_shape, dtype=jnp.float32),
            fixation_recency=jnp.zeros(matrix_shape, dtype=jnp.float32),
            is_terminal=jnp.zeros(matrix_shape, dtype=jnp.bool_),
            is_discovered=jnp.zeros(matrix_shape, dtype=jnp.bool_),
            length=jnp.array(0, dtype=jnp.int32),
        )

    def _append_move_trace(self, trace: MoveTrace, action: jax.Array, state: DecisionTreeState) -> MoveTrace:
        index = trace.length
        return MoveTrace(
            actions=trace.actions.at[index].set(action),
            activations=trace.activations.at[index].set(state.activation),
            counts=trace.counts.at[index].set(state.n_visits),
            gs=trace.gs.at[index].set(state.g_values),
            qs=trace.qs.at[index].set(state.q_values),
            fixation_recency=trace.fixation_recency.at[index].set(state.fixation_recency),
            is_terminal=trace.is_terminal.at[index].set(state.is_terminal),
            is_discovered=trace.is_discovered.at[index].set(state.is_discovered),
            length=index + 1,
        )

    def _move_to_child(self, state: DecisionTreeState, node: jax.Array, params: DecisionTreeParams):
        key, choice_key = jax.random.split(state.rng_key)
        state = state._replace(rng_key=key)

        children = state.child_nodes[node]
        q_children = state.q_values[children]
        probs = self._greedy_child_probs(q_children)
        idx = jax.random.choice(choice_key, 2, p=probs)
        child = children[idx]
        return self._look(state, child, params, skip_q_update=True), child

    def _sample_move_path(self, state: DecisionTreeState, params: DecisionTreeParams):
        path = self.empty_path
        state = self._look(state, state.root_node, params, skip_q_update=True)

        init = (
            state,
            state.root_node,
            jnp.array(0.0, dtype=jnp.float32),
            path,
        )

        def cond_fn(carry):
            state, node, _, _ = carry
            return state.child_nodes[node, 0] >= 0

        def body_fn(carry):
            state, node, cum_reward, path = carry
            state, child = self._move_to_child(state, node, params)

            cum_reward = cum_reward + state.points[child]
            path_len = jnp.sum(path >= 0)
            path = path.at[path_len].set(child)

            return state, child, cum_reward, path

        state, _, cum_reward, path = jax.lax.while_loop(cond_fn, body_fn, init)

        return cum_reward, path, state

    def _sample_move_path_with_trace(self, state: DecisionTreeState, params: DecisionTreeParams):
        path = self.empty_path
        state = self._look(state, state.root_node, params, skip_q_update=True)
        trace = self._append_move_trace(self._empty_move_trace(), state.root_node, state)

        init = (
            state,
            state.root_node,
            jnp.array(0.0, dtype=jnp.float32),
            path,
            trace,
        )

        def cond_fn(carry):
            state, node, _, _, _ = carry
            return state.child_nodes[node, 0] >= 0

        def body_fn(carry):
            state, node, cum_reward, path, trace = carry
            state, child = self._move_to_child(state, node, params)

            cum_reward = cum_reward + state.points[child]
            path_len = jnp.sum(path >= 0)
            path = path.at[path_len].set(child)
            trace = self._append_move_trace(trace, child, state)

            return state, child, cum_reward, path, trace

        state, _, cum_reward, path, trace = jax.lax.while_loop(cond_fn, body_fn, init)
        return cum_reward, path, state, trace

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

        return DecisionTreeState(
            root_node=root,
            points=points,
            child_nodes=child_nodes,
            parent_nodes=parent_nodes,
            g_values=jnp.full((self.num_nodes,), self.min_path_value, dtype=jnp.float32).at[root].set(0.0),
            fixation_node=root,
            q_values=self._zeros(),
            n_visits=self._zeros(jnp.int32),
            fixation_recency=self._zeros(),
            activation=self._zeros(),
            is_discovered=self._zeros(jnp.bool_).at[root].set(True),
            is_terminal=self._zeros(jnp.bool_),
            time_elapsed=jnp.int32(0),
            rng_key=key,
        )

    def reset(self, key: jax.Array, params: DecisionTreeParams):
        state = self._sample_initial_state(key)
        state = self._look(state, state.root_node, params)
        obs = self._get_obs(state)
        info = self._get_info(state)
        return state, obs, info

    def step(self, state: DecisionTreeState, action: jax.Array, params: DecisionTreeParams):
        state = state._replace(
            time_elapsed=state.time_elapsed + 1,
            fixation_recency=state.fixation_recency * params.recency_decay,
        )

        def look_branch():
            return self._look(state, action, params), -params.cost, self.empty_path

        def terminate_branch():
            reward, choice_path, move_state = self._sample_move_path(state, params)
            return move_state, reward * self.scale_factor, choice_path

        state, reward, choice_path = jax.lax.cond(
            action < self.num_nodes,
            look_branch,
            terminate_branch,
        )
        done = (action == self.num_nodes) | (state.time_elapsed == self.t_max)
        obs = self._get_obs(state)
        info = {
            **self._get_info(state),
            "choice_path": choice_path,
            "move_reward": jnp.where(action == self.num_nodes, reward, 0.0),
        }

        return state, obs, reward, done, info

    def step_with_move_trace(self, state: DecisionTreeState, action: jax.Array, params: DecisionTreeParams):
        """Step the cognitive architecture and record post-action movement memory for detailed simulations."""
        state = state._replace(
            time_elapsed=state.time_elapsed + 1,
            fixation_recency=state.fixation_recency * params.recency_decay,
        )

        def look_branch():
            return self._look(state, action, params), -params.cost, self.empty_path, self._empty_move_trace()

        def terminate_branch():
            reward, choice_path, move_state, move_trace = self._sample_move_path_with_trace(state, params)
            return move_state, reward * self.scale_factor, choice_path, move_trace

        state, reward, choice_path, move_trace = jax.lax.cond(
            action < self.num_nodes,
            look_branch,
            terminate_branch,
        )
        done = (action == self.num_nodes) | (state.time_elapsed == self.t_max)
        obs = self._get_obs(state)
        info = {
            **self._get_info(state),
            "choice_path": choice_path,
            "move_reward": jnp.where(action == self.num_nodes, reward, 0.0),
            "move_trace": move_trace,
        }

        return state, obs, reward, done, info
