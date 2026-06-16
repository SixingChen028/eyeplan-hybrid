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
    g_values: jax.Array
    # search state
    fixation_node: jax.Array
    q_values: jax.Array
    n_visits: jax.Array
    fixation_recency: jax.Array
    activation: jax.Array
    is_discovered: jax.Array
    is_terminal: jax.Array
    time_elapsed: jax.Array
    # implementation detail
    rng_key: jax.Array

class DecisionTreeParams(NamedTuple):
    beta_move: jax.Array
    eps_move: jax.Array
    learning_rate: jax.Array
    lamda_backup: jax.Array
    backup_steps: jax.Array
    wm_decay: jax.Array
    wm_neighbor_activation: jax.Array
    forget_rate: jax.Array
    q_drift: jax.Array
    q_decay: jax.Array
    recency_decay: jax.Array
    cost: jax.Array
    move_cost_scale: jax.Array

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
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        num_nodes: int,
        t_max: int,
        scale_factor: float,
        shuffle_nodes: bool,
        use_recency_obs: bool,
        use_g_values_obs: bool,
        use_q_values_obs: bool,
        use_n_visits_obs: bool,
        use_is_terminal_obs: bool,
        use_time_elapsed_obs: bool,
        disable_persistence: bool,
        activation_masks_actions: bool,
        activation_gates_backup_sink: bool,
        activation_gates_backup_source: bool,
        disable_corruption: bool,
        activation_masks_observation: bool,
        excluded_child_value: float | None,
        point_set: tuple,
    ):
        self.num_nodes = int(num_nodes)
        self.t_max = int(t_max)
        self.scale_factor = float(scale_factor)
        self.shuffle_nodes = bool(shuffle_nodes)
        self.disable_persistence = bool(disable_persistence)
        self.activation_masks_actions = bool(activation_masks_actions)
        self.activation_gates_backup_sink = bool(activation_gates_backup_sink)
        self.activation_gates_backup_source = bool(activation_gates_backup_source)
        self.disable_corruption = bool(disable_corruption)
        self.activation_masks_observation = bool(activation_masks_observation)
        self.excluded_child_value = None if excluded_child_value is None else float(excluded_child_value)
        self.use_recency_obs = bool(use_recency_obs)
        self.use_g_values_obs = bool(use_g_values_obs)
        self.use_q_values_obs = bool(use_q_values_obs)
        self.use_n_visits_obs = bool(use_n_visits_obs)
        self.use_is_terminal_obs = bool(use_is_terminal_obs)
        self.use_time_elapsed_obs = bool(use_time_elapsed_obs)

        # enforce parameterization rules
        if self.disable_persistence: # assumes default activation behavior
            assert self.activation_gates_backup_sink
            assert self.activation_gates_backup_source
            assert not self.disable_corruption
            assert self.activation_masks_observation
        
        # Arbitrary node fixation would bypass the current path-availability assumptions.
        assert self.activation_masks_actions

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
        beta_move: float,
        eps_move: float,
        learning_rate: float,
        lamda_backup: float,
        backup_steps: int,
        wm_decay: float,
        wm_neighbor_activation: float,
        forget_rate: float,
        q_drift: float,
        q_decay,
        recency_decay,
        cost: float,
        move_cost_scale: float = 0.0,
    ) -> DecisionTreeParams:

        assert beta_move >= 0.0, "beta_move must be non-negative."
        assert 0.0 <= eps_move <= 1.0, "eps_move must between 0 and 1"
        assert learning_rate >= 0.0, "learning_rate must be non-negative."
        assert 0.0 <= lamda_backup <= 1.0, "lamda_backup must be between 0 and 1."
        assert backup_steps >= 0, "backup_steps must be non-negative."
        assert 0.0 <= wm_decay <= 1.0, "wm_decay must be between 0 and 1."
        assert 0.0 < wm_neighbor_activation <= 1.0, "wm_neighbor_activation must be positive and at most 1."
        assert 0.0 <= forget_rate <= 1.0, "forget_rate must be between 0 and 1."
        assert q_drift >= 0.0, "q_drift must be non-negative."
        assert 0.0 <= q_decay <= 1.0, "q_decay must be between 0 and 1."
        assert 0.0 <= recency_decay <= 1.0, "recency_decay must be between 0 and 1."
        assert cost >= 0.0, "cost must be non-negative."
        assert move_cost_scale >= 0.0, "move_cost_scale must be non-negative."

        return DecisionTreeParams(
            beta_move=jnp.asarray(beta_move, dtype=jnp.float32),
            eps_move=jnp.asarray(eps_move, dtype=jnp.float32),
            learning_rate=jnp.asarray(learning_rate, dtype=jnp.float32),
            lamda_backup=jnp.asarray(lamda_backup, dtype=jnp.float32),
            backup_steps=jnp.asarray(backup_steps, dtype=jnp.int32),
            wm_decay=jnp.asarray(wm_decay, dtype=jnp.float32),
            wm_neighbor_activation=jnp.asarray(wm_neighbor_activation, dtype=jnp.float32),
            forget_rate=jnp.asarray(forget_rate, dtype=jnp.float32),
            q_drift=jnp.asarray(q_drift, dtype=jnp.float32),
            q_decay=jnp.asarray(q_decay, dtype=jnp.float32),
            recency_decay=jnp.asarray(recency_decay, dtype=jnp.float32),
            cost=jnp.asarray(cost, dtype=jnp.float32),
            move_cost_scale=jnp.asarray(move_cost_scale, dtype=jnp.float32),
        )

    def _zeros(self, dtype: jnp.dtype = jnp.float32) -> jax.Array:
        return jnp.zeros((self.num_nodes,), dtype=dtype)

    def _one_hot(self, label: jax.Array) -> jax.Array:
        label = jnp.asarray(label, dtype=jnp.int32)
        idx = jnp.maximum(label, 0)
        mask = label >= 0
        return jax.nn.one_hot(idx, self.num_nodes, dtype=jnp.float32) * mask.astype(jnp.float32)

    def _softmax(self, x: jax.Array, params: DecisionTreeParams) -> jax.Array:
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
        return state._replace(
            q_values=jnp.where(active, state.q_values, 0.0),
            n_visits=jnp.where(active, state.n_visits, 0),
            fixation_recency=jnp.where(active, state.fixation_recency, 0.0),
            is_terminal=state.is_terminal & active,
        )

    def _backup_target(self, state, node, params: DecisionTreeParams):
        children = state.child_nodes[node]
        child_q = safe_get(state.q_values, children, fill_value=0.0)
        child_active = safe_get(state.activation, children, fill_value=0.0) > 0.0

        if not self.activation_gates_backup_source:
            probs = self._softmax(child_q, params)
        elif self.excluded_child_value is None:
            active = child_active.astype(jnp.float32)
            active_count = jnp.sum(active)
            max_q = jnp.max(jnp.where(child_active, child_q, -jnp.inf))
            z = params.beta_move * jnp.where(child_active, child_q - max_q, 0.0)
            exp_z = jnp.where(child_active, jnp.exp(z), 0.0)
            softmax_probs = exp_z / jnp.maximum(jnp.sum(exp_z), 1e-20)
            random_probs = active / jnp.maximum(active_count, 1.0)
            probs = jnp.where(
                active_count > 0.0,
                (1.0 - params.eps_move) * softmax_probs + params.eps_move * random_probs,
                0.0,
            )
        else:
            child_q = jnp.where(child_active, child_q, self.excluded_child_value)
            probs = self._softmax(child_q, params)

        return state.points[node] + jnp.sum(probs * child_q)

    def _update_q(self, state, params: DecisionTreeParams):
        node = state.fixation_node
        q_values = state.q_values

        def cond_fn(carry):
            current, weight, steps, _ = carry
            has_current = current >= 0
            is_fixated_node = steps < 0
            has_budget = is_fixated_node | (steps < params.backup_steps)
            current_active = safe_get(state.activation, current, fill_value=0.0) > 0.0
            activation_allows_backup = (
                is_fixated_node | current_active if self.activation_gates_backup_sink else True
            )
            return (
                (weight > 1e-6)
                & has_current
                & has_budget
                & activation_allows_backup
            )

        def body_fn(carry):
            current, weight, steps, q_values = carry
            target = self._backup_target(state._replace(q_values=q_values), current, params)
            step_size = params.learning_rate * weight
            new_value = q_values[current] + step_size * (target - q_values[current])
            q_values = q_values.at[current].set(new_value)
            return state.parent_nodes[current], weight * params.lamda_backup, steps + 1, q_values

        _, _, _, q_values = jax.lax.while_loop(
            cond_fn,
            body_fn,
            (
                node,
                jnp.asarray(1.0, dtype=q_values.dtype),
                jnp.asarray(-1, dtype=jnp.int32),
                q_values,
            ),
        )
        return state._replace(q_values=q_values)

    def _look(self, state: DecisionTreeState, node: jax.Array, params: DecisionTreeParams, *, skip_q_update: bool = False):

        children = state.child_nodes[node]
        state = state._replace(
            fixation_node=node,
            g_values=safe_set(state.g_values, children, state.g_values[node] + state.points[node]),
            n_visits=state.n_visits.at[node].add(1),
            fixation_recency=state.fixation_recency.at[node].set(1.0),
            is_discovered=safe_set(state.is_discovered, children, True),
            is_terminal=state.is_terminal.at[node].set(state.child_nodes[node, 0] < 0),
        )
        state = self._update_activation(state, params)
        if self.disable_persistence:
            state = self._clear_inactive_memory(state)
        if not skip_q_update:
            state = self._update_q(state, params)
        if not (self.disable_persistence or self.disable_corruption):
            # skip under disable_persistence because any inactive info has already been cleared
            state = self._corrupt_memory(state, params)
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

    def _corrupt_memory(self, state: DecisionTreeState, params: DecisionTreeParams):
        key, q_drift_key, forget_key = jax.random.split(state.rng_key, 3)
        inactive = state.activation == 0.0
        corruptible = inactive & state.is_discovered

        # add noise/drift to q values outside of WM
        q_values = jnp.where(corruptible, state.q_values * params.q_decay, state.q_values)
        q_noise = jax.random.normal(q_drift_key, shape=(self.num_nodes,)) * params.q_drift
        q_values = jnp.where(corruptible, q_values + q_noise, q_values)

        # stochastically forget node-specific memory outside of WM
        forget_mask = corruptible & (
            jax.random.uniform(forget_key, shape=(self.num_nodes,)) < params.forget_rate
        )
        q_values = jnp.where(forget_mask, 0.0, q_values)
        n_visits = jnp.where(forget_mask, 0, state.n_visits)
        fixation_recency = jnp.where(forget_mask, 0.0, state.fixation_recency)

        # deterministically is_terminal for inactive nodes (never persisted)
        is_terminal = state.is_terminal & ~inactive

        return state._replace(
            q_values=q_values,
            n_visits=n_visits,
            fixation_recency=fixation_recency,
            is_terminal=is_terminal,
            rng_key=key,
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
        if self.activation_masks_observation:
            return state.activation > 0.0
        return state.is_discovered

    def _get_action_mask(self, state: DecisionTreeState) -> jax.Array:
        fixation_allowed = state.time_elapsed != (self.t_max - 1)
        if self.activation_masks_actions:
            node_mask = (state.activation > 0) & fixation_allowed
        else:
            node_mask = jnp.full((self.num_nodes,), fixation_allowed, dtype=jnp.bool_)
        node_mask = node_mask.at[state.root_node].set(fixation_allowed)
        term_mask = jnp.array([True], dtype=jnp.bool_)
        return jnp.concatenate([node_mask, term_mask], axis=0)

    def _get_info(self, state: DecisionTreeState) -> dict[str, jax.Array]:
        return {
            "mask": self._get_action_mask(state),
            "observation_mask": self._get_observation_mask(state),
        }

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
            key, choice_key = jax.random.split(state.rng_key)
            state = state._replace(rng_key=key)

            children = state.child_nodes[node]
            q_children = state.q_values[children]
            probs = self._softmax(q_children, params)
            idx = jax.random.choice(choice_key, 2, p=probs)
            child = children[idx]

            cum_reward = cum_reward + state.points[child]
            path_len = jnp.sum(path >= 0)
            path = path.at[path_len].set(child)
            state = self._look(state, child, params, skip_q_update=True)

            return state, child, cum_reward, path

        state, _, cum_reward, path = jax.lax.while_loop(cond_fn, body_fn, init)

        return cum_reward, path, state

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
            path_len = jnp.sum(choice_path >= 0)
            move_cost = params.move_cost_scale * params.cost * path_len.astype(jnp.float32)
            return move_state, reward * self.scale_factor - move_cost, choice_path

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
