import math
import jax
import jax.numpy as jnp
from typing import NamedTuple

from modules.tree_generation import build_tree_templates

BACKUP_MODES = ("full", "wm_both", "wm_zero", "wm_partial")
BACKUP_MODE_FULL = "full"
BACKUP_MODE_WM_BOTH = "wm_both"
BACKUP_MODE_WM_ZERO = "wm_zero"
BACKUP_MODE_WM_PARTIAL = "wm_partial"


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
    is_terminal: jax.Array

class JaxDecisionTreeParams(NamedTuple):
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
    best_open_value: jax.Array | None
    best_terminal_value: jax.Array | None
    recency: jax.Array | None
    time_elapsed: jax.Array | None


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
        use_g_values_obs: bool,
        use_q_values_obs: bool,
        use_n_visits_obs: bool,
        use_is_terminal_obs: bool,
        use_time_elapsed_obs: bool,
        backup_mode: str,
        point_set: tuple,
        wm_only: bool = False,
        persist_terminal: bool = False,
    ):
        self.num_nodes = int(num_nodes)
        self.t_max = int(t_max)
        self.scale_factor = float(scale_factor)
        self.shuffle_nodes = bool(shuffle_nodes)
        self.wm_only = bool(wm_only)
        self.persist_terminal = bool(persist_terminal)
        self.use_recency_obs = bool(use_recency_obs)
        self.use_best_open_value_obs = bool(use_best_open_value_obs)
        self.use_best_terminal_value_obs = bool(use_best_terminal_value_obs)
        self.use_g_values_obs = bool(use_g_values_obs)
        self.use_q_values_obs = bool(use_q_values_obs)
        self.use_n_visits_obs = bool(use_n_visits_obs)
        self.use_is_terminal_obs = bool(use_is_terminal_obs)
        self.use_time_elapsed_obs = bool(use_time_elapsed_obs)
        if backup_mode not in BACKUP_MODES:
            raise ValueError(f"backup_mode must be one of {BACKUP_MODES}.")
        self.backup_mode = backup_mode

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
    ) -> JaxDecisionTreeParams:

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

        return JaxDecisionTreeParams(
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

    def _compute_path_values(self, parent_nodes, points):
        g_values = self._zeros()

        def body_fn(_, g_values):
            parent_safe = jnp.maximum(parent_nodes, 0)
            parent_values = g_values[parent_safe] + points[parent_safe]
            return jnp.where(parent_nodes >= 0, parent_values, 0.0)

        return jax.lax.fori_loop(0, math.ceil(self.num_nodes / 2), body_fn, g_values)

    def _clear_inactive_memory(self, state: JaxDecisionTreeState):
        active = state.activation > 0.0
        if self.wm_only:
            return state._replace(
                q_values=jnp.where(active, state.q_values, 0.0),
                n_visits=jnp.where(active, state.n_visits, 0),
                fixation_recency=jnp.where(active, state.fixation_recency, 0.0),
                is_terminal=state.is_terminal & active,
            )

        # Terminality is normally working-memory dependent, but can be made persistent for ablations.
        if not self.persist_terminal:
            state = state._replace(is_terminal=state.is_terminal & active)

        return state

    def _backup_target(self, state, node, params: JaxDecisionTreeParams):
        children = state.child_nodes[node]
        child_q = safe_get(state.q_values, children, fill_value=0.0)
        child_active = safe_get(state.activation, children, fill_value=0.0) > 0.0

        if self.backup_mode == BACKUP_MODE_WM_ZERO:
            child_q = jnp.where(child_active, child_q, 0.0)
            probs = self._softmax(child_q, params)
        elif self.backup_mode == BACKUP_MODE_WM_PARTIAL:
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
        elif self.backup_mode in {BACKUP_MODE_FULL, BACKUP_MODE_WM_BOTH}:
            probs = self._softmax(child_q, params)
        else:
            raise AssertionError(f"Unexpected backup_mode: {self.backup_mode}")

        return state.points[node] + jnp.sum(probs * child_q)

    def _update_q(self, state, params: JaxDecisionTreeParams):
        node = state.fixation_node
        q_values = state.q_values
        children = state.child_nodes[node]
        child_q = safe_get(q_values, children, fill_value=0.0)
        target = state.points[node] + jnp.max(child_q)
        node_step = params.learning_rate * (target - q_values[node])
        q_values = q_values.at[node].add(node_step)

        def cond_fn(carry):
            current, weight, steps, _ = carry
            parent = state.parent_nodes[current]
            has_parent = parent >= 0
            has_budget = steps < params.backup_steps
            parent_active = safe_get(state.activation, parent, fill_value=0.0) > 0.0
            full_backup = (self.backup_mode == BACKUP_MODE_FULL) & (not self.wm_only)
            wm_allows_backup = full_backup | parent_active
            return (weight > 1e-6) & (current != state.root_node) & has_parent & has_budget & wm_allows_backup

        def body_fn(carry):
            current, weight, steps, q_values = carry
            ancestor = state.parent_nodes[current]
            target = self._backup_target(state._replace(q_values=q_values), ancestor, params)
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

    def _look(
        self,
        state: JaxDecisionTreeState,
        node: jax.Array,
        params: JaxDecisionTreeParams,
    ) -> JaxDecisionTreeState:
        state = self._update_fixation_memory(state, node, params)
        state = self._update_q(state, params)
        if not self.wm_only:
            state = self._corrupt_memory(state, params)
        return state

    def _update_fixation_memory(
        self,
        state: JaxDecisionTreeState,
        node: jax.Array,
        params: JaxDecisionTreeParams,
    ) -> JaxDecisionTreeState:
        state = state._replace(
            fixation_node=node,
            n_visits=state.n_visits.at[node].add(1),
            fixation_recency=state.fixation_recency.at[node].set(1.0),
            is_terminal=state.is_terminal.at[node].set(state.child_nodes[node, 0] < 0),
        )
        state = self._update_activation(state, params)
        state = self._clear_inactive_memory(state)
        return state

    def _look_without_q(
        self,
        state: JaxDecisionTreeState,
        node: jax.Array,
        params: JaxDecisionTreeParams,
    ) -> JaxDecisionTreeState:
        state = self._update_fixation_memory(state, node, params)
        if not self.wm_only:
            state = self._corrupt_memory(state, params)
        return state

    def _update_activation(self, state: JaxDecisionTreeState, params: JaxDecisionTreeParams) -> JaxDecisionTreeState:
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
        # root is always active
        activation = activation.at[state.root_node].set(1.0)

        return state._replace(
            rng_key=key,
            activation=activation,
        )

    def _corrupt_memory(self, state: JaxDecisionTreeState, params: JaxDecisionTreeParams):
        key, q_drift_key, forget_key = jax.random.split(state.rng_key, 3)
        inactive = state.activation == 0.0

        # add noise/drift to q values outside of WM
        q_values = jnp.where(inactive, state.q_values * params.q_decay, state.q_values)
        q_noise = jax.random.normal(q_drift_key, shape=(self.num_nodes,)) * params.q_drift
        q_values = jnp.where(inactive, q_values + q_noise, q_values)

        # stochastically forget node-specific memory outside of WM
        forget_mask = inactive & (
            jax.random.uniform(forget_key, shape=(self.num_nodes,)) < params.forget_rate
        )
        q_values = jnp.where(forget_mask, 0.0, q_values)
        n_visits = jnp.where(forget_mask, 0, state.n_visits)
        fixation_recency = jnp.where(forget_mask, 0.0, state.fixation_recency)

        return state._replace(
            q_values=q_values,
            n_visits=n_visits,
            fixation_recency=fixation_recency,
            rng_key=key,
        )

    def _get_obs(self, state: JaxDecisionTreeState) -> DecisionTreeObs:
        active_mask = state.activation > 0.0
        known_mask = safe_get(state.n_visits > 0, state.parent_nodes, fill_value=True)
        g_value_mask = active_mask if self.wm_only else known_mask

        best_open_value = None
        if self.use_best_open_value_obs:
            unseen_mask = state.n_visits == 0
            open_mask = g_value_mask & unseen_mask
            open_obs = jnp.max(jnp.where(open_mask, state.g_values, self.min_path_value))
            best_open_value = jnp.array([open_obs], dtype=jnp.float32)

        best_terminal_value = None
        if self.use_best_terminal_value_obs:
            total_values = state.g_values + state.points
            best_terminal_obs = jnp.max(jnp.where(state.is_terminal, total_values, self.min_path_value))
            best_terminal_value = jnp.array([best_terminal_obs], dtype=jnp.float32)

        child1, child2 = state.child_nodes[state.fixation_node]
        return DecisionTreeObs(
            fixation=self._one_hot(state.fixation_node),
            fixation_point=jnp.array([state.points[state.fixation_node]], dtype=jnp.float32),
            parent=self._one_hot(state.parent_nodes[state.fixation_node]),
            child=self._one_hot(child1) + self._one_hot(child2),
            root=self._one_hot(state.root_node),
            g_values=(
                jnp.where(g_value_mask, state.g_values, 0.0)
                if self.use_g_values_obs
                else None
            ),
            q_values=state.q_values if self.use_q_values_obs else None,
            n_visits=state.n_visits.astype(jnp.float32) if self.use_n_visits_obs else None,
            is_terminal=state.is_terminal.astype(jnp.float32) if self.use_is_terminal_obs else None,
            best_open_value=best_open_value,
            best_terminal_value=best_terminal_value,
            recency=state.fixation_recency if self.use_recency_obs else None,
            time_elapsed=(
                jnp.array([state.time_elapsed], dtype=jnp.float32)
                if self.use_time_elapsed_obs
                else None
            ),
        )

    def _get_action_mask(self, state: JaxDecisionTreeState) -> jax.Array:
        fixation_allowed = state.time_elapsed != (self.t_max - 1)
        node_mask = (state.activation > 0) & fixation_allowed
        term_mask = jnp.array([True], dtype=jnp.bool_)
        return jnp.concatenate([node_mask, term_mask], axis=0)

    def _sample_move_path(self, state: JaxDecisionTreeState, params: JaxDecisionTreeParams):
        path = self.empty_path
        state = self._look_without_q(state, state.root_node, params)

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
            state = self._look_without_q(state, child, params)

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
            is_terminal=self._zeros(jnp.bool_),
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
            "mask": self._get_action_mask(state),
            "choice_path": choice_path,
            "move_reward": jnp.where(action == self.num_nodes, reward, 0.0),
        }

        return state, obs, reward, done, info
