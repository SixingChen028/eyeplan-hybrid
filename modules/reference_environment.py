import numpy as np


class ReferenceDecisionTreeEnv:
    """Simple NumPy reference implementation of the decision-tree environment."""

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
        canonicalize: bool = False,
        recency_decay="off",
        wm_backup: bool = False,
        point_set=None,
        seed: int | None = None,
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
        self.q_decay_auto, self.q_decay = self._parse_q_decay(q_decay)
        self.t_max = int(t_max)
        self.cost = float(cost)
        self.scale_factor = float(scale_factor)
        self.shuffle_nodes = bool(shuffle_nodes)
        self.canonicalize = bool(canonicalize)
        self.use_recency_obs, self.recency_decay_auto, self.recency_decay = self._parse_recency_decay(recency_decay)
        self.wm_backup = bool(wm_backup)

        if point_set is None:
            point_set = [-8, -4, -2, -1, 1, 2, 4, 8]
        self.point_set = np.asarray(point_set)

        self.action_size = self.num_nodes + 1
        self.reset(seed)  # build the tree for get_obs()
        self.observation_shape = self.get_obs().shape

    def _q_prior_var(self) -> float:
        return float(np.var(self.point_set * self.scale_factor))

    def _q_decay_value(self) -> float:
        if not self.q_decay_auto:
            return float(self.q_decay)

        drift_var = self.q_drift**2
        prior_var = max(self._q_prior_var(), 1e-8)
        return drift_var / (drift_var + prior_var)

    def seed(self, seed: int | None):
        self.rng = np.random.RandomState(seed)

    def _one_hot(self, label: int) -> np.ndarray:
        out = np.zeros(self.num_nodes)
        if label >= 0:
            out[label] = 1.0
        return out

    def _canonical_one_hot(self, raw_label: int) -> np.ndarray:
        if raw_label < 0:
            return self._one_hot(-1)
        return self._one_hot(int(self.raw_to_canon[raw_label]))

    def _canonical_values(self, values: np.ndarray) -> np.ndarray:
        out = np.zeros((self.num_nodes,), dtype=values.dtype)
        for canon_node, raw_node in enumerate(self.canon_to_raw):
            if raw_node >= 0:
                out[canon_node] = values[raw_node]
        return out

    def _assign_canonical_node(self, raw_node: int):
        raw_node = int(raw_node)
        if raw_node < 0 or self.raw_to_canon[raw_node] >= 0:
            return

        canon_node = self.next_canon_id
        self.raw_to_canon[raw_node] = canon_node
        self.canon_to_raw[canon_node] = raw_node
        self.next_canon_id += 1

    def _canonicalize_visible(self):
        fixation_parent = self.parent_nodes[self.fixation_node]
        fixation_children = self.child_nodes[self.fixation_node]

        self._assign_canonical_node(self.fixation_node)
        self._assign_canonical_node(fixation_parent)
        self._assign_canonical_node(fixation_children[0])
        self._assign_canonical_node(fixation_children[1])

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        z = self.beta_move * (x - np.max(x))
        p = np.exp(z)
        p = p / np.sum(p)

        if self.eps_move > 0.0:
            p = (1.0 - self.eps_move) * p + self.eps_move * (1.0 / x.shape[0])

        return p

    def _build_tree(self):
        nodes = np.arange(self.num_nodes)
        if self.shuffle_nodes:
            nodes = self.rng.permutation(nodes)

        root = nodes[0]
        child_nodes = -np.ones((self.num_nodes, 2), dtype=int)
        parent_nodes = -np.ones(self.num_nodes, dtype=int)

        leaf_nodes = [root]
        num_edges = (self.num_nodes - 1) // 2

        for i in range(num_edges):
            parent_idx = self.rng.randint(0, len(leaf_nodes))
            node_idx = 1 + 2 * i
            left = nodes[node_idx]
            right = nodes[node_idx + 1]

            parent = leaf_nodes[parent_idx]
            child_nodes[parent] = [left, right]
            parent_nodes[left] = parent
            parent_nodes[right] = parent

            leaf_nodes[parent_idx] = leaf_nodes[-1]
            leaf_nodes.pop()
            leaf_nodes.append(left)
            leaf_nodes.append(right)

        return root, child_nodes, parent_nodes

    def _compute_path_values(self):
        g_values = np.zeros(self.num_nodes)
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            for child in self.child_nodes[node]:
                if child >= 0:
                    g_values[child] = g_values[node] + self.points[node]
                    stack.append(child)
        return g_values

    def _known_mask(self):
        expanded = self.n_visits > 0
        expanded[self.root_node] = True

        known = np.zeros((self.num_nodes,), dtype=bool)
        known[self.root_node] = True
        for node in range(self.num_nodes):
            parent = self.parent_nodes[node]
            if parent >= 0 and expanded[parent]:
                known[node] = True
        return known

    def _bellman_target(self, node: int, active_only: bool = False) -> float:
        children = self.child_nodes[node]
        if children[0] < 0:  # node is terminal
            return self.points[node]

        child_q = self.q_values[children]
        if active_only:
            child_q = np.where(self.activation[children] > 0.0, child_q, 0.0)
        return self.points[node] + np.max(child_q)

    def _update_q(self, node: int):
        target = self._bellman_target(node)
        self.q_values[node] += self.learning_rate * (target - self.q_values[node])

        current = node
        weight = self.lamda_backup
        steps = 0
        while weight > 1e-6 and current != self.root_node and steps < self.backup_steps:
            ancestor = self.parent_nodes[current]

            target = self._bellman_target(ancestor, active_only=self.wm_backup)
            step_size = self.learning_rate * weight
            self.q_values[ancestor] += step_size * (target - self.q_values[ancestor])
            weight *= self.lamda_backup
            current = ancestor
            steps += 1

    def _decay_fixation_recency(self):
        self.fixation_recency *= self._recency_decay_value()

    def _recency_decay_value(self) -> float:
        if self.recency_decay_auto:
            if self.wm_decay == 1.0:
                return 0.5
            return self.wm_decay
        return self.recency_decay

    def _look(self, node: int):
        self.n_visits[node] += 1
        self.fixation_recency[node] = 1.0 if self._recency_decay_value() > 0.0 else 0.0
        self._update_q(node)

    def _update_activation(self, node: int):
        self.activation *= self.wm_decay
        self.activation = np.clip(self.activation, 0.0, 1.0)

        self.activation[node] = 1.0

        parent = self.parent_nodes[node]
        if parent >= 0:
            self.activation[parent] = 1.0

        for child in self.child_nodes[node]:
            if child >= 0:
                self.activation[child] = 1.0

        self.activation[self.root_node] = 1.0

        keep = self.rng.uniform(size=self.num_nodes) < self.activation
        self.activation[~keep] = 0.0
        inactive = self.activation == 0.0
        self.q_values[inactive] *= 1.0 - self._q_decay_value()
        if self.q_drift > 0.0:
            self.q_values[inactive] += self.rng.normal(size=np.sum(inactive)) * self.q_drift
        q_drop_mask = (self.activation == 0.0) & (self.rng.uniform(size=self.num_nodes) < self.q_drop_rate)
        self.q_values[q_drop_mask] = 0.0

    def _expected_move_reward(self) -> float:
        expected = np.zeros(self.num_nodes)

        for _ in range(self.num_nodes):
            next_expected = np.zeros(self.num_nodes)
            for node in range(self.num_nodes):
                children = self.child_nodes[node]
                if children[0] < 0:
                    continue

                q_children = self.q_values[children]
                probs = self._softmax(q_children)
                next_expected[node] = float(np.sum(probs * (self.points[children] + expected[children])))
            expected = next_expected

        return float(expected[self.root_node])

    def _sample_move_path(self):
        node = self.root_node
        cum_reward = 0.0
        chosen_path: list[int] = []

        while self.child_nodes[node, 0] >= 0:
            children = self.child_nodes[node]
            q_children = self.q_values[children]
            probs = self._softmax(q_children)
            idx = self.rng.choice(2, p=probs)
            child = children[idx]

            cum_reward += self.points[child]
            chosen_path.append(child)
            node = child

        return cum_reward, chosen_path

    def get_obs(self) -> np.ndarray:
        fixation_parent = self.parent_nodes[self.fixation_node]
        fixation_children = self.child_nodes[self.fixation_node]
        known_mask = self._known_mask()
        visible_g_values = np.where(known_mask, self.g_values, 0.0)
        unseen_mask = self.n_visits == 0
        open_mask = known_mask & unseen_mask
        is_terminal_seen_raw = (self.child_nodes[:, 0] < 0) & (self.n_visits > 0)
        best_open_value = np.max(self.g_values[open_mask]) if np.any(open_mask) else -10.0
        total_values = self.g_values + self.points
        best_terminal_value = np.max(total_values[is_terminal_seen_raw]) if np.any(is_terminal_seen_raw) else -10.0

        if not self.canonicalize:
            is_terminal_seen = is_terminal_seen_raw.astype(float)
            fixation_child_mask = self._one_hot(fixation_children[0]) + self._one_hot(
                fixation_children[1]
            )
            parts = [
                self._one_hot(self.fixation_node),
                np.asarray([self.points[self.fixation_node]]),
                self._one_hot(fixation_parent),
                fixation_child_mask,
                self._one_hot(self.root_node),
                visible_g_values,
                self.q_values,
                self.n_visits.astype(float),
                is_terminal_seen,
                np.asarray([best_open_value], dtype=float),
                np.asarray([best_terminal_value], dtype=float),
            ]
            if self.use_recency_obs:
                parts.append(self.fixation_recency)
            parts.append(np.asarray([self.time_elapsed]))
            return np.concatenate(parts)

        fixation_child_mask = self._canonical_one_hot(fixation_children[0]) + self._canonical_one_hot(
            fixation_children[1]
        )

        parts = [
            self._canonical_one_hot(self.fixation_node),
            np.asarray([self.points[self.fixation_node]]),
            self._canonical_one_hot(fixation_parent),
            fixation_child_mask,
            self._canonical_one_hot(self.root_node),
            self._canonical_values(visible_g_values),
            self._canonical_values(self.q_values),
            self._canonical_values(self.n_visits).astype(float),
            self._canonical_values(is_terminal_seen_raw.astype(float)),
            np.asarray([best_open_value], dtype=float),
            np.asarray([best_terminal_value], dtype=float),
        ]
        if self.use_recency_obs:
            parts.append(self._canonical_values(self.fixation_recency))
        parts.append(np.asarray([self.time_elapsed]))

        obs = np.concatenate(parts)

        return obs

    def get_action_mask(self) -> np.ndarray:
        mask = np.zeros((self.action_size,), dtype=bool)
        mask[-1] = True  # can always terminate
        if self.time_elapsed == self.t_max - 1:
            return mask  # can ONLY terminate 
        raw_node_mask = self.activation > 0
        raw_node_mask[self.root_node] = True
        if not self.canonicalize:
            mask[: self.num_nodes] = raw_node_mask
            return mask

        mask[: self.num_nodes] = self._canonical_values(raw_node_mask).astype(bool)
        return mask

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.seed(seed)

        self.time_elapsed = 0

        self.root_node, self.child_nodes, self.parent_nodes = self._build_tree()

        point_idx = self.rng.randint(0, self.point_set.shape[0], size=(self.num_nodes,))
        self.points = self.point_set[point_idx].astype(float)
        self.points[self.root_node] = 0.0

        self.q_values = np.zeros(self.num_nodes)
        self.g_values = self._compute_path_values()
        self.n_visits = np.zeros(self.num_nodes, dtype=int)
        self.fixation_recency = np.zeros(self.num_nodes, dtype=float)

        self.activation = np.zeros(self.num_nodes)

        self.fixation_node = self.root_node
        if self.canonicalize:
            self.raw_to_canon = -np.ones(self.num_nodes, dtype=int)
            self.canon_to_raw = -np.ones(self.num_nodes, dtype=int)
            self.next_canon_id = 0
        else:
            self.raw_to_canon = np.arange(self.num_nodes, dtype=int)
            self.canon_to_raw = np.arange(self.num_nodes, dtype=int)
            self.next_canon_id = self.num_nodes

        self.fixation_recency[self.root_node] = 1.0 if self._recency_decay_value() > 0.0 else 0.0
        self._update_activation(self.fixation_node)
        if self.canonicalize:
            self._canonicalize_visible()

        obs = self.get_obs()
        info = {"mask": self.get_action_mask()}
        return obs, info

    def step(self, action: int):
        action = int(action)

        next_time_elapsed = self.time_elapsed + 1

        if action < 0 or action > self.num_nodes:
            raise ValueError(f"Invalid action {action}; expected 0 <= action <= {self.num_nodes}.")

        action_mask = self.get_action_mask()
        if not action_mask[action]:
            valid_actions = np.flatnonzero(action_mask).tolist()
            raise ValueError(f"Invalid action {action}; valid actions are {valid_actions}.")

        raw_action = (
            self.canon_to_raw[action]
            if self.canonicalize and action < self.num_nodes
            else action
        )

        self.time_elapsed = next_time_elapsed
        self._decay_fixation_recency()
        reward = -self.cost

        if action < self.num_nodes:
            self._look(raw_action)
            self.fixation_node = raw_action
            self._update_activation(raw_action)
            if self.canonicalize:
                self._canonicalize_visible()
        elif action == self.num_nodes:
            cum_reward = self._expected_move_reward()
            reward = cum_reward * self.scale_factor

        done = bool(action == self.num_nodes or self.time_elapsed == self.t_max)
        obs = self.get_obs()
        info = {"mask": self.get_action_mask()}

        return obs, reward, done, False, info
