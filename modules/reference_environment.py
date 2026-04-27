import numpy as np


class ReferenceDecisionTreeEnv:
    """Simple NumPy reference implementation of the decision-tree environment."""

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
        mask_fixation: bool = True,
        point_set=None,
        seed: int | None = None,
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
        self.mask_fixation = bool(mask_fixation)

        if point_set is None:
            point_set = [-8, -4, -2, -1, 1, 2, 4, 8]
        self.point_set = np.asarray(point_set, dtype=np.float32)

        self.action_size = self.num_nodes + 1
        self.reset(seed)  # build the tree for get_obs()
        self.observation_shape = self.get_obs().shape


    def seed(self, seed: int | None):
        self.rng = np.random.RandomState(seed)

    def _one_hot(self, label: int) -> np.ndarray:
        out = np.zeros((self.num_nodes,), dtype=np.float32)
        if label >= 0:
            out[label] = 1.0
        return out

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x

        z = self.beta_move * (x - np.max(x))
        p = np.exp(z)
        p = p / np.sum(p)

        if self.eps_move > 0.0:
            p = (1.0 - self.eps_move) * p + self.eps_move * (1.0 / x.shape[0])

        return p

    def _build_tree(self):
        nodes = np.arange(self.num_nodes, dtype=np.int32)
        if self.shuffle_nodes:
            nodes = self.rng.permutation(nodes)

        root = int(nodes[0])
        child_nodes = -np.ones((self.num_nodes, 2), dtype=np.int32)
        parent_nodes = -np.ones((self.num_nodes,), dtype=np.int32)

        leaf_nodes = [root]
        num_edges = (self.num_nodes - 1) // 2

        for i in range(num_edges):
            parent_idx = int(self.rng.randint(0, len(leaf_nodes)))
            node_idx = 1 + 2 * i
            left = int(nodes[node_idx])
            right = int(nodes[node_idx + 1])

            parent = int(leaf_nodes[parent_idx])
            child_nodes[parent] = np.asarray([left, right], dtype=np.int32)
            parent_nodes[left] = parent
            parent_nodes[right] = parent

            leaf_nodes[parent_idx] = leaf_nodes[-1]
            leaf_nodes.pop()
            leaf_nodes.append(left)
            leaf_nodes.append(right)

        return root, child_nodes, parent_nodes

    def _compute_path_values(self):
        g_values = np.zeros((self.num_nodes,), dtype=np.float32)
        for _ in range(self.num_nodes):
            for node in range(self.num_nodes):
                parent = int(self.parent_nodes[node])
                if parent >= 0:
                    g_values[node] = g_values[parent] + self.points[parent]
        return g_values

    def _known_mask(self):
        expanded = self.n_visits > 0
        expanded[self.root_node] = True

        known = np.zeros((self.num_nodes,), dtype=bool)
        known[self.root_node] = True
        for node in range(self.num_nodes):
            parent = int(self.parent_nodes[node])
            if parent >= 0 and expanded[parent]:
                known[node] = True
        return known

    def _bellman_target(self, node: int) -> float:
        children = self.child_nodes[node]
        if int(children[0]) < 0:
            return float(self.points[node])

        child_q = self.q_values[children]
        return float(self.points[node] + np.max(child_q))

    def _update_q(self, node: int):
        target = self._bellman_target(node)
        self.q_values[node] += self.learning_rate * (target - self.q_values[node])

        current = node
        weight = self.lamda_backup
        while weight > 1e-6 and current != self.root_node:
            ancestor = int(self.parent_nodes[current])
            if ancestor < 0:
                break

            target = self._bellman_target(ancestor)
            step_size = self.learning_rate * weight
            self.q_values[ancestor] += step_size * (target - self.q_values[ancestor])
            weight *= self.lamda_backup
            current = ancestor

    def _look(self, node: int):
        self.n_visits[node] += 1
        self._update_q(node)

    def _update_activation(self, node: int):
        self.activation *= self.wm_decay
        self.activation = np.clip(self.activation, 0.0, 1.0)

        self.activation[node] = 1.0

        parent = int(self.parent_nodes[node])
        if parent >= 0:
            self.activation[parent] = 1.0

        for child in self.child_nodes[node]:
            child = int(child)
            if child >= 0:
                self.activation[child] = 1.0

        self.activation[self.root_node] = 1.0

        keep = self.rng.uniform(size=self.num_nodes) < self.activation
        self.activation[~keep] = 0.0
        self.active_mask = keep

    def _move(self):
        node = int(self.root_node)
        cum_reward = 0.0
        chosen_path: list[int] = []

        while int(self.child_nodes[node, 0]) >= 0:
            children = self.child_nodes[node]
            q_children = self.q_values[children]
            probs = self._softmax(q_children)
            idx = int(self.rng.choice(2, p=probs))
            child = int(children[idx])

            cum_reward += float(self.points[child])
            chosen_path.append(child)
            node = child

        return cum_reward, chosen_path

    def get_obs(self) -> np.ndarray:
        fixation_parent = int(self.parent_nodes[self.fixation_node])
        fixation_children = self.child_nodes[self.fixation_node]
        fixation_child_mask = self._one_hot(int(fixation_children[0])) + self._one_hot(
            int(fixation_children[1])
        )
        visible_g_values = np.where(self._known_mask(), self.g_values, 0.0).astype(np.float32)

        obs = np.concatenate(
            [
                self._one_hot(self.fixation_node),
                np.asarray([self.points[self.fixation_node]], dtype=np.float32),
                self._one_hot(fixation_parent),
                fixation_child_mask,
                self._one_hot(self.root_node),
                visible_g_values,
                self.q_values,
                self.n_visits.astype(np.float32),
                np.asarray([self.time_elapsed], dtype=np.float32),
            ]
        )

        return obs

    def get_q_values(self) -> np.ndarray:
        return self.q_values.copy()

    def get_path_values(self) -> np.ndarray:
        return self.g_values.copy()

    def get_num_visits(self) -> np.ndarray:
        return self.n_visits.copy()

    def get_action_mask(self) -> np.ndarray:
        mask = np.zeros((self.action_size,), dtype=bool)
        mask[-1] = True  # can always terminate
        if self.time_elapsed == self.t_max - 1:
            return mask  # can ONLY terminate 
        mask[: self.num_nodes] = self.active_mask
        mask[self.root_node] = True
        return mask

    def reset(self, seed: int | None = None, options=None):
        del options
        if seed is not None:
            self.seed(seed)

        self.time_elapsed = 0
        self.chosen_path = []

        self.root_node, self.child_nodes, self.parent_nodes = self._build_tree()

        point_idx = self.rng.randint(0, self.point_set.shape[0], size=(self.num_nodes,))
        self.points = self.point_set[point_idx].astype(np.float32)
        self.points[self.root_node] = 0.0

        self.q_values = np.zeros((self.num_nodes,), dtype=np.float32)
        self.g_values = self._compute_path_values()
        self.n_visits = np.zeros((self.num_nodes,), dtype=np.int32)

        self.activation = np.zeros((self.num_nodes,), dtype=np.float32)
        self.active_mask = np.zeros((self.num_nodes,), dtype=bool)

        self.fixation_node = int(self.root_node)
        self._update_activation(self.fixation_node)

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

        self.time_elapsed = next_time_elapsed
        reward = -self.cost

        if action < self.num_nodes:
            self._look(action)
            self.fixation_node = action
            self._update_activation(action)
            self.chosen_path = []
        elif action == self.num_nodes:
            cum_reward, chosen_path = self._move()
            reward = cum_reward * self.scale_factor
            self.chosen_path = chosen_path
        else:
            self.chosen_path = []

        done = bool(action == self.num_nodes or self.time_elapsed == self.t_max)
        obs = self.get_obs()
        info = {"mask": self.get_action_mask()}

        return obs, reward, done, False, info
