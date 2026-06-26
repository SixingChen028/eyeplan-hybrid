from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import jax
import numpy as np

from .environment import DecisionTreeEnv, DecisionTreeParams
from .simulation import append_simulation_trial, empty_simulation_data


DEFAULT_MCTS_C_GRID = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
DEFAULT_MCTS_EVALS_PER_POINT = 10_000
DEFAULT_MCTS_ROLLOUTS = 32


@dataclass
class MCTSEvaluation:
    c_raw: float
    c_scaled: float
    mean_episode_reward: float
    mean_no_cost_reward_scaled: float
    mean_no_cost_reward_raw: float
    mean_episode_length: float
    num_trials: int


class MCTSSimulator:
    def __init__(
        self,
        env: DecisionTreeEnv,
        env_params: DecisionTreeParams,
        *,
        num_rollouts: int = DEFAULT_MCTS_ROLLOUTS,
    ):
        if num_rollouts <= 0:
            raise ValueError("num_rollouts must be positive")

        self.env = env
        self.env_params = env_params
        self.num_rollouts = int(num_rollouts)
        self._reset_jit = jax.jit(lambda key: self.env.reset(key, self.env_params))
        self._step_jit = jax.jit(lambda state, action: self.env.step(state, action, self.env_params))

    def _path_return(self, points: np.ndarray, path: list[int]) -> float:
        path_len = max(len(path) - 1, 0)
        movement_cost = float(self.env_params.move_cost_scale) * float(self.env_params.cost) * path_len
        fixation_cost = float(self.env_params.cost) * path_len
        return float(np.sum(points[path]) * self.env.scale_factor - fixation_cost - movement_cost)

    def _select_simulation_child(
        self,
        child_nodes: np.ndarray,
        visits: np.ndarray,
        value_sums: np.ndarray,
        node: int,
        c_scaled: float,
        rng: np.random.Generator,
    ) -> int:
        children = [int(child) for child in child_nodes[node] if child >= 0]
        unvisited = [child for child in children if visits[child] == 0]
        if unvisited:
            return int(rng.choice(unvisited))

        parent_visits = max(int(visits[node]), 1)
        best_child = children[0]
        best_score = -np.inf
        for child in children:
            mean_value = value_sums[child] / visits[child]
            explore = c_scaled * np.sqrt(np.log(parent_visits) / visits[child])
            score = mean_value + explore
            if score > best_score:
                best_child = child
                best_score = score
        return int(best_child)

    def _select_final_child(
        self,
        child_nodes: np.ndarray,
        visits: np.ndarray,
        value_sums: np.ndarray,
        node: int,
    ) -> int:
        children = [int(child) for child in child_nodes[node] if child >= 0]

        def child_key(child: int) -> tuple[int, float]:
            mean_value = value_sums[child] / max(visits[child], 1)
            return int(visits[child]), float(mean_value)

        return int(max(children, key=child_key))

    def _plan_path(self, state: Any, c_raw: float, rng: np.random.Generator) -> list[int]:
        c_scaled = float(c_raw) * self.env.scale_factor
        child_nodes = np.asarray(state.child_nodes)
        points = np.asarray(state.points)
        root = int(state.root_node)
        visits = np.zeros((self.env.num_nodes,), dtype=np.int32)
        value_sums = np.zeros((self.env.num_nodes,), dtype=np.float64)

        for _ in range(self.num_rollouts):
            node = root
            path = [node]
            while child_nodes[node, 0] >= 0:
                node = self._select_simulation_child(child_nodes, visits, value_sums, node, c_scaled, rng)
                path.append(node)

            total_return = self._path_return(points, path)
            for path_node in path:
                visits[path_node] += 1
                value_sums[path_node] += total_return

        selected = []
        node = root
        while child_nodes[node, 0] >= 0:
            node = self._select_final_child(child_nodes, visits, value_sums, node)
            selected.append(node)
        return selected

    def _run_trial(self, key: jax.Array, *, c_raw: float, rng: np.random.Generator):
        state, _, info = self._reset_jit(key)
        action_mask = np.asarray(info["mask"])
        planned_path = self._plan_path(state, c_raw, rng)

        action_seq: list[int] = []
        choice_seq: list[int] = []
        episode_reward = 0.0
        terminal_reward = 0.0
        moved = False
        done = False

        for action in planned_path:
            if len(action_seq) >= self.env.t_max - 1:
                break
            if not action_mask[int(action)]:
                raise RuntimeError(f"Planned MCTS fixation {action} is not legal from the current WM state.")
            state, _, reward, done, info = self._step_jit(state, int(action))
            action_mask = np.asarray(info["mask"])
            action_seq.append(int(action))
            episode_reward += float(reward)
            if done:
                break

        if not done:
            action = self.env.num_nodes
            state, _, reward, done, info = self._step_jit(state, int(action))
            action_seq.append(int(action))
            episode_reward += float(reward)
            terminal_reward = float(reward)
            moved = True
            choice_path = np.asarray(info["choice_path"], dtype=np.int32)
            choice_seq = choice_path[choice_path >= 0].tolist()

        scaled_no_cost, raw_no_cost = (
            (terminal_reward, terminal_reward / self.env.scale_factor)
            if moved
            else (0.0, 0.0)
        )
        return state, action_seq, choice_seq, episode_reward, scaled_no_cost, raw_no_cost

    def evaluate(
        self,
        *,
        c_raw: float,
        seed: int,
        num_trials: int,
    ) -> MCTSEvaluation:
        if num_trials <= 0:
            raise ValueError("num_trials must be positive")

        key = jax.random.PRNGKey(seed)
        rng = np.random.default_rng(seed)
        rewards = []
        no_cost_scaled = []
        no_cost_raw = []
        lengths = []

        for trial_idx in range(num_trials):
            trial_key = jax.random.fold_in(key, trial_idx)
            _, action_seq, _, reward, scaled, raw = self._run_trial(
                trial_key,
                c_raw=c_raw,
                rng=rng,
            )
            rewards.append(reward)
            no_cost_scaled.append(scaled)
            no_cost_raw.append(raw)
            lengths.append(len(action_seq))

        return MCTSEvaluation(
            c_raw=float(c_raw),
            c_scaled=float(c_raw) * self.env.scale_factor,
            mean_episode_reward=float(np.mean(rewards)),
            mean_no_cost_reward_scaled=float(np.mean(no_cost_scaled)),
            mean_no_cost_reward_raw=float(np.mean(no_cost_raw)),
            mean_episode_length=float(np.mean(lengths)),
            num_trials=int(num_trials),
        )

    def simulate(
        self,
        *,
        c_raw: float,
        seed: int,
        num_trials: int,
        skip_timeout_trials: bool = True,
    ) -> Dict[str, List[Any]]:
        if num_trials <= 0:
            raise ValueError("num_trials must be positive")

        key = jax.random.PRNGKey(seed)
        rng = np.random.default_rng(seed)
        data = empty_simulation_data(detailed=False)

        for trial_idx in range(num_trials):
            trial_key = jax.random.fold_in(key, trial_idx)
            state, action_seq, choice_seq, _, _, _ = self._run_trial(
                trial_key,
                c_raw=c_raw,
                rng=rng,
            )

            append_simulation_trial(
                data,
                child_nodes=np.asarray(state.child_nodes),
                root_node=int(state.root_node),
                points=np.asarray(state.points),
                action_seq=action_seq,
                choice_seq=choice_seq,
                num_nodes=self.env.num_nodes,
                t_max=self.env.t_max,
                skip_timeout_trials=skip_timeout_trials,
            )

        return data
