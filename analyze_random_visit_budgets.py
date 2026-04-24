import argparse
import os
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modules.environment import JaxDecisionTreeEnv


def _optimal_raw(child_nodes: np.ndarray, points: np.ndarray, root: int) -> float:
    def dfs(node: int) -> float:
        left = int(child_nodes[node, 0])
        if left < 0:
            return float(points[node])
        right = int(child_nodes[node, 1])
        return float(points[node]) + max(dfs(left), dfs(right))

    return dfs(root)


def _make_budget_grid(max_visits: int, num_points: int) -> np.ndarray:
    grid = np.geomspace(1, max_visits, num=num_points)
    grid = np.unique(np.round(grid).astype(int))
    grid = np.clip(grid, 1, max_visits)
    if grid[-1] != max_visits:
        grid = np.append(grid, max_visits)
    return np.unique(grid)


def _select_not_done(done: jax.Array, new: jax.Array, old: jax.Array) -> jax.Array:
    selector = done
    while selector.ndim < new.ndim:
        selector = selector[..., None]
    return jnp.where(selector, old, new)


def _build_budget_runner(env: JaxDecisionTreeEnv, batch_size: int):
    reset_vmapped = jax.vmap(env.reset)
    step_vmapped = jax.vmap(env.step)

    @partial(jax.jit, static_argnames=("budget",))
    def run_budget(reset_keys: jax.Array, action_key: jax.Array, budget: int):
        state, _, info = reset_vmapped(reset_keys)
        mask = info["mask"]

        done = jnp.zeros((batch_size,), dtype=jnp.bool_)
        rewards = jnp.zeros((batch_size,), dtype=jnp.float32)
        steps = jnp.zeros((batch_size,), dtype=jnp.int32)

        def body_fn(carry, _):
            state, mask, done, rewards, steps, key = carry

            key, sample_key = jax.random.split(key)
            sample_keys = jax.random.split(sample_key, batch_size)

            legal = mask[:, : env.num_nodes]
            logits = jnp.where(legal, 0.0, jnp.finfo(jnp.float32).min)
            actions = jax.vmap(lambda k, l: jax.random.categorical(k, l))(sample_keys, logits)
            has_legal = jnp.any(legal, axis=1)
            actions = jnp.where(has_legal, actions, jnp.zeros_like(actions))

            next_state, _, next_reward, next_done, _, next_info = step_vmapped(state, actions)

            active = (~done).astype(jnp.float32)

            state = jax.tree_util.tree_map(
                lambda new, old: _select_not_done(done, new, old),
                next_state,
                state,
            )
            mask = _select_not_done(done, next_info["mask"], mask)
            done = jnp.logical_or(done, next_done)

            rewards = rewards + next_reward * active
            steps = steps + active.astype(jnp.int32)

            return (state, mask, done, rewards, steps, key), None

        (state, mask, done, rewards, steps, action_key), _ = jax.lax.scan(
            body_fn,
            (state, mask, done, rewards, steps, action_key),
            xs=None,
            length=budget,
        )

        move_actions = jnp.full((batch_size,), env.num_nodes, dtype=jnp.int32)
        next_state, _, next_reward, next_done, _, next_info = step_vmapped(state, move_actions)

        active = (~done).astype(jnp.float32)

        state = jax.tree_util.tree_map(
            lambda new, old: _select_not_done(done, new, old),
            next_state,
            state,
        )
        mask = _select_not_done(done, next_info["mask"], mask)
        done = jnp.logical_or(done, next_done)

        rewards = rewards + next_reward * active
        steps = steps + active.astype(jnp.int32)

        path_safe = jnp.maximum(state.chosen_path, 0)
        gathered = jnp.take_along_axis(state.points, path_safe, axis=1)
        valid = jnp.arange(env.num_nodes)[None, :] < state.chosen_path_len[:, None]
        no_cost_raw = jnp.sum(gathered * valid.astype(gathered.dtype), axis=1)
        no_cost_scaled = no_cost_raw * env.scale_factor

        return rewards, no_cost_scaled, no_cost_raw, steps

    return run_budget


def evaluate_random_visit_curve(
    env: JaxDecisionTreeEnv,
    reset_keys: jax.Array,
    budget_grid: np.ndarray,
    action_seed: int,
    label: str,
) -> pd.DataFrame:
    batch_size = reset_keys.shape[0]
    run_budget = _build_budget_runner(env, batch_size=batch_size)

    reset_fn = jax.jit(env.reset)
    optimal_raw_rewards = []
    for key in np.asarray(reset_keys):
        state, _, _ = reset_fn(key)
        optimal_raw_rewards.append(
            _optimal_raw(
                child_nodes=np.asarray(state.child_nodes),
                points=np.asarray(state.points),
                root=int(state.root_node),
            )
        )

    optimal_raw_mean = float(np.mean(optimal_raw_rewards))
    optimal_scaled_mean = optimal_raw_mean * env.scale_factor

    rows = []
    for budget in budget_grid:
        action_key = jax.random.PRNGKey(action_seed + int(budget))
        rewards, no_cost_scaled, no_cost_raw, steps = run_budget(
            reset_keys,
            action_key,
            int(budget),
        )

        rows.append(
            {
                "curve": label,
                "visit_budget": int(budget),
                "mean_episode_reward": float(jnp.mean(rewards)),
                "mean_no_cost_reward_scaled": float(jnp.mean(no_cost_scaled)),
                "mean_no_cost_reward_raw": float(jnp.mean(no_cost_raw)),
                "mean_episode_length": float(jnp.mean(steps)),
                "optimal_no_cost_scaled_mean": optimal_scaled_mean,
                "optimal_no_cost_raw_mean": optimal_raw_mean,
            }
        )

    return pd.DataFrame(rows)


def plot_curves(df: pd.DataFrame, output_path: str):
    plt.figure(figsize=(13, 5))

    ax1 = plt.subplot(1, 2, 1)
    for curve, group in df.groupby("curve"):
        ax1.plot(
            group["visit_budget"],
            group["mean_no_cost_reward_scaled"],
            marker="o",
            linewidth=2,
            label=f"{curve} no-cost",
        )

    optimal_scaled = float(df["optimal_no_cost_scaled_mean"].iloc[0])
    ax1.axhline(
        optimal_scaled,
        color="black",
        linestyle="--",
        linewidth=2,
        label="optimal no-cost expected",
    )

    ax1.set_xscale("log")
    ax1.set_xlabel("Random fixation budget")
    ax1.set_ylabel("Scaled no-cost reward")
    ax1.set_title("No-Cost Reward vs Visit Budget")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    ax2 = plt.subplot(1, 2, 2)
    for curve, group in df.groupby("curve"):
        ax2.plot(
            group["visit_budget"],
            group["mean_episode_reward"],
            marker="o",
            linewidth=2,
            label=f"{curve} with cost",
        )

    ax2.set_xscale("log")
    ax2.set_xlabel("Random fixation budget")
    ax2.set_ylabel("Episode reward (with cost)")
    ax2.set_title("Reward With Cost vs Visit Budget")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, default=os.path.join("results", "random_visit_budget_7node"))
    parser.add_argument("--num_nodes", type=int, default=7)
    parser.add_argument("--eval_trials", type=int, default=2000)
    parser.add_argument("--max_visits", type=int, default=200)
    parser.add_argument("--grid_points", type=int, default=20)
    parser.add_argument("--seed", type=int, default=33)

    parser.add_argument("--cost", type=float, default=0.01)
    parser.add_argument("--scale_factor", type=float, default=1 / 8)
    parser.add_argument("--t_max", type=int, default=250)
    parser.add_argument("--wm_decay", type=float, default=0.8)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    budget_grid = _make_budget_grid(args.max_visits, args.grid_points)

    key = jax.random.PRNGKey(args.seed)
    reset_keys = jax.random.split(key, args.eval_trials)

    env_default = JaxDecisionTreeEnv(
        num_nodes=args.num_nodes,
        beta_move=4.0,
        eps_move=0.02,
        learning_rate=0.2,
        wm_decay=args.wm_decay,
        t_max=max(args.t_max, args.max_visits + 5),
        cost=args.cost,
        scale_factor=args.scale_factor,
        shuffle_nodes=True,
    )

    env_deterministic = JaxDecisionTreeEnv(
        num_nodes=args.num_nodes,
        beta_move=100.0,
        eps_move=0.0,
        learning_rate=1.0,
        wm_decay=args.wm_decay,
        t_max=max(args.t_max, args.max_visits + 5),
        cost=args.cost,
        scale_factor=args.scale_factor,
        shuffle_nodes=True,
    )

    df_default = evaluate_random_visit_curve(
        env=env_default,
        reset_keys=reset_keys,
        budget_grid=budget_grid,
        action_seed=args.seed + 1,
        label="default_move",
    )

    df_deterministic = evaluate_random_visit_curve(
        env=env_deterministic,
        reset_keys=reset_keys,
        budget_grid=budget_grid,
        action_seed=args.seed + 2,
        label="deterministic_move",
    )

    df = pd.concat([df_default, df_deterministic], ignore_index=True)
    df.to_csv(os.path.join(args.output_dir, "random_visit_budget_curves.csv"), index=False)

    summary = (
        df.sort_values("visit_budget")
        .groupby("curve")
        .tail(1)
        .loc[
            :,
            [
                "curve",
                "visit_budget",
                "mean_episode_reward",
                "mean_no_cost_reward_scaled",
                "mean_no_cost_reward_raw",
                "optimal_no_cost_scaled_mean",
                "optimal_no_cost_raw_mean",
            ],
        ]
        .reset_index(drop=True)
    )
    summary.to_csv(os.path.join(args.output_dir, "random_visit_budget_summary.csv"), index=False)

    plot_curves(df, os.path.join(args.output_dir, "random_visit_budget_curves.png"))

    print("Saved outputs to:", args.output_dir)
    print(summary)


if __name__ == "__main__":
    main()
