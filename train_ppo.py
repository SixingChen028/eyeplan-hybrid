import os
import json
import pickle
import time
import jax
import numpy as np

from modules.argument import ArgParser
from modules.run_dirs import (
    create_timestamped_run_dir,
    resolve_timestamped_run_dir,
    write_run_metadata,
)
from modules.a2c import save_jax_params, save_jax_tree, load_jax_tree
from modules.environment import JaxDecisionTreeEnv
from modules.ppo import JaxBatchMaskPPO
from modules.simulation import JaxSimulator


CHECKPOINT_STATE_NAME = "train_state_latest.p"
CHECKPOINT_META_NAME = "train_state_latest.json"


def _has_resume_key(jobid: str) -> bool:
    return str(jobid).strip() != ""


def _save_rolling_checkpoint(state, checkpoint_state_path: str, checkpoint_meta_path: str, next_update: int):
    temp_state_path = f"{checkpoint_state_path}.tmp"
    temp_meta_path = f"{checkpoint_meta_path}.tmp"

    save_jax_tree(state, temp_state_path)
    with open(temp_meta_path, "w") as file:
        json.dump({"next_update": int(next_update)}, file, indent=2, sort_keys=True)

    os.replace(temp_state_path, checkpoint_state_path)
    os.replace(temp_meta_path, checkpoint_meta_path)


def _load_resume_state(checkpoint_state_path: str, checkpoint_meta_path: str):
    if not os.path.exists(checkpoint_state_path) or not os.path.exists(checkpoint_meta_path):
        return None, 0

    with open(checkpoint_meta_path, "r") as file:
        checkpoint_meta = json.load(file)
    next_update = int(checkpoint_meta["next_update"])
    state = load_jax_tree(checkpoint_state_path)
    return state, next_update


if __name__ == '__main__':
    parser = ArgParser()
    args = parser.args
    num_updates = int(args.num_episodes / args.batch_size)

    state = None
    start_update = 0
    exp_path = None

    if _has_resume_key(args.jobid):
        try:
            candidate_path = resolve_timestamped_run_dir(
                path=args.path,
                experiment=args.experiment,
                jobid=args.jobid,
            )
            candidate_checkpoint_dir = os.path.join(candidate_path, "checkpoints")
            candidate_checkpoint_state = os.path.join(candidate_checkpoint_dir, CHECKPOINT_STATE_NAME)
            candidate_checkpoint_meta = os.path.join(candidate_checkpoint_dir, CHECKPOINT_META_NAME)
            state, start_update = _load_resume_state(candidate_checkpoint_state, candidate_checkpoint_meta)
            if not (0 < start_update < num_updates):
                state = None
                start_update = 0
            else:
                exp_path = candidate_path
        except (FileNotFoundError, KeyError, ValueError, OSError, json.JSONDecodeError, pickle.UnpicklingError):
            state = None
            start_update = 0

    if exp_path is None:
        exp_path = create_timestamped_run_dir(
            path=args.path,
            experiment=args.experiment,
            jobid=args.jobid,
        )
        metadata_path = write_run_metadata(run_dir=exp_path, args=args, cwd=os.getcwd())
    else:
        metadata_path = os.path.join(exp_path, "metadata.json")

    print(f"run_dir={exp_path}")
    print(f"run_metadata={metadata_path}")
    if start_update > 0:
        print(f"resuming_from_update={start_update}")

    checkpoint_dir = os.path.join(exp_path, "checkpoints")
    if args.checkpoint_frequency >= 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_state_path = os.path.join(checkpoint_dir, CHECKPOINT_STATE_NAME)
    checkpoint_meta_path = os.path.join(checkpoint_dir, CHECKPOINT_META_NAME)

    env = JaxDecisionTreeEnv(
        num_nodes=args.num_nodes,
        beta_move=args.beta_move,
        eps_move=args.eps_move,
        learning_rate=args.learning_rate,
        lamda_backup=args.lamda_backup,
        wm_decay=args.wm_decay,
        q_drop_rate=args.q_drop_rate,
        t_max=args.t_max,
        cost=args.cost,
        scale_factor=args.scale_factor,
        shuffle_nodes=args.shuffle_nodes,
        canonicalize=args.canonicalize,
        recency_decay=args.recency_decay,
    )

    trainer = JaxBatchMaskPPO(
        env=env,
        feature_size=env.observation_shape[0],
        action_size=env.action_size,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        lr=args.lr,
        max_grad_norm=args.max_grad_norm,
        gamma=args.gamma,
        lamda=args.lamda,
        beta_v=args.beta_v,
        beta_e=args.beta_e,
        clip_eps=args.ppo_clip_eps,
        ppo_epochs=args.ppo_epochs,
        normalize_advantages=args.ppo_normalize_advantages,
        network_type=args.network_type,
    )

    if state is None:
        state = trainer.init_state(seed=args.seed)

    entropy_schedule = np.linspace(
        args.beta_e_init,
        args.beta_e_final,
        num_updates,
        dtype=np.float32,
    )

    data = {
        "loss": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy_loss": [],
        "clip_fraction": [],
        "approx_kl": [],
        "episode_length": [],
        "episode_reward": [],
        "step_time_s": [],
        "cumulative_time_s": [],
    }

    print(
        "run_config "
        f"batch_size={args.batch_size} "
        f"num_episodes={args.num_episodes} "
        f"eval_episodes={args.eval_episodes} "
        f"num_updates={num_updates} "
        f"t_max={args.t_max} "
        f"ppo_epochs={args.ppo_epochs} "
        f"ppo_clip_eps={args.ppo_clip_eps} "
        f"print_frequency={args.print_frequency} "
        f"checkpoint_frequency={args.checkpoint_frequency} "
        f"log_full_metrics={args.log_full_metrics}"
    )

    col_sep = "   "
    run_total_updates = max(num_updates - start_update, 1)

    def _fmt_num(value: float, width: int = 8, decimals: int = 3) -> str:
        return f"{value: {width}.{decimals}f}"

    def _hhmmss(total_seconds: float) -> str:
        total_rounded = max(int(round(total_seconds)), 0)
        hours, remainder = divmod(total_rounded, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _eta_hhmmss(elapsed: float, completed_updates: int, total_updates: int) -> str:
        completed = max(completed_updates, 1)
        remaining = max(total_updates - completed_updates, 0)
        eta_seconds = elapsed * (remaining / completed)
        return _hhmmss(eta_seconds)

    if args.print_frequency > 0:
        header = col_sep.join(
            [
                f"{'update':>8}",
                f"{'ep_num':>10}",
                f"{'ep_rew':>8}",
                f"{'ep_len':>8}",
                f"{'loss':>8}",
                f"{'policy':>8}",
                f"{'value':>8}",
                f"{'entropy':>8}",
                f"{'clipfrac':>8}",
                f"{'approx_kl':>8}",
                f"{'elapsed':>8}",
                f"{'ETA':>8}",
            ]
        )
        print(header)
        print("-" * len(header))
    if not args.log_full_metrics:
        print("metric_mode=chunk_mean_per_update (lower host sync overhead)")

    start_time = time.time()
    eta_skip_elapsed = None
    eta_skip_updates = None
    window_start_idx = len(data["loss"])

    def _next_boundary(processed_updates: int, frequency: int) -> int:
        return ((processed_updates // frequency) + 1) * frequency

    def _next_checkpoint_boundary(processed_updates: int) -> int:
        if args.checkpoint_frequency < 0:
            return num_updates
        if args.checkpoint_frequency > 0:
            return _next_boundary(processed_updates, args.checkpoint_frequency)
        if args.print_frequency > 0:
            return _next_boundary(processed_updates, args.print_frequency)
        return num_updates

    index = start_update
    while index < num_updates:
        chunk_start = index
        chunk_end = num_updates
        if args.print_frequency > 0:
            chunk_end = min(chunk_end, _next_boundary(chunk_start, args.print_frequency))
        if args.checkpoint_frequency >= 0:
            chunk_end = min(chunk_end, _next_checkpoint_boundary(chunk_start))
        chunk_updates = chunk_end - chunk_start
        chunk_entropy = entropy_schedule[chunk_start:chunk_end]

        chunk_start_wall = time.time()
        if args.log_full_metrics:
            state, chunk_metrics = trainer.train_compiled(state, chunk_entropy)
            chunk_metrics = jax.tree_util.tree_map(
                lambda x: np.asarray(jax.device_get(x)),
                chunk_metrics,
            )
        else:
            state, chunk_metrics_mean = trainer.train_compiled_mean_metrics(state, chunk_entropy)
            chunk_metrics_mean = jax.tree_util.tree_map(
                lambda x: float(jax.device_get(x)),
                chunk_metrics_mean,
            )
            chunk_metrics = jax.tree_util.tree_map(
                lambda x: np.full((chunk_updates,), x, dtype=np.float32),
                chunk_metrics_mean,
            )
        chunk_end_wall = time.time()
        is_first_chunk = chunk_start == start_update

        chunk_elapsed = chunk_end_wall - chunk_start_wall
        avg_step_time = chunk_elapsed / chunk_updates
        chunk_cumulative_start = chunk_start_wall - start_time
        cumulative_values = (
            chunk_cumulative_start
            + (np.arange(1, chunk_updates + 1, dtype=np.float64) / chunk_updates) * chunk_elapsed
        )

        chunk_data_start = len(data["loss"])
        data["loss"].extend(chunk_metrics.loss.tolist())
        data["policy_loss"].extend(chunk_metrics.policy_loss.tolist())
        data["value_loss"].extend(chunk_metrics.value_loss.tolist())
        data["entropy_loss"].extend(chunk_metrics.entropy_loss.tolist())
        data["clip_fraction"].extend(chunk_metrics.clip_fraction.tolist())
        data["approx_kl"].extend(chunk_metrics.approx_kl.tolist())
        data["episode_length"].extend(chunk_metrics.episode_length.tolist())
        data["episode_reward"].extend(chunk_metrics.episode_reward.tolist())
        data["step_time_s"].extend([avg_step_time] * chunk_updates)
        data["cumulative_time_s"].extend(cumulative_values.tolist())

        for chunk_index in range(chunk_updates):
            update_index = chunk_start + chunk_index
            progress = (chunk_index + 1) / chunk_updates
            event_time = chunk_start_wall + progress * chunk_elapsed

            should_log = (
                args.print_frequency > 0 and (
                    update_index == 0
                    or (update_index + 1) % args.print_frequency == 0
                    or (update_index + 1) == num_updates
                )
            )
            if should_log:
                elapsed = event_time - start_time
                completed_updates = update_index + 1 - start_update
                if is_first_chunk or eta_skip_elapsed is None or eta_skip_updates is None:
                    eta_display = ""
                else:
                    eta_elapsed = max(elapsed - eta_skip_elapsed, 0.0)
                    eta_completed = max(completed_updates - eta_skip_updates, 1)
                    eta_total = max(run_total_updates - eta_skip_updates, 1)
                    eta_display = _eta_hhmmss(eta_elapsed, eta_completed, eta_total)
                elapsed_display = _hhmmss(elapsed)

                data_index = chunk_data_start + chunk_index
                window = slice(window_start_idx, data_index + 1)
                avg_episode_reward = float(np.mean(data["episode_reward"][window]))
                avg_episode_length = float(np.mean(data["episode_length"][window]))
                avg_loss = float(np.mean(data["loss"][window]))
                avg_policy_loss = float(np.mean(data["policy_loss"][window]))
                avg_value_loss = float(np.mean(data["value_loss"][window]))
                avg_entropy_loss = float(np.mean(data["entropy_loss"][window]))
                avg_clip_fraction = float(np.mean(data["clip_fraction"][window]))
                avg_approx_kl = float(np.mean(data["approx_kl"][window]))

                print(
                    col_sep.join(
                        [
                            f"{update_index + 1:>8d}",
                            f"{(update_index + 1) * args.batch_size:>10d}",
                            _fmt_num(avg_episode_reward),
                            _fmt_num(avg_episode_length),
                            _fmt_num(avg_loss),
                            _fmt_num(avg_policy_loss),
                            _fmt_num(avg_value_loss),
                            _fmt_num(avg_entropy_loss),
                            _fmt_num(avg_clip_fraction),
                            _fmt_num(avg_approx_kl),
                            f"{elapsed_display:>8}",
                            f"{eta_display:>8}",
                        ]
                    )
                )
                window_start_idx = data_index + 1

        if is_first_chunk and eta_skip_elapsed is None:
            eta_skip_elapsed = chunk_end_wall - start_time
            eta_skip_updates = chunk_end - start_update

        should_checkpoint = False
        if args.checkpoint_frequency >= 0:
            if args.checkpoint_frequency == 0:
                should_checkpoint = True
            else:
                should_checkpoint = (
                    chunk_end % args.checkpoint_frequency == 0
                    or chunk_end == num_updates
                )
        if should_checkpoint:
            _save_rolling_checkpoint(
                state=state,
                checkpoint_state_path=checkpoint_state_path,
                checkpoint_meta_path=checkpoint_meta_path,
                next_update=chunk_end,
            )

        index = chunk_end

    print(
        "run_summary "
        f"updates={num_updates} "
        f"elapsed_seconds={time.time() - start_time:.3f} "
        f"mean_step_seconds={np.mean(data['step_time_s']):.6f}"
    )

    eval_start = time.time()
    simulator = JaxSimulator(env)
    eval_stats = simulator.evaluate_policy(
        params=state.params,
        seed=args.seed,
        num_trials=args.eval_episodes,
        greedy=True,
    )
    print(
        "eval_summary "
        f"episodes={eval_stats['num_trials']} "
        f"reward_mean={eval_stats['reward_mean']:.6f} "
        f"reward_sd={eval_stats['reward_sd']:.6f} "
        f"reward_no_cost_mean={eval_stats['reward_no_cost_mean']:.6f} "
        f"reward_no_cost_sd={eval_stats['reward_no_cost_sd']:.6f} "
        f"n_steps_mean={eval_stats['n_steps_mean']:.3f} "
        f"n_steps_sd={eval_stats['n_steps_sd']:.3f} "
        f"elapsed_seconds={time.time() - eval_start:.3f}"
    )

    save_jax_params(state.params, os.path.join(exp_path, 'net_jax_ppo.p'))
    with open(os.path.join(exp_path, 'data_training_jax_ppo.p'), 'wb') as file:
        pickle.dump(data, file)
