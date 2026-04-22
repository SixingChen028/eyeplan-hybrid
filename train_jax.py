import os
import pickle
import time
import jax
import numpy as np

from modules.argument import ArgParser
from modules.jax_run_dirs import create_timestamped_run_dir, write_run_metadata
from modules.jax_environment import JaxDecisionTreeEnv
from modules.jax_a2c import JaxBatchMaskA2C, save_jax_params
from modules.jax_simulation import JaxSimulator


EVAL_EPISODES = 10_000


if __name__ == '__main__':
    parser = ArgParser()
    args = parser.args
    exp_path = create_timestamped_run_dir(path=args.path, jobid=args.jobid)
    metadata_path = write_run_metadata(run_dir=exp_path, args=args, cwd=os.getcwd())
    print(f"run_dir={exp_path}")
    print(f"run_metadata={metadata_path}")
    checkpoint_dir = os.path.join(exp_path, "checkpoints")
    if args.checkpoint_frequency > 0:
        os.makedirs(checkpoint_dir, exist_ok=True)

    env = JaxDecisionTreeEnv(
        num_nodes=args.num_nodes,
        beta_move=args.beta_move,
        eps_move=args.eps_move,
        learning_rate=args.learning_rate,
        lamda_backup=args.lamda_backup,
        wm_decay=args.wm_decay,
        t_max=args.t_max,
        cost=args.cost,
        scale_factor=args.scale_factor,
        shuffle_nodes=args.shuffle_nodes,
    )

    trainer = JaxBatchMaskA2C(
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
    )

    state = trainer.init_state(seed=args.seed)

    num_updates = int(args.num_episodes / args.batch_size)
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
        "episode_length": [],
        "episode_reward": [],
        "step_time_s": [],
        "cumulative_time_s": [],
    }

    print(
        "run_config "
        f"batch_size={args.batch_size} "
        f"num_episodes={args.num_episodes} "
        f"num_updates={num_updates} "
        f"t_max={args.t_max} "
        f"print_frequency={args.print_frequency} "
        f"checkpoint_frequency={args.checkpoint_frequency}"
    )

    if args.print_frequency > 0:
        print(
            f"{'update':>8}  "
            f"{'ep_num':>10}  "
            f"{'ep_rew':>12}  "
            f"{'ep_len':>12}  "
            f"{'loss':>12}  "
            f"{'policy':>12}  "
            f"{'value':>12}  "
            f"{'entropy':>12}  "
            f"{'beta_e':>8}  "
            f"{'step_s':>10}  "
            f"{'since_log':>10}"
        )
        print("-" * 142)

    start_time = time.time()
    last_log_time = time.time()
    window_start_idx = 0

    def _next_boundary(processed_updates: int, frequency: int) -> int:
        return ((processed_updates // frequency) + 1) * frequency

    index = 0
    while index < num_updates:
        chunk_start = index
        chunk_end = num_updates
        if args.print_frequency > 0:
            chunk_end = min(chunk_end, _next_boundary(chunk_start, args.print_frequency))
        if args.checkpoint_frequency > 0:
            chunk_end = min(chunk_end, _next_boundary(chunk_start, args.checkpoint_frequency))
        chunk_updates = chunk_end - chunk_start
        chunk_entropy = entropy_schedule[chunk_start:chunk_end]

        chunk_start_wall = time.time()
        state, chunk_metrics = trainer.train_compiled(state, chunk_entropy)
        chunk_metrics = jax.tree_util.tree_map(
            lambda x: np.asarray(jax.device_get(x)),
            chunk_metrics,
        )
        chunk_end_wall = time.time()

        chunk_elapsed = chunk_end_wall - chunk_start_wall
        avg_step_time = chunk_elapsed / chunk_updates
        chunk_cumulative_start = chunk_start_wall - start_time

        for chunk_index in range(chunk_updates):
            update_index = chunk_start + chunk_index
            progress = (chunk_index + 1) / chunk_updates
            event_time = chunk_start_wall + progress * chunk_elapsed
            cumulative_time = chunk_cumulative_start + progress * chunk_elapsed

            metrics_loss = float(chunk_metrics.loss[chunk_index])
            metrics_policy_loss = float(chunk_metrics.policy_loss[chunk_index])
            metrics_value_loss = float(chunk_metrics.value_loss[chunk_index])
            metrics_entropy_loss = float(chunk_metrics.entropy_loss[chunk_index])
            metrics_episode_length = float(chunk_metrics.episode_length[chunk_index])
            metrics_episode_reward = float(chunk_metrics.episode_reward[chunk_index])

            data["loss"].append(metrics_loss)
            data["policy_loss"].append(metrics_policy_loss)
            data["value_loss"].append(metrics_value_loss)
            data["entropy_loss"].append(metrics_entropy_loss)
            data["episode_length"].append(metrics_episode_length)
            data["episode_reward"].append(metrics_episode_reward)
            data["step_time_s"].append(avg_step_time)
            data["cumulative_time_s"].append(cumulative_time)

            should_log = (
                args.print_frequency > 0 and (
                    update_index == 0
                    or (update_index + 1) % args.print_frequency == 0
                    or (update_index + 1) == num_updates
                )
            )
            if should_log:
                since_log = event_time - last_log_time
                last_log_time = event_time

                window = slice(window_start_idx, update_index + 1)
                avg_episode_reward = float(np.mean(data["episode_reward"][window]))
                avg_episode_length = float(np.mean(data["episode_length"][window]))
                avg_loss = float(np.mean(data["loss"][window]))
                avg_policy_loss = float(np.mean(data["policy_loss"][window]))
                avg_value_loss = float(np.mean(data["value_loss"][window]))
                avg_entropy_loss = float(np.mean(data["entropy_loss"][window]))
                avg_beta_e = float(np.mean(entropy_schedule[window]))
                avg_step_time_window = float(np.mean(data["step_time_s"][window]))

                print(
                    f"{update_index + 1:>8d}  "
                    f"{(update_index + 1) * args.batch_size:>10d}  "
                    f"{avg_episode_reward:>12.5f}  "
                    f"{avg_episode_length:>12.3f}  "
                    f"{avg_loss:>12.5f}  "
                    f"{avg_policy_loss:>12.5f}  "
                    f"{avg_value_loss:>12.5f}  "
                    f"{avg_entropy_loss:>12.5f}  "
                    f"{avg_beta_e:>8.5f}  "
                    f"{avg_step_time_window:>10.4f}  "
                    f"{since_log:>10.4f}"
                )
                window_start_idx = update_index + 1

            should_checkpoint = (
                args.checkpoint_frequency > 0 and (
                    (update_index + 1) % args.checkpoint_frequency == 0
                    or (update_index + 1) == num_updates
                )
            )
            if should_checkpoint:
                if chunk_index != chunk_updates - 1:
                    raise RuntimeError(
                        "Checkpoint boundary must align with chunk end in compiled training."
                    )
                checkpoint_path = os.path.join(
                    checkpoint_dir,
                    f"net_jax_update_{update_index + 1:08d}.p",
                )
                save_jax_params(state.params, checkpoint_path)
                print(f"checkpoint_saved update={update_index + 1} path={checkpoint_path}")

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
        num_trials=EVAL_EPISODES,
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

    save_jax_params(state.params, os.path.join(exp_path, 'net_jax.p'))
    with open(os.path.join(exp_path, 'data_training_jax.p'), 'wb') as file:
        pickle.dump(data, file)
