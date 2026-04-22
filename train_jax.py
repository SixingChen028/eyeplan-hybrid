import os
import pickle
import time
import numpy as np

from modules.argument import ArgParser
from modules.jax_environment import JaxDecisionTreeEnv
from modules.jax_a2c import JaxBatchMaskA2C, save_jax_params
from modules.jax_simulation import JaxSimulator


EVAL_EPISODES = 10_000


if __name__ == '__main__':
    parser = ArgParser()
    args = parser.args

    exp_path = os.path.join(args.path, f'exp_{args.learning_rate}_{args.wm_decay}_{args.jobid}')
    os.makedirs(exp_path, exist_ok=True)

    env = JaxDecisionTreeEnv(
        num_nodes=args.num_nodes,
        beta_move=args.beta_move,
        eps_move=args.eps_move,
        learning_rate=args.learning_rate,
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
        f"print_frequency={args.print_frequency}"
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

    for index in range(num_updates):
        step_start = time.time()
        state, metrics = trainer.train_step(
            state=state,
            beta_e=float(entropy_schedule[index]),
        )
        step_time = time.time() - step_start
        cumulative_time = time.time() - start_time

        data["loss"].append(float(metrics.loss))
        data["policy_loss"].append(float(metrics.policy_loss))
        data["value_loss"].append(float(metrics.value_loss))
        data["entropy_loss"].append(float(metrics.entropy_loss))
        data["episode_length"].append(float(metrics.episode_length))
        data["episode_reward"].append(float(metrics.episode_reward))
        data["step_time_s"].append(step_time)
        data["cumulative_time_s"].append(cumulative_time)

        should_log = (
            args.print_frequency > 0 and (
                index == 0
                or (index + 1) % args.print_frequency == 0
                or (index + 1) == num_updates
            )
        )
        if should_log:
            now = time.time()
            since_log = now - last_log_time
            last_log_time = now

            window = slice(window_start_idx, index + 1)
            avg_episode_reward = float(np.mean(data["episode_reward"][window]))
            avg_episode_length = float(np.mean(data["episode_length"][window]))
            avg_loss = float(np.mean(data["loss"][window]))
            avg_policy_loss = float(np.mean(data["policy_loss"][window]))
            avg_value_loss = float(np.mean(data["value_loss"][window]))
            avg_entropy_loss = float(np.mean(data["entropy_loss"][window]))
            avg_beta_e = float(np.mean(entropy_schedule[window]))
            avg_step_time = float(np.mean(data["step_time_s"][window]))

            print(
                f"{index + 1:>8d}  "
                f"{(index + 1) * args.batch_size:>10d}  "
                f"{avg_episode_reward:>12.5f}  "
                f"{avg_episode_length:>12.3f}  "
                f"{avg_loss:>12.5f}  "
                f"{avg_policy_loss:>12.5f}  "
                f"{avg_value_loss:>12.5f}  "
                f"{avg_entropy_loss:>12.5f}  "
                f"{avg_beta_e:>8.5f}  "
                f"{avg_step_time:>10.4f}  "
                f"{since_log:>10.4f}"
            )
            window_start_idx = index + 1

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
