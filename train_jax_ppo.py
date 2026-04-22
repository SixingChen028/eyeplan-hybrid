import os
import json
import pickle
import time
import numpy as np

from modules.argument import ArgParser
from modules.jax_run_dirs import (
    create_timestamped_run_dir,
    resolve_timestamped_run_dir,
    write_run_metadata,
)
from modules.jax_a2c import save_jax_params, save_jax_tree, load_jax_tree
from modules.jax_environment import JaxDecisionTreeEnv
from modules.jax_ppo import JaxBatchMaskPPO
from modules.jax_simulation import JaxSimulator


EVAL_EPISODES = 10_000
CHECKPOINT_STATE_NAME = "train_state_latest.p"
CHECKPOINT_META_NAME = "train_state_latest.json"


def _has_resume_key(jobid: str) -> bool:
    return str(jobid).strip() not in {"", "0"}


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
            candidate_path = resolve_timestamped_run_dir(path=args.path, jobid=args.jobid)
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
        exp_path = create_timestamped_run_dir(path=args.path, jobid=args.jobid)
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
        t_max=args.t_max,
        cost=args.cost,
        scale_factor=args.scale_factor,
        shuffle_nodes=args.shuffle_nodes,
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
        f"num_updates={num_updates} "
        f"t_max={args.t_max} "
        f"ppo_epochs={args.ppo_epochs} "
        f"ppo_clip_eps={args.ppo_clip_eps} "
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
            f"{'clipfrac':>10}  "
            f"{'approx_kl':>10}  "
            f"{'beta_e':>8}  "
            f"{'step_s':>10}  "
            f"{'since_log':>10}"
        )
        print("-" * 168)

    start_time = time.time()
    last_log_time = time.time()
    window_start_idx = 0
    
    def _should_checkpoint(next_update: int) -> bool:
        if args.checkpoint_frequency < 0:
            return False
        if args.checkpoint_frequency == 0:
            return True
        return next_update % args.checkpoint_frequency == 0 or next_update == num_updates

    for index in range(start_update, num_updates):
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
        data["clip_fraction"].append(float(metrics.clip_fraction))
        data["approx_kl"].append(float(metrics.approx_kl))
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
            avg_clip_fraction = float(np.mean(data["clip_fraction"][window]))
            avg_approx_kl = float(np.mean(data["approx_kl"][window]))
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
                f"{avg_clip_fraction:>10.5f}  "
                f"{avg_approx_kl:>10.5f}  "
                f"{avg_beta_e:>8.5f}  "
                f"{avg_step_time:>10.4f}  "
                f"{since_log:>10.4f}"
            )
            window_start_idx = index + 1

        if _should_checkpoint(index + 1):
            _save_rolling_checkpoint(
                state=state,
                checkpoint_state_path=checkpoint_state_path,
                checkpoint_meta_path=checkpoint_meta_path,
                next_update=index + 1,
            )

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

    save_jax_params(state.params, os.path.join(exp_path, 'net_jax_ppo.p'))
    with open(os.path.join(exp_path, 'data_training_jax_ppo.p'), 'wb') as file:
        pickle.dump(data, file)
