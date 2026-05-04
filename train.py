import os
import atexit
import json
import pickle
import sys
import time
import math
import jax
import numpy as np

from modules.argument import ArgParser
from modules.run_dirs import (
    create_timestamped_run_dir,
    resolve_timestamped_run_dir,
    write_run_metadata,
)
from modules.environment import JaxDecisionTreeEnv
from modules.a2c import JaxBatchMaskA2C, save_jax_params, save_jax_tree, load_jax_tree
from modules.ppo import JaxBatchMaskPPO
from modules.simulation import JaxSimulator


CHECKPOINT_STATE_NAME = "train_state_latest.p"
CHECKPOINT_META_NAME = "train_state_latest.json"
EVAL_SUMMARY_NAME = "eval_summary_jax.json"
TRAINING_DATA_NAME = "data_training_jax.p"


class _TeeStream:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            try:
                stream.write(data)
            except Exception:
                continue
        return len(data)

    def flush(self):
        for stream in self._streams:
            try:
                stream.flush()
            except Exception:
                continue


def _tee_console_to_log(run_dir: str):
    log_path = os.path.join(run_dir, "training.log")
    log_file = open(log_path, "a", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = _TeeStream(original_stdout, log_file)
    sys.stderr = _TeeStream(original_stderr, log_file)

    def _restore_streams_and_close_log():
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        try:
            log_file.close()
        except Exception:
            pass

    atexit.register(_restore_streams_and_close_log)
    return log_path


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


def _args_match(saved_value, current_value) -> bool:
    if isinstance(saved_value, bool) or isinstance(current_value, bool):
        return bool(saved_value) == bool(current_value)

    numeric_types = (int, float)
    if isinstance(saved_value, numeric_types) and isinstance(current_value, numeric_types):
        return math.isclose(float(saved_value), float(current_value), rel_tol=1e-9, abs_tol=1e-12)

    return saved_value == current_value


def _validate_resume_metadata(metadata_args: dict, current_args) -> None:
    ignored_keys = {"resume", "path", "jobid", "experiment", "eval_episodes"}
    missing_defaults = {"network_type": "mlp"}
    mismatches: list[str] = []
    missing_keys: list[str] = []

    for key, current_value in vars(current_args).items():
        if key in ignored_keys:
            continue
        if key not in metadata_args:
            if key in missing_defaults and _args_match(missing_defaults[key], current_value):
                continue
            missing_keys.append(key)
            continue

        saved_value = metadata_args[key]
        if not _args_match(saved_value, current_value):
            mismatches.append(
                f"{key}: metadata={saved_value!r}, current={current_value!r}"
            )

    if missing_keys or mismatches:
        issues: list[str] = []
        if missing_keys:
            issues.append("missing keys in metadata: " + ", ".join(sorted(missing_keys)))
        if mismatches:
            issues.append("mismatched values: " + "; ".join(mismatches))
        raise ValueError(
            "Resume metadata check failed. Passed arguments must match metadata for the selected run. "
            + " | ".join(issues)
        )


def _load_existing_training_data(path: str, keys: list[str], max_updates: int) -> dict[str, list[float]]:
    with open(path, "rb") as file:
        loaded = pickle.load(file)
    if not isinstance(loaded, dict):
        raise ValueError(f"Training data at {path} must be a dictionary.")

    restored: dict[str, list[float]] = {}
    for key in keys:
        values = loaded.get(key, [0.0] * max_updates)
        if not isinstance(values, list):
            raise ValueError(f"Training data key '{key}' in {path} must be a list.")
        restored[key] = list(values[:max_updates])
    return restored


if __name__ == '__main__':
    parser = ArgParser()
    args = parser.args
    num_updates = int(args.num_updates)

    state = None
    start_update = 0
    resume_matched_run = False
    if args.resume:
        if not _has_resume_key(args.jobid):
            raise ValueError("--resume requires a non-empty --jobid.")

        try:
            exp_path = resolve_timestamped_run_dir(
                path=args.path,
                experiment=args.experiment,
                jobid=args.jobid,
            )
        except FileNotFoundError:
            exp_path = None
        if exp_path is not None:
            metadata_path = os.path.join(exp_path, "metadata.json")
            if not os.path.exists(metadata_path):
                raise FileNotFoundError(f"Missing metadata for resumed run: {metadata_path}")
            resume_matched_run = True
    else:
        exp_path = None

    if exp_path is None:
        exp_path = create_timestamped_run_dir(
            path=args.path,
            experiment=args.experiment,
            jobid=args.jobid,
        )
        metadata_path = write_run_metadata(run_dir=exp_path, args=args, cwd=os.getcwd())
    else:
        metadata_path = os.path.join(exp_path, "metadata.json")
        with open(metadata_path, "r") as file:
            metadata = json.load(file)
        metadata_args = metadata.get("args", {})
        if not isinstance(metadata_args, dict):
            raise ValueError(f"Invalid metadata args format in: {metadata_path}")
        _validate_resume_metadata(metadata_args, args)

        checkpoint_dir = os.path.join(exp_path, "checkpoints")
        checkpoint_state_path = os.path.join(checkpoint_dir, CHECKPOINT_STATE_NAME)
        checkpoint_meta_path = os.path.join(checkpoint_dir, CHECKPOINT_META_NAME)
        state, start_update = _load_resume_state(checkpoint_state_path, checkpoint_meta_path)
        if state is None:
            raise FileNotFoundError(
                "Resume requested but no checkpoint was found at "
                f"{checkpoint_state_path} and {checkpoint_meta_path}"
            )
        if start_update < 0:
            raise ValueError(f"Invalid checkpoint next_update={start_update} (must be >= 0).")
        if start_update > num_updates:
            raise ValueError(
                f"Checkpoint next_update={start_update} exceeds requested num_updates={num_updates}."
            )

    log_path = _tee_console_to_log(exp_path)
    print(f"training_log={log_path}")
    devices = ", ".join(
        f"{device.platform}:{device.device_kind}"
        for device in jax.local_devices()
    )
    print(f"jax_backend={jax.default_backend()} jax_devices=[{devices}]")
    print(f"run_dir={exp_path}")
    print(f"run_metadata={metadata_path}")
    if args.resume:
        print(f"resume_mode=true")
    if args.resume and not resume_matched_run:
        print("resume_checkpoint_not_found=true (starting new run)")
    if start_update > 0 and start_update < num_updates:
        print(f"resuming_from_update={start_update}")
    if start_update == num_updates:
        print("resume_checkpoint_already_complete=true")

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
        backup_steps=args.backup_steps,
        wm_decay=args.wm_decay,
        wm_backup=args.wm_backup,
        q_drop_rate=args.q_drop_rate,
        q_flip_rate=args.q_flip_rate,
        t_max=args.t_max,
        cost=args.cost,
        scale_factor=args.scale_factor,
        shuffle_nodes=args.shuffle_nodes,
        canonicalize=args.canonicalize,
        recency_decay=args.recency_decay,
    )

    trainer = JaxBatchMaskA2C(
        env=env,
        feature_size=env.observation_shape[0],
        action_size=env.action_size,
        hidden_size=args.hidden_size,
        num_envs=args.num_envs,
        rollout_length=args.rollout_length,
        lr=args.lr,
        max_grad_norm=args.max_grad_norm,
        gamma=args.gamma,
        lamda=args.lamda,
        beta_v=args.beta_v,
        beta_e=args.beta_e,
        network_type=args.network_type,
    ) if args.algo == "a2c" else JaxBatchMaskPPO(
        env=env,
        feature_size=env.observation_shape[0],
        action_size=env.action_size,
        hidden_size=args.hidden_size,
        num_envs=args.num_envs,
        rollout_length=args.rollout_length,
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
        "episode_length": [],
        "episode_reward": [],
        "episode_count": [],
        "episode_reward_sum": [],
        "episode_length_sum": [],
        "step_time_s": [],
        "cumulative_time_s": [],
    }
    if args.algo == "a2c":
        data["grad_norm"] = []
        data["param_norm"] = []
    else:
        data["clip_fraction"] = []
        data["approx_kl"] = []
    training_data_name = TRAINING_DATA_NAME if args.algo == "a2c" else "data_training_jax_ppo.p"
    training_data_path = os.path.join(exp_path, training_data_name)
    data_keys = list(data.keys())
    if args.resume and start_update > 0 and os.path.exists(training_data_path):
        restored_data = _load_existing_training_data(
            path=training_data_path,
            keys=data_keys,
            max_updates=start_update,
        )
        data.update(restored_data)

    print(
        "run_config "
        f"algo={args.algo} "
        f"num_envs={args.num_envs} "
        f"rollout_length={args.rollout_length} "
        f"num_updates={num_updates} "
        f"eval_episodes={args.eval_episodes} "
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

    def _fmt_k(value: float, width: int = 10) -> str:
        rounded_thousands = int(round(value / 1000.0))
        return f"{rounded_thousands}K".rjust(width)

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
        metric_col_1 = "grad_n" if args.algo == "a2c" else "clipfrac"
        metric_col_2 = "param_n" if args.algo == "a2c" else "approx_kl"
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
                f"{metric_col_1:>8}",
                f"{metric_col_2:>8}",
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
    resume_cumulative_time_offset = (
        float(data["cumulative_time_s"][start_update - 1])
        if start_update > 0 and len(data["cumulative_time_s"]) >= start_update
        else 0.0
    )

    def _next_boundary(processed_updates: int, frequency: int) -> int:
        return ((processed_updates // frequency) + 1) * frequency

    def _should_checkpoint(next_update: int) -> bool:
        if args.checkpoint_frequency < 0:
            return False
        if args.checkpoint_frequency == 0:
            return True
        return next_update % args.checkpoint_frequency == 0 or next_update == num_updates

    index = start_update
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
            episode_count = np.zeros((chunk_updates,), dtype=np.float32)
            episode_reward_sum = np.zeros((chunk_updates,), dtype=np.float32)
            episode_length_sum = np.zeros((chunk_updates,), dtype=np.float32)
            episode_count[-1] = chunk_metrics_mean.episode_count
            episode_reward_sum[-1] = chunk_metrics_mean.episode_reward_sum
            episode_length_sum[-1] = chunk_metrics_mean.episode_length_sum
            chunk_metrics = chunk_metrics._replace(
                episode_count=episode_count,
                episode_reward_sum=episode_reward_sum,
                episode_length_sum=episode_length_sum,
            )
        chunk_end_wall = time.time()
        is_first_chunk = chunk_start == start_update

        chunk_elapsed = chunk_end_wall - chunk_start_wall
        avg_step_time = chunk_elapsed / chunk_updates
        chunk_cumulative_start = resume_cumulative_time_offset + (chunk_start_wall - start_time)

        cumulative_values = (
            chunk_cumulative_start
            + (np.arange(1, chunk_updates + 1, dtype=np.float64) / chunk_updates) * chunk_elapsed
        )
        chunk_data_start = len(data["loss"])
        data["loss"].extend(chunk_metrics.loss.tolist())
        data["policy_loss"].extend(chunk_metrics.policy_loss.tolist())
        data["value_loss"].extend(chunk_metrics.value_loss.tolist())
        data["entropy_loss"].extend(chunk_metrics.entropy_loss.tolist())
        data["episode_length"].extend(chunk_metrics.episode_length.tolist())
        data["episode_reward"].extend(chunk_metrics.episode_reward.tolist())
        data["episode_count"].extend(chunk_metrics.episode_count.tolist())
        data["episode_reward_sum"].extend(chunk_metrics.episode_reward_sum.tolist())
        data["episode_length_sum"].extend(chunk_metrics.episode_length_sum.tolist())
        if args.algo == "a2c":
            data["grad_norm"].extend(chunk_metrics.grad_norm.tolist())
            data["param_norm"].extend(chunk_metrics.param_norm.tolist())
        else:
            data["clip_fraction"].extend(chunk_metrics.clip_fraction.tolist())
            data["approx_kl"].extend(chunk_metrics.approx_kl.tolist())
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
                cumulative_episode_count = float(np.sum(data["episode_count"][:data_index + 1]))
                episode_count = float(np.sum(data["episode_count"][window]))
                avg_episode_reward = (
                    float(np.sum(data["episode_reward_sum"][window]) / episode_count)
                    if episode_count > 0
                    else float("nan")
                )
                avg_episode_length = (
                    float(np.sum(data["episode_length_sum"][window]) / episode_count)
                    if episode_count > 0
                    else float("nan")
                )
                avg_loss = float(np.mean(data["loss"][window]))
                avg_policy_loss = float(np.mean(data["policy_loss"][window]))
                avg_value_loss = float(np.mean(data["value_loss"][window]))
                avg_entropy_loss = float(np.mean(data["entropy_loss"][window]))
                if args.algo == "a2c":
                    avg_grad_norm = float(np.mean(data["grad_norm"][window]))
                    avg_param_norm = float(np.mean(data["param_norm"][window]))
                    extra_col_1 = _fmt_num(avg_grad_norm)
                    extra_col_2 = _fmt_num(avg_param_norm)
                else:
                    avg_clip_fraction = float(np.mean(data["clip_fraction"][window]))
                    avg_approx_kl = float(np.mean(data["approx_kl"][window]))
                    extra_col_1 = _fmt_num(avg_clip_fraction)
                    extra_col_2 = _fmt_num(avg_approx_kl)

                print(
                    col_sep.join(
                        [
                            f"{update_index + 1:>8d}",
                            _fmt_k(cumulative_episode_count),
                            _fmt_num(avg_episode_reward),
                            _fmt_num(avg_episode_length),
                            _fmt_num(avg_loss),
                            _fmt_num(avg_policy_loss),
                            _fmt_num(avg_value_loss),
                            _fmt_num(avg_entropy_loss),
                            extra_col_1,
                            extra_col_2,
                            f"{elapsed_display:>8}",
                            f"{eta_display:>8}",
                        ]
                    )
                )
                window_start_idx = data_index + 1

        if is_first_chunk and eta_skip_elapsed is None:
            eta_skip_elapsed = chunk_end_wall - start_time
            eta_skip_updates = chunk_end - start_update

        if _should_checkpoint(chunk_end):
            _save_rolling_checkpoint(
                state=state,
                checkpoint_state_path=checkpoint_state_path,
                checkpoint_meta_path=checkpoint_meta_path,
                next_update=chunk_end,
            )

        index = chunk_end

    train_elapsed_seconds = time.time() - start_time
    mean_step_seconds = (
        float(np.mean(data["step_time_s"]))
        if len(data["step_time_s"]) > 0
        else float("nan")
    )
    print(
        "run_summary "
        f"updates={num_updates} "
        f"elapsed_seconds={train_elapsed_seconds:.3f} "
        f"mean_step_seconds={mean_step_seconds:.6f}"
    )

    eval_start = time.time()
    simulator = JaxSimulator(env)
    eval_stats = simulator.evaluate_policy(
        params=state.params,
        seed=args.seed,
        num_trials=args.eval_episodes,
        greedy=True,
    )
    eval_elapsed_seconds = time.time() - eval_start
    print(
        "eval_summary "
        f"episodes={eval_stats['num_trials']} "
        f"reward_mean={eval_stats['reward_mean']:.6f} "
        f"reward_sd={eval_stats['reward_sd']:.6f} "
        f"reward_no_cost_mean={eval_stats['reward_no_cost_mean']:.6f} "
        f"reward_no_cost_sd={eval_stats['reward_no_cost_sd']:.6f} "
        f"n_steps_mean={eval_stats['n_steps_mean']:.3f} "
        f"n_steps_sd={eval_stats['n_steps_sd']:.3f} "
        f"elapsed_seconds={eval_elapsed_seconds:.3f}"
    )

    eval_summary = {
        "num_trials": int(eval_stats["num_trials"]),
        "reward_mean": float(eval_stats["reward_mean"]),
        "reward_sd": float(eval_stats["reward_sd"]),
        "reward_no_cost_mean": float(eval_stats["reward_no_cost_mean"]),
        "reward_no_cost_sd": float(eval_stats["reward_no_cost_sd"]),
        "n_steps_mean": float(eval_stats["n_steps_mean"]),
        "n_steps_sd": float(eval_stats["n_steps_sd"]),
        "train_elapsed_seconds": float(train_elapsed_seconds),
        "eval_elapsed_seconds": float(eval_elapsed_seconds),
        "num_updates": int(num_updates),
    }
    eval_summary_path = os.path.join(exp_path, EVAL_SUMMARY_NAME)
    with open(eval_summary_path, "w") as file:
        json.dump(eval_summary, file, indent=2, sort_keys=True)
    print(f"eval_summary_path={eval_summary_path}")

    model_params_name = "net_jax.p" if args.algo == "a2c" else "net_jax_ppo.p"
    save_jax_params(state.params, os.path.join(exp_path, model_params_name))
    with open(training_data_path, 'wb') as file:
        pickle.dump(data, file)
