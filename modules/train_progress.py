from __future__ import annotations

import os
import shutil
import statistics
import subprocess
import threading
import time

import jax
import jax.numpy as jnp
import numpy as np

from modules.a2c_sweep import A2CHyperParams, A2CSweepResult, VmappedA2CTrainer

RUN_LOG_COLUMNS = (
    ("update", 8),
    ("ep_num", 10),
    ("ep_rew", 8),
    ("ep_len", 8),
    ("loss", 8),
    ("policy", 8),
    ("value", 8),
    ("entropy", 8),
    ("grad_n", 8),
    ("param_n", 8),
)

PARALLEL_PROGRESS_COLUMNS = (
    ("update", 8),
    ("ep_num", 10),
    ("elapsed", 8),
    ("ETA", 8),
    ("upd/s", 8),
    ("gpu%", 8),
    ("mem%", 8),
)


def _log(*args, **kwargs) -> None:
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def _header(columns: tuple[tuple[str, int], ...]) -> str:
    return "   ".join(f"{label:>{width}}" for label, width in columns)


def _query_gpu_stats() -> dict[str, float] | None:
    if shutil.which("nvidia-smi") is None:
        return None
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None

    lines = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
    if not lines:
        return None

    gpu_utils: list[float] = []
    mem_utils: list[float] = []
    for line in lines:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        gpu_util = float(parts[0])
        mem_used = float(parts[2])
        mem_total = float(parts[3])
        gpu_utils.append(gpu_util)
        mem_utils.append((mem_used / mem_total) * 100.0 if mem_total > 0 else 0.0)

    if not gpu_utils:
        return None
    return {
        "gpu_util_mean": float(sum(gpu_utils) / len(gpu_utils)),
        "gpu_util_max": float(max(gpu_utils)),
        "gpu_mem_util_mean": float(sum(mem_utils) / len(mem_utils)),
        "gpu_mem_util_max": float(max(mem_utils)),
    }


def _summarize_gpu_samples(samples: list[dict[str, float]]) -> str:
    if not samples:
        return "parallel_train_gpu_summary unavailable=true"

    gpu_util_mean = [sample["gpu_util_mean"] for sample in samples]
    gpu_util_max = [sample["gpu_util_max"] for sample in samples]
    gpu_mem_mean = [sample["gpu_mem_util_mean"] for sample in samples]
    gpu_mem_max = [sample["gpu_mem_util_max"] for sample in samples]

    return (
        "parallel_train_gpu_summary "
        f"samples={len(samples)} "
        f"gpu_util_mean={statistics.mean(gpu_util_mean):.1f} "
        f"gpu_util_p95={np.percentile(np.asarray(gpu_util_mean), 95):.1f} "
        f"gpu_util_peak={max(gpu_util_max):.1f} "
        f"gpu_mem_util_mean={statistics.mean(gpu_mem_mean):.1f} "
        f"gpu_mem_util_peak={max(gpu_mem_max):.1f}"
    )


class _GpuSampler:
    def __init__(self, interval_seconds: float = 0.5):
        self._interval_seconds = float(interval_seconds)
        self._samples: list[dict[str, float]] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if shutil.which("nvidia-smi") is None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=self._interval_seconds * 3.0)
        self._thread = None

    def snapshot_from(self, start_index: int) -> list[dict[str, float]]:
        return list(self._samples[start_index:])

    def count(self) -> int:
        return len(self._samples)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            sample = _query_gpu_stats()
            if sample is not None:
                self._samples.append(sample)
            self._stop_event.wait(self._interval_seconds)


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{seconds:02d}s"
    return f"{minutes:d}m{seconds:02d}s"


def _entropy_schedule(hypers: A2CHyperParams, num_updates: int) -> jax.Array:
    progress = jnp.linspace(0.0, 1.0, num_updates, dtype=jnp.float32)
    return (
        hypers.beta_e_init[:, None]
        + (hypers.beta_e_final - hypers.beta_e_init)[:, None] * progress[None, :]
    ).astype(jnp.float32)


def _concat_metric_chunks(chunks: list):
    return jax.tree_util.tree_map(
        lambda *values: jnp.concatenate(values, axis=2),
        *chunks,
    )


def _resolve_compiled_updates_per_chunk(
    requested_updates: int,
    *,
    max_compiled_updates_per_chunk: int,
) -> int:
    requested_updates = int(requested_updates)
    if requested_updates <= 0:
        raise ValueError("requested_updates must be positive")
    max_compiled_updates_per_chunk = int(max_compiled_updates_per_chunk)
    if max_compiled_updates_per_chunk <= 0:
        return requested_updates
    return min(requested_updates, max_compiled_updates_per_chunk)


def _init_per_run_training_logs(run_dirs: list[str]) -> None:
    header = _header(RUN_LOG_COLUMNS)
    for run_dir in run_dirs:
        log_path = os.path.join(run_dir, "training.log")
        with open(log_path, "w") as file:
            file.write(f"run_dir={run_dir}\n")
            file.write(header + "\n")
            file.write("-" * len(header) + "\n")


def _fmt_num(value: float, width: int = 8, decimals: int = 3) -> str:
    return f"{value: {width}.{decimals}f}"


def _fmt_ep_num(value: int, width: int = 10) -> str:
    ep_num_k = int(round(value / 1_000.0))
    return f"{ep_num_k:>{width}d}K"


def _append_per_run_progress_logs(
    run_dirs: list[str],
    chunk_metrics,
    *,
    num_hypers: int,
    num_seeds: int,
    update_end: int,
    env_steps_per_update: int = 1,
    cumulative_episode_counts: np.ndarray | None = None,
    emit_stdout: bool = False,
) -> None:
    metrics = jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), chunk_metrics)
    run_dirs_by_index = np.asarray(run_dirs, dtype=object).reshape((num_hypers, num_seeds))

    for hyper_index in range(num_hypers):
        for seed_index in range(num_seeds):
            run_dir = str(run_dirs_by_index[hyper_index, seed_index])
            log_path = os.path.join(run_dir, "training.log")
            chunk_episode_count = float(np.sum(metrics.episode_count[hyper_index, seed_index]))
            if cumulative_episode_counts is not None:
                cumulative_episode_counts[hyper_index, seed_index] += chunk_episode_count
                ep_num = int(round(cumulative_episode_counts[hyper_index, seed_index]))
            else:
                ep_num = update_end * env_steps_per_update
            line = "   ".join(
                [
                    f"{update_end:>8d}",
                    _fmt_ep_num(ep_num),
                    _fmt_num(float(np.mean(metrics.episode_reward[hyper_index, seed_index]))),
                    _fmt_num(float(np.mean(metrics.episode_length[hyper_index, seed_index]))),
                    _fmt_num(float(np.mean(metrics.loss[hyper_index, seed_index]))),
                    _fmt_num(float(np.mean(metrics.policy_loss[hyper_index, seed_index]))),
                    _fmt_num(float(np.mean(metrics.value_loss[hyper_index, seed_index]))),
                    _fmt_num(float(np.mean(metrics.entropy_loss[hyper_index, seed_index]))),
                    _fmt_num(float(np.mean(metrics.grad_norm[hyper_index, seed_index]))),
                    _fmt_num(float(np.mean(metrics.param_norm[hyper_index, seed_index]))),
                ]
            )
            with open(log_path, "a") as file:
                file.write(line + "\n")
            if emit_stdout:
                _log(line)


def _summarize_chunk_gpu(samples: list[dict[str, float]]) -> dict[str, float] | None:
    if not samples:
        return None
    return {
        "gpu_util_mean": float(statistics.mean([s["gpu_util_mean"] for s in samples])),
        "gpu_util_max": float(max(s["gpu_util_max"] for s in samples)),
        "gpu_mem_util_mean": float(statistics.mean([s["gpu_mem_util_mean"] for s in samples])),
        "gpu_mem_util_max": float(max(s["gpu_mem_util_max"] for s in samples)),
    }


def train_with_progress(
    trainer: VmappedA2CTrainer,
    hypers: A2CHyperParams,
    seeds: list[int],
    *,
    run_dirs: list[str] | None = None,
    num_hypers: int | None = None,
    num_updates: int,
    env_steps_per_update: int = 1,
    print_frequency: int,
    max_compiled_updates_per_chunk: int = -1,
    include_gpu_summary: bool = False,
    emit_single_run_progress_to_stdout: bool = False,
):
    has_run_dirs = run_dirs is not None
    if run_dirs is None:
        run_dirs = []
    if num_hypers is None:
        num_hypers = int(getattr(hypers.lr, "shape", [1])[0])

    emit_progress = print_frequency > 0
    progress_frequency = int(print_frequency) if emit_progress else int(num_updates)
    schedule = _entropy_schedule(hypers, num_updates)
    compiled_updates_per_chunk = _resolve_compiled_updates_per_chunk(
        min(progress_frequency, num_updates),
        max_compiled_updates_per_chunk=max_compiled_updates_per_chunk,
    )

    compile_start = time.time()
    states = jax.block_until_ready(trainer.init_sweep_states(hypers, seeds))
    trainer.compile_train_sweep_chunk(
        states,
        hypers,
        schedule[:, :compiled_updates_per_chunk],
    )
    if emit_progress:
        _log(f"parallel_train_compile_seconds={time.time() - compile_start:.3f}")

    if emit_progress and has_run_dirs:
        _init_per_run_training_logs(run_dirs)
        if emit_single_run_progress_to_stdout:
            run_header = _header(RUN_LOG_COLUMNS)
            _log(run_header)
            _log("-" * len(run_header))

    start = time.time()
    metrics_chunks = []
    gpu_samples: list[dict[str, float]] = []
    cumulative_episode_counts = np.zeros((num_hypers, len(seeds)), dtype=np.float64)
    cumulative_episode_count_total = 0.0
    gpu_sampler = _GpuSampler(interval_seconds=0.5)
    if emit_progress or include_gpu_summary:
        gpu_sampler.start()

    show_parallel_table = emit_progress and not emit_single_run_progress_to_stdout
    if show_parallel_table:
        header = _header(PARALLEL_PROGRESS_COLUMNS)
        _log(header)
        _log("-" * len(header))

    try:
        for update_start in range(0, num_updates, progress_frequency):
            update_end = min(update_start + progress_frequency, num_updates)
            chunk_sample_start = gpu_sampler.count()
            window_metric_chunks = []

            for exec_start in range(update_start, update_end, compiled_updates_per_chunk):
                exec_end = min(exec_start + compiled_updates_per_chunk, update_end)
                chunk = schedule[:, exec_start:exec_end]
                result = jax.block_until_ready(trainer.train_sweep_chunk(states, hypers, chunk))
                states = result.states
                metrics_chunks.append(result.metrics)
                window_metric_chunks.append(result.metrics)

            window_metrics = _concat_metric_chunks(window_metric_chunks)
            chunk_episode_count_total = float(np.sum(np.asarray(jax.device_get(window_metrics.episode_count))))
            cumulative_episode_count_total += chunk_episode_count_total
            if emit_progress and has_run_dirs:
                _append_per_run_progress_logs(
                    run_dirs,
                    window_metrics,
                    num_hypers=num_hypers,
                    num_seeds=len(seeds),
                    update_end=update_end,
                    env_steps_per_update=env_steps_per_update,
                    cumulative_episode_counts=cumulative_episode_counts,
                    emit_stdout=emit_single_run_progress_to_stdout,
                )

            elapsed_seconds = time.time() - start
            updates_done = update_end
            updates_per_second = updates_done / elapsed_seconds
            eta_seconds = (num_updates - updates_done) / updates_per_second

            gpu_stats = _summarize_chunk_gpu(gpu_sampler.snapshot_from(chunk_sample_start))
            if gpu_stats is not None:
                gpu_samples.append(gpu_stats)
                gpu_util_text = f"{gpu_stats['gpu_util_mean']:.1f}"
                gpu_mem_text = f"{gpu_stats['gpu_mem_util_mean']:.1f}"
            else:
                gpu_util_text = "n/a"
                gpu_mem_text = "n/a"

            if emit_progress and not has_run_dirs:
                _log(
                    "parallel_train_progress "
                    f"updates={updates_done}/{num_updates} "
                    f"elapsed={elapsed_seconds:.3f} "
                    f"updates_per_second={updates_per_second:.6f}",
                    flush=True,
                )
            if show_parallel_table:
                _log(
                    "   ".join(
                        [
                            f"{updates_done:>8d}",
                            f"{int(round(cumulative_episode_count_total / 1_000.0)):>9d}K",
                            f"{_format_duration(elapsed_seconds):>8}",
                            f"{_format_duration(eta_seconds):>8}",
                            f"{updates_per_second:>8.3f}",
                            f"{gpu_util_text:>8}",
                            f"{gpu_mem_text:>8}",
                        ]
                    ),
                    flush=True,
                )
    finally:
        gpu_sampler.stop()

    if include_gpu_summary and not gpu_samples:
        gpu_stats = _query_gpu_stats()
        if gpu_stats is not None:
            gpu_samples.append(gpu_stats)

    result = A2CSweepResult(states=states, metrics=_concat_metric_chunks(metrics_chunks))
    elapsed_seconds = time.time() - start
    if include_gpu_summary:
        return result, elapsed_seconds, _summarize_gpu_samples(gpu_samples)
    return result, elapsed_seconds
