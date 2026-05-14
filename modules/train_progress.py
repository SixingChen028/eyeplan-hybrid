from __future__ import annotations

import faulthandler
import os
import shutil
import statistics
import subprocess
import sys
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


class StartupTrainingTimeout:
    def __init__(
        self,
        seconds: float,
        *,
        exit_code: int = 124,
        exit_fn=os._exit,
        diagnostic_fn=None,
    ):
        self.seconds = float(seconds)
        self.exit_code = int(exit_code)
        self._exit_fn = exit_fn
        self._diagnostic_fn = diagnostic_fn
        self._stage = "startup"
        self._stage_start = time.time()
        self._timer: threading.Timer | None = None

    def start(self) -> None:
        if self.seconds <= 0 or self._timer is not None:
            return
        self._timer = threading.Timer(self.seconds, self._expire)
        self._timer.daemon = True
        self._timer.start()

    def cancel(self) -> None:
        if self._timer is None:
            return
        self._timer.cancel()
        self._timer = None

    def set_stage(self, stage: str) -> None:
        self._stage = str(stage)
        self._stage_start = time.time()

    def _expire(self) -> None:
        print(
            "parallel_train_startup_timeout "
            f"seconds={self.seconds:g} "
            f"stage={self._stage} "
            f"stage_elapsed={time.time() - self._stage_start:.1f} "
            "reason=training_not_started",
            file=sys.stderr,
            flush=True,
        )
        if self._diagnostic_fn is not None:
            self._diagnostic_fn()
        try:
            faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        except Exception as error:
            print(
                f"parallel_train_diagnostic_traceback error={type(error).__name__}:{error}",
                file=sys.stderr,
                flush=True,
            )
        self._exit_fn(self.exit_code)


class _CompileProgressLogger:
    def __init__(self, start_time: float, *, interval_seconds: float = 60.0):
        self._start_time = float(start_time)
        self._interval_seconds = float(interval_seconds)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._interval_seconds <= 0:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        self._thread = None

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_seconds):
            _log(f"parallel_train_compile_progress elapsed={time.time() - self._start_time:.1f}")


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


def _run_diagnostic_command(command: list[str], *, timeout_seconds: float = 10.0) -> str:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (subprocess.SubprocessError, FileNotFoundError) as error:
        return f"error={type(error).__name__}:{error}"

    output = result.stdout.strip()
    stderr = result.stderr.strip()
    parts = [f"returncode={result.returncode}"]
    if output:
        parts.append(f"stdout={output}")
    if stderr:
        parts.append(f"stderr={stderr}")
    return " ".join(parts)


def log_jax_gpu_diagnostics() -> None:
    env_keys = (
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "JAX_PLATFORMS",
        "JAX_PLATFORM_NAME",
        "XLA_FLAGS",
        "XLA_PYTHON_CLIENT_ALLOCATOR",
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
    )
    env = " ".join(f"{key}={os.environ.get(key, '<unset>')}" for key in env_keys)
    print(f"parallel_train_diagnostic_env {env}", file=sys.stderr, flush=True)
    print(f"parallel_train_diagnostic_jax version={jax.__version__}", file=sys.stderr, flush=True)
    try:
        devices = ", ".join(
            f"{device.platform}:{device.device_kind}:id={device.id}"
            for device in jax.local_devices()
        )
        print(
            f"parallel_train_diagnostic_jax backend={jax.default_backend()} devices=[{devices}]",
            file=sys.stderr,
            flush=True,
        )
    except Exception as error:
        print(
            f"parallel_train_diagnostic_jax error={type(error).__name__}:{error}",
            file=sys.stderr,
            flush=True,
        )

    if shutil.which("nvidia-smi") is None:
        print("parallel_train_diagnostic_nvidia_smi unavailable=true", file=sys.stderr, flush=True)
        return

    print(
        "parallel_train_diagnostic_nvidia_smi_L "
        + _run_diagnostic_command(["nvidia-smi", "-L"]),
        file=sys.stderr,
        flush=True,
    )
    print(
        "parallel_train_diagnostic_nvidia_smi_query "
        + _run_diagnostic_command(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,pci.bus_id,driver_version,memory.total,memory.used,"
                "compute_mode,persistence_mode,pstate,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
        ),
        file=sys.stderr,
        flush=True,
    )


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
        lambda *values: jnp.concatenate(values, axis=1),
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
    update_end: int,
    env_steps_per_update: int = 1,
    cumulative_episode_counts: np.ndarray | None = None,
    emit_stdout: bool = False,
) -> None:
    metrics = jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), chunk_metrics)

    for run_index, run_dir in enumerate(run_dirs):
        log_path = os.path.join(run_dir, "training.log")
        chunk_episode_count = float(np.sum(metrics.episode_count[run_index]))
        if cumulative_episode_counts is not None:
            cumulative_episode_counts[run_index] += chunk_episode_count
            ep_num = int(round(cumulative_episode_counts[run_index]))
        else:
            ep_num = update_end * env_steps_per_update
        line = "   ".join(
            [
                f"{update_end:>8d}",
                _fmt_ep_num(ep_num),
                _fmt_num(float(np.mean(metrics.episode_reward[run_index]))),
                _fmt_num(float(np.mean(metrics.episode_length[run_index]))),
                _fmt_num(float(np.mean(metrics.loss[run_index]))),
                _fmt_num(float(np.mean(metrics.policy_loss[run_index]))),
                _fmt_num(float(np.mean(metrics.value_loss[run_index]))),
                _fmt_num(float(np.mean(metrics.entropy_loss[run_index]))),
                _fmt_num(float(np.mean(metrics.grad_norm[run_index]))),
                _fmt_num(float(np.mean(metrics.param_norm[run_index]))),
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
    *,
    run_dirs: list[str] | None = None,
    num_updates: int,
    env_steps_per_update: int = 1,
    print_frequency: int,
    max_compiled_updates_per_chunk: int = -1,
    startup_timeout: StartupTrainingTimeout | None = None,
    include_gpu_summary: bool = False,
    emit_single_run_progress_to_stdout: bool = False,
):
    has_run_dirs = run_dirs is not None
    if run_dirs is None:
        run_dirs = []
    num_runs = int(getattr(hypers.lr, "shape", [1])[0])

    emit_progress = print_frequency > 0
    progress_frequency = int(print_frequency) if emit_progress else int(num_updates)
    schedule = _entropy_schedule(hypers, num_updates)
    compiled_updates_per_chunk = _resolve_compiled_updates_per_chunk(
        min(progress_frequency, num_updates),
        max_compiled_updates_per_chunk=max_compiled_updates_per_chunk,
    )

    compile_start = time.time()
    _log(f"compiling jax graph; updates_per_chunk={compiled_updates_per_chunk}")
    if startup_timeout is not None:
        startup_timeout.set_stage("init_sweep_states")
    compile_progress = _CompileProgressLogger(compile_start)
    compile_progress.start()
    try:
        states = jax.block_until_ready(trainer.init_sweep_states(hypers))
        _log(f"  block_until_ready done after {time.time() - compile_start:.1f} seconds")
        if startup_timeout is not None:
            startup_timeout.set_stage("compile_train_sweep_chunk")
        trainer.compile_train_sweep_chunk(
            states,
            hypers,
            schedule[:, :compiled_updates_per_chunk],
        )
    finally:
        compile_progress.stop()
    _log(f"  compilation done after {time.time() - compile_start:.1f} seconds")

    if emit_progress and has_run_dirs:
        _init_per_run_training_logs(run_dirs)
        if emit_single_run_progress_to_stdout:
            run_header = _header(RUN_LOG_COLUMNS)
            _log(run_header)
            _log("-" * len(run_header))

    start = time.time()
    if startup_timeout is not None:
        startup_timeout.cancel()
    _log("parallel_train_started")
    metrics_chunks = []
    gpu_samples: list[dict[str, float]] = []
    cumulative_episode_counts = np.zeros((num_runs,), dtype=np.float64)
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
