#!/usr/bin/env python3
import argparse
import os

import jax

jax.config.update("jax_compiler_enable_remat_pass", False)

from modules.a2c_sweep import VmappedA2CTrainer, build_hypers
from modules.config import (
    DEFAULT_META,
    apply_cli_param_overrides,
    expand_sweep,
    load_config,
    resolve_training_geometry,
)
from modules.train_progress import train_with_progress
from modules.train_results import (
    env_from_args,
    log_run_dirs_preview,
    prepare_run_dirs,
    save_results,
)


def _log(*args, **kwargs) -> None:
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a parallelized RL TOML sweep (A2C).")
    parser.add_argument("config", help="TOML config path or config stem under ./config.")
    parser.add_argument("--path", help="Override output path from [meta].result_path.")
    parser.add_argument("--experiment", help="Override experiment name. Defaults to [meta].experiment or config stem.")
    args, override_tokens = parser.parse_known_args()

    config_path, config = load_config(args.config)
    meta = dict(DEFAULT_META)
    meta.update(config.get("meta", {}))
    params = apply_cli_param_overrides(config.get("params", {}), override_tokens)

    fixed, runs, varied_keys = expand_sweep(params)
    output_path = args.path or str(meta["result_path"])
    experiment = args.experiment or str(meta.get("experiment") or config_path.stem)
    results_dir_display = os.path.join(output_path, "runs", experiment, "")
    if results_dir_display.startswith("./"):
        results_dir_display = results_dir_display[2:]
    _log(f"writing results to {results_dir_display}")

    num_updates, num_envs, rollout_length = resolve_training_geometry(fixed)
    env = env_from_args(runs[0])
    trainer = VmappedA2CTrainer(
        env=env,
        feature_size=env.observation_shape[0],
        action_size=env.action_size,
        hidden_size=fixed["hidden_size"],
        num_envs=num_envs,
        num_updates=num_updates,
        rollout_length=rollout_length,
        network_type=fixed["network_type"],
    )
    hypers = build_hypers(runs)

    devices = ", ".join(
        f"{device.platform}:{device.device_kind}"
        for device in jax.local_devices()
    )
    _log(f"jax_backend={jax.default_backend()} jax_devices=[{devices}]")
    _log(
        "parallel_run_config "
        f"runs={len(runs)} "
        f"num_updates={num_updates} "
        f"num_envs={num_envs} "
        f"rollout_length={rollout_length} "
        f"t_max={fixed['t_max']} "
        f"varied_keys={','.join(varied_keys)}"
    )
    single_run_mode = len(runs) == 1 and len(varied_keys) == 0
    run_dirs = prepare_run_dirs(
        runs,
        path=output_path,
        experiment=experiment,
        config_path=config_path,
        varied_keys=varied_keys,
    )
    log_run_dirs_preview(run_dirs)

    result, elapsed_seconds, gpu_summary_line = train_with_progress(
        trainer,
        hypers,
        run_dirs=run_dirs,
        num_updates=num_updates,
        env_steps_per_update=num_envs * rollout_length,
        print_frequency=int(meta["print_frequency"]),
        max_compiled_updates_per_chunk=int(meta["max_compiled_updates_per_chunk"]),
        include_gpu_summary=True,
        emit_single_run_progress_to_stdout=single_run_mode,
    )
    _log(f"parallel_train_elapsed_seconds={elapsed_seconds:.3f}")
    _log(gpu_summary_line)

    save_results(
        result,
        runs,
        run_dirs,
        elapsed_seconds=elapsed_seconds,
    )


if __name__ == "__main__":
    main()
