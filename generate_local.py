#!/usr/bin/env python3
import argparse
import shlex
import subprocess
import tomllib
from pathlib import Path

from generate_sbatch import (
    DEFAULT_META,
    _as_dict,
    _format_summary_value,
    _format_cli_overrides,
    _launcher_task_overrides,
    _resolve_config_path,
    _run_overrides,
    _selected_array_axes,
    _split_params,
    _split_python_command,
    _to_bash_name,
    _to_shell_scalar,
)
from modules.config import normalize_config

DEFAULT_LOCAL = {
    "cpus_per_task": 4,
    "gpus": [0],
    "log": "./log/local",
    "processes_per_gpu": 1,
}


def _parse_gpus(raw: str) -> list[str]:
    gpus = [item.strip() for item in raw.split(",")]
    if any(item == "" for item in gpus):
        raise ValueError("GPU list must be comma-separated values like '0,1'.")
    return gpus


def _local_gpus(local: dict, cli_gpus: str | None) -> list[str]:
    if cli_gpus is not None:
        return _parse_gpus(cli_gpus)

    raw = local.get("gpus", DEFAULT_LOCAL["gpus"])
    if isinstance(raw, str):
        return _parse_gpus(raw)
    if not isinstance(raw, list):
        raise ValueError("local.gpus must be an array or comma-separated string.")
    if len(raw) == 0:
        raise ValueError("local.gpus must not be empty.")
    return [str(gpu) for gpu in raw]


def _render_script(config: dict, config_path: Path, *, gpus: list[str] | None = None) -> str:
    normalized_config = normalize_config(config)
    meta = _as_dict(normalized_config.get("meta"), "meta", default=DEFAULT_META)
    local = _as_dict(config.get("local"), "local", default=DEFAULT_LOCAL)
    params = _as_dict(normalized_config.get("params"), "params")

    _, array_params = _split_params(params)
    selected_axes = _selected_array_axes(meta, array_params)
    run_overrides = _run_overrides(config)
    task_overrides = _launcher_task_overrides(run_overrides, selected_axes, array_params)

    experiment = str(meta.get("experiment") or config_path.stem)
    python_exec, python_extra_args = _split_python_command(str(meta["python"]))
    entrypoint = str(meta.get("entrypoint", "train.py"))
    result_path = str(meta["result_path"])
    log_path = str(local.get("log", DEFAULT_LOCAL["log"]))
    cpus_per_task = int(local.get("cpus_per_task", DEFAULT_LOCAL["cpus_per_task"]))
    processes_per_gpu = int(local.get("processes_per_gpu", DEFAULT_LOCAL["processes_per_gpu"]))
    if processes_per_gpu < 1:
        raise ValueError("local.processes_per_gpu must be >= 1.")

    local_gpus = gpus if gpus is not None else _local_gpus(local, None)
    if len(local_gpus) == 0:
        raise ValueError("At least one GPU must be provided.")

    task_count = len(task_overrides)
    if not run_overrides:
        task_count = 1
        for key in selected_axes:
            task_count *= len(array_params[key])

    lines: list[str] = []
    lines.append("#!/bin/bash")
    lines.append("set -euo pipefail")
    lines.append("")

    lines.append(f"PYTHON_BIN={shlex.quote(python_exec)}")
    lines.append('if [[ -x ".venv/bin/python" ]]; then')
    lines.append('    if [[ -f ".venv/bin/activate" ]]; then')
    lines.append('        source ".venv/bin/activate"')
    lines.append("    fi")
    lines.append('    PYTHON_BIN=".venv/bin/python"')
    lines.append("fi")
    lines.append("")

    lines.append(f"THREADS={cpus_per_task}")
    lines.append("export OMP_NUM_THREADS=${THREADS}")
    lines.append("export MKL_NUM_THREADS=${THREADS}")
    lines.append("export OPENBLAS_NUM_THREADS=${THREADS}")
    lines.append("export VECLIB_MAXIMUM_THREADS=${THREADS}")
    lines.append("export NUMEXPR_NUM_THREADS=${THREADS}")
    lines.append('export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${THREADS}"')
    lines.append("")

    lines.append(f"RESULT_PATH={shlex.quote(result_path)}")
    lines.append(f"EXPERIMENT={shlex.quote(experiment)}")
    if meta.get("label") is not None:
        lines.append(f"LABEL={shlex.quote(str(meta['label']))}")
    lines.append(f"LOG_DIR={shlex.quote(log_path)}")
    lines.append('mkdir -p "${RESULT_PATH}" "${LOG_DIR}"')
    lines.append("")

    lines.append("GPUS=(" + " ".join(shlex.quote(gpu) for gpu in local_gpus) + ")")
    lines.append(f"PROCESSES_PER_GPU={processes_per_gpu}")
    lines.append('SLOTS=("${GPUS[@]}")')
    lines.append("if ((PROCESSES_PER_GPU > 1)); then")
    lines.append('    SLOTS=()')
    lines.append('    for gpu in "${GPUS[@]}"; do')
    lines.append("        for ((slot_idx = 0; slot_idx < PROCESSES_PER_GPU; slot_idx++)); do")
    lines.append('            SLOTS+=("${gpu}")')
    lines.append("        done")
    lines.append("    done")
    lines.append("fi")
    lines.append("MAX_PARALLEL=${#SLOTS[@]}")
    lines.append(f"TOTAL={task_count}")
    lines.append('echo "local_grid total=${TOTAL} gpus=${GPUS[*]} processes_per_gpu=${PROCESSES_PER_GPU}"')
    lines.append("")

    if task_overrides:
        lines.append("# Local-grid tasks from [[runs]].")
        lines.append("TASK_ARGS=(")
        for task in task_overrides:
            lines.append(f"    {shlex.quote(_format_cli_overrides(task))}")
        lines.append(")")
        lines.append("")
    elif selected_axes:
        lines.append("# Local-grid axes from config.")
        for key in selected_axes:
            array_name = _to_bash_name(key, "VALUES")
            shell_values = " ".join(shlex.quote(_to_shell_scalar(value)) for value in array_params[key])
            lines.append(f"{array_name}=({shell_values})")
        lines.append("")

        for key in selected_axes:
            n_name = _to_bash_name(key, "N")
            array_name = _to_bash_name(key, "VALUES")
            lines.append(f"{n_name}=${{#{array_name}[@]}}")
        lines.append("")

    lines.append("wait_for_batch() {")
    lines.append('    for pid in "${PIDS[@]}"; do')
    lines.append('        wait "${pid}"')
    lines.append("    done")
    lines.append("    PIDS=()")
    lines.append("}")
    lines.append("")

    lines.append("PIDS=()")
    lines.append("for ((TASK_ID = 0; TASK_ID < TOTAL; TASK_ID++)); do")
    lines.append("    SLOT_INDEX=$((TASK_ID % MAX_PARALLEL))")
    lines.append("    GPU=${SLOTS[${SLOT_INDEX}]}")
    if task_overrides:
        lines.append("    TASK_ARGS_VALUE=${TASK_ARGS[${TASK_ID}]}")
    elif selected_axes:
        lines.append("    idx=${TASK_ID}")
        for key in reversed(selected_axes):
            idx_name = _to_bash_name(key, "IDX").lower()
            n_name = _to_bash_name(key, "N")
            lines.append(f"    {idx_name}=$((idx % {n_name})); idx=$((idx / {n_name}))")
        for key in selected_axes:
            value_name = _to_bash_name(key, "VALUE")
            array_name = _to_bash_name(key, "VALUES")
            idx_name = _to_bash_name(key, "IDX").lower()
            lines.append(f"    {value_name}=${{{array_name}[${idx_name}]}}")
    lines.append('    LOG_FILE="${LOG_DIR}/task_${TASK_ID}.log"')

    echo_parts = ['    echo "local_task task_id=${TASK_ID} gpu=${GPU}']
    if task_overrides:
        echo_parts.append(' args=${TASK_ARGS_VALUE}')
    else:
        for key in selected_axes:
            echo_parts.append(f" {key}=${{{_to_bash_name(key, 'VALUE')}}}")
    echo_parts.append(' log=${LOG_FILE}"')
    lines.append("".join(echo_parts))

    cmd_parts = ['    CUDA_VISIBLE_DEVICES="${GPU}"', '"${PYTHON_BIN}"']
    cmd_parts.extend(shlex.quote(arg) for arg in python_extra_args)
    cmd_parts.append(shlex.quote(entrypoint))
    cmd_parts.append(shlex.quote(str(config_path)))
    cmd_parts.append('--path="${RESULT_PATH}"')
    cmd_parts.append('--experiment="${EXPERIMENT}"')
    if meta.get("label") is not None:
        cmd_parts.append('--label="${LABEL}"')
    if bool(meta.get("skip_existing", False)):
        cmd_parts.append("--skip-existing")
    cmd_parts.append("--skipeval")
    if task_overrides:
        cmd_parts.append("${TASK_ARGS_VALUE}")
    selected_axis_cmd_keys = [] if task_overrides else selected_axes
    for key in selected_axis_cmd_keys:
        cmd_parts.append(f'--{key}="${{{_to_bash_name(key, "VALUE")}}}"')
    lines.append(" \\\n        ".join(cmd_parts) + ' >"${LOG_FILE}" 2>&1 &')
    lines.append('    PIDS+=("$!")')
    lines.append("    if (( ${#PIDS[@]} == MAX_PARALLEL )); then")
    lines.append("        wait_for_batch")
    lines.append("    fi")
    lines.append("done")
    lines.append("if (( ${#PIDS[@]} > 0 )); then")
    lines.append("    wait_for_batch")
    lines.append("fi")
    lines.append('echo "local_grid_complete total=${TOTAL}"')
    lines.append("")
    return "\n".join(lines)


def _default_output_path(config_path: Path) -> Path:
    return Path("local") / f"{config_path.stem}.sh"


def _build_local_summary_lines(config: dict, config_path: Path, gpus: list[str]) -> list[str]:
    normalized_config = normalize_config(config)
    meta = _as_dict(normalized_config.get("meta"), "meta", default=DEFAULT_META)
    local = _as_dict(config.get("local"), "local", default=DEFAULT_LOCAL)
    params = _as_dict(normalized_config.get("params"), "params")

    _, array_params = _split_params(params)
    selected_axes = set(_selected_array_axes(meta, array_params))
    run_overrides = _run_overrides(config)
    task_overrides = _launcher_task_overrides(run_overrides, list(selected_axes), array_params)

    array_combination_count = 1
    for key in sorted(selected_axes):
        array_combination_count *= len(array_params[key])
    if task_overrides:
        array_combination_count = len(task_overrides)

    vmap_keys = sorted(key for key in array_params if key not in selected_axes)
    vmap_combination_count = 1
    for key in vmap_keys:
        vmap_combination_count *= len(array_params[key])

    lines: list[str] = []
    if run_overrides:
        lines.append(f"Run tables: {len(run_overrides)}")
    lines.append(f"Local tasks: {array_combination_count} process launches")
    if selected_axes:
        for key in sorted(selected_axes):
            lines.append(f"  - {key}: {_format_summary_value(array_params[key])}")

    lines.append(f"Per-process vmap sweep: {vmap_combination_count} combinations")
    if vmap_keys:
        for key in vmap_keys:
            lines.append(f"  - {key}: {_format_summary_value(array_params[key])}")

    lines.append(
        "Local resources: "
        f"gpus={','.join(gpus)}, processes-per-gpu={local['processes_per_gpu']}, "
        f"cpus-per-task={local['cpus_per_task']}"
    )
    if meta.get("array_vars") is None:
        lines.append("Local axes default to shape-changing sweep parameters, matching generate_sbatch.py.")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a local multi-GPU launcher for train.py sweeps.")
    parser.add_argument(
        "config",
        nargs="?",
        help="Config file path or config stem (e.g. wm_cost). Defaults to the most recently modified file in config/.",
    )
    parser.add_argument("-o", "--output", help="Optional output shell script path.")
    parser.add_argument("--gpus", help="Comma-separated GPU IDs for CUDA_VISIBLE_DEVICES, e.g. 0,1,2,3.")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the generated script after writing it.",
    )
    args = parser.parse_args()

    config_path = _resolve_config_path(args.config)
    with config_path.open("rb") as file:
        config = tomllib.load(file)

    local = _as_dict(config.get("local"), "local", default=DEFAULT_LOCAL)
    gpus = _local_gpus(local, args.gpus)
    script_text = _render_script(config, config_path=config_path, gpus=gpus)
    output_path = Path(args.output) if args.output else _default_output_path(config_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(script_text, encoding="utf-8")
    output_path.chmod(0o755)
    print(f"Wrote {output_path}")

    for line in _build_local_summary_lines(config, config_path, gpus):
        print(line)

    if args.run:
        subprocess.run(["bash", str(output_path)], check=True)


if __name__ == "__main__":
    main()
