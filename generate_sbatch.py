#!/usr/bin/env python3
import argparse
import math
import re
import shlex
import subprocess
import tomllib
from pathlib import Path

from modules.config import SHAPE_KEYS, normalize_config

DEFAULT_META = {
    "python": "python -u",
    "entrypoint": "train.py",
    "result_path": "./results",
}

DEFAULT_SBATCH = {
    "cpus_per_task": 1,
    "time": "08:00:00",
    "mem_per_cpu": "1G",
    "log": "./log/%A_%a",
}

def _as_dict(value, name: str, default: dict | None = None) -> dict:
    if value is None:
        return dict(default or {})
    if not isinstance(value, dict):
        raise ValueError(f"Expected [{name}] to be a table.")
    merged = dict(default or {})
    merged.update(value)
    return merged


def _is_scalar(value) -> bool:
    return isinstance(value, (bool, int, float, str))


def _to_shell_scalar(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isfinite(value):
            return repr(value)
        raise ValueError("Float values in config must be finite.")
    if isinstance(value, str):
        return value
    raise ValueError(f"Unsupported value type: {type(value).__name__}")


def _to_bash_name(key: str, suffix: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", key).strip("_").upper()
    if not token:
        token = "VALUE"
    if token[0].isdigit():
        token = f"V_{token}"
    return f"{token}_{suffix}"


def _split_python_command(command: str) -> tuple[str, list[str]]:
    tokens = shlex.split(command)
    if len(tokens) == 0:
        raise ValueError("meta.python must contain a python command.")
    return tokens[0], tokens[1:]


def _latest_config_path() -> Path:
    config_dir = Path("config")
    if not config_dir.exists() or not config_dir.is_dir():
        raise FileNotFoundError("No config provided and config/ directory was not found.")
    tomls = sorted(config_dir.glob("*.toml"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not tomls:
        raise FileNotFoundError("No config provided and no .toml files found in config/.")
    return tomls[0].resolve()


def _resolve_config_path(config_arg: str | None) -> Path:
    if config_arg is None:
        return _latest_config_path()

    raw = Path(config_arg)
    candidates: list[Path] = []

    if raw.suffix == ".toml":
        candidates.append(raw)
    else:
        candidates.append(raw.with_suffix(".toml"))
        candidates.append(raw)

    if not raw.is_absolute():
        stem = raw.stem if raw.suffix == ".toml" else raw.name
        candidates.append(Path("config") / f"{stem}.toml")

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    options = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(f"Could not find config file. Tried:\n{options}")


def _split_params(params: dict) -> tuple[dict[str, object], dict[str, list[object]]]:
    scalars: dict[str, object] = {}
    arrays: dict[str, list[object]] = {}
    for key, value in params.items():
        if isinstance(value, list):
            items = list(value)
            if len(items) == 0:
                raise ValueError(f"params.{key} must be a non-empty array.")
            for idx, item in enumerate(items):
                if not _is_scalar(item):
                    raise ValueError(f"params.{key}[{idx}] must be scalar, got {type(item).__name__}.")
            arrays[key] = items
            continue
        if isinstance(value, tuple):
            items = list(value)
            if len(items) == 0:
                raise ValueError(f"params.{key} tuple must be non-empty.")
            for idx, item in enumerate(items):
                if not _is_scalar(item):
                    raise ValueError(f"params.{key}[{idx}] must be scalar, got {type(item).__name__}.")
            scalars[key] = value
            continue
        if not _is_scalar(value):
            raise ValueError(f"params.{key} must be scalar or array, got {type(value).__name__}.")
        scalars[key] = value
    return scalars, arrays


def _selected_array_axes(meta: dict, array_params: dict[str, list[object]]) -> list[str]:
    axes: list[str] = [key for key in array_params if key in SHAPE_KEYS]

    array_vars_raw = meta.get("array_vars")
    if array_vars_raw is not None:
        if not isinstance(array_vars_raw, list):
            raise ValueError("meta.array_vars must be an array when provided.")
        for item in array_vars_raw:
            if not isinstance(item, str):
                raise ValueError("meta.array_vars entries must be strings.")
            if item not in array_params:
                raise ValueError(f"meta.array_vars contains '{item}', but params.{item} is not an array.")
            if item not in axes:
                axes.append(item)

    return axes


def _render_script(config: dict, config_path: Path) -> str:
    normalized_config = normalize_config(config)
    meta = _as_dict(normalized_config.get("meta"), "meta", default=DEFAULT_META)
    sbatch = _as_dict(config.get("sbatch"), "sbatch", default=DEFAULT_SBATCH)
    params = _as_dict(normalized_config.get("params"), "params")

    _, array_params = _split_params(params)
    selected_axes = _selected_array_axes(meta, array_params)

    experiment = str(meta.get("experiment") or config_path.stem)
    job_name = str(sbatch.get("job_name", experiment))
    python_exec, python_extra_args = _split_python_command(str(meta["python"]))
    entrypoint = str(meta.get("entrypoint", "train.py"))
    result_path = str(meta["result_path"])
    log_path = str(sbatch.get("log", "./log/%A_%a.log"))
    gpu_enabled = bool(sbatch.get("gpu", False))

    array_size = 1
    for key in selected_axes:
        array_size *= len(array_params[key])

    lines: list[str] = []
    lines.append("#!/bin/bash")
    lines.append(f"#SBATCH --job-name={job_name}")
    lines.append(f"#SBATCH --cpus-per-task={sbatch['cpus_per_task']}")
    lines.append(f"#SBATCH --time={sbatch['time']}")
    lines.append(f"#SBATCH --mem-per-cpu={sbatch['mem_per_cpu']}")
    lines.append(f"#SBATCH -e {log_path}")
    lines.append(f"#SBATCH -o {log_path}")
    if gpu_enabled:
        lines.append("#SBATCH --gres=gpu:1")
    for directive in sbatch.get("extra_directives", []):
        lines.append(f"#SBATCH {directive}")
    lines.append(f"#SBATCH --array=0-{array_size - 1}")
    lines.append("")

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

    if not gpu_enabled:
        lines.append("# Force CPU execution for JAX.")
        lines.append("export JAX_PLATFORMS=cpu")
    lines.append("THREADS=${SLURM_CPUS_PER_TASK:-1}")
    lines.append("export OMP_NUM_THREADS=${THREADS}")
    lines.append("export MKL_NUM_THREADS=${THREADS}")
    lines.append("export OPENBLAS_NUM_THREADS=${THREADS}")
    lines.append("export VECLIB_MAXIMUM_THREADS=${THREADS}")
    lines.append("export NUMEXPR_NUM_THREADS=${THREADS}")
    lines.append('export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${THREADS}"')
    lines.append("")

    lines.append(f"RESULT_PATH={shlex.quote(result_path)}")
    lines.append(f"EXPERIMENT={shlex.quote(experiment)}")
    lines.append('mkdir -p "${RESULT_PATH}"')
    lines.append("")

    if selected_axes:
        lines.append("# Slurm-array axes from config.")
        for key in selected_axes:
            array_name = _to_bash_name(key, "VALUES")
            shell_values = " ".join(shlex.quote(_to_shell_scalar(v)) for v in array_params[key])
            lines.append(f"{array_name}=({shell_values})")
        lines.append("")

        for key in selected_axes:
            n_name = _to_bash_name(key, "N")
            array_name = _to_bash_name(key, "VALUES")
            lines.append(f"{n_name}=${{#{array_name}[@]}}")
        n_terms = [_to_bash_name(key, "N") for key in selected_axes]
        lines.append(f"TOTAL=$(({' * '.join(n_terms)}))")
        lines.append("")

        lines.append("TASK_ID=${SLURM_ARRAY_TASK_ID}")
        lines.append("if ((TASK_ID < 0 || TASK_ID >= TOTAL)); then")
        lines.append('    echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_ID}; expected [0, $((TOTAL - 1))]"')
        lines.append("    exit 1")
        lines.append("fi")
        lines.append("")

        lines.append("idx=${TASK_ID}")
        for key in reversed(selected_axes):
            idx_name = _to_bash_name(key, "IDX").lower()
            n_name = _to_bash_name(key, "N")
            lines.append(f"{idx_name}=$((idx % {n_name})); idx=$((idx / {n_name}))")
        lines.append("")

        for key in selected_axes:
            value_name = _to_bash_name(key, "VALUE")
            array_name = _to_bash_name(key, "VALUES")
            idx_name = _to_bash_name(key, "IDX").lower()
            lines.append(f"{value_name}=${{{array_name}[${idx_name}]}}")
        lines.append("")

        log_parts = ['echo "grid_task task_id=${TASK_ID}']
        for key in selected_axes:
            log_parts.append(f" {key}=${{{_to_bash_name(key, 'VALUE')}}}")
        log_parts.append('"')
        lines.append("".join(log_parts))
    else:
        lines.append("TASK_ID=${SLURM_ARRAY_TASK_ID:-0}")

    lines.append("")

    cmd_parts = ['"${PYTHON_BIN}"']
    cmd_parts.extend(shlex.quote(arg) for arg in python_extra_args)
    cmd_parts.append(shlex.quote(entrypoint))
    cmd_parts.append(shlex.quote(str(config_path)))
    cmd_parts.append('--path="${RESULT_PATH}"')
    cmd_parts.append('--experiment="${EXPERIMENT}"')
    if bool(meta.get("skip_existing", False)):
        cmd_parts.append("--skip-existing")
    
    # TODO: wire this up!
    # if bool(meta.get("skipeval", False)):
    cmd_parts.append("--skipeval")
    for key in selected_axes:
        cmd_parts.append(f'--{key}="${{{_to_bash_name(key, "VALUE")}}}"')

    lines.append(" \\\n    ".join(cmd_parts))
    lines.append("")
    return "\n".join(lines)


def _default_output_path(config_path: Path) -> Path:
    return Path("sbatch") / f"{config_path.stem}.sbatch"


def _format_summary_value(values: list[object]) -> str:
    return "[" + ", ".join(_to_shell_scalar(value) for value in values) + "]"


def _build_job_summary_lines(config: dict, config_path: Path) -> list[str]:
    normalized_config = normalize_config(config)
    meta = _as_dict(normalized_config.get("meta"), "meta", default=DEFAULT_META)
    sbatch = _as_dict(config.get("sbatch"), "sbatch", default=DEFAULT_SBATCH)
    params = _as_dict(normalized_config.get("params"), "params")

    _, array_params = _split_params(params)
    selected_axes = set(_selected_array_axes(meta, array_params))

    array_combination_count = 1
    for key in sorted(selected_axes):
        array_combination_count *= len(array_params[key])

    vmap_keys = sorted(key for key in array_params if key not in selected_axes)
    vmap_combination_count = 1
    for key in vmap_keys:
        vmap_combination_count *= len(array_params[key])

    lines: list[str] = []
    lines.append(f"Array parameters: {array_combination_count} combinations")
    if selected_axes:
        for key in sorted(selected_axes):
            lines.append(f"  - {key}: {_format_summary_value(array_params[key])}")

    lines.append(f"Vmap parameters: {vmap_combination_count} combinations")
    if vmap_keys:
        for key in vmap_keys:
            lines.append(f"  - {key}: {_format_summary_value(array_params[key])}")

    resource_label = "gpu:1" if bool(sbatch.get("gpu", False)) else f"cpu:{sbatch['cpus_per_task']}"
    lines.append(
        "Resources: "
        f"{resource_label}, time={sbatch['time']}, mem-per-cpu={sbatch['mem_per_cpu']}"
    )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a lightweight sbatch script for train.py sweeps.")
    parser.add_argument("config", nargs="?", help="Config file path or config stem (e.g. wm_cost). Defaults to the most recently modified file in config/.")
    parser.add_argument("-o", "--output", help="Optional output sbatch path.")
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit the generated sbatch script with `sbatch`.",
    )
    args = parser.parse_args()

    config_path = _resolve_config_path(args.config)
    with config_path.open("rb") as file:
        config = tomllib.load(file)

    script_text = _render_script(config, config_path=config_path)
    output_path = Path(args.output) if args.output else _default_output_path(config_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(script_text, encoding="utf-8")
    output_path.chmod(0o755)
    print(f"Wrote {output_path}")

    if args.submit:
        for line in _build_job_summary_lines(config, config_path):
            print(line)
        submit_cmd = ["sbatch", str(output_path)]
        confirm = input("Submit job? [y/N] ").strip().lower()
        if confirm != "y":
            print("Cancelled.")
            return

        result = subprocess.run(
            submit_cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        stdout = result.stdout.strip()
        if stdout:
            print(stdout)


if __name__ == "__main__":
    main()
