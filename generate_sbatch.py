#!/usr/bin/env python3
import argparse
import math
import re
import shlex
import subprocess
import tomllib
from pathlib import Path


DEFAULT_META = {
    "python": "python -u",
    "entrypoint": "train.py",
    "result_path": "./results",
    "resume": True,
    "jobid_from_task_id": True,
    "eval_episodes": 102_400,
    "print_frequency": 100,
    "parallel": False,
    "array_vars": [],
    "parallel_config_arg": "path",
    "post_simulate": None,
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


def _to_arg_literal(value) -> str:
    if isinstance(value, str):
        return shlex.quote(value)
    return _to_shell_scalar(value)


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


def _resolve_config_path(config_arg: str) -> Path:
    raw = Path(config_arg)
    candidates: list[Path] = []

    if raw.suffix == ".toml":
        candidates.append(raw)
    else:
        candidates.append(raw.with_suffix(".toml"))
        candidates.append(raw)

    if not raw.is_absolute():
        stem = raw.stem if raw.suffix == ".toml" else raw.name
        candidates.extend(
            [
                Path("config") / f"{stem}.toml",
            ]
        )

    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    options = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(f"Could not find config file. Tried:\n{options}")


def _split_params(params: dict) -> tuple[list[tuple[str, object]], list[tuple[str, list[object]]]]:
    fixed: list[tuple[str, object]] = []
    vary: list[tuple[str, list[object]]] = []
    for key, value in params.items():
        if isinstance(value, list):
            if len(value) == 0:
                raise ValueError(f"params.{key} must be a non-empty array.")
            for idx, item in enumerate(value):
                if not _is_scalar(item):
                    raise ValueError(f"params.{key}[{idx}] must be scalar, got {type(item).__name__}.")
            vary.append((key, value))
            continue
        if not _is_scalar(value):
            raise ValueError(f"params.{key} must be scalar or array, got {type(value).__name__}.")
        fixed.append((key, value))
    return fixed, vary


def _render_script(config: dict, config_path: Path) -> str:
    meta = _as_dict(config.get("meta"), "meta", default=DEFAULT_META)
    sbatch = _as_dict(config.get("sbatch"), "sbatch", default=DEFAULT_SBATCH)
    params = _as_dict(config.get("params"), "params")
    gpu_enabled = bool(sbatch.get("gpu", False))

    experiment = str(meta.get("experiment", config_path.stem))
    job_name = str(sbatch.get("job_name", experiment))
    python_exec, python_extra_args = _split_python_command(str(meta["python"]))
    log_path = str(sbatch.get("log", sbatch.get("output", sbatch.get("error", "./log/%A_%a.log"))))

    fixed, vary = _split_params(params)
    vary_by_key = {key: values for key, values in vary}
    parallel_mode = bool(meta.get("parallel", False))
    array_vars_raw = meta.get("array_vars", [])
    if array_vars_raw is None:
        array_vars_raw = []
    if not isinstance(array_vars_raw, list):
        raise ValueError("meta.array_vars must be an array of parameter names.")
    array_vars = [str(item) for item in array_vars_raw]
    invalid_array_vars = [key for key in array_vars if key not in vary_by_key]
    if invalid_array_vars:
        invalid_keys = ", ".join(invalid_array_vars)
        raise ValueError(f"meta.array_vars contains keys that are not sweep params: {invalid_keys}")
    if parallel_mode and not array_vars:
        raise ValueError("meta.parallel=true requires meta.array_vars to be non-empty.")

    if parallel_mode:
        selected_vary = [(key, vary_by_key[key]) for key in array_vars]
        remaining_vary = [(key, values) for key, values in vary if key not in set(array_vars)]
    else:
        selected_vary = list(vary)
        remaining_vary = []

    entrypoint = str(meta["entrypoint"])
    if parallel_mode and entrypoint == DEFAULT_META["entrypoint"]:
        entrypoint = "train.py"
    if meta.get("post_simulate") is None:
        post_simulate = not parallel_mode
    else:
        post_simulate = bool(meta["post_simulate"])
    n_terms = [_to_bash_name(key, "N") for key, _ in selected_vary]
    array_size = 1
    for _, values in selected_vary:
        array_size *= len(values)

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

    lines.append(f"RESULT_PATH={shlex.quote(str(meta['result_path']))}")
    lines.append(f"EXPERIMENT={shlex.quote(experiment)}")
    lines.append('mkdir -p "${RESULT_PATH}"')
    lines.append("")

    if selected_vary:
        lines.append("# Grid values from config.")
        for key, values in selected_vary:
            array_name = _to_bash_name(key, "VALUES")
            shell_values = " ".join(shlex.quote(_to_shell_scalar(v)) for v in values)
            lines.append(f"{array_name}=({shell_values})")
        lines.append("")

        for key, _ in selected_vary:
            n_name = _to_bash_name(key, "N")
            array_name = _to_bash_name(key, "VALUES")
            lines.append(f"{n_name}=${{#{array_name}[@]}}")
        lines.append(f"TOTAL=$(({' * '.join(n_terms)}))")
        lines.append("")

        lines.append("TASK_ID=${SLURM_ARRAY_TASK_ID}")
        lines.append("if ((TASK_ID < 0 || TASK_ID >= TOTAL)); then")
        lines.append('    echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_ID}; expected [0, $((TOTAL - 1))]"')
        lines.append("    exit 1")
        lines.append("fi")
        lines.append("")

        lines.append("idx=${TASK_ID}")
        for key, _ in reversed(selected_vary):
            idx_name = _to_bash_name(key, "IDX").lower()
            n_name = _to_bash_name(key, "N")
            lines.append(f"{idx_name}=$((idx % {n_name})); idx=$((idx / {n_name}))")
        lines.append("")

        for key, _ in selected_vary:
            value_name = _to_bash_name(key, "VALUE")
            array_name = _to_bash_name(key, "VALUES")
            idx_name = _to_bash_name(key, "IDX").lower()
            lines.append(f"{value_name}=${{{array_name}[${idx_name}]}}")
        lines.append("")

        log_parts = ['echo "grid_task task_id=${TASK_ID}']
        for key, _ in selected_vary:
            log_parts.append(f" {key}=${{{_to_bash_name(key, 'VALUE')}}}")
        log_parts.append('"')
        lines.append("".join(log_parts))
    else:
        lines.append("TASK_ID=${SLURM_ARRAY_TASK_ID:-0}")
    lines.append("")

    python_cmd_parts = ['"${PYTHON_BIN}"']
    python_cmd_parts.extend(shlex.quote(arg) for arg in python_extra_args)
    python_cmd_parts.append(shlex.quote(entrypoint))
    lines.append(" ".join(python_cmd_parts) + " \\")
    if parallel_mode:
        config_arg_mode = str(meta.get("parallel_config_arg", "path")).lower()
        if config_arg_mode == "stem":
            parallel_config_arg = config_path.stem
        elif config_arg_mode == "path":
            parallel_config_arg = str(config_path)
        else:
            raise ValueError("meta.parallel_config_arg must be 'path' or 'stem'.")
        lines.append(f"    {shlex.quote(parallel_config_arg)} \\")
        lines.append('    --path="${RESULT_PATH}" \\')
        lines.append('    --experiment="${EXPERIMENT}" \\')
        for key, _ in selected_vary:
            lines.append(f'    --{key}="${{{_to_bash_name(key, "VALUE")}}}" \\')
    else:
        if bool(meta["jobid_from_task_id"]):
            lines.append('    --jobid="${TASK_ID}" \\')
        lines.append('    --path="${RESULT_PATH}" \\')
        lines.append('    --experiment="${EXPERIMENT}" \\')
        lines.append(f"    --resume={_to_shell_scalar(bool(meta['resume']))} \\")
        lines.append(f"    --eval_episodes={_to_arg_literal(meta['eval_episodes'])} \\")
        lines.append(f"    --print_frequency={_to_arg_literal(meta['print_frequency'])} \\")

        for key, value in fixed:
            lines.append(f"    --{key}={_to_arg_literal(value)} \\")
        for key, _ in selected_vary:
            lines.append(f'    --{key}="${{{_to_bash_name(key, "VALUE")}}}" \\')
        for key, _ in remaining_vary:
            lines.append(f"    --{key}={_to_arg_literal(vary_by_key[key])} \\")

    if lines[-1].endswith(" \\"):
        lines[-1] = lines[-1][:-2]
    lines.append("")
    if post_simulate:
        lines.append('RUN=$(ls -td "${RESULT_PATH}/runs/${EXPERIMENT}/${TASK_ID}"_* | head -n 1)')
        lines.append('"${PYTHON_BIN}" simulate.py "${RUN}"')
        lines.append('"${PYTHON_BIN}" simulate.py "${RUN}" --detailed')
        lines.append("")
    return "\n".join(lines)


def _default_output_path(config_path: Path) -> Path:
    return Path("sbatch") / f"{config_path.stem}.sbatch"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an sbatch script from TOML.")
    parser.add_argument("config", help="Config file path or config stem (e.g. hybrid-dyna).")
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
        result = subprocess.run(
            ["sbatch", str(output_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        stdout = result.stdout.strip()
        if stdout:
            print(stdout)


if __name__ == "__main__":
    main()
