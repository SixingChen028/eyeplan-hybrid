#!/usr/bin/env python3
import argparse
import math
import re
import shlex
import tomllib
from pathlib import Path


def _as_dict(value, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"Expected [{name}] to be a table.")
    return value


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
    raise ValueError(f"Unsupported value type in config: {type(value).__name__}")


def _to_shell_arg_literal(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return _to_shell_scalar(value)
    if isinstance(value, str):
        return shlex.quote(value)
    raise ValueError(f"Unsupported arg value type: {type(value).__name__}")


def _to_bash_name(key: str, suffix: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", key).strip("_").upper()
    if not token:
        token = "VALUE"
    if token[0].isdigit():
        token = f"V_{token}"
    return f"{token}_{suffix}"


def _render_sbatch_script(config: dict) -> str:
    sbatch = _as_dict(config.get("sbatch"), "sbatch")
    runtime = _as_dict(config.get("runtime", {}), "runtime")
    train = _as_dict(config.get("train"), "train")
    grid = _as_dict(config.get("grid"), "grid")
    fixed_args = _as_dict(grid.get("fixed", {}), "grid.fixed")
    vary_args = _as_dict(grid.get("vary", {}), "grid.vary")

    for key in ("job_name", "cpus_per_task", "time", "mem_per_cpu"):
        if key not in sbatch:
            raise ValueError(f"Missing required key sbatch.{key}")

    for key in ("entrypoint",):
        if key not in train:
            raise ValueError(f"Missing required key train.{key}")

    vary_items = list(vary_args.items())
    for key, values in vary_items:
        if not isinstance(values, list) or len(values) == 0:
            raise ValueError(f"grid.vary.{key} must be a non-empty array.")

    python_cmd = str(train.get("python", "python -u"))
    entrypoint = str(train["entrypoint"])
    result_path = str(train.get("result_path", "./results"))
    experiment = str(train.get("experiment", "default"))
    resume = bool(train.get("resume", False))
    jobid_from_task_id = bool(train.get("jobid_from_task_id", True))

    set_euo = bool(runtime.get("set_euo_pipefail", True))
    force_jax_cpu = bool(runtime.get("force_jax_cpu", True))
    threads_from_slurm = bool(runtime.get("threads_from_slurm", True))

    n_terms = []
    for key, _ in vary_items:
        n_name = _to_bash_name(key, "N")
        n_terms.append(n_name)

    lines: list[str] = []
    lines.append("#!/bin/bash")
    lines.append(f"#SBATCH --job-name={sbatch['job_name']}")
    lines.append(f"#SBATCH --cpus-per-task={sbatch['cpus_per_task']}")
    lines.append(f"#SBATCH --time={sbatch['time']}")
    lines.append(f"#SBATCH --mem-per-cpu={sbatch['mem_per_cpu']}")
    if "error" in sbatch:
        lines.append(f"#SBATCH -e {sbatch['error']}")
    if "output" in sbatch:
        lines.append(f"#SBATCH -o {sbatch['output']}")
    for extra in sbatch.get("extra_directives", []):
        lines.append(f"#SBATCH {extra}")
    array_end = 1
    for _, values in vary_items:
        array_end *= len(values)
    lines.append(f"#SBATCH --array=0-{array_end - 1}")
    lines.append("")

    if set_euo:
        lines.append("set -euo pipefail")
        lines.append("")

    if force_jax_cpu:
        lines.append("# Force CPU execution for JAX.")
        lines.append("export JAX_PLATFORMS=cpu")
    if threads_from_slurm:
        lines.append("THREADS=${SLURM_CPUS_PER_TASK:-1}")
        lines.append("export OMP_NUM_THREADS=${THREADS}")
        lines.append("export MKL_NUM_THREADS=${THREADS}")
        lines.append("export OPENBLAS_NUM_THREADS=${THREADS}")
        lines.append("export VECLIB_MAXIMUM_THREADS=${THREADS}")
        lines.append("export NUMEXPR_NUM_THREADS=${THREADS}")
        lines.append('export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${THREADS}"')
    if force_jax_cpu or threads_from_slurm:
        lines.append("")

    lines.append(f"RESULT_PATH={shlex.quote(result_path)}")
    lines.append(f"EXPERIMENT={shlex.quote(experiment)}")
    lines.append('mkdir -p "${RESULT_PATH}"')
    lines.append("")

    if vary_items:
        lines.append("# Grid values from config.")
        for key, values in vary_items:
            arr_name = _to_bash_name(key, "VALUES")
            shell_values = " ".join(shlex.quote(_to_shell_scalar(v)) for v in values)
            lines.append(f"{arr_name}=({shell_values})")
        lines.append("")
        for key, _ in vary_items:
            n_name = _to_bash_name(key, "N")
            arr_name = _to_bash_name(key, "VALUES")
            lines.append(f"{n_name}=${{#{arr_name}[@]}}")

        if n_terms:
            lines.append(f"TOTAL=$(({' * '.join(n_terms)}))")
        else:
            lines.append("TOTAL=1")
        lines.append("")
        lines.append("TASK_ID=${SLURM_ARRAY_TASK_ID}")
        lines.append('if ((TASK_ID < 0 || TASK_ID >= TOTAL)); then')
        lines.append('    echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_ID}; expected [0, $((TOTAL - 1))]"')
        lines.append("    exit 1")
        lines.append("fi")
        lines.append("")
        lines.append("idx=${TASK_ID}")
        for key, _ in reversed(vary_items):
            idx_name = _to_bash_name(key, "IDX").lower()
            n_name = _to_bash_name(key, "N")
            lines.append(f"{idx_name}=$((idx % {n_name})); idx=$((idx / {n_name}))")
        lines.append("")
        for key, _ in vary_items:
            value_name = _to_bash_name(key, "VALUE")
            arr_name = _to_bash_name(key, "VALUES")
            idx_name = _to_bash_name(key, "IDX").lower()
            lines.append(f"{value_name}=${{{arr_name}[${idx_name}]}}")
        lines.append("")

        echo_parts = ['echo "grid_task task_id=${TASK_ID}']
        for key, _ in vary_items:
            value_name = _to_bash_name(key, "VALUE")
            echo_parts.append(f" {key}=${{{value_name}}}")
        echo_parts.append('"')
        lines.append("".join(echo_parts))
        lines.append("")
    else:
        lines.append("TASK_ID=${SLURM_ARRAY_TASK_ID:-0}")
        lines.append("")

    lines.append(f"{python_cmd} {shlex.quote(entrypoint)} \\")

    arg_lines: list[str] = []
    if jobid_from_task_id:
        arg_lines.append('    --jobid="${TASK_ID}" \\')
    arg_lines.append('    --path="${RESULT_PATH}" \\')
    arg_lines.append('    --experiment="${EXPERIMENT}" \\')
    if resume:
        arg_lines.append("    --resume=true \\")
    else:
        arg_lines.append("    --resume=false \\")

    for key, value in fixed_args.items():
        literal = _to_shell_arg_literal(value)
        arg_lines.append(f"    --{key}={literal} \\")

    for key, _ in vary_items:
        value_name = _to_bash_name(key, "VALUE")
        arg_lines.append(f'    --{key}="${{{value_name}}}" \\')

    if arg_lines:
        arg_lines[-1] = arg_lines[-1][:-2]
    lines.extend(arg_lines)
    lines.append("")
    return "\n".join(lines)


def _default_output_path(config_path: Path) -> Path:
    if config_path.suffix == ".toml":
        return config_path.with_suffix(".sbatch")
    return config_path.with_name(config_path.name + ".sbatch")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an sbatch script from a TOML config."
    )
    parser.add_argument("config", help="Path to TOML config file.")
    parser.add_argument(
        "-o",
        "--output",
        help="Output sbatch path (default: same stem as TOML, .sbatch).",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("rb") as file:
        config = tomllib.load(file)

    script_text = _render_sbatch_script(config)
    output_path = Path(args.output) if args.output else _default_output_path(config_path)
    output_path.write_text(script_text, encoding="utf-8")
    output_path.chmod(0o755)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
