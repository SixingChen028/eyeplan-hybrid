import argparse
import csv
import json
import math
import os
import sys
import tomllib

from modules.analysis_targets import get_summary_analysis_dir, resolve_analysis_target


EVAL_FIELDS = [
    "n_steps_mean",
    "n_steps_sd",
    "reward_mean",
    "reward_sd",
    "reward_no_cost_mean",
    "reward_no_cost_sd",
    "train_elapsed_seconds",
]


def _read_json(path: str) -> dict:
    with open(path, "r") as file:
        return json.load(file)


def _read_toml(path: str) -> dict:
    with open(path, "rb") as file:
        return tomllib.load(file)


def _varying_params_from_config(config_path: str) -> list[str]:
    config = _read_toml(config_path)
    if "params" not in config or not isinstance(config["params"], dict):
        raise ValueError(f"Config file must contain a [params] table: {config_path}")

    params = config["params"]
    varying = [
        key for key, value in params.items()
        if isinstance(value, list) and len(value) > 1
    ]
    return varying


def _resolve_experiment_and_runs(
    target: str,
    results_root: str,
    config_dir: str,
) -> tuple[str, list[str], str]:
    if target.endswith(".toml"):
        config_path = os.path.abspath(os.path.expanduser(target))
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        experiment = os.path.splitext(os.path.basename(config_path))[0]
        resolved = resolve_analysis_target(experiment, results_root=results_root)
        return experiment, resolved.run_dirs, config_path

    resolved = resolve_analysis_target(target, results_root=results_root)
    config_path = os.path.join(config_dir, f"{resolved.experiment}.toml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config not found for experiment '{resolved.experiment}': {config_path}"
        )
    return resolved.experiment, resolved.run_dirs, config_path


def _build_row(run_dir: str, varying_params: list[str], eval_file: str) -> dict:
    metadata_path = os.path.join(run_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Missing metadata file for run: {metadata_path}")

    eval_path = os.path.join(run_dir, eval_file)
    if not os.path.exists(eval_path):
        raise FileNotFoundError(f"Missing evaluation file for run: {eval_path}")

    metadata = _read_json(metadata_path)
    eval_summary = _read_json(eval_path)
    args = metadata.get("args")
    if not isinstance(args, dict):
        raise ValueError(f"metadata.json must contain an object at key 'args': {metadata_path}")

    row = {"run_id": os.path.basename(run_dir)}
    for param in varying_params:
        if param not in args:
            raise ValueError(f"Parameter '{param}' not found in metadata args for run: {run_dir}")
        row[param] = args[param]

    for field in EVAL_FIELDS:
        if field not in eval_summary:
            raise ValueError(f"Field '{field}' not found in eval summary for run: {run_dir}")
        row[field] = eval_summary[field]

    return row


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _format_value(value) -> str:
    return str(value)


def _format_mean_sd(values: list[float]) -> str:
    return f"{_mean(values):.3f} ± {_sample_sd(values):.3f}"


def _summary_table_rows(rows: list[dict], group_params: list[str]) -> list[dict]:
    groups = {}
    for row in rows:
        key = tuple(row[param] for param in group_params)
        groups.setdefault(key, []).append(row)

    summary_rows = []
    for key in sorted(groups):
        group_rows = groups[key]
        summary_row = {
            param: value
            for param, value in zip(group_params, key, strict=True)
        }
        summary_row["reward"] = _format_mean_sd(
            [float(row["reward_mean"]) for row in group_rows]
        )
        summary_row["n_steps"] = _format_mean_sd(
            [float(row["n_steps_mean"]) for row in group_rows]
        )
        summary_rows.append(summary_row)
    return summary_rows


def _print_aligned_table(rows: list[dict], columns: list[str]) -> None:
    table = [
        {column: _format_value(row[column]) for column in columns}
        for row in rows
    ]
    widths = {
        column: max(len(column), *(len(row[column]) for row in table))
        for column in columns
    }
    print("  ".join(column.rjust(widths[column]) for column in columns))
    for row in table:
        print("  ".join(row[column].rjust(widths[column]) for column in columns))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize an experiment into results/analysis/<experiment>/summary/evaluation.csv"
    )
    parser.add_argument(
        "target",
        type=str,
        help="Experiment target or config file path, e.g. apr24, apr24/*, apr24/run_id, config/apr24.toml",
    )
    parser.add_argument("--results_root", type=str, default=os.path.join(os.getcwd(), "results"))
    parser.add_argument("--config_dir", type=str, default=os.path.join(os.getcwd(), "config"))
    parser.add_argument("--eval_file", type=str, default="eval_summary_jax.json")
    args = parser.parse_args()

    experiment, run_dirs, config_path = _resolve_experiment_and_runs(
        target=args.target,
        results_root=args.results_root,
        config_dir=args.config_dir,
    )
    varying_params = _varying_params_from_config(config_path)
    if not run_dirs:
        raise FileNotFoundError(
            f"No run directories found for experiment '{experiment}' under '{args.results_root}'."
        )

    rows = []
    missing_eval_paths = []
    for run_dir in sorted(run_dirs):
        eval_path = os.path.join(run_dir, args.eval_file)
        if not os.path.exists(eval_path):
            missing_eval_paths.append(eval_path)
            print(f"Missing evaluation file for run: {eval_path}", file=sys.stderr)
            continue
        rows.append(
            _build_row(run_dir=run_dir, varying_params=varying_params, eval_file=args.eval_file)
        )

    missing_fraction = len(missing_eval_paths) / len(run_dirs)
    if missing_fraction > 0.5:
        print(
            f"Missing {len(missing_eval_paths)} of {len(run_dirs)} evaluation files; "
            "not writing summary output.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    output_dir = get_summary_analysis_dir(results_root=args.results_root, experiment=experiment)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "evaluation.csv")

    fieldnames = ["run_id"] + varying_params + EVAL_FIELDS
    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Runs summarized: {len(rows)}")
    print(f"Varying parameters: {', '.join(varying_params) if varying_params else '(none)'}")
    print(f"Wrote: {output_path}")

    group_params = [param for param in varying_params if param != "seed"]
    table_rows = _summary_table_rows(rows=rows, group_params=group_params)
    print()
    _print_aligned_table(table_rows, group_params + ["reward", "n_steps"])


if __name__ == "__main__":
    main()
