import argparse
import os
import sys


def _configure_jax_platform(argv: list[str], environ: dict[str, str]) -> None:
    if "--gpu" in argv[1:]:
        return

    environ["JAX_PLATFORMS"] = "cpu"
    environ["JAX_PLATFORM_NAME"] = "cpu"


_configure_jax_platform(sys.argv, os.environ)

from modules.evaluation import EVAL_SUMMARY_NAME, evaluate_run_dir
from modules.results_layout import resolve_analysis_target


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate eval_summary_jax.json for completed runs.")
    parser.add_argument(
        "targets",
        nargs="+",
        type=str,
        help="One or more targets: <experiment>, results/runs/<experiment>, or results/runs/<experiment>/<run_id>.",
    )
    parser.add_argument("--results_root", type=str, default=os.path.join(os.getcwd(), "results"))
    parser.add_argument("--eval_episodes", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--gpu", action="store_true", help="Allow JAX to use GPU devices.")
    args = parser.parse_args()

    run_dirs: list[str] = []
    seen: set[str] = set()
    for target in args.targets:
        resolved = resolve_analysis_target(target, results_root=args.results_root)
        for run_dir in resolved.run_dirs:
            if run_dir not in seen:
                seen.add(run_dir)
                run_dirs.append(run_dir)

    for run_dir in run_dirs:
        eval_summary_path = os.path.join(run_dir, EVAL_SUMMARY_NAME)
        if os.path.exists(eval_summary_path) and not args.overwrite:
            print(f"skip existing eval_summary={eval_summary_path}", flush=True)
            continue

        try:
            path, summary = evaluate_run_dir(
                run_dir,
                overwrite=args.overwrite,
                eval_episodes=args.eval_episodes,
                batch_size=args.batch_size,
            )
        except FileNotFoundError as error:
            print(f"skip incomplete run={run_dir} reason={error}", flush=True)
            continue
        print(
            "wrote_eval_summary "
            f"path={path} "
            f"episodes={summary['num_trials']} "
            f"reward_mean={summary['reward_mean']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
