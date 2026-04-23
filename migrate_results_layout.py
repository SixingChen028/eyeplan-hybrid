import argparse
import os
import re
import shutil
from dataclasses import dataclass


RESERVED_TOP_LEVEL = {"runs", "analysis", "archive"}
ARCHIVE_PREFIXES = ("exp", "tmp", "test", "smoke")
RUN_WITH_SUFFIX_RE = re.compile(r"^(?:(?P<prefix>.+)_)?(?P<timestamp>\d{8}_\d{6})_(?P<suffix>[a-z0-9]{4})$")
RUN_NO_SUFFIX_RE = re.compile(r"^(?:(?P<prefix>.+)_)?(?P<timestamp>\d{8}_\d{6})$")


@dataclass(frozen=True)
class MoveOp:
    src: str
    dst: str
    reason: str


def _to_abs(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _is_run_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "metadata.json"))


def _list_top_level_dirs(results_root: str) -> list[str]:
    dirs: list[str] = []
    for entry in os.scandir(results_root):
        if entry.is_dir():
            dirs.append(entry.path)
    return sorted(dirs)


def _classify_run_id(run_id: str) -> str:
    if RUN_WITH_SUFFIX_RE.match(run_id):
        return "with_suffix"
    if RUN_NO_SUFFIX_RE.match(run_id):
        return "no_suffix"
    return "other"


def _is_legacy_experiment_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    run_children = 0
    for entry in os.scandir(path):
        if entry.is_dir() and _is_run_dir(entry.path):
            run_children += 1
    return run_children > 0


def _archive_path(results_root: str, name: str) -> str:
    return os.path.join(results_root, "archive", name)


def _run_target_path(results_root: str, experiment: str, run_id: str) -> str:
    return os.path.join(results_root, "runs", experiment, run_id)


def _unique_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    base = path
    idx = 2
    while True:
        candidate = f"{base}__{idx}"
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def _append_move(ops: list[MoveOp], src: str, dst: str, reason: str) -> None:
    src_abs = _to_abs(src)
    dst_abs = _to_abs(dst)
    if src_abs == dst_abs:
        return
    ops.append(MoveOp(src=src_abs, dst=dst_abs, reason=reason))


def build_migration_plan(results_root: str) -> tuple[list[MoveOp], list[str]]:
    results_root = _to_abs(results_root)
    ops: list[MoveOp] = []
    skipped: list[str] = []

    for entry_path in _list_top_level_dirs(results_root):
        name = os.path.basename(entry_path)
        lower_name = name.lower()

        if name in RESERVED_TOP_LEVEL:
            continue

        if lower_name.startswith(ARCHIVE_PREFIXES):
            _append_move(
                ops,
                src=entry_path,
                dst=_archive_path(results_root, name),
                reason="prefix_archive_rule",
            )
            continue

        if _is_run_dir(entry_path):
            run_class = _classify_run_id(name)
            if run_class == "with_suffix":
                _append_move(
                    ops,
                    src=entry_path,
                    dst=_run_target_path(results_root, "default", name),
                    reason="legacy_default_run_with_suffix",
                )
            elif run_class == "no_suffix":
                _append_move(
                    ops,
                    src=entry_path,
                    dst=_archive_path(results_root, name),
                    reason="legacy_timestamp_without_suffix",
                )
            else:
                skipped.append(f"{entry_path} (unrecognized_run_name)")
            continue

        if _is_legacy_experiment_dir(entry_path):
            experiment = name
            for child in sorted(os.scandir(entry_path), key=lambda e: e.name):
                if not child.is_dir() or not _is_run_dir(child.path):
                    continue
                run_id = child.name
                run_class = _classify_run_id(run_id)
                if run_class == "with_suffix":
                    _append_move(
                        ops,
                        src=child.path,
                        dst=_run_target_path(results_root, experiment, run_id),
                        reason="legacy_experiment_run_with_suffix",
                    )
                elif run_class == "no_suffix":
                    _append_move(
                        ops,
                        src=child.path,
                        dst=_archive_path(results_root, os.path.join(experiment, run_id)),
                        reason="legacy_experiment_run_without_suffix",
                    )
                else:
                    _append_move(
                        ops,
                        src=child.path,
                        dst=_archive_path(results_root, os.path.join(experiment, run_id)),
                        reason="legacy_experiment_run_unrecognized_name",
                    )

            # If the old experiment container ends up empty after child migration, archive it.
            _append_move(
                ops,
                src=entry_path,
                dst=_archive_path(results_root, experiment),
                reason="legacy_experiment_container",
            )
            continue

        skipped.append(f"{entry_path} (unclassified_top_level)")

    return ops, skipped


def apply_migration_plan(ops: list[MoveOp], *, execute: bool) -> list[MoveOp]:
    resolved_ops: list[MoveOp] = []
    for op in ops:
        if not os.path.exists(op.src):
            continue
        final_dst = _unique_path(op.dst)
        resolved_op = MoveOp(src=op.src, dst=final_dst, reason=op.reason)
        resolved_ops.append(resolved_op)

        if not execute:
            continue

        os.makedirs(os.path.dirname(final_dst), exist_ok=True)
        shutil.move(op.src, final_dst)

    return resolved_ops


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate results directory into runs/archive structure.")
    parser.add_argument("--results_root", type=str, default=os.path.join(os.getcwd(), "results"))
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply moves. Without --apply, only print migration plan (dry run).",
    )
    args = parser.parse_args()

    plan, skipped = build_migration_plan(args.results_root)
    resolved = apply_migration_plan(plan, execute=args.apply)

    mode = "APPLY" if args.apply else "DRY_RUN"
    print(f"mode={mode}")
    print(f"results_root={_to_abs(args.results_root)}")
    print(f"moves={len(resolved)}")
    for op in resolved:
        print(f"[{op.reason}] {op.src} -> {op.dst}")

    print(f"skipped={len(skipped)}")
    for item in skipped:
        print(f"SKIP {item}")


if __name__ == "__main__":
    main()
